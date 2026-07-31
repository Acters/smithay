//! Vulkan-assisted cross-GPU copy bridge.
//!
//! Some driver stacks (notably proprietary NVIDIA) cannot share dmabufs with other
//! vendors through GL/EGL: the render GPU can only render into its own tiled
//! modifiers, which foreign GPUs cannot import, and it cannot import linear or
//! foreign-tiled buffers either. In that situation smithay's multigpu renderer
//! falls back to a cpu-copy (glReadPixels + upload) for every frame.
//!
//! Vulkan transfer operations have no such limitation: a foreign dmabuf can be
//! imported via `VK_EXT_external_memory_dmabuf`, copied on-GPU into a LINEAR image,
//! and re-exported as a dmabuf which other vendors (e.g. Mesa/i915) can import.
//! This module implements that copy. It is used as a fast path between the direct
//! dmabuf share and the cpu-copy fallback.
//!
//! Synchronization is the caller's responsibility for now: the source buffer's
//! rendering must have completed before [`VkBridge::copy_to_linear`] is invoked
//! (e.g. by blocking on the render [`crate::backend::renderer::sync::SyncPoint`]),
//! and this bridge blocks (`vkQueueWaitIdle`) before returning the exported dmabuf.

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

use ash::{ext, khr, vk};
use drm_fourcc::DrmModifier;
use tracing::{debug, info, warn};

use crate::backend::{
    allocator::{
        dmabuf::{Dmabuf, DmabufFlags},
        vulkan::format::get_vk_format,
        Buffer,
    },
    drm::DrmNode,
};

// ---------------------------------------------------------------------------
// Early (pre-DRM-master) initialization support.
//
// On proprietary NVIDIA, creating a vulkan instance from *inside* a compositor
// process that already holds DRM master deadlocks the ICD. Initializing on a
// background thread spawned before the compositor opens its DRM devices avoids
// this. niri (or another compositor) should call [`preinit`] as early as
// possible in main(); the multigpu renderer picks the result up later.
// ---------------------------------------------------------------------------

type InitResult = Result<VkBridge, VkBridgeError>;
static PREINIT: std::sync::Mutex<Option<std::sync::mpsc::Receiver<InitResult>>> =
    std::sync::Mutex::new(None);

/// Spawn the bridge initialization thread early. `vendor_id` selects the
/// physical device to use (e.g. 0x10de for NVIDIA); pass `None` to use the
/// first device that reports a render node.
pub fn preinit(vendor_id: Option<u32>) {
    let mut guard = PREINIT.lock().unwrap();
    if guard.is_some() {
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("vkbridge-init".into())
        .spawn(move || {
            let _ = tx.send(VkBridge::new_for_vendor(vendor_id));
        })
        .expect("failed to spawn vkbridge init thread");
    *guard = Some(rx);
}

/// Take the pre-initialization receiver, if [`preinit`] was called.
pub fn take_preinit() -> Option<std::sync::mpsc::Receiver<InitResult>> {
    PREINIT.lock().unwrap().take()
}

/// Error type for [`VkBridge`] operations.
#[derive(Debug, thiserror::Error)]
pub enum VkBridgeError {
    /// Vulkan instance/device setup failed
    #[error("vulkan setup failed: {0}")]
    Setup(String),
    /// a vulkan call failed
    #[error("vulkan error: {0:?}")]
    Vk(vk::Result),
    /// the source dmabuf cannot be bridged (multi-plane or unknown format)
    #[error("unsupported dmabuf for bridging")]
    Unsupported,
    /// failed to build the exported dmabuf
    #[error("failed to build exported dmabuf")]
    Export,
}

impl From<vk::Result> for VkBridgeError {
    fn from(err: vk::Result) -> Self {
        VkBridgeError::Vk(err)
    }
}

/// Vulkan copy engine for a single (render) DRM node.
///
/// Cheap to keep around once created; all heavy objects are allocated per copy and
/// released immediately, the exported dmabuf fd owns the underlying memory.
/// Thread-safe vulkan handles shared between the compositor thread and the
/// copy worker. All command submission happens on the worker.
/// Per-frame reusable vulkan state. The staging dmabuf is persistent across
/// frames, so its import image is created once and reused; creating/importing
/// images every frame caused heavy nvidia driver lock contention.
#[derive(Default)]
struct CopyState {
    cmd: Option<vk::CommandBuffer>,
    /// (fd, modifier, format, w, h) of the currently imported source
    src_key: Option<(i32, u64, vk::Format, u32, u32)>,
    src_image: Option<(vk::Image, vk::DeviceMemory)>,
    /// Reusable destination (linear, exportable) images. Ring of 3: the
    /// consumer on the target gpu is at most ~2 frames behind, so reuse is
    /// safe. Avoids per-frame create/alloc/free churn in the nvidia driver.
    dst_key: Option<(vk::Format, u32, u32)>,
    dst_ring: Vec<(vk::Image, vk::DeviceMemory)>,
    dst_index: usize,
    mem_type_cache: std::collections::HashMap<u32, u32>,
}

/// Thread-safe vulkan handles shared between the compositor thread and the
/// copy worker. All command submission happens on the worker.
struct BridgeCore {
    entry: ash::Entry,
    instance: ash::Instance,
    phd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    cmd_pool: vk::CommandPool,
    get_memory_fd: khr::external_memory_fd::Device,
    state: std::sync::Mutex<CopyState>,
}

impl Drop for BridgeCore {
    fn drop(&mut self) {
        unsafe {
            let mut state = self.state.lock().unwrap();
            if let Some((img, mem)) = state.src_image.take() {
                self.device.destroy_image(img, None);
                self.device.free_memory(mem, None);
            }
            if let Some(cmd) = state.cmd.take() {
                self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            }
            for (img, mem) in state.dst_ring.drain(..) {
                self.device.destroy_image(img, None);
                self.device.free_memory(mem, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

struct CopyJob {
    src: Dmabuf,
    sync: crate::backend::renderer::sync::SyncPoint,
    /// Exported native fence fd for the render sync point. The worker polls
    /// it instead of calling into EGL's client wait, which spins a cpu core
    /// on the proprietary nvidia driver.
    wait_fd: Option<std::os::fd::OwnedFd>,
    regions: Vec<crate::utils::Rectangle<i32, crate::utils::Buffer>>,
}

/// Vulkan copy engine for a single (render) DRM node.
///
/// Copies run on a background worker thread: the compositor submits jobs
/// (staging dmabuf + render sync point) and picks up the latest completed
/// linear dmabuf a frame later. No cross-driver fence fds are used — the
/// nvidia proprietary driver cannot export pollable fence fds nor import
/// EGL fence fds — and the compositor never blocks on the copy, at the cost
/// of one frame of latency on the target output.
pub struct VkBridge {
    core: std::sync::Arc<BridgeCore>,
    sender: std::sync::mpsc::SyncSender<CopyJob>,
    latest: std::sync::Arc<std::sync::Mutex<Option<Dmabuf>>>,
    /// Number of submitted copy jobs (shared with the worker's completed
    /// counter). Used to keep the compositor from rendering into the staging
    /// buffer while the worker is still copying it.
    submitted: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    completed: std::sync::Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl BridgeCore {
    /// Import (or reuse the cached import of) the staging dmabuf as a vulkan
    /// image. The staging buffer is persistent, so this almost always hits
    /// the cache; the per-frame create/import/destroy churn contended badly
    /// with GL rendering inside the nvidia driver.
    fn ensure_src_image(&self, src: &Dmabuf, vk_format: vk::Format) -> Result<vk::Image, VkBridgeError> {
        let (w, h) = (src.size().w as u32, src.size().h as u32);
        let src_modifier = src.format().modifier;
        let src_fd = src.handles().next().ok_or(VkBridgeError::Unsupported)?;
        let key = (src_fd.as_raw_fd(), src_modifier.into(), vk_format, w, h);

        // cache check WITHOUT holding the lock across driver calls; mem_type
        // locks the same mutex internally and would self-deadlock otherwise
        let cached = { self.state.lock().unwrap().src_key };
        if cached == Some(key) {
            return Ok(self.state.lock().unwrap().src_image.unwrap().0);
        }
        {
            let mut state = self.state.lock().unwrap();
            if let Some((img, mem)) = state.src_image.take() {
                unsafe {
                    self.device.destroy_image(img, None);
                    self.device.free_memory(mem, None);
                }
            }
        }

        let src_modifiers = [src_modifier.into()];
        let mut modifier_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&src_modifiers);
        let mut src_external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .push_next(&mut modifier_list)
            .push_next(&mut src_external)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk_format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let img = unsafe { self.device.create_image(&info, None)? };

        let req = unsafe { self.device.get_image_memory_requirements(img) };
        let dup_fd = unsafe { libc::dup(src_fd.as_raw_fd()) };
        if dup_fd < 0 {
            unsafe { self.device.destroy_image(img, None) };
            return Err(VkBridgeError::Setup("failed to dup dmabuf fd".into()));
        }
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(dup_fd);
        let mem_type = self
            .mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
            .ok_or(VkBridgeError::Setup("no memory type for src import".into()))?;
        let alloc = vk::MemoryAllocateInfo::default()
            .push_next(&mut import_info)
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let mem = match unsafe { self.device.allocate_memory(&alloc, None) } {
            Ok(mem) => mem,
            Err(err) => {
                unsafe {
                    libc::close(dup_fd);
                    self.device.destroy_image(img, None);
                }
                return Err(err.into());
            }
        };
        unsafe { self.device.bind_image_memory(img, mem, 0)? };

        debug!("vkbridge: imported staging dmabuf (new cache entry)");
        let mut state = self.state.lock().unwrap();
        state.src_key = Some(key);
        state.src_image = Some((img, mem));
        Ok(img)
    }

    fn copy_to_linear_blocking(
        &self,
        src: &Dmabuf,
        regions: &[crate::utils::Rectangle<i32, crate::utils::Buffer>],
    ) -> Result<Dmabuf, VkBridgeError> {
        if src.num_planes() != 1 {
            return Err(VkBridgeError::Unsupported);
        }
        let format = src.format().code;
        let vk_format = get_vk_format(format).ok_or(VkBridgeError::Unsupported)?;
        let (w, h) = (src.size().w as u32, src.size().h as u32);
        let src_modifier = src.format().modifier;
        let src_stride = src.strides().next().ok_or(VkBridgeError::Unsupported)?;
        let src_offset = src.offsets().next().ok_or(VkBridgeError::Unsupported)?;
        let src_fd = src.handles().next().ok_or(VkBridgeError::Unsupported)?;
        // stride/offset are not passed to vulkan: the layout is fully described by
        // the drm format modifier used for the import.
        let _ = (src_stride, src_offset);

        // -- destination image: reuse a ring slot (created on demand)
        const DST_RING_SIZE: usize = 3;
        let (dst, dst_mem) = {
            // never call into the driver while holding the state lock:
            // mem_type() locks the same mutex and would self-deadlock
            {
                let mut state = self.state.lock().unwrap();
                if state.dst_key != Some((vk_format, w, h)) {
                    for (img, mem) in state.dst_ring.drain(..) {
                        unsafe {
                            self.device.destroy_image(img, None);
                            self.device.free_memory(mem, None);
                        }
                    }
                    state.dst_key = Some((vk_format, w, h));
                    state.dst_index = 0;
                }
            }
            let idx = {
                let mut state = self.state.lock().unwrap();
                let idx = state.dst_index;
                state.dst_index = (idx + 1) % DST_RING_SIZE;
                idx
            };
            let existing = { self.state.lock().unwrap().dst_ring.get(idx).copied() };
            if let Some((img, mem)) = existing {
                (img, mem)
            } else {
                let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let info = vk::ImageCreateInfo::default()
                    .push_next(&mut external_memory_info)
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk_format)
                    .extent(vk::Extent3D { width: w, height: h, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::LINEAR)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::SAMPLED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                let img = unsafe { self.device.create_image(&info, None)? };
                let req = unsafe { self.device.get_image_memory_requirements(img) };
                let mut export_alloc = vk::ExportMemoryAllocateInfo::default()
                    .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
                let mem_type = self
                    .mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::empty())
                    .ok_or(VkBridgeError::Setup("no memory type for dst".into()))?;
                let alloc = vk::MemoryAllocateInfo::default()
                    .push_next(&mut export_alloc)
                    .allocation_size(req.size)
                    .memory_type_index(mem_type);
                let mem = unsafe { self.device.allocate_memory(&alloc, None)? };
                unsafe { self.device.bind_image_memory(img, mem, 0)? };
                self.state.lock().unwrap().dst_ring.push((img, mem));
                (img, mem)
            }
        };

        let _ = vk_format; // layout is fully described by the drm modifier
        let src_img = self.ensure_src_image(src, vk_format)?;
        self.copy_inner(src, src_img, dst, dst_mem, regions)
    }

    fn copy_inner(
        &self,
        src: &Dmabuf,
        src_img: vk::Image,
        dst: vk::Image,
        dst_mem: vk::DeviceMemory,
        regions: &[crate::utils::Rectangle<i32, crate::utils::Buffer>],
    ) -> Result<Dmabuf, VkBridgeError> {
        let (w, h) = (src.size().w as u32, src.size().h as u32);
        let format = src.format().code;
        let flags = DmabufFlags::empty();

        // record and submit the copy (reusing the pooled command buffer)
        let cmd = {
            let mut state = self.state.lock().unwrap();
            match state.cmd {
                Some(cmd) => {
                    unsafe { self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())? };
                    cmd
                }
                None => {
                    let cmd = unsafe {
                        self.device.allocate_command_buffers(
                            &vk::CommandBufferAllocateInfo::default()
                                .command_pool(self.cmd_pool)
                                .level(vk::CommandBufferLevel::PRIMARY)
                                .command_buffer_count(1),
                        )?[0]
                    };
                    state.cmd = Some(cmd);
                    cmd
                }
            }
        };
        let submit_result = (|| -> Result<(), VkBridgeError> {
            unsafe {
                self.device.begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?;
                let barriers = [
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                        .dst_queue_family_index(self.queue_family)
                        .image(src_img)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    vk::ImageMemoryBarrier::default()
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                        .dst_queue_family_index(self.queue_family)
                        .image(dst)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                ];
                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &barriers,
                );
                let subresource = vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let copy_regions: Vec<vk::ImageCopy> = if regions.is_empty() {
                    vec![vk::ImageCopy::default()
                        .src_subresource(subresource)
                        .dst_subresource(subresource)
                        .extent(vk::Extent3D { width: w, height: h, depth: 1 })]
                } else {
                    regions
                        .iter()
                        .map(|rect| {
                            vk::ImageCopy::default()
                                .src_subresource(subresource)
                                .src_offset(vk::Offset3D { x: rect.loc.x, y: rect.loc.y, z: 0 })
                                .dst_subresource(subresource)
                                .dst_offset(vk::Offset3D { x: rect.loc.x, y: rect.loc.y, z: 0 })
                                .extent(vk::Extent3D {
                                    width: rect.size.w as u32,
                                    height: rect.size.h as u32,
                                    depth: 1,
                                })
                        })
                        .collect()
                };
                self.device.cmd_copy_image(
                    cmd,
                    src_img,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copy_regions,
                );
                self.device.end_command_buffer(cmd)?;
                let cmds = [cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                self.device.queue_submit(self.queue, &[submit], vk::Fence::null())?;
                self.device.queue_wait_idle(self.queue)?;
            }
            Ok(())
        })();

        submit_result?;

        // export dst as dmabuf
        let fd = unsafe {
            self.get_memory_fd.get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(dst_mem)
                    .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
            )
        }?;
        let subresource = vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR);
        let layout = unsafe { self.device.get_image_subresource_layout(dst, subresource) };
        // the ring keeps the image and memory alive across frames

        let mut builder = Dmabuf::builder(
            src.size(),
            format,
            DrmModifier::Linear,
            flags,
        );
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        if !builder.add_plane(owned, 0, layout.offset as u32, layout.row_pitch as u32) {
            return Err(VkBridgeError::Export);
        }
        let dmabuf = builder.build().ok_or(VkBridgeError::Export)?;
        Ok(dmabuf)
    }

    fn mem_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Option<u32> {
        if let Some(&cached) = self.state.lock().unwrap().mem_type_cache.get(&bits) {
            return Some(cached);
        }
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.phd) };
        let found = (0..props.memory_type_count).find(|&i| {
            (bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(required)
        });
        if let Some(idx) = found {
            self.state.lock().unwrap().mem_type_cache.insert(bits, idx);
        }
        found
    }
}

fn worker_main(
    core: std::sync::Arc<BridgeCore>,
    receiver: std::sync::mpsc::Receiver<CopyJob>,
    latest: std::sync::Arc<std::sync::Mutex<Option<Dmabuf>>>,
    completed: std::sync::Arc<(std::sync::Mutex<usize>, std::sync::Condvar)>,
) {
    debug!("vkbridge: copy worker started");
    while let Ok(job) = receiver.recv() {
        match &job.wait_fd {
            Some(fd) => {
                // poll the exported fence fd: sleeps in the kernel, unlike
                // the nvidia EGL client wait which spins
                let mut pfd = libc::pollfd {
                    fd: fd.as_raw_fd(),
                    events: libc::POLLIN,
                    revents: 0,
                };
                let ret = unsafe { libc::poll(&mut pfd, 1, 500) };
                if ret < 0 {
                    warn!("vkbridge: fence poll failed: {}", std::io::Error::last_os_error());
                    continue;
                }
            }
            None => {
                if let Err(err) = job.sync.wait() {
                    warn!("vkbridge: render sync interrupted: {err:?}");
                    continue;
                }
            }
        }
        match core.copy_to_linear_blocking(&job.src, &job.regions) {
            Ok(dmabuf) => {
                *latest.lock().unwrap() = Some(dmabuf);
            }
            Err(err) => warn!("vkbridge: copy failed: {err}"),
        }
        // the staging buffer is fully consumed; unblock the compositor
        let (lock, cvar) = &*completed;
        *lock.lock().unwrap() += 1;
        cvar.notify_all();
    }
    debug!("vkbridge: copy worker exiting");
}

impl VkBridge {
    /// Create a bridge for the given DRM (render) node.
    ///
    /// Returns an error if no matching vulkan physical device exists or device
    /// creation fails; callers should fall back to the cpu-copy in that case.
    pub fn new(node: DrmNode) -> Result<Self, VkBridgeError> {
        Self::init_with(|_vendor, major, minor| major == node.major() as i64 && minor == node.minor() as i64)
    }

    /// Create a bridge selecting the physical device by PCI vendor id.
    ///
    /// `None` picks the first device that is not Intel (0x8086) — i.e. the
    /// "discrete" gpu on hybrid laptops — falling back to the first device
    /// with a render node. Used by [`preinit`], where no DRM node is known yet.
    pub fn new_for_vendor(vendor_id: Option<u32>) -> Result<Self, VkBridgeError> {
        Self::init_with(|vendor, _major, _minor| match vendor_id {
            Some(want) => vendor == want,
            None => vendor != 0x8086,
        })
    }

    fn init_with(matcher: impl Fn(u32, i64, i64) -> bool) -> Result<Self, VkBridgeError> {
        info!("vkbridge: init start");
        let entry = unsafe { ash::Entry::load() }
            .map_err(|err| VkBridgeError::Setup(format!("failed to load vulkan: {err}")))?;
        info!("vkbridge: loader ok");

        let app_info = vk::ApplicationInfo::default()
            .application_name(c"smithay-vkbridge")
            .api_version(vk::API_VERSION_1_3);
        // NOTE: VK_EXT_physical_device_drm is intentionally *not* enabled here.
        // The proprietary NVIDIA driver does not advertise it as an instance
        // extension (instance creation fails with ERROR_EXTENSION_NOT_PRESENT),
        // yet it fills in VkPhysicalDeviceDrmPropertiesEXT just fine without it.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app_info),
                None,
            )
        }?;
        info!("vkbridge: instance created");

        let phds = unsafe { instance.enumerate_physical_devices()? };
        info!("vkbridge: enumerated {} physical devices", phds.len());
        let mut phd = None;
        for candidate in phds {
            let (has_render, render_major, render_minor, vendor_id) = {
                let mut drm_props = vk::PhysicalDeviceDrmPropertiesEXT::default();
                let vendor_id = {
                    let mut props =
                        vk::PhysicalDeviceProperties2::default().push_next(&mut drm_props);
                    unsafe { instance.get_physical_device_properties2(candidate, &mut props) };
                    props.properties.vendor_id
                };
                (
                    drm_props.has_render,
                    drm_props.render_major,
                    drm_props.render_minor,
                    vendor_id,
                )
            };
            debug!(
                "vkbridge: phd vendor=0x{vendor_id:04x} render={render_major}:{render_minor}",
            );
            if has_render == vk::TRUE && matcher(vendor_id, render_major, render_minor) {
                phd = Some(candidate);
                break;
            }
        }
        let phd = phd.ok_or_else(|| VkBridgeError::Setup("no matching physical device".into()))?;

        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(phd) };
        let queue_family = queue_families
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .or_else(|| {
                queue_families
                    .iter()
                    .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            })
            .ok_or_else(|| VkBridgeError::Setup("no transfer/graphics queue".into()))?
            as u32;

        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device_extensions = [
            ext::image_drm_format_modifier::NAME.as_ptr(),
            ext::external_memory_dma_buf::NAME.as_ptr(),
            khr::external_memory_fd::NAME.as_ptr(),
        ];
        let device = unsafe {
            instance.create_device(
                phd,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_info)
                    .enabled_extension_names(&device_extensions),
                None,
            )
        }?;
        info!("vkbridge: device created");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };
        let cmd_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }?;
        info!("vkbridge: queue + cmd pool ok");
        let get_memory_fd = khr::external_memory_fd::Device::new(&instance, &device);

        info!("vkbridge: initialized");
        let core = std::sync::Arc::new(BridgeCore {
            entry,
            instance,
            phd,
            device,
            queue,
            queue_family,
            cmd_pool,
            get_memory_fd,
            state: std::sync::Mutex::new(CopyState::default()),
        });
        let (sender, receiver) = std::sync::mpsc::sync_channel::<CopyJob>(2);
        let latest = std::sync::Arc::new(std::sync::Mutex::new(None));
        let submitted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let completed = std::sync::Arc::new((std::sync::Mutex::new(0), std::sync::Condvar::new()));
        let worker = {
            let core = core.clone();
            let latest = latest.clone();
            let completed = completed.clone();
            std::thread::Builder::new()
                .name("vkbridge-copy".into())
                .spawn(move || worker_main(core, receiver, latest, completed))
                .map_err(|err| VkBridgeError::Setup(format!("failed to spawn copy worker: {err}")))?
        };
        Ok(VkBridge {
            core,
            sender,
            latest,
            submitted,
            completed,
            worker: Some(worker),
        })
    }

    /// Queue a copy of `src` to linear. Returns immediately; the result is
    /// available via [`VkBridge::latest_completed`] once the worker finishes it
    /// (typically next frame). The job's sync point is awaited on the worker
    /// thread, never on the compositor.
    pub fn submit_copy(
        &self,
        src: Dmabuf,
        sync: crate::backend::renderer::sync::SyncPoint,
        regions: Vec<crate::utils::Rectangle<i32, crate::utils::Buffer>>,
    ) {
        let wait_fd = sync.export();
        self.submitted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if self
            .sender
            .try_send(CopyJob { src, sync, wait_fd, regions })
            .is_err()
        {
            debug!("vkbridge: copy queue full, skipping frame copy");
        }
    }

    /// Block until all submitted copies have completed. Called before the
    /// compositor re-renders into the staging buffer, so the copy worker is
    /// never reading it concurrently. Usually a no-op: the worker finishes a
    /// copy in ~1ms, long before the next frame starts rendering.
    pub fn wait_for_pending_copies(&self) {
        let submitted = self.submitted.load(std::sync::atomic::Ordering::SeqCst);
        let (lock, cvar) = &*self.completed;
        let mut completed = lock.lock().unwrap();
        while *completed < submitted {
            completed = cvar.wait(completed).unwrap();
        }
    }

    /// The most recent completed linear dmabuf, if any.
    pub fn latest_completed(&self) -> Option<Dmabuf> {
        self.latest.lock().unwrap().clone()
    }
}

impl Drop for VkBridge {
    fn drop(&mut self) {
        // Detach the worker (drop the join handle) instead of joining it:
        // the worker blocks in recv() and the channel never closes while we
        // hold a sender, so joining would deadlock the compositor's shutdown.
        // The bridge only drops at compositor exit, where the process
        // teardown reaps the thread anyway.
        drop(self.worker.take());
    }
}

impl std::fmt::Debug for VkBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkBridge").finish_non_exhaustive()
    }
}
