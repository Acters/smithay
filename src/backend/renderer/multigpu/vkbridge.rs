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

use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd};

use ash::{ext, khr, vk};
use drm_fourcc::DrmModifier;
use tracing::{debug, info};

use crate::backend::{
    allocator::{
        dmabuf::{Dmabuf, DmabufFlags},
        vulkan::format::get_vk_format,
        Buffer,
    },
    drm::DrmNode,
};
use crate::utils::{Buffer as BufferCoord, Rectangle};

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
pub struct VkBridge {
    entry: ash::Entry,
    instance: ash::Instance,
    phd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    cmd_pool: vk::CommandPool,
    /// Fences exported to consumers; kept alive for a frame because the nvidia
    /// driver orphans exported fence fds if the VkFence is destroyed too early.
    pending_fences: Vec<vk::Fence>,
    get_memory_fd: khr::external_memory_fd::Device,
    get_semaphore_fd: khr::external_semaphore_fd::Device,
    get_fence_fd: khr::external_fence_fd::Device,
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
            khr::external_semaphore::NAME.as_ptr(),
            khr::external_semaphore_fd::NAME.as_ptr(),
            khr::external_fence::NAME.as_ptr(),
            khr::external_fence_fd::NAME.as_ptr(),
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
        let get_semaphore_fd = khr::external_semaphore_fd::Device::new(&instance, &device);
        let get_fence_fd = khr::external_fence_fd::Device::new(&instance, &device);

        info!("vkbridge: initialized");
        Ok(VkBridge {
            entry,
            instance,
            phd,
            device,
            queue,
            queue_family,
            cmd_pool,
            pending_fences: Vec::new(),
            get_memory_fd,
            get_semaphore_fd,
            get_fence_fd,
        })
    }

    /// Copy `src` (any modifier, single plane) into a newly allocated LINEAR dmabuf
    /// of identical format and size, entirely on-GPU, without blocking the CPU.
    ///
    /// `wait_fd` is an optional native fence fd the copy must wait on (the render
    /// frame's sync point, exported); it is consumed by this call. The returned
    /// fd is a native fence fd that signals when the copy is complete; the caller
    /// must make the consumer wait on it before sampling the returned dmabuf.
    ///
    /// `regions` limits the copy to the given rectangles (empty = full buffer).
    /// The blocking `vkQueueWaitIdle` of v1 is gone: all synchronization is
    /// fd-based (v2 explicit sync).
    pub fn copy_to_linear_async(
        &mut self,
        src: &Dmabuf,
        wait_fd: Option<OwnedFd>,
        regions: &[Rectangle<i32, BufferCoord>],
    ) -> Result<(Dmabuf, OwnedFd), VkBridgeError> {
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

        // -- destination image: LINEAR, exportable
        let mut external_memory_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let dst_image_info = vk::ImageCreateInfo::default()
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
        let dst = unsafe { self.device.create_image(&dst_image_info, None)? };
        let mut dst_image = Some(dst);

        // -- source image wrapping the imported dmabuf
        let src_modifiers = [src_modifier.into()];
        let mut modifier_list = vk::ImageDrmFormatModifierListCreateInfoEXT::default()
            .drm_format_modifiers(&src_modifiers);
        let mut src_external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let src_image_info = vk::ImageCreateInfo::default()
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
        let src_image = unsafe { self.device.create_image(&src_image_info, None)? };
        let mut src_image = Some(src_image);

        let _ = vk_format; // layout is fully described by the drm modifier
        let result = self.copy_inner(
            src,
            src_fd.as_raw_fd(),
            &mut src_image,
            &mut dst_image,
            wait_fd,
            regions,
        );

        // cleanup images regardless of outcome
        if let Some(img) = src_image.take() {
            unsafe { self.device.destroy_image(img, None) };
        }
        if let Some(img) = dst_image.take() {
            unsafe { self.device.destroy_image(img, None) };
        }
        result
    }

    fn copy_inner(
        &mut self,
        src: &Dmabuf,
        src_fd: std::os::fd::RawFd,
        src_image: &mut Option<vk::Image>,
        dst_image: &mut Option<vk::Image>,
        wait_fd: Option<OwnedFd>,
        regions: &[Rectangle<i32, BufferCoord>],
    ) -> Result<(Dmabuf, OwnedFd), VkBridgeError> {
        let (w, h) = (src.size().w as u32, src.size().h as u32);
        let format = src.format().code;
        let flags = DmabufFlags::empty();

        // bind memory to dst (exportable)
        let dst = dst_image.unwrap();
        let dst_req = unsafe { self.device.get_image_memory_requirements(dst) };
        let mut export_alloc = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let dst_mem_type = self
            .mem_type(dst_req.memory_type_bits, vk::MemoryPropertyFlags::empty())
            .ok_or(VkBridgeError::Setup("no memory type for dst".into()))?;
        let dst_alloc = vk::MemoryAllocateInfo::default()
            .push_next(&mut export_alloc)
            .allocation_size(dst_req.size)
            .memory_type_index(dst_mem_type);
        let dst_mem = unsafe { self.device.allocate_memory(&dst_alloc, None)? };
        unsafe { self.device.bind_image_memory(dst, dst_mem, 0)? };

        // bind imported memory to src
        let src_img = src_image.unwrap();
        let src_req = unsafe { self.device.get_image_memory_requirements(src_img) };
        let dup_fd = unsafe { libc::dup(src_fd) };
        if dup_fd < 0 {
            unsafe { self.device.free_memory(dst_mem, None) };
            return Err(VkBridgeError::Setup("failed to dup dmabuf fd".into()));
        }
        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(dup_fd);
        let src_mem_type = self
            .mem_type(src_req.memory_type_bits, vk::MemoryPropertyFlags::empty())
            .ok_or(VkBridgeError::Setup("no memory type for src import".into()))?;
        let src_alloc = vk::MemoryAllocateInfo::default()
            .push_next(&mut import_info)
            .allocation_size(src_req.size)
            .memory_type_index(src_mem_type);
        let src_mem = unsafe { self.device.allocate_memory(&src_alloc, None) };
        let src_mem = match src_mem {
            Ok(mem) => mem,
            Err(err) => {
                unsafe {
                    libc::close(dup_fd);
                    self.device.free_memory(dst_mem, None);
                }
                return Err(err.into());
            }
        };
        unsafe { self.device.bind_image_memory(src_img, src_mem, 0)? };

        // record and submit the copy
        // retire previous frames' fences; consumers had a frame to import them
        for fence in self.pending_fences.drain(..) {
            unsafe { self.device.destroy_fence(fence, None) };
        }

        let cmd = unsafe {
            self.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.cmd_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0]
        };
        let submit_result = (|| -> Result<OwnedFd, VkBridgeError> {
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
                let subresource = |mip_level| vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let copy_regions: Vec<vk::ImageCopy> = if regions.is_empty() {
                    vec![vk::ImageCopy::default()
                        .src_subresource(subresource(0))
                        .dst_subresource(subresource(0))
                        .extent(vk::Extent3D { width: w, height: h, depth: 1 })]
                } else {
                    regions
                        .iter()
                        .map(|rect| {
                            vk::ImageCopy::default()
                                .src_subresource(subresource(0))
                                .src_offset(vk::Offset3D { x: rect.loc.x, y: rect.loc.y, z: 0 })
                                .dst_subresource(subresource(0))
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

                // explicit sync: wait on the render fence (imported as a temporary
                // semaphore), signal an exportable fence for the consumer.
                let semaphore = wait_fd
                    .map(|fd| -> Result<vk::Semaphore, VkBridgeError> {
                        debug!("vkbridge: creating semaphore + importing fence fd");
                        let semaphore = unsafe {
                            self.device
                                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?
                        };
                        debug!("vkbridge: semaphore created, importing fd");
                        let import = vk::ImportSemaphoreFdInfoKHR::default()
                            .semaphore(semaphore)
                            .flags(vk::SemaphoreImportFlags::TEMPORARY)
                            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_FD)
                            .fd(fd.into_raw_fd());
                        unsafe { self.get_semaphore_fd.import_semaphore_fd(&import)? };
                        debug!("vkbridge: fence fd imported into semaphore");
                        Ok(semaphore)
                    })
                    .transpose()?;
                debug!("vkbridge: render fence semaphore ready: {}", semaphore.is_some());

                let mut export_fence_info = vk::ExportFenceCreateInfo::default()
                    .handle_types(vk::ExternalFenceHandleTypeFlags::OPAQUE_FD);
                let fence = unsafe {
                    self.device.create_fence(
                        &vk::FenceCreateInfo::default().push_next(&mut export_fence_info),
                        None,
                    )?
                };

                let cmds = [cmd];
                let wait_semaphores: Vec<vk::Semaphore> = semaphore.into_iter().collect();
                let wait_stages = [vk::PipelineStageFlags::TRANSFER];
                let mut submit = vk::SubmitInfo::default().command_buffers(&cmds);
                if !wait_semaphores.is_empty() {
                    // wait stages must correspond 1:1 with wait semaphores;
                    // setting a mask without semaphores is a spec violation
                    // that crashes the nvidia driver.
                    submit = submit
                        .wait_semaphores(&wait_semaphores)
                        .wait_dst_stage_mask(&wait_stages);
                }
                debug!(
                    "vkbridge: submitting copy (wait_semaphore={}, export fence)",
                    !wait_semaphores.is_empty()
                );
                self.device.queue_submit(self.queue, &[submit], fence)?;
                debug!("vkbridge: submit ok, exporting fence fd");

                if let Some(sem) = semaphore {
                    unsafe { self.device.destroy_semaphore(sem, None) };
                }
                let out_fd = unsafe {
                    self.get_fence_fd.get_fence_fd(
                        &vk::FenceGetFdInfoKHR::default()
                            .fence(fence)
                            .handle_type(vk::ExternalFenceHandleTypeFlags::OPAQUE_FD),
                    )?
                };
                self.pending_fences.push(fence);
                Ok(unsafe { OwnedFd::from_raw_fd(out_fd) })
            }
        })();

        // free command buffer + src memory/image handles; export dst regardless of outcome only on success
        unsafe {
            self.device.free_command_buffers(self.cmd_pool, &[cmd]);
            self.device.free_memory(src_mem, None);
        }
        let copy_fence = submit_result?;

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
        // free the vulkan memory handle; the exported fd keeps the memory alive
        unsafe { self.device.free_memory(dst_mem, None) };

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
        debug!("vkbridge: copied {}x{} {:?} -> linear dmabuf", w, h, format);
        Ok((dmabuf, copy_fence))
    }

    fn mem_type(&self, bits: u32, required: vk::MemoryPropertyFlags) -> Option<u32> {
        let props = unsafe { self.instance.get_physical_device_memory_properties(self.phd) };
        (0..props.memory_type_count).find(|&i| {
            (bits & (1 << i)) != 0
                && props.memory_types[i as usize].property_flags.contains(required)
        })
    }
}

impl Drop for VkBridge {
    fn drop(&mut self) {
        unsafe {
            for fence in self.pending_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
        let _ = &self.entry;
    }
}

impl std::fmt::Debug for VkBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkBridge").finish_non_exhaustive()
    }
}

/// A native fence fd wrapped as a smithay [`Fence`](crate::backend::renderer::sync::Fence),
/// letting consumers wait on a vulkan-produced fence through the usual
/// [`SyncPoint`](crate::backend::renderer::sync::SyncPoint) paths (including
/// server-side waits after EGL import).
#[derive(Debug)]
pub struct NativeFdFence(OwnedFd);

impl NativeFdFence {
    /// Wrap an owned native fence fd.
    pub fn new(fd: OwnedFd) -> Self {
        Self(fd)
    }
}

impl crate::backend::renderer::sync::Fence for NativeFdFence {
    fn is_signaled(&self) -> bool {
        let mut pfd = libc::pollfd {
            fd: self.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        unsafe { libc::poll(&mut pfd, 1, 0) > 0 }
    }

    fn wait(&self) -> Result<(), crate::backend::renderer::sync::Interrupted> {
        let mut pfd = libc::pollfd {
            fd: self.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        if unsafe { libc::poll(&mut pfd, 1, -1) } < 0 {
            return Err(crate::backend::renderer::sync::Interrupted);
        }
        Ok(())
    }

    fn is_exportable(&self) -> bool {
        true
    }

    fn export(&self) -> Option<OwnedFd> {
        let fd = unsafe { libc::dup(self.0.as_raw_fd()) };
        (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
    }
}
