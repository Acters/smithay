//! Same-frame Vulkan dma-buf transfer engine.
//!
//! The caller owns both allocations and their reuse fences. A successful [`VkBridge::copy`]
//! submits the requested frame, returning its fence without waiting for the copy to finish.
//! Wait that fence before sampling the destination AND before rendering into the source again.
//! Pass the previous destination reader's release fence on its next use. Damage is in raw
//! buffer coordinates; callers handle empty damage and initialize newly allocated destinations.
//!
//! Imported images use explicit modifiers and GENERAL at the foreign API boundary. Both APIs
//! must relinquish the buffers while the copy owns them. This is not an implicit-sync adapter.

use std::{
    collections::VecDeque,
    ffi::CStr,
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::Arc,
};

use ash::{ext, khr, vk};
use tracing::warn;

use crate::{
    backend::{
        allocator::{
            Buffer as AllocatorBuffer, Fourcc, Modifier,
            dmabuf::{Dmabuf, DmabufFlags},
            vulkan::format::get_vk_format,
        },
        drm::DrmNode,
        renderer::sync::{Fence, Interrupted, SyncPoint},
    },
    utils::{Buffer, Rectangle},
};

/// Failure to initialize or submit a transfer. No successful submission is hidden by an error.
#[derive(Debug, thiserror::Error)]
pub enum VkBridgeError {
    /// Loader, capability or device selection failure.
    #[error("vulkan setup failed: {0}")]
    Setup(String),
    /// Vulkan rejected an operation.
    #[error("vulkan error: {0:?}")]
    Vk(#[from] vk::Result),
    /// Unsupported or inconsistent buffer descriptors or damage.
    #[error("unsupported dma-buf transfer: {0}")]
    Unsupported(&'static str),
    /// An input fence could not be waited.
    #[error("input fence wait interrupted")]
    Wait(#[from] Interrupted),
    /// Duplicating a dma-buf failed.
    #[error("dma-buf fd error: {0}")]
    Io(#[from] std::io::Error),
    /// Bound outstanding work instead of accumulating unbounded retained allocations.
    #[error("too many outstanding Vulkan transfers")]
    Busy,
}

impl VkBridgeError {
    /// Device loss invalidates imported memory contents, not just the transfer route.
    pub fn is_device_lost(&self) -> bool {
        matches!(self, Self::Vk(vk::Result::ERROR_DEVICE_LOST))
    }
}

// Only instance creation happens before DRM master acquisition. Device creation remains in
// new(): moving it into preinit can interfere with direct scanout on proprietary NVIDIA.
struct Preinit {
    _entry: ash::Entry,
    instance: ash::Instance,
    preferred: Option<u32>,
}

impl Drop for Preinit {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

type InitResult = Result<Preinit, VkBridgeError>;
static PREINIT: std::sync::Mutex<Option<std::sync::mpsc::Receiver<InitResult>>> = std::sync::Mutex::new(None);

/// Create the Vulkan instance before acquiring DRM master; defer logical device creation.
/// `vendor_id` is a preference only: [`VkBridge::new`] always verifies the exact render node.
pub fn preinit(vendor_id: Option<u32>) {
    let mut slot = PREINIT.lock().unwrap();
    if slot.is_some() {
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    match std::thread::Builder::new()
        .name("vkbridge-init".into())
        .spawn(move || {
            let _ = tx.send(early_init(vendor_id));
        }) {
        Ok(_) => *slot = Some(rx),
        Err(err) => warn!(%err, "could not spawn Vulkan preinit"),
    }
}

fn early_init(preferred: Option<u32>) -> InitResult {
    let started = std::time::Instant::now();
    tracing::info!("Vulkan transfer instance initialization started");
    let entry = unsafe { ash::Entry::load() }.map_err(|e| VkBridgeError::Setup(e.to_string()))?;
    let app = vk::ApplicationInfo::default()
        .application_name(c"smithay-vkbridge")
        .api_version(vk::API_VERSION_1_2);
    let instance =
        unsafe { entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None) }?;
    tracing::info!(
        elapsed_ms = started.elapsed().as_millis(),
        "Vulkan transfer instance created"
    );
    Ok(Preinit {
        _entry: entry,
        instance,
        preferred,
    })
}

struct Core {
    init: Preinit,
    phd: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    memory_fd: khr::external_memory_fd::Device,
    semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    import_sync_fd: bool,
    export_sync_fd: bool,
}

impl Core {
    fn modifier_properties(&self, format: vk::Format) -> Vec<vk::DrmFormatModifierPropertiesEXT> {
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props = vk::FormatProperties2::default().push_next(&mut list);
        unsafe {
            self.init
                .instance
                .get_physical_device_format_properties2(self.phd, format, &mut props)
        };
        let mut modifiers =
            vec![vk::DrmFormatModifierPropertiesEXT::default(); list.drm_format_modifier_count as usize];
        list.p_drm_format_modifier_properties = modifiers.as_mut_ptr();
        let mut props = vk::FormatProperties2::default().push_next(&mut list);
        unsafe {
            self.init
                .instance
                .get_physical_device_format_properties2(self.phd, format, &mut props)
        };
        modifiers.truncate(list.drm_format_modifier_count as usize);
        modifiers
    }

    // Match the image created by import exactly: 2D, explicit modifier, exclusive sharing,
    // no image-create flags, one mip/layer/sample and DMA_BUF external memory.
    fn import_properties(
        &self,
        format: vk::Format,
        modifier: u64,
        usage: vk::ImageUsageFlags,
    ) -> Result<Option<vk::ImageFormatProperties>, VkBridgeError> {
        let mut modifier_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
            .drm_format_modifier(modifier)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut external_info = vk::PhysicalDeviceExternalImageFormatInfo::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::PhysicalDeviceImageFormatInfo2::default()
            .format(format)
            .ty(vk::ImageType::TYPE_2D)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .push_next(&mut modifier_info)
            .push_next(&mut external_info);
        let mut external_props = vk::ExternalImageFormatProperties::default();
        let mut image_props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
        match unsafe {
            self.init
                .instance
                .get_physical_device_image_format_properties2(self.phd, &info, &mut image_props)
        } {
            Ok(()) => {}
            Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => return Ok(None),
            Err(err) => return Err(err.into()),
        }
        let limits = image_props.image_format_properties;
        let external = external_props.external_memory_properties;
        Ok((external
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            && external
                .compatible_handle_types
                .contains(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT))
        .then_some(limits))
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Every image and batch owns this core. The last reference can disappear only after
        // all submitted batches have retired; no global queue-idle wait is needed here.
        unsafe { self.device.destroy_device(None) };
    }
}

struct Imported {
    core: Arc<Core>,
    // Pointer-stable Dmabuf identity, not an fd number. Retaining it prevents allocation reuse.
    dmabuf: Dmabuf,
    source: bool,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Drop for Imported {
    fn drop(&mut self) {
        unsafe {
            self.core.device.destroy_image(self.image, None);
            self.core.device.free_memory(self.memory, None);
        }
    }
}

// One pool per batch avoids externally synchronized command-pool operations between the
// compositor and a SyncPoint dropped on another thread. There is no Core -> Batch reference.
struct Resources {
    core: Arc<Core>,
    _images: [Arc<Imported>; 2],
    pool: vk::CommandPool,
    fence: vk::Fence,
    semaphores: Vec<vk::Semaphore>,
}

impl Drop for Resources {
    fn drop(&mut self) {
        unsafe {
            for sem in self.semaphores.drain(..) {
                self.core.device.destroy_semaphore(sem, None);
            }
            self.core.device.destroy_fence(self.fence, None);
            self.core.device.destroy_command_pool(self.pool, None);
        }
    }
}

struct Batch {
    resources: Option<Resources>,
    submitted: bool,
}

impl Batch {
    fn resources(&self) -> &Resources {
        self.resources.as_ref().unwrap()
    }

    fn complete(&self) -> Result<bool, vk::Result> {
        let r = self.resources();
        unsafe { r.core.device.get_fence_status(r.fence) }
    }

    fn wait(&self) -> Result<(), vk::Result> {
        let r = self.resources();
        unsafe { r.core.device.wait_for_fences(&[r.fence], true, u64::MAX) }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        if self.submitted {
            match self.wait() {
                Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => {}
                Err(err) => {
                    // An unexpected host-side wait failure is NOT completion. Leak the whole
                    // ownership tree rather than free resources the GPU might still access.
                    warn!(?err, "Vulkan retirement wait failed; retaining batch for safety");
                    std::mem::forget(self.resources.take());
                }
            }
        }
    }
}

struct TransferFence {
    batch: Arc<Batch>,
    fd: Option<OwnedFd>,
}

impl std::fmt::Debug for TransferFence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanTransferFence")
            .field("exportable", &self.fd.is_some())
            .finish()
    }
}

impl Fence for TransferFence {
    fn is_signaled(&self) -> bool {
        matches!(
            self.batch.complete(),
            Ok(true) | Err(vk::Result::ERROR_DEVICE_LOST)
        )
    }
    fn wait(&self) -> Result<(), Interrupted> {
        retirement_wait(self.batch.wait())
    }
    fn is_exportable(&self) -> bool {
        self.fd.is_some()
    }
    fn export(&self) -> Option<OwnedFd> {
        self.fd.as_ref()?.try_clone().ok()
    }
}

fn retirement_wait(result: Result<(), vk::Result>) -> Result<(), Interrupted> {
    match result {
        // Device loss retires resource access, but does NOT make pixels valid.
        // The engine keeps reporting the raw device-loss error from copy().
        Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => Ok(()),
        Err(_) => Err(Interrupted),
    }
}

const MAX_IMPORTS: usize = 8;
const MAX_PENDING: usize = 8;
const MAX_MODIFIER_QUERIES: usize = 8;

struct SourceModifiers {
    fourcc: Fourcc,
    width: u32,
    height: u32,
    modifiers: Vec<Modifier>,
}

/// Transfer-only engine; allocations and presentation state belong to the caller.
/// Dropping the engine or its last outstanding fence may wait for GPU retirement.
pub struct VkBridge {
    core: Arc<Core>,
    imports: VecDeque<Arc<Imported>>,
    pending: Vec<Arc<Batch>>,
    source_modifiers: VecDeque<SourceModifiers>,
}

impl std::fmt::Debug for VkBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkBridge")
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl VkBridge {
    /// Create the logical device on exactly `node`, consuming preinitialization if available.
    pub fn new(node: DrmNode) -> Result<Self, VkBridgeError> {
        let started = std::time::Instant::now();
        let receiver = PREINIT.lock().unwrap().take();
        tracing::info!(
            ?node,
            preinitialized = receiver.is_some(),
            "Vulkan transfer device initialization started"
        );
        let init = match receiver {
            Some(rx) => rx
                .recv()
                .map_err(|_| VkBridgeError::Setup("preinit channel closed".into()))??,
            None => early_init(None)?,
        };
        let mut devices = unsafe { init.instance.enumerate_physical_devices() }?;
        devices.sort_by_key(|&phd| {
            let props = unsafe { init.instance.get_physical_device_properties(phd) };
            init.preferred.is_some_and(|v| v != props.vendor_id)
        });
        let required = [
            ext::physical_device_drm::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
            ext::queue_family_foreign::NAME,
            khr::external_memory_fd::NAME,
        ];
        let mut selected = None;
        for phd in devices {
            let extensions = unsafe { init.instance.enumerate_device_extension_properties(phd) }?;
            let supports = |name: &CStr| {
                extensions
                    .iter()
                    .any(|p| unsafe { CStr::from_ptr(p.extension_name.as_ptr()) == name })
            };
            if !required.iter().all(|name| supports(name)) {
                continue;
            }
            let mut drm = vk::PhysicalDeviceDrmPropertiesEXT::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut drm);
            unsafe { init.instance.get_physical_device_properties2(phd, &mut props) };
            if props.properties.api_version < vk::API_VERSION_1_2 {
                continue;
            }
            if drm.has_render == vk::TRUE
                && drm.render_major == node.major() as i64
                && drm.render_minor == node.minor() as i64
            {
                selected = Some((phd, supports(khr::external_semaphore_fd::NAME)));
                break;
            }
        }
        let (phd, semaphore_extension) = selected.ok_or_else(|| {
            VkBridgeError::Setup(
                "no matching Vulkan 1.2 render node with explicit dma-buf/foreign ownership support".into(),
            )
        })?;
        // Graphics queues have unrestricted image-transfer granularity. Dedicated transfer
        // queues may require whole mip levels or aligned regions, incompatible with damage.
        let families = unsafe { init.instance.get_physical_device_queue_family_properties(phd) };
        let family = families
            .iter()
            .position(|p| p.queue_count > 0 && p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| VkBridgeError::Setup("no graphics transfer queue".into()))?
            as u32;
        let mut semaphore_props = vk::ExternalSemaphoreProperties::default();
        if semaphore_extension {
            unsafe {
                init.instance.get_physical_device_external_semaphore_properties(
                    phd,
                    &vk::PhysicalDeviceExternalSemaphoreInfo::default()
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD),
                    &mut semaphore_props,
                )
            };
        }
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut names: Vec<_> = required.iter().map(|n| n.as_ptr()).collect();
        if semaphore_extension {
            names.push(khr::external_semaphore_fd::NAME.as_ptr());
        }
        let device = unsafe {
            init.instance.create_device(
                phd,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&names),
                None,
            )
        }?;
        // All operations after device creation are infallible until ownership reaches Core.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory_fd = khr::external_memory_fd::Device::new(&init.instance, &device);
        let semaphore_fd =
            semaphore_extension.then(|| khr::external_semaphore_fd::Device::new(&init.instance, &device));
        let features = semaphore_props.external_semaphore_features;
        tracing::info!(
            ?node,
            elapsed_ms = started.elapsed().as_millis(),
            "Vulkan transfer device ready"
        );
        Ok(Self {
            core: Arc::new(Core {
                init,
                phd,
                device,
                queue,
                family,
                memory_fd,
                semaphore_fd,
                import_sync_fd: features.contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE),
                export_sync_fd: features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
            }),
            imports: VecDeque::new(),
            pending: Vec::new(),
            source_modifiers: VecDeque::new(),
        })
    }

    /// Enumerate source modifiers this engine can import for an exact format and size.
    ///
    /// Only explicit single-plane modifiers supporting `TRANSFER_SRC` and DMA_BUF import
    /// are returned. Intersect these with the source EGL render formats BEFORE allocating
    /// the source; a GPU's native GBM modifier is not necessarily Vulkan-importable.
    /// The result is numerically sorted, not a performance preference. An empty result means
    /// no supported combination. Unknown FourCCs and zero/non-i32 dimensions are errors.
    ///
    /// Up to eight FourCC/size queries (including empty results) are cached. Negotiation does
    /// not guarantee a particular allocation can be imported: actual fd memory types, plane
    /// layout and descriptor compatibility are still validated by [`Self::copy`].
    pub fn source_modifiers(
        &mut self,
        fourcc: Fourcc,
        width: u32,
        height: u32,
    ) -> Result<Vec<Modifier>, VkBridgeError> {
        if width == 0 || height == 0 || width > i32::MAX as u32 || height > i32::MAX as u32 {
            return Err(VkBridgeError::Unsupported(
                "invalid source modifier query dimensions",
            ));
        }
        let format = get_vk_format(fourcc).ok_or(VkBridgeError::Unsupported("unknown FourCC"))?;
        if let Some(index) = self
            .source_modifiers
            .iter()
            .position(|entry| entry.fourcc == fourcc && entry.width == width && entry.height == height)
        {
            let entry = self.source_modifiers.remove(index).unwrap();
            let result = entry.modifiers.clone();
            self.source_modifiers.push_back(entry);
            return Ok(result);
        }
        let mut modifiers = Vec::new();
        for candidate in self.core.modifier_properties(format) {
            if !single_plane_transfer(&candidate, true) {
                continue;
            }
            if let Some(limits) = self.core.import_properties(
                format,
                candidate.drm_format_modifier,
                vk::ImageUsageFlags::TRANSFER_SRC,
            )? {
                if fits_image(&limits, width, height) {
                    modifiers.push(Modifier::from(candidate.drm_format_modifier));
                }
            }
        }
        modifiers.sort_by_key(|&modifier| u64::from(modifier));
        modifiers.dedup();
        if self.source_modifiers.len() == MAX_MODIFIER_QUERIES {
            self.source_modifiers.pop_front();
        }
        self.source_modifiers.push_back(SourceModifiers {
            fourcc,
            width,
            height,
            modifiers: modifiers.clone(),
        });
        Ok(modifiers)
    }

    /// Submit this frame's regions and return a fence for that exact submission.
    ///
    /// Input fences are imported as temporary SYNC_FD semaphore payloads when possible;
    /// otherwise they are explicitly CPU-waited. Copy completion is never CPU-waited here.
    /// A failed native completion export still returns a valid, nonexportable Vulkan fence.
    /// `src` and `dst` must be distinct, non-aliasing allocations with matching dimensions,
    /// FourCC and orientation. Only explicit single-plane modifiers are supported.
    pub fn copy(
        &mut self,
        src: &Dmabuf,
        dst: &Dmabuf,
        acquire: &SyncPoint,
        destination_release: Option<&SyncPoint>,
        regions: &[Rectangle<i32, Buffer>],
    ) -> Result<SyncPoint, VkBridgeError> {
        let format = validate(src, dst, regions)?;
        // Device loss is a real error, not a successfully completed frame.
        for batch in &self.pending {
            batch.complete()?;
        }
        self.pending.retain(|batch| !batch.complete().unwrap_or(false));
        if self.pending.len() >= MAX_PENDING {
            return Err(VkBridgeError::Busy);
        }
        let source = self.import(src, format, true)?;
        let destination = self.import(dst, format, false)?;
        let mut batch = Batch {
            submitted: false,
            resources: Some(Resources {
                core: self.core.clone(),
                _images: [source, destination],
                // Imported SYNC_FD payloads own their dependency. CPU-waited inputs
                // have retired. Retaining SyncPoints here would make fence chains
                // retain every earlier batch, defeating bounded retirement.
                pool: vk::CommandPool::null(),
                fence: vk::Fence::null(),
                semaphores: Vec::new(),
            }),
        };
        let r = batch.resources.as_mut().unwrap();
        let device = &self.core.device;
        r.pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.core.family),
                None,
            )
        }?;
        r.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let mut waits = Vec::new();
        for input in std::iter::once(acquire).chain(destination_release) {
            if !input.contains_fence() {
                continue;
            }
            let mut imported = false;
            if self.core.import_sync_fd {
                if let Some(fd) = input.export() {
                    let sem = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
                    r.semaphores.push(sem);
                    let info = vk::ImportSemaphoreFdInfoKHR::default()
                        .semaphore(sem)
                        .flags(vk::SemaphoreImportFlags::TEMPORARY)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                        .fd(fd.as_raw_fd());
                    if unsafe {
                        self.core
                            .semaphore_fd
                            .as_ref()
                            .unwrap()
                            .import_semaphore_fd(&info)
                    }
                    .is_ok()
                    {
                        // Vulkan owns the descriptor only after a successful import.
                        let _ = fd.into_raw_fd();
                        waits.push(sem);
                        imported = true;
                    }
                }
            }
            if !imported {
                input.wait()?;
            }
        }
        let mut signals = Vec::new();
        if self.core.export_sync_fd {
            let mut export = vk::ExportSemaphoreCreateInfo::default()
                .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            let sem = unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None)
            }?;
            r.semaphores.push(sem);
            signals.push(sem);
        }
        let command = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(r.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }?[0];
        record(
            &self.core,
            command,
            r._images[0].image,
            r._images[1].image,
            regions,
        )?;
        let commands = [command];
        let stages = vec![vk::PipelineStageFlags::ALL_COMMANDS; waits.len()];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .signal_semaphores(&signals);
        unsafe { device.queue_submit(self.core.queue, &[submit], r.fence) }?;
        batch.submitted = true;
        // From this point there must be no fallible early return: caller must receive the
        // source-release fence even when native fd export fails after successful submission.
        let fd = signals.first().and_then(|&sem| {
            let info = vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            match unsafe { self.core.semaphore_fd.as_ref().unwrap().get_semaphore_fd(&info) } {
                Ok(fd) if fd >= 0 => Some(unsafe { OwnedFd::from_raw_fd(fd) }),
                // -1 denotes an already-signaled payload. Keep using VkFence for retirement.
                Ok(_) => None,
                Err(err) => {
                    warn!(?err, "native copy fence export failed; using Vulkan wait");
                    None
                }
            }
        });
        let batch = Arc::new(batch);
        self.pending.push(batch.clone());
        Ok(TransferFence { batch, fd }.into())
    }

    fn import(
        &mut self,
        dmabuf: &Dmabuf,
        format: vk::Format,
        source: bool,
    ) -> Result<Arc<Imported>, VkBridgeError> {
        if let Some(index) = self
            .imports
            .iter()
            .position(|i| i.dmabuf == *dmabuf && i.source == source)
        {
            let image = self.imports.remove(index).unwrap();
            self.imports.push_back(image.clone());
            return Ok(image);
        }
        let core = &self.core;
        let usage = if source {
            vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::TRANSFER_DST
        };
        let modifier = u64::from(dmabuf.format().modifier);
        // Use the same exact modifier/usage/importability checks as allocation negotiation.
        let modifiers = core.modifier_properties(format);
        if !modifiers
            .iter()
            .any(|m| m.drm_format_modifier == modifier && single_plane_transfer(m, source))
        {
            tracing::debug!(source, ?format, modifier, available = ?modifiers.iter().map(|m| (m.drm_format_modifier, m.drm_format_modifier_plane_count)).collect::<Vec<_>>(), "Vulkan DMA-BUF modifier rejected");
            return Err(VkBridgeError::Unsupported(
                "modifier is unsupported, has auxiliary planes or lacks transfer support",
            ));
        }
        let limits = core
            .import_properties(format, modifier, usage)?
            .ok_or(VkBridgeError::Unsupported("image is not importable"))?;
        if !fits_image(&limits, dmabuf.size().w as u32, dmabuf.size().h as u32) {
            return Err(VkBridgeError::Unsupported("image is not importable at this size"));
        }
        let layouts = [vk::SubresourceLayout {
            offset: dmabuf.offsets().next().unwrap() as u64,
            row_pitch: dmabuf.strides().next().unwrap() as u64,
            ..Default::default()
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(modifier)
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .push_next(&mut explicit)
            .push_next(&mut external)
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: dmabuf.size().w as u32,
                height: dmabuf.size().h as u32,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // Establish RAII immediately, before any subsequent driver call can fail.
        let mut image = Imported {
            core: core.clone(),
            dmabuf: dmabuf.clone(),
            source,
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
        };
        image.image = unsafe { core.device.create_image(&info, None) }?;
        let req = unsafe { core.device.get_image_memory_requirements(image.image) };
        let fd = dmabuf.handles().next().unwrap().try_clone_to_owned()?;
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        unsafe {
            core.memory_fd.get_memory_fd_properties(
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                fd.as_raw_fd(),
                &mut fd_props,
            )
        }?;
        let bits = req.memory_type_bits & fd_props.memory_type_bits;
        let memory_type = select_memory_type(bits).ok_or(VkBridgeError::Unsupported(
            "no common memory type for dma-buf and image",
        ))?;
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image.image);
        let allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(memory_type)
            .push_next(&mut import)
            .push_next(&mut dedicated);
        image.memory = unsafe { core.device.allocate_memory(&allocation, None) }?;
        let _ = fd.into_raw_fd();
        unsafe { core.device.bind_image_memory(image.image, image.memory, 0) }?;
        let image = Arc::new(image);
        if self.imports.len() == MAX_IMPORTS {
            self.imports.pop_front();
        }
        self.imports.push_back(image.clone());
        Ok(image)
    }
}

fn single_plane_transfer(modifier: &vk::DrmFormatModifierPropertiesEXT, source: bool) -> bool {
    let required = if source {
        vk::FormatFeatureFlags::TRANSFER_SRC
    } else {
        vk::FormatFeatureFlags::TRANSFER_DST
    };
    Modifier::from(modifier.drm_format_modifier) != Modifier::Invalid
        && modifier.drm_format_modifier_plane_count == 1
        && modifier.drm_format_modifier_tiling_features.contains(required)
}

fn fits_image(limits: &vk::ImageFormatProperties, width: u32, height: u32) -> bool {
    width > 0
        && height > 0
        && width <= limits.max_extent.width
        && height <= limits.max_extent.height
        && limits.max_extent.depth >= 1
        && limits.max_mip_levels >= 1
        && limits.max_array_layers >= 1
        && limits.sample_counts.contains(vk::SampleCountFlags::TYPE_1)
}

fn select_memory_type(bits: u32) -> Option<u32> {
    (bits != 0).then(|| bits.trailing_zeros())
}

fn valid_region(width: i32, height: i32, rect: &Rectangle<i32, Buffer>) -> bool {
    rect.loc.x >= 0
        && rect.loc.y >= 0
        && rect.size.w > 0
        && rect.size.h > 0
        && rect.loc.x.checked_add(rect.size.w).is_some_and(|x| x <= width)
        && rect.loc.y.checked_add(rect.size.h).is_some_and(|y| y <= height)
}

fn valid_descriptor(width: i32, height: i32, planes: usize, modifier: Modifier, stride: u32) -> bool {
    // All currently mapped Vulkan formats are uncompressed four-byte pixels. Tiled
    // modifiers have opaque pitch rules; explicit layouts are additionally driver-validated.
    width > 0
        && height > 0
        && planes == 1
        && modifier != Modifier::Invalid
        && stride > 0
        && (modifier != Modifier::Linear || u64::from(stride) >= width as u64 * 4)
}

fn validate(
    src: &Dmabuf,
    dst: &Dmabuf,
    regions: &[Rectangle<i32, Buffer>],
) -> Result<vk::Format, VkBridgeError> {
    if src == dst {
        return Err(VkBridgeError::Unsupported("source aliases destination"));
    }
    if src.size() != dst.size()
        || src.format().code != dst.format().code
        || src.y_inverted() != dst.y_inverted()
    {
        return Err(VkBridgeError::Unsupported("size, FourCC or orientation mismatch"));
    }
    for buffer in [src, dst] {
        if !valid_descriptor(
            buffer.size().w,
            buffer.size().h,
            buffer.num_planes(),
            buffer.format().modifier,
            buffer.strides().next().unwrap_or(0),
        ) || buffer
            .flags()
            .intersects(DmabufFlags::INTERLACED | DmabufFlags::BOTTOM_FIRST)
        {
            return Err(VkBridgeError::Unsupported("invalid single-plane descriptor"));
        }
    }
    // Distinct Dmabuf wrappers can still contain dup'd descriptors for one allocation.
    // Identity is used for caching; kernel identity is additionally checked for aliasing.
    let identity = |buffer: &Dmabuf| -> Result<_, std::io::Error> {
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        let fd = buffer.handles().next().unwrap();
        if unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error());
        }
        let stat = unsafe { stat.assume_init() };
        Ok((stat.st_dev, stat.st_ino))
    };
    if identity(src)? == identity(dst)? {
        return Err(VkBridgeError::Unsupported(
            "source and destination share an allocation",
        ));
    }
    if regions.is_empty()
        || !regions
            .iter()
            .all(|r| valid_region(src.size().w, src.size().h, r))
    {
        return Err(VkBridgeError::Unsupported("empty or out-of-bounds damage"));
    }
    get_vk_format(src.format().code).ok_or(VkBridgeError::Unsupported("unknown FourCC"))
}

fn record(
    core: &Core,
    cmd: vk::CommandBuffer,
    src: vk::Image,
    dst: vk::Image,
    regions: &[Rectangle<i32, Buffer>],
) -> Result<(), VkBridgeError> {
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);
    let specs = [
        (
            src,
            vk::AccessFlags::TRANSFER_READ,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ),
        (
            dst,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ),
    ];
    let acquire: Vec<_> = specs
        .iter()
        .map(|&(image, access, layout)| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(access)
                // Never discard source contents (or untouched destination damage) with UNDEFINED.
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(layout)
                .src_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
                .dst_queue_family_index(core.family)
        })
        .collect();
    let release: Vec<_> = specs
        .iter()
        .map(|&(image, access, layout)| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(range)
                .src_access_mask(access)
                .dst_access_mask(vk::AccessFlags::empty())
                .old_layout(layout)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(core.family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_FOREIGN_EXT)
        })
        .collect();
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .layer_count(1);
    unsafe {
        core.device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        core.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &acquire,
        );
        // Separate commands permit overlapping damage rectangles without violating the
        // non-overlap rule for a single vkCmdCopyImage region array. Serialize their writes.
        for (index, rect) in regions.iter().enumerate() {
            if index != 0 {
                let barrier = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                core.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                );
            }
            let offset = vk::Offset3D {
                x: rect.loc.x,
                y: rect.loc.y,
                z: 0,
            };
            let region = vk::ImageCopy::default()
                .src_subresource(layers)
                .dst_subresource(layers)
                .src_offset(offset)
                .dst_offset(offset)
                .extent(vk::Extent3D {
                    width: rect.size.w as u32,
                    height: rect.size.h as u32,
                    depth: 1,
                });
            core.device.cmd_copy_image(
                cmd,
                src,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        core.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &release,
        );
        core.device.end_command_buffer(cmd)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modifier_negotiation_requires_explicit_single_plane_and_exact_usage() {
        let mut modifier = vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: u64::from(Modifier::Linear),
            drm_format_modifier_plane_count: 1,
            drm_format_modifier_tiling_features: vk::FormatFeatureFlags::TRANSFER_SRC,
        };
        assert!(single_plane_transfer(&modifier, true));
        assert!(!single_plane_transfer(&modifier, false));
        modifier.drm_format_modifier_tiling_features = vk::FormatFeatureFlags::TRANSFER_DST;
        assert!(!single_plane_transfer(&modifier, true));
        assert!(single_plane_transfer(&modifier, false));
        modifier.drm_format_modifier_plane_count = 2;
        assert!(!single_plane_transfer(&modifier, false));
        modifier.drm_format_modifier_plane_count = 1;
        modifier.drm_format_modifier = u64::from(Modifier::Invalid);
        assert!(!single_plane_transfer(&modifier, false));
    }

    #[test]
    fn modifier_negotiation_checks_exact_extent_and_image_shape() {
        let mut limits = vk::ImageFormatProperties {
            max_extent: vk::Extent3D {
                width: 256,
                height: 160,
                depth: 1,
            },
            max_mip_levels: 1,
            max_array_layers: 1,
            sample_counts: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        assert!(fits_image(&limits, 256, 160));
        assert!(fits_image(&limits, 1, 1));
        assert!(!fits_image(&limits, 257, 160));
        assert!(!fits_image(&limits, 256, 161));
        assert!(!fits_image(&limits, 0, 160));
        assert!(!fits_image(&limits, 256, 0));
        limits.sample_counts = vk::SampleCountFlags::TYPE_4;
        assert!(!fits_image(&limits, 256, 160));
        limits.sample_counts = vk::SampleCountFlags::TYPE_1;
        limits.max_array_layers = 0;
        assert!(!fits_image(&limits, 256, 160));
    }

    #[test]
    fn device_loss_retires_fences_but_remains_a_transfer_error() {
        assert!(retirement_wait(Ok(())).is_ok());
        assert!(retirement_wait(Err(vk::Result::ERROR_DEVICE_LOST)).is_ok());
        assert!(retirement_wait(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)).is_err());
        assert!(VkBridgeError::Vk(vk::Result::ERROR_DEVICE_LOST).is_device_lost());
        assert!(!VkBridgeError::Busy.is_device_lost());
    }

    #[test]
    fn region_edges_and_overflow() {
        let rect = |x, y, w, h| {
            // Size's constructor rejects negatives before our validator sees them.
            let mut rect = Rectangle::<i32, Buffer>::new((x, y).into(), (0, 0).into());
            rect.size.w = w;
            rect.size.h = h;
            rect
        };
        assert!(valid_region(100, 50, &rect(0, 0, 100, 50)));
        assert!(valid_region(100, 50, &rect(99, 49, 1, 1)));
        for bad in [
            rect(-1, 0, 1, 1),
            rect(0, -1, 1, 1),
            rect(0, 0, 0, 1),
            rect(0, 0, 1, -1),
            rect(99, 0, 2, 1),
            rect(0, 49, 1, 2),
            rect(i32::MAX, 0, 1, 1),
        ] {
            assert!(!valid_region(100, 50, &bad));
        }
    }

    #[test]
    fn descriptors_reject_implicit_multiplane_and_short_rows() {
        assert!(valid_descriptor(64, 32, 1, Modifier::Linear, 256));
        assert!(!valid_descriptor(64, 32, 1, Modifier::Linear, 255));
        assert!(!valid_descriptor(64, 32, 2, Modifier::Linear, 256));
        assert!(!valid_descriptor(64, 32, 1, Modifier::Invalid, 256));
        assert!(!valid_descriptor(0, 32, 1, Modifier::Linear, 256));
        assert!(!valid_descriptor(64, -1, 1, Modifier::Linear, 256));
        assert!(!valid_descriptor(i32::MAX, 32, 1, Modifier::Linear, u32::MAX));
    }

    #[test]
    fn memory_type_intersection() {
        assert_eq!(select_memory_type(0b1010 & 0b1100), Some(3));
        assert_eq!(select_memory_type(1 & 2), None);
        assert_eq!(select_memory_type(1 << 31), Some(31));
    }
}
