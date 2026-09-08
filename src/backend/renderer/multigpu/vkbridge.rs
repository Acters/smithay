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
    os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};

use super::timing::{self, Counter, Stage};
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
        vulkan::{AppInfo, Instance, InstanceError, PhysicalDevice, version::Version},
    },
    utils::{Buffer, Rectangle},
};

/// Failure to initialize or submit a transfer. No successful submission is hidden by an error.
/// Device loss can leave work submitted and invalidates allocation contents: it is never
/// evidence that buffers may be reused. Potentially submitted ownership is retired or retained.
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

impl From<InstanceError> for VkBridgeError {
    fn from(err: InstanceError) -> Self {
        match err {
            // Preserve raw Vulkan errors, including the device-loss classification.
            InstanceError::Vk(err) => Self::Vk(err),
            err => Self::Setup(err.to_string()),
        }
    }
}

struct Core {
    // PhysicalDevice owns an Arc-backed Instance clone, keeping the loader/instance alive
    // until this logical device and every batch/import using it have been destroyed.
    phd: PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    memory_fd: khr::external_memory_fd::Device,
    semaphore_fd: Option<khr::external_semaphore_fd::Device>,
    import_sync_fd: bool,
    export_sync_fd: bool,
    // Consumers can discover loss while the owner skips their locked completion.
    device_lost: AtomicBool,
}

impl Core {
    fn modifier_properties(
        &self,
        format: vk::Format,
    ) -> Result<Vec<vk::DrmFormatModifierPropertiesEXT>, VkBridgeError> {
        self.phd
            .get_format_modifier_properties(format)
            .map_err(|err| VkBridgeError::Setup(err.to_string()))
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
            self.phd
                .instance()
                .handle()
                .get_physical_device_image_format_properties2(self.phd.handle(), &info, &mut image_props)
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
        let _timing = timing::time(Stage::DeviceDestroy);
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
        let _timing = timing::time(Stage::ImportedDestroy);
        unsafe {
            self.core.device.destroy_image(self.image, None);
            self.core.device.free_memory(self.memory, None);
        }
    }
}

// Pool operations belong exclusively to the engine's mutable owner. SyncPoints never
// own reusable handles directly: a completion mutex fences every query/wait against
// extraction, and extraction freezes the logical outcome before a handle can be reset.
struct Resources {
    core: Arc<Core>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    waits: [vk::Semaphore; 2],
    signal: vk::Semaphore,
    // Temporary imports not consumed by a submission must be replaced on checkout.
    dirty_waits: [bool; 2],
    // A submitted signal is clean only after successful SYNC_FD copy export.
    dirty_signal: bool,
}

impl Resources {
    fn new(core: Arc<Core>) -> Result<Self, vk::Result> {
        // Partial construction is RAII-safe; null Vulkan destruction is permitted.
        let mut r = Self {
            core,
            pool: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            waits: [vk::Semaphore::null(); 2],
            signal: vk::Semaphore::null(),
            dirty_waits: [false; 2],
            dirty_signal: false,
        };
        timing::count(Counter::ResourceSetsCreated, 1);
        unsafe {
            let device = &r.core.device;
            r.pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(r.core.family),
                None,
            )?;
            r.command = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(r.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            r.fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
            for sem in &mut r.waits {
                *sem = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            }
        }
        r.signal = r.create_signal()?;
        Ok(r)
    }

    fn create_signal(&self) -> Result<vk::Semaphore, vk::Result> {
        if !self.core.export_sync_fd {
            return Ok(vk::Semaphore::null());
        }
        let mut export = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
        unsafe {
            self.core
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut export), None)
        }
    }

    fn reset(&mut self) -> Result<(), vk::Result> {
        let _timing = timing::time(Stage::VulkanPoolReset);
        // Only extracted, GPU-retired or never-submitted sets reach here. No consumer
        // can still query this original fence. Keep command-pool backing allocations.
        unsafe {
            self.core
                .device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.core.device.reset_fences(&[self.fence])?;
            for index in 0..self.waits.len() {
                if self.dirty_waits[index] {
                    let replacement = self
                        .core
                        .device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                    self.core.device.destroy_semaphore(self.waits[index], None);
                    self.waits[index] = replacement;
                    self.dirty_waits[index] = false;
                }
            }
            if self.dirty_signal {
                // Rare failed export: retirement alone does not unsignal a binary semaphore.
                let replacement = self.create_signal()?;
                self.core.device.destroy_semaphore(self.signal, None);
                self.signal = replacement;
                self.dirty_signal = false;
                timing::count(Counter::SignalSemaphoresReplaced, 1);
            }
        }
        Ok(())
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        let _timing = timing::time(Stage::ResourcesDestroy);
        timing::count(Counter::ResourceSetsDestroyed, 1);
        unsafe {
            for sem in self.waits {
                self.core.device.destroy_semaphore(sem, None);
            }
            self.core.device.destroy_semaphore(self.signal, None);
            self.core.device.destroy_fence(self.fence, None);
            self.core.device.destroy_command_pool(self.pool, None);
        }
    }
}

// Returning a pre-submit error never loses a slot. Unconsumed temporary payloads and
// partially recorded command buffers are repaired at the next checkout, without GPU waits.
struct Checkout<'a, T> {
    idle: &'a mut Vec<T>,
    resources: Option<T>,
}

impl<T> Drop for Checkout<'_, T> {
    fn drop(&mut self) {
        if let Some(resources) = self.resources.take() {
            self.idle.push(resources);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    Pending,
    Complete,
    DeviceLost,
}

// Generic ownership transition is also exercised without a Vulkan device in tests.
struct CompletionState<T> {
    outcome: Outcome,
    owned: Option<T>,
}

impl<T> CompletionState<T> {
    fn pending(owned: T) -> Self {
        Self {
            outcome: Outcome::Pending,
            owned: Some(owned),
        }
    }

    fn observe(&mut self, result: Result<bool, vk::Result>) -> Result<bool, vk::Result> {
        match self.outcome {
            Outcome::Complete => return Ok(true),
            Outcome::DeviceLost => return Err(vk::Result::ERROR_DEVICE_LOST),
            Outcome::Pending => {}
        }
        match result {
            Ok(true) => self.outcome = Outcome::Complete,
            Err(vk::Result::ERROR_DEVICE_LOST) => self.outcome = Outcome::DeviceLost,
            _ => {}
        }
        result
    }

    fn take_retired(&mut self) -> Option<T> {
        if self.outcome == Outcome::Pending {
            None
        } else {
            self.owned.take()
        }
    }
}

struct Submission {
    resources: Resources,
    // No inputs/SyncPoints are retained: their imported fd payloads own dependencies.
    _images: [Arc<Imported>; 2],
}

type State = CompletionState<Submission>;

impl State {
    fn poll(&mut self, wait: bool, retired: &AtomicBool) -> Result<bool, vk::Result> {
        if self.outcome != Outcome::Pending {
            retired.store(true, Ordering::Release);
            return self.observe(Ok(false));
        }
        let r = &self.owned.as_ref().unwrap().resources;
        // Global loss only stops new work. Each pending fence still needs its own
        // driver result to prove retired access; another command's loss is not proof.
        let result = if wait {
            let _timing = timing::time(Stage::VulkanFenceWait);
            unsafe { r.core.device.wait_for_fences(&[r.fence], true, u64::MAX) }.map(|()| true)
        } else {
            unsafe { r.core.device.get_fence_status(r.fence) }
        };
        if result == Err(vk::Result::ERROR_DEVICE_LOST) {
            r.core.device_lost.store(true, Ordering::Relaxed);
        }
        let result = self.observe(result);
        if self.outcome != Outcome::Pending {
            // Publish while holding the state lock, before extraction/reset is possible.
            // This cache is monotonic and treats loss as retired access, not valid pixels.
            retired.store(true, Ordering::Release);
        }
        result
    }
}

struct Completion {
    state: Mutex<State>,
    retired: AtomicBool,
}

impl Completion {
    fn is_signaled(&self) -> bool {
        if self.retired.load(Ordering::Acquire) {
            return true;
        }
        let result = match self.state.try_lock() {
            Ok(mut state) => state.poll(false, &self.retired),
            Err(std::sync::TryLockError::Poisoned(err)) => err.into_inner().poll(false, &self.retired),
            Err(std::sync::TryLockError::WouldBlock) => {
                // A waiter may have published completion since the first load. Never
                // report false merely because a terminal state's mutex is contended.
                return self.retired.load(Ordering::Acquire);
            }
        };
        matches!(result, Ok(true) | Err(vk::Result::ERROR_DEVICE_LOST))
    }
}

impl Drop for Completion {
    fn drop(&mut self) {
        let _timing = timing::time(Stage::BatchDestroy);
        // Normally the engine has already extracted ownership. This also protects
        // unwinding: an unexpected retirement failure retains the entire Core/image tree.
        let state = self.state.get_mut().unwrap_or_else(|err| err.into_inner());
        if state.owned.is_some() {
            match state.poll(true, &self.retired) {
                Ok(true) | Err(vk::Result::ERROR_DEVICE_LOST) => {
                    timing::count(Counter::BatchesRetired, 1);
                }
                result => {
                    warn!(
                        ?result,
                        "Vulkan retirement wait failed; retaining submission for safety"
                    );
                    std::mem::forget(state.owned.take());
                }
            }
        }
    }
}

struct TransferFence {
    completion: Arc<Completion>,
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
        self.completion.is_signaled()
    }
    fn wait(&self) -> Result<(), Interrupted> {
        if self.completion.retired.load(Ordering::Acquire) {
            return Ok(());
        }
        retirement_wait(
            self.completion
                .state
                .lock()
                .unwrap_or_else(|err| err.into_inner())
                .poll(true, &self.completion.retired)
                .map(|_| ()),
        )
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

fn pool_busy(pending: usize) -> bool {
    pending >= MAX_PENDING
}

struct SourceModifiers {
    fourcc: Fourcc,
    width: u32,
    height: u32,
    modifiers: Vec<Modifier>,
}

/// Transfer-only engine; allocations and presentation state belong to the caller.
/// Dropping the engine may wait for GPU retirement. Old fences keep their logical
/// completion after retirement and never own recycled Vulkan handles.
pub struct VkBridge {
    core: Arc<Core>,
    imports: VecDeque<Arc<Imported>>,
    pending: Vec<Arc<Completion>>,
    // idle + pending <= MAX_PENDING. Idle slots contain no images or old completions.
    idle: Vec<Resources>,
    source_modifiers: VecDeque<SourceModifiers>,
}

impl Drop for VkBridge {
    fn drop(&mut self) {
        for completion in self.pending.drain(..) {
            let mut state = completion.state.lock().unwrap_or_else(|err| err.into_inner());
            match state.poll(true, &completion.retired) {
                Ok(true) | Err(vk::Result::ERROR_DEVICE_LOST) => {
                    let retired = state.take_retired();
                    drop(state);
                    if retired.is_some() {
                        timing::count(Counter::BatchesRetired, 1);
                    }
                    // Teardown is on the engine owner. Old SyncPoints retain only immutable
                    // logical outcome + their independent fd, never Vulkan handles/images.
                    drop(retired);
                }
                result => {
                    warn!(
                        ?result,
                        "Vulkan engine retirement failed; retaining completion for safety"
                    );
                    drop(state);
                    // Preserve callable old fences AND the entire ownership tree. Do not
                    // leave a Pending state without its resources, or free live GPU work.
                    std::mem::forget(completion);
                }
            }
        }
    }
}

impl std::fmt::Debug for VkBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VkBridge")
            .field("pending", &self.pending.len())
            .finish_non_exhaustive()
    }
}

impl VkBridge {
    /// Initialize an instance and logical device on exactly the supplied render `node`.
    ///
    /// MultiRenderer invokes this on its existing lazy background init thread, without a
    /// global pre-DRM-master hook or vendor preference. Keep initialization off the compositor
    /// event-loop thread: late background initialization has been observed to work, but this
    /// does not establish that synchronous initialization on that thread is safe on all ICDs
    /// or resolve the cause of historical initialization hangs.
    /// The Smithay instance wrapper may enable debug utilities and, in debug builds, available
    /// validation layers. Vulkan 1.2 remains required even if `SMITHAY_VK_VERSION` lowers the
    /// wrapper's negotiated instance version.
    pub fn new(node: DrmNode) -> Result<Self, VkBridgeError> {
        let started = std::time::Instant::now();
        tracing::info!(?node, "Vulkan transfer device initialization started");
        let instance_started = std::time::Instant::now();
        tracing::info!(?node, "Vulkan transfer instance initialization started");
        let instance = Instance::new(
            Version::VERSION_1_2,
            Some(AppInfo {
                name: "smithay-vkbridge".into(),
                version: Version::VERSION_1_0,
            }),
        )?;
        tracing::info!(
            ?node,
            elapsed_ms = instance_started.elapsed().as_millis(),
            version = %instance.api_version(),
            "Vulkan transfer instance created"
        );
        // Instance::new takes a MAX version and honors the loader/environment limit. Our
        // extension dependency strategy relies on features promoted to Vulkan core 1.2.
        if instance.api_version() < Version::VERSION_1_2 {
            return Err(VkBridgeError::Setup(
                "Vulkan transfer requires instance version 1.2".into(),
            ));
        }
        let required = [
            ext::physical_device_drm::NAME,
            ext::image_drm_format_modifier::NAME,
            ext::external_memory_dma_buf::NAME,
            ext::queue_family_foreign::NAME,
            khr::external_memory_fd::NAME,
        ];
        let phd = PhysicalDevice::enumerate(&instance)?
            .find(|phd| {
                phd.api_version() >= Version::VERSION_1_2
                    && required.iter().all(|name| phd.has_device_extension(name))
                    // No primary-node or vendor fallback: the bridge must use the render GPU.
                    && phd.render_node().ok().flatten() == Some(node)
            })
            .ok_or_else(|| {
                VkBridgeError::Setup(
                    "no matching Vulkan 1.2 render node with explicit dma-buf/foreign ownership support"
                        .into(),
                )
            })?;
        let semaphore_extension = phd.has_device_extension(khr::external_semaphore_fd::NAME);
        let instance = phd.instance().handle();
        // Graphics queues have unrestricted image-transfer granularity. Dedicated transfer
        // queues may require whole mip levels or aligned regions, incompatible with damage.
        let families = unsafe { instance.get_physical_device_queue_family_properties(phd.handle()) };
        let family = families
            .iter()
            .position(|p| p.queue_count > 0 && p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or_else(|| VkBridgeError::Setup("no graphics transfer queue".into()))?
            as u32;
        let mut semaphore_props = vk::ExternalSemaphoreProperties::default();
        if semaphore_extension {
            unsafe {
                instance.get_physical_device_external_semaphore_properties(
                    phd.handle(),
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
            instance.create_device(
                phd.handle(),
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_extension_names(&names),
                None,
            )
        }?;
        // All operations after device creation are infallible until ownership reaches Core.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let memory_fd = khr::external_memory_fd::Device::new(instance, &device);
        let semaphore_fd =
            semaphore_extension.then(|| khr::external_semaphore_fd::Device::new(instance, &device));
        let features = semaphore_props.external_semaphore_features;
        tracing::info!(
            ?node,
            device = phd.name(),
            elapsed_ms = started.elapsed().as_millis(),
            "Vulkan transfer device ready"
        );
        Ok(Self {
            core: Arc::new(Core {
                phd,
                device,
                queue,
                family,
                memory_fd,
                semaphore_fd,
                import_sync_fd: features.contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE),
                export_sync_fd: features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE),
                device_lost: AtomicBool::new(false),
            }),
            imports: VecDeque::new(),
            pending: Vec::with_capacity(MAX_PENDING),
            idle: Vec::with_capacity(MAX_PENDING),
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
        for candidate in self.core.modifier_properties(format)? {
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
        let _copy_timing = timing::time(Stage::VulkanCopy);
        if self.core.device_lost.load(Ordering::Relaxed) {
            return Err(vk::Result::ERROR_DEVICE_LOST.into());
        }
        let result = self.copy_inner(src, dst, acquire, destination_release, regions);
        if result.as_ref().is_err_and(|err| err.is_device_lost()) {
            self.core.device_lost.store(true, Ordering::Relaxed);
        }
        result
    }

    // Poll at most once per pending submission. A consumer may hold the state mutex
    // across an infinite wait; do not stall rendering on it. There is no pool mutex.
    fn reap(&mut self) -> Result<(), vk::Result> {
        let mut index = 0;
        while index < self.pending.len() {
            let retired = match self.pending[index].state.try_lock() {
                Ok(mut state) => {
                    state.poll(false, &self.pending[index].retired)?;
                    state.take_retired()
                }
                Err(std::sync::TryLockError::WouldBlock) => None,
                Err(std::sync::TryLockError::Poisoned(err)) => {
                    let mut state = err.into_inner();
                    state.poll(false, &self.pending[index].retired)?;
                    state.take_retired()
                }
            };
            if let Some(submission) = retired {
                // State is now permanently Complete. Destroy images / return resources
                // outside its lock, on the engine owner, not a KMS SyncPoint drop.
                self.pending.swap_remove(index);
                self.idle.push(submission.resources);
                timing::count(Counter::BatchesRetired, 1);
                timing::count(Counter::ResourceSetsRecycled, 1);
            } else {
                index += 1;
            }
        }
        if self.core.device_lost.load(Ordering::Relaxed) {
            return Err(vk::Result::ERROR_DEVICE_LOST);
        }
        Ok(())
    }

    fn copy_inner(
        &mut self,
        src: &Dmabuf,
        dst: &Dmabuf,
        acquire: &SyncPoint,
        destination_release: Option<&SyncPoint>,
        regions: &[Rectangle<i32, Buffer>],
    ) -> Result<SyncPoint, VkBridgeError> {
        let validation_timing = timing::time(Stage::VulkanValidation);
        let format = validate(src, dst, regions)?;
        drop(validation_timing);
        let retire_timing = timing::time(Stage::VulkanRetire);
        self.reap()?;
        if pool_busy(self.pending.len()) {
            timing::count(Counter::ResourcePoolBusy, 1);
            return Err(VkBridgeError::Busy);
        }
        drop(retire_timing);
        let source = self.import(src, format, true)?;
        let destination = self.import(dst, format, false)?;
        let resources_timing = timing::time(Stage::VulkanResources);
        let reused = !self.idle.is_empty();
        let resources = match self.idle.pop() {
            Some(resources) => resources,
            None => Resources::new(self.core.clone())?,
        };
        let mut checkout = Checkout {
            idle: &mut self.idle,
            resources: Some(resources),
        };
        let r = checkout.resources.as_mut().unwrap();
        if reused {
            r.reset()?;
            timing::count(Counter::ResourceSetsReused, 1);
        }
        let device = &self.core.device;
        drop(resources_timing);
        let input_timing = timing::time(Stage::VulkanInputSetup);
        let mut waits = [vk::Semaphore::null(); 2];
        let mut wait_count = 0;
        for (index, input) in std::iter::once(acquire).chain(destination_release).enumerate() {
            if !input.contains_fence() {
                continue;
            }
            let mut imported = false;
            if self.core.import_sync_fd {
                if let Some(fd) = input.export() {
                    let sem = r.waits[index];
                    let info = vk::ImportSemaphoreFdInfoKHR::default()
                        .semaphore(sem)
                        .flags(vk::SemaphoreImportFlags::TEMPORARY)
                        .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD)
                        .fd(fd.as_raw_fd());
                    r.dirty_waits[index] = true;
                    match unsafe {
                        self.core
                            .semaphore_fd
                            .as_ref()
                            .unwrap()
                            .import_semaphore_fd(&info)
                    } {
                        Ok(()) => {
                            // Vulkan owns the descriptor only after a successful import.
                            let _ = fd.into_raw_fd();
                            waits[wait_count] = sem;
                            wait_count += 1;
                            imported = true;
                            timing::count(Counter::NativeInputImports, 1);
                        }
                        Err(vk::Result::ERROR_DEVICE_LOST) => {
                            return Err(vk::Result::ERROR_DEVICE_LOST.into());
                        }
                        Err(_) => {} // Keep the explicit CPU-wait fallback.
                    }
                }
            }
            if !imported {
                timing::count(Counter::CpuInputWaits, 1);
                let _wait_timing = timing::time(Stage::CpuInputFenceWait);
                input.wait()?;
            }
        }
        drop(input_timing);
        let signal_timing = timing::time(Stage::VulkanSignalSetup);
        let signal_storage = [r.signal];
        let signals = &signal_storage[..usize::from(self.core.export_sync_fd)];
        drop(signal_timing);
        let record_timing = timing::time(Stage::VulkanRecord);
        let command = r.command;
        record(&self.core, command, source.image, destination.image, regions)?;
        drop(record_timing);
        let commands = [command];
        let waits = &waits[..wait_count];
        let stages = [vk::PipelineStageFlags::ALL_COMMANDS; 2];
        let submit = vk::SubmitInfo::default()
            .command_buffers(&commands)
            .wait_semaphores(waits)
            .wait_dst_stage_mask(&stages[..wait_count])
            .signal_semaphores(signals);
        let submit_timing = timing::time(Stage::VulkanSubmit);
        // Allocate the logical owner BEFORE submission, including retained images. On
        // success ownership must never return to the pre-submit Checkout guard.
        let mut completion = Arc::new(Completion {
            retired: AtomicBool::new(false),
            state: Mutex::new(CompletionState::pending(Submission {
                resources: checkout.resources.take().unwrap(),
                _images: [source, destination],
            })),
        });
        // Not published yet: unique access avoids a mutex across submit/export. Only
        // errors guaranteeing unchanged submission state may use pre-submit rollback.
        let state = Arc::get_mut(&mut completion)
            .unwrap()
            .state
            .get_mut()
            .unwrap_or_else(|err| err.into_inner());
        let r = &mut state.owned.as_mut().unwrap().resources;
        if let Err(err) = unsafe { device.queue_submit(self.core.queue, &[submit], r.fence) } {
            if err == vk::Result::ERROR_DEVICE_LOST {
                self.core.device_lost.store(true, Ordering::Relaxed);
                // Loss can leave work submitted. Never put these handles into idle.
                // Completion::drop performs this fence's actual wait before destruction;
                // an unexpected wait failure leaks its complete ownership tree instead.
                drop(completion);
            } else {
                let submission = state.owned.take().unwrap();
                checkout.resources = Some(submission.resources);
            }
            return Err(err.into());
        }
        for (index, sem) in r.waits.iter().enumerate() {
            if waits.contains(sem) {
                r.dirty_waits[index] = false; // Temporary payload consumed before retirement.
            }
        }
        r.dirty_signal = !signals.is_empty();
        timing::count(Counter::BatchesCreated, 1);
        drop(submit_timing);
        // From this point there must be no fallible early return: caller must receive the
        // source-release fence even when native fd export fails after successful submission.
        let export_timing = timing::time(Stage::VulkanExport);
        let fd = signals.first().and_then(|&sem| {
            let info = vk::SemaphoreGetFdInfoKHR::default()
                .semaphore(sem)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::SYNC_FD);
            match unsafe { self.core.semaphore_fd.as_ref().unwrap().get_semaphore_fd(&info) } {
                Ok(fd) => {
                    // Copy export resets the semaphore payload, including fd == -1.
                    // The owned fd is independent of all future semaphore/fence reuse.
                    r.dirty_signal = false;
                    (fd >= 0).then(|| unsafe { OwnedFd::from_raw_fd(fd) })
                }
                Err(err) => {
                    if err == vk::Result::ERROR_DEVICE_LOST {
                        self.core.device_lost.store(true, Ordering::Relaxed);
                    }
                    warn!(?err, "native copy fence export failed; using Vulkan wait");
                    None
                }
            }
        });
        drop(export_timing);
        self.pending.push(completion.clone());
        Ok(TransferFence { completion, fd }.into())
    }

    fn import(
        &mut self,
        dmabuf: &Dmabuf,
        format: vk::Format,
        source: bool,
    ) -> Result<Arc<Imported>, VkBridgeError> {
        let _timing = timing::time(if source {
            Stage::SourceImport
        } else {
            Stage::TargetImport
        });
        if let Some(index) = self
            .imports
            .iter()
            .position(|i| i.dmabuf == *dmabuf && i.source == source)
        {
            let image = self.imports.remove(index).unwrap();
            self.imports.push_back(image.clone());
            return Ok(image);
        }
        timing::count(
            if source {
                Counter::SourceImportMisses
            } else {
                Counter::TargetImportMisses
            },
            1,
        );
        let core = &self.core;
        let usage = if source {
            vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::TRANSFER_DST
        };
        let modifier = u64::from(dmabuf.format().modifier);
        // Use the same exact modifier/usage/importability checks as allocation negotiation.
        let modifiers = core.modifier_properties(format)?;
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
    fn pending_status_does_not_block_behind_a_waiting_consumer() {
        // No driver is needed: the pending state remains locked until it is made terminal.
        let completion = Arc::new(Completion {
            state: Mutex::new(State {
                outcome: Outcome::Pending,
                owned: None,
            }),
            retired: AtomicBool::new(false),
        });
        let mut waiter = completion.state.lock().unwrap();
        let fence = TransferFence {
            completion: completion.clone(),
            fd: None,
        };
        let (sender, receiver) = std::sync::mpsc::channel();
        let query = std::thread::spawn(move || sender.send(fence.is_signaled()).unwrap());
        let result = receiver.recv_timeout(std::time::Duration::from_secs(1));
        // Always release before asserting, so a blocking regression fails without hanging.
        waiter.outcome = Outcome::Complete;
        assert_eq!(waiter.poll(false, &completion.retired), Ok(true));
        drop(waiter);
        query.join().unwrap();
        assert_eq!(result, Ok(false));
        assert!(completion.is_signaled());
    }

    #[test]
    fn cached_terminal_status_stays_true_even_while_state_is_locked() {
        for outcome in [Outcome::Complete, Outcome::DeviceLost] {
            let completion = Arc::new(Completion {
                state: Mutex::new(State { outcome, owned: None }),
                retired: AtomicBool::new(false),
            });
            let mut owner = completion.state.lock().unwrap();
            let result = owner.poll(false, &completion.retired);
            assert!(matches!(result, Ok(true) | Err(vk::Result::ERROR_DEVICE_LOST)));
            assert!(owner.take_retired().is_none());
            let fence = TransferFence {
                completion: completion.clone(),
                fd: None,
            };
            let (sender, receiver) = std::sync::mpsc::channel();
            let query = std::thread::spawn(move || {
                sender.send((fence.is_signaled(), fence.wait().is_ok())).unwrap();
            });
            let result = receiver.recv_timeout(std::time::Duration::from_secs(1));
            drop(owner);
            query.join().unwrap();
            assert_eq!(result, Ok((true, true)));
            assert!(completion.is_signaled());
        }
    }

    #[test]
    fn old_completion_stays_true_after_resource_reuse_and_engine_retirement() {
        let old = Arc::new(Mutex::new(CompletionState::pending(7usize)));
        let retained = old.clone();
        let resource = {
            let mut state = old.lock().unwrap();
            assert_eq!(state.observe(Ok(true)), Ok(true));
            state.take_retired().unwrap()
        };
        let mut next = CompletionState::pending(resource);
        assert_eq!(next.observe(Ok(false)), Ok(false));
        drop(old); // Engine no longer keeps this logical completion.
        let mut old = retained.lock().unwrap();
        assert_eq!(old.observe(Ok(false)), Ok(true));
        assert_eq!(old.observe(Err(vk::Result::ERROR_DEVICE_LOST)), Ok(true));
        assert!(old.owned.is_none());
        assert!(old.take_retired().is_none());
        assert_eq!(next.observe(Ok(true)), Ok(true));
        assert_eq!(next.take_retired(), Some(7));
    }

    #[test]
    fn pending_and_unexpected_errors_cannot_release_ownership() {
        let owner = Arc::new(());
        let mut state = CompletionState::pending(owner.clone());
        assert_eq!(state.observe(Ok(false)), Ok(false));
        assert!(state.take_retired().is_none());
        assert_eq!(
            state.observe(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)),
            Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY)
        );
        assert_eq!(state.outcome, Outcome::Pending);
        assert!(state.take_retired().is_none());
        assert_eq!(Arc::strong_count(&owner), 2);
        assert_eq!(state.observe(Ok(true)), Ok(true));
        drop(state.take_retired());
        assert_eq!(Arc::strong_count(&owner), 1);
    }

    #[test]
    fn loss_freezes_retired_access_without_becoming_success() {
        let mut state = CompletionState::pending(5usize);
        assert_eq!(
            state.observe(Err(vk::Result::ERROR_DEVICE_LOST)),
            Err(vk::Result::ERROR_DEVICE_LOST)
        );
        assert_eq!(state.outcome, Outcome::DeviceLost);
        // Owner may destroy, not recycle: reap propagates this error before extraction.
        assert_eq!(state.observe(Ok(true)), Err(vk::Result::ERROR_DEVICE_LOST));
        assert_eq!(state.take_retired(), Some(5));
        assert_eq!(state.observe(Ok(false)), Err(vk::Result::ERROR_DEVICE_LOST));
        assert!(state.take_retired().is_none());
    }

    #[test]
    fn waiting_consumer_prevents_nonblocking_reaper_extraction() {
        let state = Arc::new(Mutex::new(CompletionState::pending(9usize)));
        let consumer = state.lock().unwrap();
        let engine = state.clone();
        std::thread::spawn(move || {
            assert!(matches!(
                engine.try_lock(),
                Err(std::sync::TryLockError::WouldBlock)
            ));
        })
        .join()
        .unwrap();
        drop(consumer);
        let mut engine = state.try_lock().unwrap();
        assert!(engine.take_retired().is_none());
        assert_eq!(engine.observe(Ok(true)), Ok(true));
        assert_eq!(engine.take_retired(), Some(9));
    }

    #[test]
    fn pre_submit_guard_returns_slot_on_error_but_not_after_submission() {
        let mut idle = vec![3usize];
        let result: Result<(), ()> = (|| {
            let resources = idle.pop();
            let _checkout = Checkout {
                idle: &mut idle,
                resources,
            };
            Err(())?;
            Ok(())
        })();
        assert!(result.is_err());
        assert_eq!(idle, [3]);
        let resources = idle.pop();
        let mut checkout = Checkout {
            idle: &mut idle,
            resources,
        };
        let submitted = checkout.resources.take().unwrap();
        drop(checkout);
        assert!(idle.is_empty());
        assert_eq!(submitted, 3);
    }

    #[test]
    fn bounded_slots_recycle_with_arbitrarily_retained_old_completions() {
        let mut idle = Vec::new();
        let mut pending = Vec::new();
        let mut old = Vec::new();
        let mut created = 0;
        for _ in 0..32 {
            while !pool_busy(pending.len()) {
                let resource = idle.pop().unwrap_or_else(|| {
                    created += 1;
                    created
                });
                pending.push(CompletionState::pending(resource));
                assert!(idle.len() + pending.len() <= MAX_PENDING);
            }
            assert_eq!(pending.len(), MAX_PENDING);
            for mut state in pending.drain(..) {
                assert!(state.take_retired().is_none());
                assert_eq!(state.observe(Ok(true)), Ok(true));
                idle.push(state.take_retired().unwrap());
                old.push(state);
            }
        }
        assert_eq!(created, MAX_PENDING);
        assert_eq!(idle.len(), MAX_PENDING);
        assert_eq!(old.len(), 32 * MAX_PENDING);
        assert!(
            old.iter_mut()
                .all(|state| state.owned.is_none() && state.observe(Ok(false)) == Ok(true))
        );
    }

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
    fn wrapper_instance_errors_preserve_vulkan_classification() {
        let lost = VkBridgeError::from(InstanceError::Vk(vk::Result::ERROR_DEVICE_LOST));
        assert!(lost.is_device_lost());
        assert!(matches!(
            VkBridgeError::from(InstanceError::UnsupportedVersion),
            VkBridgeError::Setup(_)
        ));
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
