//! Same-frame transfer storage. This module deliberately has no output or presentation state.

use crate::backend::{
    allocator::{
        Buffer as _, Fourcc, Modifier,
        dmabuf::{Dmabuf, WeakDmabuf},
    },
    drm::DrmNode,
    renderer::sync::SyncPoint,
};
use crate::utils::{Buffer, Rectangle, Size};
use std::collections::HashSet;
use tracing::warn;

use super::{
    timing::{self, Stage},
    vkbridge::{VkBridge, VkBridgeError},
};

#[derive(Debug, Default)]
enum Engine {
    #[default]
    Uninitialized,
    Initializing(std::sync::mpsc::Receiver<Result<VkBridge, VkBridgeError>>),
    Ready(VkBridge),
    Failed,
}

const CACHE_LIMIT: usize = 8;

/// Small bounded LRU for capability results and retry throttling, including negatives.
#[derive(Debug)]
struct Cache<K, V>(std::collections::VecDeque<(K, V)>);

impl<K, V> Default for Cache<K, V> {
    fn default() -> Self {
        Self(std::collections::VecDeque::new())
    }
}

impl<K: PartialEq, V: Clone> Cache<K, V> {
    fn get(&mut self, key: &K) -> Option<V> {
        let index = self.0.iter().position(|(entry, _)| entry == key)?;
        let entry = self.0.remove(index).unwrap();
        let value = entry.1.clone();
        self.0.push_back(entry);
        Some(value)
    }

    fn insert(&mut self, key: K, value: V) {
        if let Some(index) = self.0.iter().position(|(entry, _)| entry == &key) {
            self.0.remove(index);
        } else if self.0.len() == CACHE_LIMIT {
            self.0.pop_front();
        }
        self.0.push_back((key, value));
    }

    fn clear(&mut self) {
        self.0.clear();
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[cfg(test)]
    fn contains_key(&self, key: &K) -> bool {
        self.0.iter().any(|(entry, _)| entry == key)
    }
}

type IntermediateKey = (Fourcc, i32, i32, Modifier, Modifier);

/// Resources shared by sequential transfers between one source/target device pair.
///
/// A single destination suffices because its previous reader's fence is an explicit
/// dependency of the next write. No ring-size or output-timing assumption is made.
#[derive(Debug, Default)]
pub(super) struct TransferState {
    engine: Engine,
    pub source_generation: Option<std::sync::Arc<()>>,
    pub device_lost: bool,
    pub direct_target_enabled: bool,
    pub copy_device: super::VulkanCopyDevice,
    copy_node: Option<DrmNode>,
    source_caps: Cache<(Fourcc, i32, i32), Vec<Modifier>>,
    destination_caps: Cache<(Fourcc, i32, i32), Vec<Modifier>>,
    intermediate_retries: Cache<IntermediateKey, std::time::Instant>,
    rejected_direct_targets: HashSet<WeakDmabuf>,
    source: Option<Dmabuf>,
    pub destination: Option<Dmabuf>,
    pub source_release: SyncPoint,
    pub destination_release: SyncPoint,
}

impl TransferState {
    pub fn matches_generation(&self, generation: &std::sync::Arc<()>) -> bool {
        self.source_generation
            .as_ref()
            .is_some_and(|source| std::sync::Arc::ptr_eq(source, generation))
    }

    /// Retire cached resources without changing the policy of an already borrowed renderer.
    pub fn invalidate(&mut self) {
        let generation = self.source_generation.clone();
        let direct_target_enabled = self.direct_target_enabled;
        let copy_device = self.copy_device;
        *self = Self::default();
        self.source_generation = generation;
        self.direct_target_enabled = direct_target_enabled;
        self.copy_device = copy_device;
    }

    /// Cache rejection by allocation identity, without keeping a swapchain allocation alive.
    pub fn direct_target_rejected(&mut self, target: &Dmabuf) -> bool {
        self.rejected_direct_targets.retain(|target| !target.is_gone());
        self.rejected_direct_targets.contains(&target.weak())
    }

    pub fn reject_direct_target(&mut self, target: &Dmabuf) {
        self.rejected_direct_targets.insert(target.weak());
    }

    /// Adopt a new staging allocation, invalidating the matching destination as well.
    pub fn set_source(&mut self, source: &Dmabuf) {
        if self.source.as_ref() != Some(source) {
            self.retire_buffers();
            self.source = Some(source.clone());
        }
    }

    /// Keep the existing lazy initialization ordering, without async frame presentation.
    pub fn engine(&mut self, node: DrmNode) -> Option<&mut VkBridge> {
        if self.copy_node != Some(node) {
            self.invalidate();
            self.copy_node = Some(node);
        }
        if self.device_lost {
            return None;
        }
        if matches!(self.engine, Engine::Uninitialized) {
            let (sender, receiver) = std::sync::mpsc::channel();
            match std::thread::Builder::new()
                .name("vktransfer-init".into())
                .spawn(move || {
                    let _ = sender.send(VkBridge::new(node));
                }) {
                Ok(_) => self.engine = Engine::Initializing(receiver),
                Err(err) => {
                    warn!("failed to spawn Vulkan transfer initialization: {err}");
                    self.engine = Engine::Failed;
                }
            }
        }
        if let Engine::Initializing(receiver) = &self.engine {
            match receiver.try_recv() {
                Ok(Ok(engine)) => self.engine = Engine::Ready(engine),
                Ok(Err(err)) => {
                    self.device_lost |= err.is_device_lost();
                    warn!("Vulkan transfer initialization failed: {err}");
                    self.engine = Engine::Failed;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    warn!("Vulkan transfer initialization disconnected");
                    self.engine = Engine::Failed;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        match &mut self.engine {
            Engine::Ready(engine) => Some(engine),
            _ => None,
        }
    }

    pub fn source_modifiers(
        &mut self,
        node: DrmNode,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Option<Vec<Modifier>> {
        self.engine(node)?;
        let key = (format, size.w, size.h);
        if let Some(caps) = self.source_caps.get(&key) {
            return Some(caps);
        }
        let result = self
            .engine(node)?
            .source_modifiers(format, size.w as u32, size.h as u32);
        match result {
            Ok(caps) => {
                self.source_caps.insert(key, caps.clone());
                Some(caps)
            }
            Err(err) => {
                self.device_lost |= err.is_device_lost();
                warn!("Vulkan source modifier negotiation failed: {err}");
                if self.device_lost {
                    self.disable();
                } else if matches!(err, VkBridgeError::Unsupported(_)) {
                    self.source_caps.insert(key, Vec::new());
                }
                Some(Vec::new())
            }
        }
    }

    pub fn prefers_direct(&self) -> bool {
        self.direct_target_enabled && self.copy_device == super::VulkanCopyDevice::Target
    }

    pub fn copy_node(&self, render: DrmNode, target: DrmNode) -> DrmNode {
        self.copy_device.node(render, target)
    }

    pub fn destination_modifiers(
        &mut self,
        node: DrmNode,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Option<Vec<Modifier>> {
        if self.engine(node).is_none() {
            return matches!(self.engine, Engine::Failed).then(Vec::new);
        }
        let key = (format, size.w, size.h);
        if let Some(caps) = self.destination_caps.get(&key) {
            return Some(caps);
        }
        match self
            .engine(node)?
            .destination_modifiers(format, size.w as u32, size.h as u32)
        {
            Ok(caps) => {
                self.destination_caps.insert(key, caps.clone());
                Some(caps)
            }
            Err(err) => {
                self.device_lost |= err.is_device_lost();
                warn!("Vulkan destination modifier negotiation failed: {err}");
                if self.device_lost {
                    self.disable();
                } else if matches!(err, VkBridgeError::Unsupported(_)) {
                    self.destination_caps.insert(key, Vec::new());
                }
                Some(Vec::new())
            }
        }
    }

    /// Rate-limit retries by exact format, extent and source/destination layout.
    /// Cleared on source replacement, so a new allocation is never rejected by an old fd.
    pub fn intermediate_allowed(&mut self, source: &Dmabuf, modifier: Modifier) -> bool {
        self.allow_intermediate_key((
            source.format().code,
            source.size().w,
            source.size().h,
            source.format().modifier,
            modifier,
        ))
    }

    fn allow_intermediate_key(&mut self, key: IntermediateKey) -> bool {
        self.intermediate_retries
            .get(&key)
            .is_none_or(|deadline| std::time::Instant::now() >= deadline)
    }

    pub fn defer_intermediate(&mut self, source: &Dmabuf, modifier: Modifier, deterministic: bool) {
        // Retry transient failures after one second; descriptor rejections after ten.
        // Wall time avoids nested layout/whole-attempt backoffs multiplying each other.
        self.intermediate_retries.insert(
            (
                source.format().code,
                source.size().w,
                source.size().h,
                source.format().modifier,
                modifier,
            ),
            std::time::Instant::now() + std::time::Duration::from_secs(if deterministic { 10 } else { 1 }),
        );
    }

    /// Discard only the intermediate allocation, never its still-running reader.
    /// The source's copy/reuse fence remains intact for the next GLES write.
    pub fn discard_destination(&mut self) {
        wait(&self.destination_release);
        self.destination_release = SyncPoint::signaled();
        self.destination = None;
    }

    pub fn disable(&mut self) {
        self.retire_buffers();
        self.engine = Engine::Failed;
    }

    fn retire_buffers(&mut self) {
        let _timing = timing::time(Stage::TransferRetire);
        // This runs only when replacing/invalidation drops storage, not for each frame.
        // A source copy fence and the downstream target-reader fence protect different
        // allocations; neither can stand in for the other.
        wait(&self.source_release);
        wait(&self.destination_release);
        self.source_release = SyncPoint::signaled();
        self.destination_release = SyncPoint::signaled();
        self.destination = None;
        self.source = None;
        self.rejected_direct_targets.clear();
        self.intermediate_retries.clear();
    }
}

/// Prefer portable forward copies, but device-native reverse destinations.
/// Callers exclude implicit/unsupported modifiers before ordering.
pub(super) fn order_destination_modifiers(role: super::VulkanCopyDevice, modifiers: &mut [Modifier]) {
    modifiers.sort_by_key(|modifier| match role {
        super::VulkanCopyDevice::Render => *modifier != Modifier::Linear,
        super::VulkanCopyDevice::Target => *modifier == Modifier::Linear,
    });
}

/// A direct write must not enlarge damage: the shared staging allocation is valid only
/// inside these regions. Make an exact, clipped, disjoint union instead of the CPU path's
/// bounding-box merges/full-frame shortcut. Disjoint regions also avoid overlapping writes.
pub(super) fn direct_damage(
    damage: &[Rectangle<i32, Buffer>],
    size: Size<i32, Buffer>,
) -> Vec<Rectangle<i32, Buffer>> {
    let bounds = Rectangle::from_size(size);
    let mut regions: Vec<Rectangle<i32, Buffer>> = Vec::new();
    for rect in damage.iter().filter(|rect| rect.size.w > 0 && rect.size.h > 0) {
        if let Some(clipped) = rect.intersection(bounds) {
            if clipped.size.w > 0 && clipped.size.h > 0 {
                let uncovered =
                    Rectangle::subtract_rects_many_in_place(vec![clipped], regions.iter().copied());
                regions.extend(uncovered);
            }
        }
    }
    regions
}

/// Only cache the engine's deterministic descriptor/capability rejection. Raw Vulkan
/// errors (including memory pressure) need not describe this target allocation at all.
/// Invalid damage describes a single request, not persistent buffer incompatibility.
/// Rejections are also cleared whenever the source allocation changes.
pub(super) fn cache_direct_rejection(error: &VkBridgeError) -> bool {
    match error {
        VkBridgeError::Unsupported("empty or out-of-bounds damage") => false,
        VkBridgeError::Unsupported(_) => true,
        _ => false,
    }
}

pub(super) fn wait(sync: &SyncPoint) {
    let _timing = timing::time(Stage::TransferWait);
    // Interrupted explicitly means retry, not completion (the Fence contract).
    while sync.wait().is_err() {}
}

impl Drop for TransferState {
    fn drop(&mut self) {
        self.retire_buffers();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::renderer::sync::{Fence, Interrupted};
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    #[derive(Debug)]
    struct TrackingFence {
        name: &'static str,
        calls: Arc<Mutex<Vec<&'static str>>>,
        interrupt: AtomicBool,
    }

    impl Fence for TrackingFence {
        fn is_signaled(&self) -> bool {
            false
        }
        fn wait(&self) -> Result<(), Interrupted> {
            self.calls.lock().unwrap().push(self.name);
            if self.interrupt.swap(false, Ordering::SeqCst) {
                Err(Interrupted)
            } else {
                Ok(())
            }
        }
        fn is_exportable(&self) -> bool {
            false
        }
        fn export(&self) -> Option<std::os::fd::OwnedFd> {
            None
        }
    }

    #[test]
    fn retirement_waits_both_roles_and_retries_interruption() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut state = TransferState::default();
        state.source_release = TrackingFence {
            name: "source",
            calls: calls.clone(),
            interrupt: AtomicBool::new(true),
        }
        .into();
        state.destination_release = TrackingFence {
            name: "target",
            calls: calls.clone(),
            interrupt: AtomicBool::new(false),
        }
        .into();
        state.retire_buffers();
        assert_eq!(*calls.lock().unwrap(), ["source", "source", "target"]);
        assert!(state.source_release.is_reached());
        assert!(state.destination_release.is_reached());
        drop(state);
        assert_eq!(calls.lock().unwrap().len(), 3);
    }

    #[test]
    fn target_frame_completion_survives_flush_blit_and_empty_finish() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fence = |name| {
            SyncPoint::from(TrackingFence {
                name,
                calls: calls.clone(),
                interrupt: AtomicBool::new(false),
            })
        };
        let mut completion = super::super::TargetFrameCompletion::default();
        assert!(!completion.current().contains_fence());

        // First source flush submitted a copy; later target blit is ordered after it.
        completion.record(fence("copy"));
        assert_eq!(completion.current().get::<TrackingFence>().unwrap().name, "copy");
        completion.record(fence("blit_from"));
        // Repeated empty source finishes/flushes must not manufacture a signaled fence.
        for _ in 0..3 {
            let empty_finish = completion.current();
            assert!(empty_finish.contains_fence());
            assert_eq!(empty_finish.get::<TrackingFence>().unwrap().name, "blit_from");
            completion.record(empty_finish);
        }
        completion.current().wait().unwrap();
        assert_eq!(*calls.lock().unwrap(), ["blit_from"]);

        // A subsequent ordered target write can replace the pending completion.
        completion.record(fence("next_target_write"));
        completion.current().wait().unwrap();
        assert_eq!(*calls.lock().unwrap(), ["blit_from", "next_target_write"]);
    }

    fn assert_exact_damage(
        damage: &[Rectangle<i32, Buffer>],
        size: Size<i32, Buffer>,
    ) -> Vec<Rectangle<i32, Buffer>> {
        let regions = direct_damage(damage, size);
        let contains = |rect: &Rectangle<i32, Buffer>, x: i32, y: i32| {
            rect.size.w > 0
                && rect.size.h > 0
                && x >= rect.loc.x
                && x < rect.loc.x + rect.size.w
                && y >= rect.loc.y
                && y < rect.loc.y + rect.size.h
        };
        let mut expected_pixels = 0;
        for y in 0..size.h {
            for x in 0..size.w {
                let expected = damage.iter().any(|rect| contains(rect, x, y));
                let copies = regions.iter().filter(|rect| contains(rect, x, y)).count();
                assert_eq!(
                    copies,
                    usize::from(expected),
                    "incorrect direct damage at ({x}, {y})"
                );
                expected_pixels += usize::from(expected);
            }
        }
        let mut copied_pixels = 0;
        for rect in &regions {
            assert!(rect.size.w > 0 && rect.size.h > 0);
            assert!(rect.loc.x >= 0 && rect.loc.y >= 0);
            assert!(rect.loc.x + rect.size.w <= size.w);
            assert!(rect.loc.y + rect.size.h <= size.h);
            copied_pixels += (rect.size.w * rect.size.h) as usize;
        }
        assert_eq!(copied_pixels, expected_pixels);
        regions
    }

    #[test]
    fn direct_damage_preserves_l_shaped_union_without_bounding_box() {
        let damage = [
            Rectangle::new((1, 1).into(), (6, 2).into()),
            Rectangle::new((1, 1).into(), (2, 6).into()),
        ];
        let regions = assert_exact_damage(&damage, (9, 9).into());
        assert!(regions.len() > 1);
    }

    #[test]
    fn direct_damage_clips_disjoint_regions_and_discards_empty_regions() {
        // Size/Rectangle constructors intentionally reject negative sizes in debug builds.
        // Inject the malformed public fields only after constructing a valid rectangle.
        let mut malformed = Rectangle::new((4, 3).into(), (0, 0).into());
        malformed.size.w = -1;
        malformed.size.h = 2;
        let damage = [
            Rectangle::new((-2, -1).into(), (4, 3).into()),
            Rectangle::new((5, 5).into(), (7, 6).into()),
            Rectangle::new((12, 3).into(), (2, 2).into()),
            Rectangle::new((3, 3).into(), (0, 4).into()),
            malformed,
        ];
        let regions = assert_exact_damage(&damage, (8, 8).into());
        assert_eq!(regions.len(), 2);
        assert!(direct_damage(&[], (8, 8).into()).is_empty());
    }

    #[test]
    fn direct_damage_never_uses_cpu_full_frame_shortcut() {
        let count = super::super::MAX_CPU_COPIES + 2;
        let damage = (0..count)
            .map(|i| Rectangle::new((i as i32 * 2 + 1, 2).into(), (1, 1).into()))
            .collect::<Vec<_>>();
        let regions = assert_exact_damage(&damage, (count as i32 * 2 + 2, 6).into());
        assert_eq!(regions.len(), count);
    }

    #[test]
    fn direct_damage_deduplicates_overlapping_writes() {
        let rect = Rectangle::new((1, 1).into(), (3, 3).into());
        let regions = assert_exact_damage(&[rect, rect], (6, 6).into());
        assert_eq!(regions, [rect]);
    }

    #[test]
    fn direct_rejection_cache_only_remembers_descriptor_capabilities() {
        assert!(cache_direct_rejection(&VkBridgeError::Unsupported(
            "unknown FourCC"
        )));
        assert!(cache_direct_rejection(&VkBridgeError::Unsupported(
            "image is not importable"
        )));
        assert!(!cache_direct_rejection(&VkBridgeError::Unsupported(
            "empty or out-of-bounds damage"
        )));
        assert!(!cache_direct_rejection(&VkBridgeError::Busy));
        assert!(!cache_direct_rejection(&VkBridgeError::Wait(Interrupted)));
        assert!(!cache_direct_rejection(&VkBridgeError::Io(
            std::io::Error::other("temporary fd failure")
        )));
        assert!(!cache_direct_rejection(&VkBridgeError::Setup(
            "setup failure".into()
        )));
        for error in [
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            ash::vk::Result::ERROR_TOO_MANY_OBJECTS,
            ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED,
            ash::vk::Result::ERROR_DEVICE_LOST,
        ] {
            assert!(!cache_direct_rejection(&VkBridgeError::Vk(error)));
        }
    }

    #[test]
    fn invalidation_preserves_direct_policy_and_source_generation() {
        let generation = Arc::new(());
        let mut state = TransferState::default();
        assert!(!state.direct_target_enabled);
        state.direct_target_enabled = true;
        state.source_generation = Some(generation.clone());
        state.invalidate();
        assert!(state.direct_target_enabled);
        assert!(state.matches_generation(&generation));
    }

    #[test]
    fn explicit_target_direct_policy_and_invalidation() {
        use super::super::VulkanCopyDevice;
        let mut state = TransferState::default();
        assert_eq!(state.copy_device, VulkanCopyDevice::Render);
        assert!(!state.prefers_direct());
        state.direct_target_enabled = true;
        assert!(!state.prefers_direct());
        state.copy_device = VulkanCopyDevice::Target;
        assert!(state.prefers_direct());
        state
            .source_caps
            .insert((Fourcc::Xrgb8888, 1920, 1080), Vec::new());
        state.invalidate();
        assert!(state.prefers_direct());
        assert!(state.source_caps.is_empty());
        state.direct_target_enabled = false;
        assert!(!state.prefers_direct());
    }

    #[test]
    fn destination_order_is_role_specific_and_stable() {
        use super::super::VulkanCopyDevice;
        let native_a = Modifier::from(0x0300_0000_0000_0010u64);
        let native_b = Modifier::from(0x0300_0000_0000_0011u64);
        let mut modifiers = [native_a, Modifier::Linear, native_b];
        order_destination_modifiers(VulkanCopyDevice::Render, &mut modifiers);
        assert_eq!(modifiers, [Modifier::Linear, native_a, native_b]);
        order_destination_modifiers(VulkanCopyDevice::Target, &mut modifiers);
        assert_eq!(modifiers, [native_a, native_b, Modifier::Linear]);
    }

    #[test]
    fn source_rejection_cache_is_format_and_extent_scoped() {
        let mut state = TransferState::default();
        state
            .source_caps
            .insert((Fourcc::Xrgb8888, 1920, 1080), Vec::new());
        assert!(state.source_caps.contains_key(&(Fourcc::Xrgb8888, 1920, 1080)));
        assert!(!state.source_caps.contains_key(&(Fourcc::Xrgb2101010, 1920, 1080)));
        assert!(!state.source_caps.contains_key(&(Fourcc::Xrgb8888, 1280, 720)));
        assert!(matches!(state.engine, Engine::Uninitialized));
    }

    #[test]
    fn capability_cache_is_bounded_lru_including_negative_results() {
        let mut cache = Cache::default();
        for key in 0..CACHE_LIMIT {
            cache.insert(key, Vec::<Modifier>::new());
        }
        assert_eq!(cache.get(&0), Some(Vec::new()));
        cache.insert(CACHE_LIMIT, vec![Modifier::Linear]);
        assert_eq!(cache.0.len(), CACHE_LIMIT);
        assert!(cache.contains_key(&0));
        assert!(!cache.contains_key(&1));
        cache.insert(0, vec![Modifier::Linear]);
        assert_eq!(cache.0.len(), CACHE_LIMIT);
        assert_eq!(cache.get(&0), Some(vec![Modifier::Linear]));
    }

    #[test]
    fn intermediate_retry_is_scoped_bounded_and_not_engine_failure() {
        let mut state = TransferState::default();
        let key = (Fourcc::Xrgb8888, 1920, 1080, Modifier::Linear, Modifier::Linear);
        let later = std::time::Instant::now() + std::time::Duration::from_secs(10);
        state.intermediate_retries.insert(key, later);
        assert!(!state.allow_intermediate_key(key));
        state.intermediate_retries.insert(key, std::time::Instant::now());
        assert!(state.allow_intermediate_key(key));
        let other = (
            Fourcc::Xrgb2101010,
            1920,
            1080,
            Modifier::Linear,
            Modifier::Linear,
        );
        assert!(state.allow_intermediate_key(other));
        for width in 0..32 {
            state.intermediate_retries.insert(
                (Fourcc::Xrgb8888, width, 1080, Modifier::Linear, Modifier::Linear),
                later,
            );
        }
        assert_eq!(state.intermediate_retries.0.len(), CACHE_LIMIT);
        assert!(matches!(state.engine, Engine::Uninitialized));
        state.retire_buffers();
        assert!(state.intermediate_retries.is_empty());
    }

    #[test]
    fn destination_discard_waits_reader_and_preserves_source_fence() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut state = TransferState::default();
        state.source_release = TrackingFence {
            name: "source",
            calls: calls.clone(),
            interrupt: AtomicBool::new(false),
        }
        .into();
        state.destination_release = TrackingFence {
            name: "target",
            calls: calls.clone(),
            interrupt: AtomicBool::new(true),
        }
        .into();
        state.discard_destination();
        assert_eq!(*calls.lock().unwrap(), ["target", "target"]);
        assert!(state.source_release.contains_fence());
        assert!(state.destination_release.is_reached());
        drop(state);
        assert_eq!(*calls.lock().unwrap(), ["target", "target", "source"]);
    }

    #[test]
    fn source_generation_uses_identity_not_node_or_value() {
        let generation = Arc::new(());
        let mut state = TransferState::default();
        assert!(!state.matches_generation(&generation));
        state.source_generation = Some(generation.clone());
        assert!(state.matches_generation(&generation.clone()));
        assert!(!state.matches_generation(&Arc::new(())));
    }
}
