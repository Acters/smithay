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

/// Two independently node-bound slots; VkBridge bounds resources to eight per
/// engine, hence a fixed sixteen-resource budget for a two-leg pair.
#[derive(Debug, Default)]
struct EngineSlot {
    engine: Engine,
    node: Option<DrmNode>,
    source_caps: Cache<(Fourcc, i32, i32), Vec<Modifier>>,
    destination_caps: Cache<(Fourcc, i32, i32), Vec<Modifier>>,
}

impl EngineSlot {
    fn initialization_pending(&self) -> bool {
        matches!(self.engine, Engine::Initializing(_))
    }

    fn engine(&mut self, node: DrmNode, device_lost: &mut bool) -> Option<&mut VkBridge> {
        debug_assert!(self.node.is_none_or(|old| old == node));
        self.node = Some(node);
        if *device_lost {
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
                    *device_lost |= err.is_device_lost();
                    warn!("Vulkan transfer initialization failed: {err}");
                    self.engine = Engine::Failed;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.engine = Engine::Failed,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        match &mut self.engine {
            Engine::Ready(engine) => Some(engine),
            _ => None,
        }
    }
}

/// The payload and its reader-release role travel together through both legs.
pub(super) struct PreparedTransferInput {
    pub dmabuf: Dmabuf,
    pub acquire: SyncPoint,
    pub detiled: bool,
}

impl PreparedTransferInput {
    pub fn publish_reader(&self, state: &mut TransferState, fence: SyncPoint) {
        state.record_reader(self.detiled, fence);
    }
}

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
    target_engine: EngineSlot,
    detile_engine: EngineSlot,
    pub source_detile_enabled: bool,
    pub linear: Option<Dmabuf>,
    pub linear_release: SyncPoint,
    pub source_generation: Option<std::sync::Arc<()>>,
    pub device_lost: bool,
    pub direct_target_enabled: bool,
    pub copy_device: super::VulkanCopyDevice,
    copy_node: Option<DrmNode>,
    intermediate_retries: Cache<IntermediateKey, std::time::Instant>,
    detile_retries: Cache<(Fourcc, i32, i32), std::time::Instant>,
    detile_route: Option<(Fourcc, i32, i32)>,
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
        let source_detile_enabled = self.source_detile_enabled;
        *self = Self::default();
        self.source_detile_enabled = source_detile_enabled;
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
        self.target_engine.engine(node, &mut self.device_lost)
    }

    /// Only an in-flight optional initialization delays baseline target caps.
    /// Failed initialization preserves the target engine's working single-copy route.
    pub fn detile_initialization_pending(&self) -> bool {
        self.detile_engine.initialization_pending()
    }

    pub fn detile_active(&self) -> bool {
        self.source_detile_enabled && self.prefers_direct()
    }

    pub fn detile_engine(&mut self, node: DrmNode) -> Option<&mut VkBridge> {
        if self.detile_engine.node.is_some_and(|old| old != node) {
            self.invalidate();
        }
        self.detile_engine.engine(node, &mut self.device_lost)
    }

    /// Poll both engines before advertising a native source layout. Target caps
    /// always come from the target slot; Intel only supplies the first leg.
    pub fn render_modifiers(
        &mut self,
        render: DrmNode,
        target: DrmNode,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Option<Vec<Modifier>> {
        self.detile_route = None;
        let node = self.copy_node(render, target);
        let baseline = self.source_modifiers(node, format, size);
        if !self.detile_active() {
            return baseline;
        }
        let ready = self.detile_engine(render).is_some();
        if !ready || baseline.is_none() {
            return baseline;
        }
        let key = (format, size.w, size.h);
        if self
            .detile_retries
            .get(&key)
            .is_some_and(|deadline| std::time::Instant::now() < deadline)
        {
            return baseline;
        }
        if self
            .destination_modifiers(node, format, size)
            .is_none_or(|caps| caps.is_empty())
        {
            return baseline;
        }
        let src = self.detile_engine.source_caps.get(&key);
        let dst = self.detile_engine.destination_caps.get(&key);
        let src = src.or_else(|| {
            match self
                .detile_engine(render)?
                .source_modifiers(format, size.w as u32, size.h as u32)
            {
                Ok(caps) => {
                    self.detile_engine.source_caps.insert(key, caps.clone());
                    Some(caps)
                }
                Err(err) => {
                    self.device_lost |= err.is_device_lost();
                    self.defer_detile(format, size);
                    None
                }
            }
        });
        let dst = dst.or_else(|| {
            match self
                .detile_engine(render)?
                .destination_modifiers(format, size.w as u32, size.h as u32)
            {
                Ok(caps) => {
                    self.detile_engine.destination_caps.insert(key, caps.clone());
                    Some(caps)
                }
                Err(err) => {
                    self.device_lost |= err.is_device_lost();
                    self.defer_detile(format, size);
                    None
                }
            }
        });
        if self.device_lost {
            self.disable();
            return Some(Vec::new());
        }
        if !baseline.as_ref().is_some_and(|m| m.contains(&Modifier::Linear))
            || !dst.is_some_and(|m| m.contains(&Modifier::Linear))
        {
            return baseline;
        }
        let native = src
            .unwrap_or_default()
            .into_iter()
            .filter(|m| *m != Modifier::Linear && *m != Modifier::Invalid)
            .collect::<Vec<_>>();
        if native.is_empty() {
            baseline
        } else {
            self.detile_route = Some(key);
            Some(native)
        }
    }

    pub fn detile_route_ready(&self, format: Fourcc, size: Size<i32, Buffer>) -> bool {
        self.detile_active() && self.detile_route == Some((format, size.w, size.h))
    }

    pub fn source_modifiers(
        &mut self,
        node: DrmNode,
        format: Fourcc,
        size: Size<i32, Buffer>,
    ) -> Option<Vec<Modifier>> {
        self.engine(node)?;
        let key = (format, size.w, size.h);
        if let Some(caps) = self.target_engine.source_caps.get(&key) {
            return Some(caps);
        }
        let result = self
            .engine(node)?
            .source_modifiers(format, size.w as u32, size.h as u32);
        match result {
            Ok(caps) => {
                self.target_engine.source_caps.insert(key, caps.clone());
                Some(caps)
            }
            Err(err) => {
                self.device_lost |= err.is_device_lost();
                warn!("Vulkan source modifier negotiation failed: {err}");
                if self.device_lost {
                    self.disable();
                } else if matches!(err, VkBridgeError::Unsupported(_)) {
                    self.target_engine.source_caps.insert(key, Vec::new());
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
            return matches!(self.target_engine.engine, Engine::Failed).then(Vec::new);
        }
        let key = (format, size.w, size.h);
        if let Some(caps) = self.target_engine.destination_caps.get(&key) {
            return Some(caps);
        }
        match self
            .engine(node)?
            .destination_modifiers(format, size.w as u32, size.h as u32)
        {
            Ok(caps) => {
                self.target_engine.destination_caps.insert(key, caps.clone());
                Some(caps)
            }
            Err(err) => {
                self.device_lost |= err.is_device_lost();
                warn!("Vulkan destination modifier negotiation failed: {err}");
                if self.device_lost {
                    self.disable();
                } else if matches!(err, VkBridgeError::Unsupported(_)) {
                    self.target_engine.destination_caps.insert(key, Vec::new());
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
        self.target_engine.engine = Engine::Failed;
        self.detile_engine.engine = Engine::Failed;
    }

    /// Format/extent-scoped retry, never a permanent engine disable. Eight entries
    /// cover allocation, query and first-leg pre-submit failures without frame churn.
    pub fn defer_detile(&mut self, format: Fourcc, size: Size<i32, Buffer>) {
        self.detile_retries.insert(
            (format, size.w, size.h),
            std::time::Instant::now() + std::time::Duration::from_secs(1),
        );
    }

    pub fn discard_linear(&mut self) {
        wait(&self.linear_release);
        self.linear_release = SyncPoint::signaled();
        self.linear = None;
    }

    fn record_reader(&mut self, detiled: bool, fence: SyncPoint) {
        if detiled {
            self.linear_release = fence;
        } else {
            self.source_release = fence;
        }
    }

    pub fn record_detile(&mut self, fence: SyncPoint) {
        self.source_release = fence.clone();
        self.linear_release = fence;
    }

    fn retire_buffers(&mut self) {
        let _timing = timing::time(Stage::TransferRetire);
        // This runs only when replacing/invalidation drops storage, not for each frame.
        // A source copy fence and the downstream target-reader fence protect different
        // allocations; neither can stand in for the other.
        wait(&self.source_release);
        self.discard_linear();
        wait(&self.destination_release);
        self.source_release = SyncPoint::signaled();
        self.destination_release = SyncPoint::signaled();
        self.destination = None;
        self.source = None;
        self.rejected_direct_targets.clear();
        self.intermediate_retries.clear();
    }
}

/// A cold or size/format-changed source can use the same native S+L transaction
/// immediately once the exact route is ready; it needs no baseline-only first frame.
pub(super) fn source_allocation_needed(
    modifiers: &[Modifier],
    compatible_source: Option<Modifier>,
    detile_ready: bool,
    linear_compatible: bool,
) -> bool {
    match compatible_source {
        Some(source) => source_migration_needed(modifiers, source, detile_ready, linear_compatible),
        None => detile_ready && !modifiers.is_empty(),
    }
}

/// A startup source may already have the native layout before either engine is
/// ready. Once the exact route is negotiated, it still needs its transactional L.
pub(super) fn source_migration_needed(
    modifiers: &[Modifier],
    source: Modifier,
    detile_ready: bool,
    linear_compatible: bool,
) -> bool {
    !modifiers.is_empty()
        && (!modifiers.contains(&source)
            || (detile_ready
                && source != Modifier::Linear
                && source != Modifier::Invalid
                && !linear_compatible))
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
            .target_engine
            .source_caps
            .insert((Fourcc::Xrgb8888, 1920, 1080), Vec::new());
        state.invalidate();
        assert!(state.prefers_direct());
        assert!(state.target_engine.source_caps.is_empty());
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
            .target_engine
            .source_caps
            .insert((Fourcc::Xrgb8888, 1920, 1080), Vec::new());
        assert!(
            state
                .target_engine
                .source_caps
                .contains_key(&(Fourcc::Xrgb8888, 1920, 1080))
        );
        assert!(
            !state
                .target_engine
                .source_caps
                .contains_key(&(Fourcc::Xrgb2101010, 1920, 1080))
        );
        assert!(
            !state
                .target_engine
                .source_caps
                .contains_key(&(Fourcc::Xrgb8888, 1280, 720))
        );
        assert!(matches!(state.target_engine.engine, Engine::Uninitialized));
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
        assert!(matches!(state.target_engine.engine, Engine::Uninitialized));
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
    fn cold_and_mixed_size_sources_allocate_native_immediately_when_ready() {
        let native = Modifier::from(72057594037927938u64);
        // Missing source (including incompatible size/format) does not need a
        // second render to upgrade from LINEAR after capability warmup.
        assert!(source_allocation_needed(&[native], None, true, false));
        let sizes = [(1920, 1080), (1280, 720), (960, 540), (1920, 1080)];
        let mut current = None;
        for size in sizes {
            let compatible_source = (current == Some(size)).then_some(native);
            assert!(source_allocation_needed(
                &[native],
                compatible_source,
                true,
                false
            ));
            current = Some(size);
        }
        assert!(!source_allocation_needed(&[native], Some(native), true, true));
        // Warming, unavailable and disabled routes retain normal cold allocation.
        assert!(!source_allocation_needed(&[native], None, false, false));
        assert!(!source_allocation_needed(&[], None, true, false));
        assert!(!source_allocation_needed(&[Modifier::Linear], None, false, false));
    }

    #[test]
    fn startup_native_source_gets_linear_only_after_route_ready() {
        let native = Modifier::from(72057594037927938u64);
        // Already-native startup allocation must not hide the missing L after warmup.
        assert!(!source_migration_needed(&[native], native, false, false));
        assert!(source_migration_needed(&[native], native, true, false));
        assert!(!source_migration_needed(&[native], native, true, true));
        // Failed, warming and disabled routes cannot force an optional allocation.
        assert!(!source_migration_needed(&[], native, true, false));
        assert!(!source_migration_needed(
            &[Modifier::Linear],
            Modifier::Linear,
            false,
            false
        ));
        assert!(!source_migration_needed(
            &[Modifier::Linear],
            Modifier::Linear,
            true,
            false
        ));
        // Existing modifier migration remains unchanged for the single-copy route.
        assert!(source_migration_needed(&[Modifier::Linear], native, false, false));
    }

    #[test]
    fn detile_route_readiness_is_policy_format_extent_scoped() {
        let mut state = TransferState::default();
        let size = (1920, 1080).into();
        state.source_detile_enabled = true;
        state.direct_target_enabled = true;
        state.copy_device = super::super::VulkanCopyDevice::Target;
        assert!(!state.detile_route_ready(Fourcc::Xrgb8888, size));
        state.detile_route = Some((Fourcc::Xrgb8888, 1920, 1080));
        assert!(state.detile_route_ready(Fourcc::Xrgb8888, size));
        assert!(!state.detile_route_ready(Fourcc::Xrgb2101010, size));
        assert!(!state.detile_route_ready(Fourcc::Xrgb8888, (1280, 720).into()));
        state.source_detile_enabled = false;
        assert!(!state.detile_route_ready(Fourcc::Xrgb8888, size));
        state.source_detile_enabled = true;
        state.invalidate();
        assert!(!state.detile_route_ready(Fourcc::Xrgb8888, size));
    }

    #[test]
    fn optional_engine_failure_preserves_baseline_cap_availability() {
        let mut state = TransferState::default();
        let baseline = Some(vec![Modifier::Linear]);
        let visible_caps = |state: &TransferState| {
            if state.detile_initialization_pending() {
                None
            } else {
                baseline.clone()
            }
        };
        assert_eq!(visible_caps(&state), baseline);
        let (_sender, receiver) = std::sync::mpsc::channel();
        state.detile_engine.engine = Engine::Initializing(receiver);
        assert_eq!(visible_caps(&state), None);
        // Recoverable optional initialization failure must not masquerade as
        // pending forever and hide already negotiated target-engine layouts.
        state.detile_engine.engine = Engine::Failed;
        assert_eq!(visible_caps(&state), baseline);
        assert!(!state.device_lost);
        // A lost device remains a separate fatal condition checked by the API.
        state.device_lost = true;
        assert!(state.device_lost);
        assert!(!state.detile_initialization_pending());
    }

    #[test]
    fn detile_policy_is_opt_in_target_direct_and_survives_invalidation() {
        use super::super::VulkanCopyDevice;
        let mut state = TransferState::default();
        assert!(!state.source_detile_enabled);
        for enabled in [false, true] {
            for direct in [false, true] {
                for role in [VulkanCopyDevice::Render, VulkanCopyDevice::Target] {
                    state.source_detile_enabled = enabled;
                    state.direct_target_enabled = direct;
                    state.copy_device = role;
                    let active = enabled && direct && role == VulkanCopyDevice::Target;
                    assert_eq!(state.detile_active(), active);
                    state.invalidate();
                    assert_eq!(state.detile_active(), active);
                }
            }
        }
    }

    #[test]
    fn engine_capability_roles_are_independent_and_epoch_retired() {
        let mut state = TransferState::default();
        let key = (Fourcc::Xrgb8888, 1920, 1080);
        state
            .detile_engine
            .source_caps
            .insert(key, vec![Modifier::Invalid]);
        state
            .detile_engine
            .destination_caps
            .insert(key, vec![Modifier::Linear]);
        state.target_engine.source_caps.insert(key, vec![]);
        assert_eq!(state.target_engine.source_caps.get(&key), Some(vec![]));
        assert_eq!(
            state.detile_engine.destination_caps.get(&key),
            Some(vec![Modifier::Linear])
        );
        assert!(
            state
                .detile_engine
                .source_caps
                .get(&(Fourcc::Xrgb2101010, 1920, 1080))
                .is_none()
        );
        assert!(
            state
                .detile_engine
                .source_caps
                .get(&(Fourcc::Xrgb8888, 1280, 720))
                .is_none()
        );
        state.invalidate();
        assert!(state.detile_engine.source_caps.is_empty());
        assert!(state.detile_engine.destination_caps.is_empty());
        assert!(state.target_engine.source_caps.is_empty());
    }

    #[test]
    fn detile_fence_ledger_failure_success_readers_and_retirement() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let fence = |name| {
            SyncPoint::from(TrackingFence {
                name,
                calls: calls.clone(),
                interrupt: AtomicBool::new(false),
            })
        };
        let mut state = TransferState::default();
        // Pre-submit failure changes no owner.
        assert!(!state.source_release.contains_fence());
        assert!(!state.linear_release.contains_fence());
        state.record_detile(fence("f1"));
        // Stage two pre-submit failure leaves BOTH f1 owners intact.
        assert_eq!(state.source_release.get::<TrackingFence>().unwrap().name, "f1");
        assert_eq!(state.linear_release.get::<TrackingFence>().unwrap().name, "f1");
        state.record_reader(true, fence("f2"));
        assert_eq!(state.source_release.get::<TrackingFence>().unwrap().name, "f1");
        assert_eq!(state.linear_release.get::<TrackingFence>().unwrap().name, "f2");
        // A GLES reader of L replaces only L's release; D has its own reader.
        state.record_reader(true, fence("linear-gles"));
        state.destination_release = fence("destination-gles");
        state.retire_buffers();
        assert_eq!(*calls.lock().unwrap(), ["f1", "linear-gles", "destination-gles"]);
        drop(state);
        assert_eq!(calls.lock().unwrap().len(), 3);
    }

    #[test]
    fn failed_second_leg_retirement_keeps_both_submitted_owners() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut state = TransferState::default();
        state.record_detile(SyncPoint::from(TrackingFence {
            name: "f1",
            calls: calls.clone(),
            interrupt: AtomicBool::new(false),
        }));
        state.retire_buffers();
        assert_eq!(*calls.lock().unwrap(), ["f1", "f1"]);
        assert!(!state.source_release.contains_fence());
        assert!(!state.linear_release.contains_fence());
    }

    #[test]
    fn detile_retry_is_bounded_and_format_extent_scoped() {
        let mut state = TransferState::default();
        state.defer_detile(Fourcc::Xrgb8888, (1920, 1080).into());
        assert!(state.detile_retries.contains_key(&(Fourcc::Xrgb8888, 1920, 1080)));
        assert!(
            !state
                .detile_retries
                .contains_key(&(Fourcc::Xrgb2101010, 1920, 1080))
        );
        assert!(!state.detile_retries.contains_key(&(Fourcc::Xrgb8888, 1280, 720)));
        for width in 1..32 {
            state.defer_detile(Fourcc::Xrgb8888, (width, 1080).into());
        }
        assert_eq!(state.detile_retries.0.len(), CACHE_LIMIT);
        state.invalidate();
        assert!(state.detile_retries.is_empty());
    }

    #[test]
    fn linear_discard_does_not_retire_other_roles() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut state = TransferState::default();
        state.record_detile(SyncPoint::from(TrackingFence {
            name: "f1",
            calls: calls.clone(),
            interrupt: AtomicBool::new(true),
        }));
        state.discard_linear();
        assert_eq!(*calls.lock().unwrap(), ["f1", "f1"]);
        assert!(state.source_release.contains_fence());
        assert!(!state.linear_release.contains_fence());
        state.retire_buffers();
        assert_eq!(*calls.lock().unwrap(), ["f1", "f1", "f1"]);
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
