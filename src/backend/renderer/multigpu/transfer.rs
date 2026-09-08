//! Same-frame transfer storage. This module deliberately has no output or presentation state.

use crate::backend::{
    allocator::{
        Fourcc, Modifier,
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
        *self = Self::default();
        self.source_generation = generation;
        self.direct_target_enabled = direct_target_enabled;
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
        match self
            .engine(node)?
            .source_modifiers(format, size.w as u32, size.h as u32)
        {
            Ok(modifiers) if !modifiers.is_empty() => Some(modifiers),
            result => {
                if let Err(err) = result {
                    self.device_lost |= err.is_device_lost();
                    warn!("Vulkan source modifier negotiation failed: {err}");
                }
                self.disable();
                None
            }
        }
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
    }
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
    fn source_generation_uses_identity_not_node_or_value() {
        let generation = Arc::new(());
        let mut state = TransferState::default();
        assert!(!state.matches_generation(&generation));
        state.source_generation = Some(generation.clone());
        assert!(state.matches_generation(&generation.clone()));
        assert!(!state.matches_generation(&Arc::new(())));
    }
}
