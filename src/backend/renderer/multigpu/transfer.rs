//! Same-frame transfer storage. This module deliberately has no output or presentation state.

use crate::backend::{
    allocator::{Fourcc, Modifier, dmabuf::Dmabuf},
    drm::DrmNode,
    renderer::sync::SyncPoint,
};
use crate::utils::{Buffer, Size};
use tracing::warn;

use super::vkbridge::{VkBridge, VkBridgeError};

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
        // This runs only when replacing/invalidation drops storage, not for each frame.
        // A source copy fence and the downstream target-reader fence protect different
        // allocations; neither can stand in for the other.
        wait(&self.source_release);
        wait(&self.destination_release);
        self.source_release = SyncPoint::signaled();
        self.destination_release = SyncPoint::signaled();
        self.destination = None;
        self.source = None;
    }
}

fn wait(sync: &SyncPoint) {
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
    fn source_generation_uses_identity_not_node_or_value() {
        let generation = Arc::new(());
        let mut state = TransferState::default();
        assert!(!state.matches_generation(&generation));
        state.source_generation = Some(generation.clone());
        assert!(state.matches_generation(&generation.clone()));
        assert!(!state.matches_generation(&Arc::new(())));
    }
}
