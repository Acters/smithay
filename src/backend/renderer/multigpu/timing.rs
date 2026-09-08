//! Opt-in, current-thread frame timing aggregation.
//!
//! Register streams on the event-loop thread during topology setup, then enter a
//! scope around frame/presentation work and drain on that same thread. Workers
//! without a scope do not record or allocate. Registration is bounded to 16 streams
//! for the lifetime of the thread; excess/unknown streams suppress recording until
//! their scope exits. Guards must be dropped in nesting order and cannot cross threads.
//!
//! Enabled recording uses clocks and short TLS borrows, never locks, formatting,
//! allocation or GPU synchronization. Disabled recording does not query clocks.
//! Nested stages overlap: do not sum them. Timings measure host elapsed time, not
//! GPU execution. Drain allocates/copies raw snapshots; summarize and serialize off
//! the event loop. Histogram quantiles are bucket upper bounds, not exact samples.
//! Destruction stages measure explicit Drop bodies only, excluding subsequent
//! automatic field destruction. VulkanFenceWait includes both live fence waits and
//! retirement waits. Registering an existing ID resets its histograms/counters,
//! even if the label is unchanged. Each registration sets StreamRegistrations to
//! one; consumers should discard that partial reporting window. Register only
//! outside active timing scopes so an old guard cannot cross this reset boundary.

use std::{
    cell::RefCell,
    marker::PhantomData,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        OnceLock,
    },
    time::{Duration, Instant},
};

/// Device/output identity chosen by the caller.
pub type StreamId = (u64, u32);
const MAX_STREAMS: usize = 16;
const BUCKETS: usize = 141;

macro_rules! names {
    ($(#[$meta:meta])* $name:ident { $($variant:ident),+ $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        pub enum $name { $(#[doc = stringify!($variant)] $variant),+ }
        impl $name {
            /// All entries in stable declaration order.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            /// Stable serialization name.
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$variant => stringify!($variant)),+ }
            }
        }
    }
}
names! {
    /// Host timing boundaries (nested entries overlap).
    Stage {
        NiriFrame, NiriPrepare, NiriElements, NiriRender, NiriSyncWait, NiriQueue,
        FrameInterval, RenderLead, QueueLead, QueueLate, PresentLateness,
        PresentEarliness, PresentationInterval, PresentCallbackDelay, QueueToPresentation,
        SourceReuseWait, TargetAcquire, TargetPublish, VulkanCopy, VulkanRetire,
        SourceImport, TargetImport, VulkanInputSetup, CpuInputFenceWait, VulkanRecord,
        VulkanSubmit, VulkanExport, ResourcesDestroy, VulkanFenceWait, GlesWait,
        CpuGlesWait, VulkanValidation, VulkanResources, ImportedDestroy, DeviceDestroy,
        BatchDestroy, TransferRetire, TransferWait, GlesFinish, TargetTextureImport,
        NiriRedraw, NiriCallbacks, NiriScreenCast, VulkanSignalSetup,
        QueueReturnLead, QueueReturnLate, NiriSceneUpdate, NiriVblank, NiriPresentRetire
    }
}
names! {
    /// Aggregated events, not necessarily successful frames.
    Counter {
        FrameAttempts, FramesSubmitted, FramesNoDamage, RenderErrors, QueueErrors,
        ClientScanoutFrames, SwapchainFrames, PresentEvents, SequenceGaps, PresentLate,
        QueuePastDeadline, UnknownClock, DirectCopies, IntermediateCopies, CpuCopies,
        SourceImportMisses, TargetImportMisses, NativeInputImports, CpuInputWaits,
        CpuGlesWaits, GlesFinishFallbacks, BatchesCreated, BatchesRetired, TextureCopies,
        StreamRegistrations, QueueReturnedPastDeadline, FuturePresentationTimestamp,
        TransitionPresentationsIgnored
    }
}

/// Fixed histogram: 25us buckets to 1ms, 100us to 5ms, 250us to 20ms,
/// then overflow. Totals and extrema retain exact nanoseconds (saturating u64).
#[derive(Debug, Clone)]
pub struct Histogram {
    bins: [u64; BUCKETS],
    count: u64,
    total_ns: u64,
    min_ns: u64,
    max_ns: u64,
}
impl Default for Histogram {
    fn default() -> Self {
        Self {
            bins: [0; BUCKETS],
            count: 0,
            total_ns: 0,
            min_ns: u64::MAX,
            max_ns: 0,
        }
    }
}
/// Summary suitable for serialization on a background thread.
#[derive(Debug, Clone, Copy, Default)]
pub struct HistogramSummary {
    /// Number of samples.
    pub count: u64,
    /// Sum of sample durations in nanoseconds.
    pub total_ns: u64,
    /// Exact smallest sample in nanoseconds, or zero when empty.
    pub min_ns: u64,
    /// Exact largest sample in nanoseconds.
    pub max_ns: u64,
    /// 1st percentile bucket upper bound, capped at maximum.
    pub p01_ns: u64,
    /// 5th percentile bucket upper bound, capped at maximum.
    pub p05_ns: u64,
    /// Median bucket upper bound, capped at maximum.
    pub p50_ns: u64,
    /// 95th percentile bucket upper bound, capped at maximum.
    pub p95_ns: u64,
    /// 99th percentile bucket upper bound, capped at maximum.
    pub p99_ns: u64,
}
impl Histogram {
    fn record(&mut self, duration: Duration) {
        let ns = duration.as_nanos().min(u64::MAX as u128) as u64;
        let index = if ns <= 1_000_000 {
            ns.saturating_sub(1) / 25_000
        } else if ns <= 5_000_000 {
            40 + (ns - 1_000_001) / 100_000
        } else if ns <= 20_000_000 {
            80 + (ns - 5_000_001) / 250_000
        } else {
            140
        } as usize;
        self.bins[index] = self.bins[index].saturating_add(1);
        self.count = self.count.saturating_add(1);
        self.total_ns = self.total_ns.saturating_add(ns);
        self.min_ns = self.min_ns.min(ns);
        self.max_ns = self.max_ns.max(ns);
    }
    fn quantile(&self, percentile: u64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let rank = ((self.count as u128 * percentile as u128).div_ceil(100)) as u64;
        let mut seen = 0u64;
        for (i, count) in self.bins.iter().enumerate() {
            seen = seen.saturating_add(*count);
            if seen >= rank {
                let upper = if i < 40 {
                    (i as u64 + 1) * 25_000
                } else if i < 80 {
                    1_000_000 + (i as u64 - 39) * 100_000
                } else if i < 140 {
                    5_000_000 + (i as u64 - 79) * 250_000
                } else {
                    self.max_ns
                };
                return upper.min(self.max_ns);
            }
        }
        self.max_ns
    }
    /// Compute approximate quantiles; overflow quantiles use the exact maximum.
    pub fn summary(&self) -> HistogramSummary {
        HistogramSummary {
            count: self.count,
            total_ns: self.total_ns,
            min_ns: if self.count == 0 { 0 } else { self.min_ns },
            max_ns: self.max_ns,
            p01_ns: self.quantile(1),
            p05_ns: self.quantile(5),
            p50_ns: self.quantile(50),
            p95_ns: self.quantile(95),
            p99_ns: self.quantile(99),
        }
    }
}

/// Owned, Send snapshot. Only nonempty metrics/counters are emitted.
#[derive(Debug)]
pub struct StreamSnapshot {
    /// Registered stream identity.
    pub id: StreamId,
    /// Label copied at drain, never during recording.
    pub label: String,
    /// Raw histograms; summarize off-thread.
    pub metrics: Vec<(Stage, Histogram)>,
    /// Nonzero event counts.
    pub counters: Vec<(Counter, u64)>,
}
struct Stream {
    id: StreamId,
    label: String,
    metrics: Box<[Histogram]>,
    counters: [u64; Counter::ALL.len()],
}
struct State {
    streams: [Option<Stream>; MAX_STREAMS],
    current: Option<usize>,
}
impl State {
    const fn new() -> Self {
        Self {
            streams: [const { None }; MAX_STREAMS],
            current: None,
        }
    }
    fn register(&mut self, id: StreamId, label: &str) {
        if let Some(stream) = self.streams.iter_mut().flatten().find(|s| s.id == id) {
            stream.label.clear();
            stream.label.push_str(label);
            stream.metrics.fill_with(Histogram::default);
            stream.counters.fill(0);
            stream.counters[Counter::StreamRegistrations as usize] = 1;
        } else if let Some(slot) = self.streams.iter_mut().find(|s| s.is_none()) {
            *slot = Some(Stream {
                id,
                label: label.into(),
                metrics: (0..Stage::ALL.len()).map(|_| Histogram::default()).collect(),
                counters: {
                    let mut counters = [0; Counter::ALL.len()];
                    counters[Counter::StreamRegistrations as usize] = 1;
                    counters
                },
            });
        }
    }
    fn select(&mut self, id: StreamId) -> Option<usize> {
        let previous = self.current;
        self.current = self
            .streams
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.id == id));
        previous
    }
    fn drain(&mut self) -> Vec<StreamSnapshot> {
        self.streams
            .iter_mut()
            .flatten()
            .filter_map(|s| {
                let metrics: Vec<_> = Stage::ALL
                    .iter()
                    .copied()
                    .filter_map(|stage| {
                        let h = &mut s.metrics[stage as usize];
                        (h.count != 0).then(|| (stage, std::mem::take(h)))
                    })
                    .collect();
                let counters: Vec<_> = Counter::ALL
                    .iter()
                    .copied()
                    .filter_map(|counter| {
                        let value = std::mem::take(&mut s.counters[counter as usize]);
                        (value != 0).then_some((counter, value))
                    })
                    .collect();
                (!metrics.is_empty() || !counters.is_empty()).then(|| StreamSnapshot {
                    id: s.id,
                    label: s.label.clone(),
                    metrics,
                    counters,
                })
            })
            .collect()
    }
}
thread_local! { static STATE: RefCell<State> = const { RefCell::new(State::new()) }; }

fn enabled_flag() -> &'static AtomicBool {
    static ENABLED: OnceLock<AtomicBool> = OnceLock::new();
    ENABLED.get_or_init(|| {
        AtomicBool::new(std::env::var_os("SMITHAY_FRAME_TIMING").is_some_and(|v| v == "1"))
    })
}

/// Current recording state, initialized from SMITHAY_FRAME_TIMING exactly `1`.
pub fn enabled() -> bool {
    enabled_flag().load(Ordering::Relaxed)
}

/// Toggle recording between frames on the event-loop thread, outside all timing
/// scopes. Preserves TLS registrations and pending samples; the caller must reset
/// stream/reporting windows on transition. Does not start a reporter or alter GPU
/// state. Relaxed ordering is sufficient: this flag publishes no associated data.
pub fn set_enabled(enabled: bool) {
    enabled_flag().store(enabled, Ordering::Relaxed);
}
/// Register/reset a stream at topology setup outside timing scopes on the recording
/// thread. Clears previous samples and sets StreamRegistrations to one. No-op when disabled.
pub fn register_stream(id: StreamId, label: &str) {
    if enabled() {
        STATE.with(|s| s.borrow_mut().register(id, label));
    }
}
/// Restores the previous stream on drop; deliberately !Send and !Sync.
#[derive(Debug)]
pub struct ScopeGuard {
    previous: Option<usize>,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for ScopeGuard {
    fn drop(&mut self) {
        let _ = STATE.try_with(|s| s.borrow_mut().current = self.previous);
    }
}
/// Enter a registered stream. Unknown IDs enter an inactive scope, restoring on drop.
/// Returns None only when disabled; registration never happens here.
pub fn enter(id: StreamId) -> Option<ScopeGuard> {
    if !enabled() {
        return None;
    }
    Some(enter_active(id))
}
fn enter_active(id: StreamId) -> ScopeGuard {
    ScopeGuard {
        previous: STATE.with(|s| s.borrow_mut().select(id)),
        _thread: PhantomData,
    }
}
/// Records elapsed host time on drop to the stream active at creation.
#[derive(Debug)]
pub struct TimerGuard {
    stream: usize,
    stage: Stage,
    start: Instant,
    _thread: PhantomData<Rc<()>>,
}
impl Drop for TimerGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        let _ = STATE.try_with(|s| {
            if let Some(stream) = &mut s.borrow_mut().streams[self.stream] {
                stream.metrics[self.stage as usize].record(elapsed);
            }
        });
    }
}
/// Start a clock only when enabled and inside an active registered stream.
pub fn time(stage: Stage) -> Option<TimerGuard> {
    if !enabled() {
        return None;
    }
    time_active(stage)
}
fn time_active(stage: Stage) -> Option<TimerGuard> {
    let stream = STATE.with(|s| s.borrow().current)?;
    Some(TimerGuard {
        stream,
        stage,
        start: Instant::now(),
        _thread: PhantomData,
    })
}
/// Add an externally measured duration to the active stream.
pub fn observe(stage: Stage, duration: Duration) {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(i) = s.current {
            s.streams[i].as_mut().unwrap().metrics[stage as usize].record(duration);
        }
    });
}
/// Add events to the active stream using saturating arithmetic.
pub fn count(counter: Counter, value: u64) {
    if !enabled() {
        return;
    }
    STATE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(i) = s.current {
            let c = &mut s.streams[i].as_mut().unwrap().counters[counter as usize];
            *c = c.saturating_add(value);
        }
    });
}
/// Drain/reset samples on this thread, retaining registrations. Allocates only here/setup.
pub fn drain() -> Vec<StreamSnapshot> {
    if !enabled() {
        return Vec::new();
    }
    STATE.with(|s| s.borrow_mut().drain())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn histogram_quantiles_overflow_and_reset() {
        let mut h = Histogram::default();
        assert_eq!(h.summary().p99_ns, 0);
        for n in 1..=100 {
            h.record(Duration::from_micros(n * 100));
        }
        let summary = h.summary();
        assert_eq!(summary.count, 100);
        assert_eq!(summary.total_ns, 505_000_000);
        assert_eq!(summary.p50_ns, 5_000_000);
        assert_eq!(summary.p95_ns, 9_500_000);
        assert_eq!(summary.p99_ns, 10_000_000);
        h.record(Duration::from_millis(30));
        assert_eq!(h.bins[140], 1);
        assert_eq!(h.summary().max_ns, 30_000_000);
        assert_eq!(h.quantile(100), 30_000_000);
        let _ = std::mem::take(&mut h);
        assert_eq!(h.summary().count, 0);
        assert!(h.bins.iter().all(|n| *n == 0));
        h.record(Duration::ZERO);
        assert_eq!(h.summary().p50_ns, 0);
    }
    #[test]
    fn exact_minimum_lower_tails_and_reset() {
        let mut h = Histogram::default();
        let empty = h.summary();
        assert_eq!((empty.min_ns, empty.p01_ns, empty.p05_ns), (0, 0, 0));
        // Descending insertion verifies minimum tracking is independent of order.
        for n in (1..=100).rev() {
            h.record(Duration::from_micros(n * 100));
        }
        let summary = h.summary();
        assert_eq!(
            (summary.min_ns, summary.p01_ns, summary.p05_ns),
            (100_000, 100_000, 500_000)
        );
        h.record(Duration::from_nanos(123));
        let summary = h.summary();
        assert_eq!(summary.min_ns, 123);
        assert_eq!(summary.p01_ns, 100_000); // ceil(101 * 1%) = second sample
        assert_eq!(summary.p05_ns, 500_000);
        h.record(Duration::ZERO);
        assert_eq!(h.summary().min_ns, 0);
        let _ = std::mem::take(&mut h);
        assert_eq!(h.summary().min_ns, 0);
        h.record(Duration::from_nanos(42));
        assert_eq!(
            (h.summary().min_ns, h.summary().p01_ns, h.summary().p05_ns),
            (42, 42, 42)
        );
        let _ = std::mem::take(&mut h);
        h.record(Duration::MAX);
        assert_eq!(h.summary().min_ns, u64::MAX);
    }
    #[test]
    fn bucket_boundaries_and_saturation() {
        for (ns, bin) in [
            (0, 0),
            (25_000, 0),
            (25_001, 1),
            (1_000_000, 39),
            (1_000_001, 40),
            (5_000_000, 79),
            (5_000_001, 80),
            (20_000_000, 139),
            (20_000_001, 140),
        ] {
            let mut h = Histogram::default();
            h.record(Duration::from_nanos(ns));
            assert_eq!(h.bins[bin], 1);
            assert_eq!(h.summary().p99_ns, ns);
        }
        let mut h = Histogram::default();
        h.record(Duration::MAX);
        h.record(Duration::MAX);
        assert_eq!(h.summary().total_ns, u64::MAX);
        assert_eq!(h.summary().max_ns, u64::MAX);
    }
    #[test]
    fn reregistration_resets_samples_and_labels() {
        let mut state = State::new();
        state.register((1, 2), "old");
        let initial = state.drain();
        assert_eq!(initial[0].counters, [(Counter::StreamRegistrations, 1)]);
        for label in ["new", "new"] {
            let stream = state.streams[0].as_mut().unwrap();
            stream.metrics[Stage::NiriFrame as usize].record(Duration::from_nanos(42));
            stream.counters[Counter::FrameAttempts as usize] = 7;
            state.register((1, 2), label);
            let snapshots = state.drain();
            assert_eq!(snapshots.len(), 1);
            assert_eq!(snapshots[0].label, label);
            assert!(snapshots[0].metrics.is_empty());
            assert_eq!(snapshots[0].counters, [(Counter::StreamRegistrations, 1)]);
            assert!(state.drain().is_empty());
            let summary = state.streams[0].as_ref().unwrap().metrics[Stage::NiriFrame as usize].summary();
            assert_eq!((summary.count, summary.min_ns, summary.max_ns), (0, 0, 0));
        }
        assert_eq!(state.streams.iter().flatten().count(), 1);
    }
    #[test]
    fn public_recording_and_worker_isolation() {
        STATE.with(|s| *s.borrow_mut() = State::new());
        if !enabled() {
            return;
        }
        register_stream((7, 8), "output");
        let _scope = enter((7, 8));
        observe(Stage::NiriFrame, Duration::from_nanos(123));
        count(Counter::FrameAttempts, u64::MAX);
        count(Counter::FrameAttempts, 1);
        std::thread::spawn(|| {
            assert!(time(Stage::NiriFrame).is_none());
            observe(Stage::NiriFrame, Duration::from_secs(1));
            count(Counter::FrameAttempts, 1);
            assert!(STATE.with(|s| s.borrow().streams.iter().all(Option::is_none)));
            assert!(drain().is_empty());
        })
        .join()
        .unwrap();
        let snapshots = drain();
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].label, "output");
        assert_eq!(snapshots[0].metrics[0].1.summary().total_ns, 123);
        assert_eq!(
            snapshots[0].counters,
            [
                (Counter::FrameAttempts, u64::MAX),
                (Counter::StreamRegistrations, 1)
            ]
        );
        assert!(drain().is_empty());
    }
    #[test]
    fn nested_context_inactive_and_bounded_slots() {
        STATE.with(|s| *s.borrow_mut() = State::new());
        assert!(time_active(Stage::NiriFrame).is_none());
        STATE.with(|s| {
            let mut s = s.borrow_mut();
            for i in 0..MAX_STREAMS + 1 {
                s.register((0, i as u32), "test");
            }
            assert_eq!(s.streams.iter().flatten().count(), MAX_STREAMS);
        });
        // Discard topology markers before testing frame attribution.
        let registrations = STATE.with(|s| s.borrow_mut().drain());
        assert_eq!(registrations.len(), MAX_STREAMS);
        assert!(
            registrations
                .iter()
                .all(|s| s.counters == [(Counter::StreamRegistrations, 1)])
        );
        let outer = enter_active((0, 0));
        let timer = time_active(Stage::NiriFrame).unwrap();
        {
            let _inner = enter_active((0, 1));
            assert_eq!(STATE.with(|s| s.borrow().current), Some(1));
            drop(timer); // attributes to creation stream, not current stream
            let _unknown = enter_active((0, 99));
            assert!(time_active(Stage::NiriFrame).is_none());
        }
        assert_eq!(STATE.with(|s| s.borrow().current), Some(0));
        drop(outer);
        assert!(time_active(Stage::NiriFrame).is_none());
        let snapshots = STATE.with(|s| s.borrow_mut().drain());
        assert_eq!(snapshots.len(), 1);
        assert_eq!(snapshots[0].id, (0, 0));
        assert_eq!(snapshots[0].metrics[0].1.summary().count, 1);
        assert!(STATE.with(|s| s.borrow_mut().drain()).is_empty());
        assert_eq!(
            STATE.with(|s| s.borrow().streams.iter().flatten().count()),
            MAX_STREAMS
        );
    }
    #[test]
    fn disabled_or_inactive_public_api_is_noop() {
        STATE.with(|s| *s.borrow_mut() = State::new());
        assert!(time(Stage::NiriFrame).is_none());
        observe(Stage::NiriFrame, Duration::from_millis(1));
        count(Counter::FrameAttempts, 1);
        assert!(drain().is_empty());
        if !enabled() {
            register_stream((1, 1), "disabled");
            assert!(enter((1, 1)).is_none());
            assert!(STATE.with(|s| s.borrow().streams.iter().all(Option::is_none)));
        }
        fn assert_send<T: Send>() {}
        assert_send::<StreamSnapshot>();
    }
}
