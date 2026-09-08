//! CPU-only timing-recorder microbenchmark; opens no GPU or display.
//! Run the SAME release binary with SMITHAY_FRAME_TIMING=0 and =1.
use smithay::backend::renderer::multigpu::timing::{self, Counter, Stage};
use std::{hint::black_box, time::Instant};

fn main() {
    let frames = 200_000u64;
    timing::register_stream((1, 1), "cpu-only");
    let started = Instant::now();
    for i in 0..frames {
        let _scope = timing::enter((1, 1));
        let _frame = timing::time(Stage::NiriFrame);
        timing::count(Counter::FrameAttempts, 1);
        for stage in [
            Stage::NiriPrepare,
            Stage::NiriElements,
            Stage::SourceReuseWait,
            Stage::TargetAcquire,
            Stage::VulkanValidation,
            Stage::VulkanRetire,
            Stage::SourceImport,
            Stage::TargetImport,
            Stage::VulkanInputSetup,
            Stage::VulkanRecord,
            Stage::VulkanSubmit,
            Stage::VulkanExport,
            Stage::TargetPublish,
            Stage::ResourcesDestroy,
            Stage::NiriQueue,
        ] {
            let _span = timing::time(stage);
            black_box(i);
        }
        timing::count(Counter::FramesSubmitted, 1);
    }
    let elapsed = started.elapsed();
    // Snapshot/summary is deliberately outside the measured recording interval.
    let snapshots = timing::drain();
    println!(
        "enabled={} frames={} elapsed_ns={} ns_per_simulated_frame={:.1} snapshots={}",
        timing::enabled(),
        frames,
        elapsed.as_nanos(),
        elapsed.as_nanos() as f64 / frames as f64,
        snapshots.len()
    );
}
