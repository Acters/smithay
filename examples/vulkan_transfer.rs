//! Explicit render-node-only hardware probe for the same-frame Vulkan transfer engine.
//!
//! Build (does not run hardware):
//! ```text
//! cargo build --example vulkan_transfer --no-default-features \
//!   --features backend_gbm,renderer_gl,renderer_multi,backend_vulkan
//! ```
//! Run with explicitly selected render nodes, never card nodes:
//! ```text
//! timeout --signal=TERM --kill-after=5s 60s target/debug/examples/vulkan_transfer \
//!   --source /dev/dri/renderD129 --target /dev/dri/renderD128 --format abgr8888
//! ```
//! No session backend, DrmDeviceFd, KMS object, Wayland/X11 connection, or DRM master ioctl
//! is used. EGL displays are created directly from GBM devices backed by the supplied render
//! fds. Vendor loaders/ICDs still run inside this process; run only when hardware testing is
//! authorized. Direct-engine mode never substitutes CPU copies for Vulkan transfers.
//!
//! Add `--multigpu` to exercise GpuManager/MultiRenderer instead, with two alternating
//! offscreen outputs, partial damage, mixed sizes and cache invalidation. That mode permits
//! the real renderer's route selection/fallbacks: pixel success alone does NOT prove Vulkan
//! was used. Run with `RUST_LOG=smithay::backend::renderer::multigpu=debug` and inspect the
//! successful Vulkan-copy route messages, including after each cache invalidation.
//! Add `--multigpu --direct-target` to use three original target LINEAR dma-bufs instead of
//! renderbuffers. This verifies exact outside-damage preservation (including L-shaped and
//! >3-region damage), same-target GLES capture, fallback-to-direct transitions and MultiFrame
//! blit flushes. Proof of the no-intermediate-blit route additionally requires the log message
//! `submitted direct Vulkan framebuffer transfer`. This mode never activates scanout.
//! Add `--pool-stress --frames 64` in direct-engine mode with `SMITHAY_FRAME_TIMING=1`
//! to check bounded resource reuse and retained old fences across both sizes and engine
//! destruction, with one concurrent fence-wait worker. This does not assert CPU submit latency.

use std::{
    error::Error,
    fs::{File, OpenOptions},
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::fs::{FileTypeExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
    sync::mpsc::{SyncSender, TrySendError, sync_channel},
    thread::JoinHandle,
    time::{Duration, Instant},
};

use clap::{Parser, ValueEnum};
use smithay::{
    backend::{
        allocator::{
            Allocator, Buffer as AllocatorBuffer, Format, Fourcc, Modifier,
            dmabuf::{AsDmabuf, Dmabuf},
            gbm::{GbmAllocator, GbmBufferFlags, GbmDevice},
        },
        drm::{DrmNode, NodeType},
        egl::{EGLContext, EGLDisplay},
        renderer::{
            Bind, BlitFrame, Color32F, ExportMem, Frame, ImportDma, Offscreen, Renderer, TextureFilter,
            TextureMapping,
            gles::{GlesRenderbuffer, GlesRenderer, GlesTexture},
            multigpu::{
                ApiDevice, GpuManager,
                gbm::GbmGlesBackend,
                timing::{self, Counter},
                vkbridge::VkBridge,
            },
            sync::SyncPoint,
        },
    },
    utils::{Buffer, DeviceFd, Physical, Rectangle, Transform},
};

type ProbeResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PixelFormat {
    Abgr8888,
    Argb8888,
    Abgr2101010,
    Argb2101010,
}

impl From<PixelFormat> for Fourcc {
    fn from(value: PixelFormat) -> Self {
        match value {
            PixelFormat::Abgr8888 => Fourcc::Abgr8888,
            PixelFormat::Argb8888 => Fourcc::Argb8888,
            PixelFormat::Abgr2101010 => Fourcc::Abgr2101010,
            PixelFormat::Argb2101010 => Fourcc::Argb2101010,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Non-master GBM/EGL/Vulkan cross-GPU pixel and fence probe")]
struct Args {
    /// Source GPU render node. Primary/card nodes and symlinks are refused before open.
    #[arg(long)]
    source: PathBuf,
    /// Target GPU render node, distinct from source.
    #[arg(long)]
    target: PathBuf,
    #[arg(long, value_enum, default_value = "abgr8888")]
    format: PixelFormat,
    /// Frames per size (direct) or per output per round (multigpu), excluding warmup.
    #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u32).range(6..=256))]
    frames: u32,
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(64..=2048))]
    width: u32,
    #[arg(long, default_value_t = 160, value_parser = clap::value_parser!(u32).range(64..=2048))]
    height: u32,
    /// Force an explicit source modifier (decimal or 0xHEX); otherwise use native EGL formats.
    #[arg(long, value_parser = parse_modifier, conflicts_with = "multigpu")]
    source_modifier: Option<u64>,
    /// Test the real MultiRenderer path; Vulkan route proof additionally requires debug logs.
    #[arg(long)]
    multigpu: bool,
    /// Request SCANOUT|RENDERING destination allocations (still no KMS/display access).
    #[arg(long, conflicts_with = "multigpu")]
    scanout_candidate: bool,
    /// Write three original target LINEAR dma-bufs via MultiRenderer; inspect direct-route logs.
    #[arg(long, requires = "multigpu", conflicts_with = "scanout_candidate")]
    direct_target: bool,
    /// Stress pooled fences/reuse; requires --frames >=64 and SMITHAY_FRAME_TIMING=1.
    #[arg(long, conflicts_with_all = ["multigpu", "direct_target"])]
    pool_stress: bool,
}

fn parse_modifier(value: &str) -> Result<u64, std::num::ParseIntError> {
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse(),
    }
}

// Do not replace this with DrmDeviceFd::new: that constructor attempts SET_MASTER.
fn render_node(path: &Path) -> ProbeResult<(DrmNode, File)> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let suffix = name.strip_prefix("renderD").unwrap_or("");
    if path.parent() != Some(Path::new("/dev/dri"))
        || suffix.is_empty()
        || !suffix.bytes().all(|b| b.is_ascii_digit())
    {
        return Err(format!("refusing non-render-node path: {}", path.display()).into());
    }
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_char_device() {
        return Err(format!("refusing symlink/non-device: {}", path.display()).into());
    }
    let node = DrmNode::from_path(path)?;
    if node.ty() != NodeType::Render {
        return Err(format!("refusing non-render DRM device: {}", path.display()).into());
    }
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    if DrmNode::from_file(&file)? != node {
        return Err("render-node identity changed while opening".into());
    }
    Ok((node, file))
}

struct Gpu {
    renderer: GlesRenderer,
    allocator: GbmAllocator<DeviceFd>,
    render_formats: Vec<Format>,
    texture_formats: Vec<Format>,
}

impl Gpu {
    fn new(file: File) -> ProbeResult<Self> {
        let fd = DeviceFd::from(OwnedFd::from(file));
        let gbm = GbmDevice::new(fd)?;
        let allocator = GbmAllocator::new(gbm.clone(), GbmBufferFlags::RENDERING);
        // GBM platform only: no default EGL display or EGL device enumeration.
        let display = unsafe { EGLDisplay::new(gbm) }?;
        let render_formats = display.dmabuf_render_formats().iter().copied().collect();
        let texture_formats = display.dmabuf_texture_formats().iter().copied().collect();
        let context = EGLContext::new(&display)?;
        let renderer = unsafe { GlesRenderer::new(context) }?;
        Ok(Self {
            renderer,
            allocator,
            render_formats,
            texture_formats,
        })
    }
}

const COLORS: [[u8; 4]; 10] = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
    [0, 255, 255, 255],
    [255, 0, 255, 255],
    [255, 255, 255, 255],
    [0, 0, 0, 255],
    [32, 96, 160, 255],
    [192, 48, 112, 255],
];

fn quadrants(w: i32, h: i32) -> [Rectangle<i32, Buffer>; 4] {
    // Unequal widths/heights make flips, swaps and stale subregions unambiguous.
    let x = w / 3;
    let y = h * 2 / 5;
    [
        Rectangle::new((0, 0).into(), (x, y).into()),
        Rectangle::new((x, 0).into(), (w - x, y).into()),
        Rectangle::new((0, y).into(), (x, h - y).into()),
        Rectangle::new((x, y).into(), (w - x, h - y).into()),
    ]
}

fn render_source(
    renderer: &mut GlesRenderer,
    src: &mut Dmabuf,
    rects: &[Rectangle<i32, Buffer>; 4],
    colors: &[usize; 4],
    update: Option<usize>,
) -> ProbeResult<SyncPoint> {
    let size = src.size();
    let mut target = renderer.bind(src)?;
    let mut frame = renderer.render(&mut target, (size.w, size.h).into(), Transform::Normal)?;
    paint(&mut frame, rects, colors, update)?;
    Ok(frame.finish()?)
}

fn paint<F: Frame>(
    frame: &mut F,
    rects: &[Rectangle<i32, Buffer>; 4],
    colors: &[usize; 4],
    update: Option<usize>,
) -> Result<(), F::Error> {
    for (index, rect) in rects.iter().enumerate() {
        if update.is_some_and(|only| only != index) {
            continue;
        }
        let rgba = COLORS[colors[index]];
        let color = Color32F::new(
            rgba[0] as f32 / 255.,
            rgba[1] as f32 / 255.,
            rgba[2] as f32 / 255.,
            1.,
        );
        let physical = Rectangle::<i32, Physical>::new(
            (rect.loc.x, rect.loc.y).into(),
            (rect.size.w, rect.size.h).into(),
        );
        frame.clear(color, &[physical])?;
    }
    Ok(())
}

fn verify<T>(
    renderer: &mut GlesRenderer,
    offscreen: &mut T,
    w: i32,
    h: i32,
    colors: &[usize; 4],
    release: &SyncPoint,
    frame: u32,
) -> ProbeResult<()>
where
    GlesRenderer: Bind<T>,
{
    // This readback is deliberately delayed until AFTER the next Vulkan submission. Its
    // CPU wait therefore cannot accidentally replace that submission's destination-reader
    // dependency. No direct reading of the copied dma-buf is used as the test oracle.
    release.wait()?;
    let bytes = capture(renderer, offscreen, w, h)?;
    // With Smithay's Normal GLES projection, buffer-coordinate y=0 is GL row zero.
    // Compare raw rows; mapping.flipped() describes reimport orientation, not a reason to
    // silently accept a vertically flipped transfer. Fourcc ABGR8888 is RGBA bytes on LE.
    for y in 0..h {
        for x in 0..w {
            let quadrant = usize::from(x >= w / 3) + 2 * usize::from(y >= h * 2 / 5);
            let expected = COLORS[colors[quadrant]];
            let offset = (y as usize * w as usize + x as usize) * 4;
            let actual = &bytes[offset..offset + 4];
            if actual.iter().zip(expected).any(|(&a, e)| a.abs_diff(e) > 2) {
                return Err(format!(
                    "frame {frame} {w}x{h}: pixel ({x},{y}) got {actual:?}, expected {expected:?}"
                )
                .into());
            }
        }
    }
    println!("PASS pixels frame={frame} size={w}x{h} (all pixels, RGBA tolerance=2)");
    Ok(())
}

// No fence wait is inserted here. Direct-target callers deliberately rely on MultiRenderer
// having queued the external completion wait before this GLES readback of the SAME target.
fn capture<T>(renderer: &mut GlesRenderer, target: &mut T, w: i32, h: i32) -> ProbeResult<Vec<u8>>
where
    GlesRenderer: Bind<T>,
{
    let target = renderer.bind(target)?;
    let mapping =
        renderer.copy_framebuffer(&target, Rectangle::from_size((w, h).into()), Fourcc::Abgr8888)?;
    if TextureMapping::format(&mapping) != Fourcc::Abgr8888 {
        return Err("readback did not provide requested RGBA8 layout".into());
    }
    let bytes = renderer.map_texture(&mapping)?;
    if bytes.len() != w as usize * h as usize * 4 {
        return Err(format!("unexpected readback length {}", bytes.len()).into());
    }
    Ok(bytes.to_vec())
}

// Declared after the original allocations: even an oracle/submit error must retire
// published producer, Vulkan and target-reader work before those allocations drop.
#[derive(Default)]
struct RetireFences {
    producer: SyncPoint,
    copy: SyncPoint,
    reader: SyncPoint,
}

impl Drop for RetireFences {
    fn drop(&mut self) {
        for fence in [&self.producer, &self.copy, &self.reader] {
            while fence.wait().is_err() {
                std::thread::yield_now();
            }
        }
    }
}

fn check_logical_fence(fence: &SyncPoint) -> ProbeResult<()> {
    // Check BEFORE wait: never silently wait for a reset/reused newer submission.
    if !fence.is_reached() {
        return Err("completed pooled fence regressed to unsignaled".into());
    }
    fence.wait()?;
    if !fence.is_reached() {
        return Err("completed pooled fence regressed after wait".into());
    }
    Ok(())
}

// Separate first native publication from regression of a previously observed signal.
// This diagnostic allowance does NOT establish that driver publication lag is legal.
const INITIAL_NATIVE_DEADLINE: Duration = Duration::from_millis(500);

#[derive(Default)]
struct NativeStats {
    checks: u64,
    initial_misses: u64,
    max_observed_latency: Duration,
}

struct OldFence {
    logical: SyncPoint,
    // Exactly one export per record: never replace an fd after a miss or timeout.
    native: Option<OwnedFd>,
    ever_native_signaled: bool,
    initial_miss: bool,
    logical_completed_at: Instant,
}

impl OldFence {
    fn new(fence: &SyncPoint) -> ProbeResult<Self> {
        check_logical_fence(fence)?;
        let logical_completed_at = Instant::now();
        let native = fence.export();
        if native.is_none() && fence.is_exportable() {
            return Err("exportable old fence failed native export".into());
        }
        Ok(Self {
            logical: fence.clone(),
            native,
            ever_native_signaled: false,
            initial_miss: false,
            logical_completed_at,
        })
    }

    fn check(&mut self, stats: &mut NativeStats, settle_initial: bool) -> ProbeResult<()> {
        check_logical_fence(&self.logical)?;
        let Some(fd) = self.native.as_ref() else {
            return Ok(());
        };
        let deadline = self.logical_completed_at + INITIAL_NATIVE_DEADLINE;
        let mut timeout = 0;
        loop {
            let mut pollfd = libc::pollfd {
                fd: fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: the single pollfd and its owned descriptor remain live for poll.
            let result = unsafe { libc::poll(&mut pollfd, 1, timeout) };
            if result < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::Interrupted {
                    return Err(err.into());
                }
                if Instant::now() >= deadline && !self.ever_native_signaled {
                    return Err("initial native publication deadline expired during EINTR".into());
                }
                // Recompute remaining time below, never restart the deadline.
            } else {
                stats.checks += 1;
                if pollfd.revents & (libc::POLLERR | libc::POLLNVAL) != 0
                    || (result > 0 && pollfd.revents & libc::POLLIN == 0)
                {
                    return Err(format!(
                        "native fence poll error: result={result} events={}",
                        pollfd.revents
                    )
                    .into());
                }
                if result == 1 && pollfd.revents & libc::POLLIN != 0 {
                    if !self.ever_native_signaled {
                        let latency = self.logical_completed_at.elapsed();
                        if latency >= INITIAL_NATIVE_DEADLINE {
                            return Err(format!(
                                "initial native publication observed after deadline: {latency:?}"
                            )
                            .into());
                        }
                        stats.max_observed_latency = stats.max_observed_latency.max(latency);
                        self.ever_native_signaled = true;
                    }
                    return Ok(());
                }
                if self.ever_native_signaled {
                    return Err("previously signaled old native fence regressed to not ready".into());
                }
                if !self.initial_miss {
                    self.initial_miss = true;
                    stats.initial_misses += 1;
                }
                if Instant::now() >= deadline {
                    return Err("initial native publication timed out (500ms, same fd)".into());
                }
                if !settle_initial {
                    return Ok(());
                }
            }
            timeout = if settle_initial && !self.ever_native_signaled {
                let remaining = deadline.saturating_duration_since(Instant::now());
                remaining.as_millis().saturating_add(1).min(500) as i32
            } else {
                0
            };
        }
    }
}

#[derive(Default)]
struct WorkerReport {
    completed: u64,
    native: NativeStats,
}

struct PoolStress {
    old: Vec<OldFence>,
    sender: Option<SyncSender<SyncPoint>>,
    worker: Option<JoinHandle<Result<WorkerReport, String>>>,
    // Created, reused, recycled, destroyed, busy, dirty signal replacements.
    // Logical batches are NOT allocations.
    counts: [u64; 6],
    submitted: u64,
    worker_queued: u64,
    native: NativeStats,
    worker_native: NativeStats,
}

impl PoolStress {
    fn new() -> ProbeResult<Self> {
        let (sender, receiver) = sync_channel::<SyncPoint>(2);
        let worker = std::thread::Builder::new()
            .name("pool-fence-wait".into())
            .spawn(move || {
                let mut report = WorkerReport::default();
                for fence in receiver {
                    while fence.wait().is_err() {
                        std::thread::yield_now();
                    }
                    let mut old = OldFence::new(&fence).map_err(|err| err.to_string())?;
                    // Only this worker may block for initial native publication while
                    // production continues. Keep and poll the SAME export throughout.
                    old.check(&mut report.native, true)
                        .map_err(|err| err.to_string())?;
                    old.check(&mut report.native, false)
                        .map_err(|err| err.to_string())?;
                    report.completed += 1;
                }
                Ok(report)
            })?;
        Ok(Self {
            old: Vec::new(),
            sender: Some(sender),
            worker: Some(worker),
            counts: [0; 6],
            submitted: 0,
            worker_queued: 0,
            native: NativeStats::default(),
            worker_native: NativeStats::default(),
        })
    }

    fn check_old(&mut self) -> ProbeResult<()> {
        for fence in &mut self.old {
            fence.check(&mut self.native, false)?;
        }
        Ok(())
    }

    fn submitted(&mut self, fence: &SyncPoint) -> ProbeResult<()> {
        self.submitted += 1;
        // Never wait for the worker/queue on the rendering thread. A busy worker samples
        // fewer submissions; every completion still enters the main-thread old-fence oracle.
        match self.sender.as_ref().unwrap().try_send(fence.clone()) {
            Ok(()) => self.worker_queued += 1,
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Disconnected(_)) => {
                self.sender.take();
                let reason = match self.worker.take().unwrap().join() {
                    Ok(Err(error)) => error,
                    Ok(Ok(report)) => format!("worker exited early after {} waits", report.completed),
                    Err(_) => "worker panicked".to_owned(),
                };
                return Err(format!("fence worker disconnected: {reason}").into());
            }
        }
        self.check_old()
    }

    fn completed(&mut self, fence: &SyncPoint) -> ProbeResult<()> {
        let mut old = OldFence::new(fence)?;
        old.check(&mut self.native, false)?;
        self.old.push(old);
        Ok(())
    }

    fn snapshot(&mut self) -> [u64; 6] {
        let counters = [
            Counter::ResourceSetsCreated,
            Counter::ResourceSetsReused,
            Counter::ResourceSetsRecycled,
            Counter::ResourceSetsDestroyed,
            Counter::ResourcePoolBusy,
            Counter::SignalSemaphoresReplaced,
        ];
        let mut delta = [0; 6];
        for snapshot in timing::drain() {
            for (counter, value) in snapshot.counters {
                if let Some(index) = counters.iter().position(|&c| c == counter) {
                    delta[index] += value;
                }
            }
        }
        for (total, value) in self.counts.iter_mut().zip(delta) {
            *total += value;
        }
        delta
    }

    fn settle(&mut self) -> ProbeResult<u64> {
        self.sender.take();
        match self.worker.take().unwrap().join() {
            Ok(Ok(report)) => {
                self.worker_native = report.native;
                Ok(report.completed)
            }
            Ok(Err(err)) => Err(err.into()),
            Err(_) => Err("fence worker panicked".into()),
        }
    }

    fn prove_before_drop(&mut self) -> ProbeResult<u64> {
        // Production has stopped: settle any remaining initial publications against
        // their ORIGINAL deadlines before joining and before destroying the engine.
        for fence in &mut self.old {
            fence.check(&mut self.native, true)?;
        }
        let waited = self.settle()?;
        self.check_old()?;
        self.snapshot();
        let [created, reused, _, destroyed, busy, signals_replaced] = self.counts;
        if created == 0
            || created > 8
            || reused < self.submitted.saturating_sub(8)
            || destroyed != 0
            || busy != 0
            || signals_replaced != 0
            || waited == 0
            || waited != self.worker_queued
            || self.old.len() as u64 != self.submitted
        {
            return Err(format!(
                "pool proof failed: counts={:?} submitted={} old={} worker={waited}/{}",
                self.counts,
                self.submitted,
                self.old.len(),
                self.worker_queued
            )
            .into());
        }
        Ok(waited)
    }
}

impl Drop for PoolStress {
    fn drop(&mut self) {
        // On every error path close the bounded channel and settle the only worker.
        self.sender.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn run_size(
    bridge: &mut VkBridge,
    source: &mut Gpu,
    target: &mut Gpu,
    args: &Args,
    w: i32,
    h: i32,
    mut stress: Option<&mut PoolStress>,
) -> ProbeResult<()> {
    let format = Fourcc::from(args.format);
    let mut modifiers: Vec<_> = source
        .render_formats
        .iter()
        .filter(|f| f.code == format && f.modifier != Modifier::Invalid)
        .map(|f| f.modifier)
        .collect();
    let transfer_modifiers = bridge.source_modifiers(format, w as u32, h as u32)?;
    modifiers.retain(|modifier| transfer_modifiers.contains(modifier));
    modifiers.sort_by_key(|&m| (m == Modifier::Linear, u64::from(m)));
    modifiers.dedup();
    println!(
        "NEGOTIATE source modifiers shared by EGL/Vulkan: {:?}",
        modifiers.iter().copied().map(u64::from).collect::<Vec<_>>()
    );
    if let Some(modifier) = args.source_modifier {
        let modifier = Modifier::from(modifier);
        if !modifiers.contains(&modifier) {
            return Err(format!(
                "requested source modifier {modifier:?} is not an explicit EGL render format"
            )
            .into());
        }
        modifiers = vec![modifier];
    }
    if modifiers.is_empty() {
        return Err("no explicit native source render modifiers".into());
    }
    if !target.texture_formats.contains(&Format {
        code: format,
        modifier: Modifier::Linear,
    }) {
        return Err("target EGL does not advertise requested LINEAR sample format".into());
    }
    let source_bo = source
        .allocator
        .create_buffer(w as u32, h as u32, format, &modifiers)?;
    // Match MultiRenderer's allocation path: request the explicit modifier,
    // not GBM's legacy LINEAR usage flag (which produces an implicit descriptor
    // on builds without create_with_modifiers2).
    let destination_bo = if args.scanout_candidate {
        if !cfg!(feature = "backend_gbm_has_create_with_modifiers2") {
            return Err("--scanout-candidate requires backend_gbm_has_create_with_modifiers2 so GBM receives usage flags together with explicit modifiers".into());
        }
        println!(
            "SCANOUT CANDIDATE: requesting SCANOUT|RENDERING with explicit LINEAR; KMS admissibility is NOT tested"
        );
        target.allocator.create_buffer_with_flags(
            w as u32,
            h as u32,
            format,
            &[Modifier::Linear],
            GbmBufferFlags::SCANOUT | GbmBufferFlags::RENDERING,
        )?
    } else {
        target
            .allocator
            .create_buffer(w as u32, h as u32, format, &[Modifier::Linear])?
    };
    let mut src = source_bo.export()?;
    let dst = destination_bo.export()?;
    if dst.format().modifier != Modifier::Linear || src.format().modifier == Modifier::Invalid {
        return Err("GBM returned implicit source or non-LINEAR destination".into());
    }
    println!(
        "ALLOC size={w}x{h} source={:?}/{} planes={} destination={:?}/{} planes={}",
        src.format().code,
        u64::from(src.format().modifier),
        src.num_planes(),
        dst.format().code,
        u64::from(dst.format().modifier),
        dst.num_planes()
    );
    // A separate renderbuffer proves target-GLES sampling, not merely Vulkan write success.
    let mut offscreen: GlesRenderbuffer =
        Offscreen::create_buffer(&mut target.renderer, Fourcc::Abgr8888, (w, h).into())?;
    let rects = quadrants(w, h);
    let full = Rectangle::<i32, Buffer>::from_size((w, h).into());
    // Include mid-tones from the first frame, including the minimum --frames run.
    let mut colors = [0, 1, 8, 9];
    let mut previous_colors = colors;
    let mut source_release = SyncPoint::signaled();
    let mut target_release: Option<SyncPoint> = None;
    let mut retire = RetireFences::default();
    for sequence in 0..args.frames {
        if let Some(stress) = stress.as_deref_mut() {
            stress.check_old()?;
        }
        // Server wait when native export is available; explicit CPU wait otherwise.
        source.renderer.wait(&source_release)?;
        let update = if sequence == 0 {
            None
        } else {
            Some((sequence as usize - 1) % 4)
        };
        if let Some(index) = update {
            colors[index] = (colors[index] + 1 + sequence as usize % 7) % COLORS.len();
        }
        let acquire = render_source(&mut source.renderer, &mut src, &rects, &colors, update)?;
        retire.producer = acquire.clone();
        let damage = update.map(|index| rects[index]).unwrap_or(full);
        let started = Instant::now();
        let copy = bridge.copy(&src, &dst, &acquire, target_release.as_ref(), &[damage])?;
        retire.copy = copy.clone();
        if let Some(stress) = stress.as_deref_mut() {
            stress.submitted(&copy)?;
        }
        println!(
            "SUBMIT frame={sequence} damage={damage:?} elapsed_us={} producer_native={} copy_native={} complete_at_return={}",
            started.elapsed().as_micros(),
            acquire.is_exportable(),
            copy.is_exportable(),
            copy.is_reached()
        );
        // Old target read is still represented by its release when this copy is submitted.
        // Only now validate the previous output, before overwriting the offscreen storage.
        if let Some(release) = &target_release {
            verify(
                &mut target.renderer,
                &mut offscreen,
                w,
                h,
                &previous_colors,
                release,
                sequence - 1,
            )?;
            if let Some(stress) = stress.as_deref_mut() {
                // Target readback proves the preceding Vulkan copy completed. Do not
                // add a CPU wait before the NEXT submission to manufacture completion.
                stress.completed(&source_release)?;
                stress.check_old()?;
            }
        }
        source_release = copy.clone();
        target.renderer.wait(&copy)?;
        let texture = target.renderer.import_dmabuf(&dst, Some(&[damage]))?;
        let mut framebuffer = target.renderer.bind(&mut offscreen)?;
        let mut frame = target
            .renderer
            .render(&mut framebuffer, (w, h).into(), Transform::Normal)?;
        frame.clear(
            Color32F::new(0., 0., 0., 1.),
            &[Rectangle::from_size((w, h).into())],
        )?;
        frame.render_texture_at(
            &texture,
            (0, 0).into(),
            1,
            1.,
            Transform::Normal,
            &[Rectangle::from_size((w, h).into())],
            &[],
            1.,
        )?;
        target_release = Some(frame.finish()?);
        retire.reader = target_release.as_ref().unwrap().clone();
        previous_colors = colors;
        if let Some(stress) = stress.as_deref_mut() {
            let delta = stress.snapshot();
            // Allow a generous 16-submission warmup at each size, but no steady-state
            // resource-set churn. Resize imports/staging are not resource-set counters.
            if sequence >= 16 && (delta[0] != 0 || delta[3] != 0 || delta[5] != 0) {
                return Err(format!("pool resource churn after warmup: {delta:?}").into());
            }
        }
    }
    verify(
        &mut target.renderer,
        &mut offscreen,
        w,
        h,
        &previous_colors,
        target_release.as_ref().unwrap(),
        args.frames - 1,
    )?;
    source_release.wait()?;
    if let Some(stress) = stress {
        stress.completed(&source_release)?;
        stress.check_old()?;
    }
    // Both allocations are recreated on the next run_size call; the same engine keeps its
    // bounded stable-identity cache. Original GBM BOs remain alive for this entire round.
    Ok(())
}

type MultiGpuManager = GpuManager<GbmGlesBackend<GlesRenderer, DeviceFd>>;

struct MultiOutput {
    buffer: GlesRenderbuffer,
    width: i32,
    height: i32,
    colors: [usize; 4],
    release: SyncPoint,
    sequence: u32,
}

fn draw_multi_output(
    manager: &mut MultiGpuManager,
    source: &DrmNode,
    target: &DrmNode,
    format: Fourcc,
    output: &mut MultiOutput,
    update: Option<usize>,
) -> ProbeResult<()> {
    let mut renderer = manager.renderer(source, target, format)?;
    let mut framebuffer = renderer.bind(&mut output.buffer)?;
    let mut frame = renderer.render(
        &mut framebuffer,
        (output.width, output.height).into(),
        Transform::Normal,
    )?;
    paint(
        &mut frame,
        &quadrants(output.width, output.height),
        &output.colors,
        update,
    )?;
    output.release = frame.finish()?;
    Ok(())
}

fn verify_multi_output(
    manager: &mut MultiGpuManager,
    target: &DrmNode,
    output: &mut MultiOutput,
    round: usize,
    index: usize,
) -> ProbeResult<()> {
    let device = manager
        .devices_mut()?
        .find(|device| device.node() == target)
        .ok_or("target renderer disappeared")?;
    println!(
        "VERIFY multigpu round={round} output={index} frame={}",
        output.sequence
    );
    verify(
        device.renderer_mut(),
        &mut output.buffer,
        output.width,
        output.height,
        &output.colors,
        &output.release,
        output.sequence,
    )
}

// Direct-target mode treats these dma-bufs as leased original output buffers. They are
// retained across writes/captures and never submitted to KMS by this render-node-only probe.
struct DirectOutput {
    dmabuf: Dmabuf,
    scratch: GlesRenderbuffer,
    scratch_expected: Option<Vec<[u8; 4]>>,
    shared_texture: Option<GlesTexture>,
    shared_capture: GlesRenderbuffer,
    width: i32,
    height: i32,
    expected: Vec<[u8; 4]>,
    before: Vec<u8>,
    damage: Vec<Rectangle<i32, Buffer>>,
    release: SyncPoint,
    blit_release: Option<SyncPoint>,
}

impl Drop for DirectOutput {
    fn drop(&mut self) {
        for release in std::iter::once(&self.release).chain(self.blit_release.as_ref()) {
            while release.wait().is_err() {
                std::thread::yield_now();
            }
        }
    }
}

fn rgba_color(rgba: [u8; 4]) -> Color32F {
    Color32F::new(
        rgba[0] as f32 / 255.,
        rgba[1] as f32 / 255.,
        rgba[2] as f32 / 255.,
        rgba[3] as f32 / 255.,
    )
}

fn clipped(rect: Rectangle<i32, Buffer>, width: i32, height: i32) -> Option<Rectangle<i32, Buffer>> {
    let left = rect.loc.x.max(0);
    let top = rect.loc.y.max(0);
    let right = rect.loc.x.saturating_add(rect.size.w).min(width);
    let bottom = rect.loc.y.saturating_add(rect.size.h).min(height);
    (left < right && top < bottom)
        .then(|| Rectangle::new((left, top).into(), (right - left, bottom - top).into()))
}

fn contains_pixel(rect: &Rectangle<i32, Buffer>, x: i32, y: i32) -> bool {
    x >= rect.loc.x && y >= rect.loc.y && x < rect.loc.x + rect.size.w && y < rect.loc.y + rect.size.h
}

fn model_patch(
    pixels: &mut [[u8; 4]],
    width: i32,
    height: i32,
    rect: Rectangle<i32, Buffer>,
    color: [u8; 4],
) {
    if let Some(rect) = clipped(rect, width, height) {
        for y in rect.loc.y..rect.loc.y + rect.size.h {
            for x in rect.loc.x..rect.loc.x + rect.size.w {
                pixels[(y * width + x) as usize] = color;
            }
        }
    }
}

fn direct_damage(width: i32, height: i32, sequence: u32) -> Vec<Rectangle<i32, Buffer>> {
    let rect = |x, y, w, h| Rectangle::new((x, y).into(), (w, h).into());
    match sequence % 4 {
        0 => vec![
            rect(width / 8, height / 8, width / 8, height / 8),
            rect(width * 5 / 8, height * 5 / 8, width / 6, height / 6),
        ],
        // Touching L: merging these to a bounding rectangle destroys the valid inner gap.
        1 => vec![
            rect(width / 4, height / 4, width / 8, height / 2),
            rect(width * 3 / 8, height * 5 / 8, width / 3, height / 8),
        ],
        // Five separated regions must NEVER become full-frame due to a CPU damage cap.
        2 => (0..5)
            .map(|i| {
                rect(
                    width * (2 * i + 1) / 12,
                    if i % 2 == 0 { height / 6 } else { height * 2 / 3 },
                    width / 16,
                    height / 8,
                )
            })
            .collect(),
        // Include out-of-bounds input. Only clipped pixels are part of actual damage.
        _ => vec![
            rect(-width / 16, height / 3, width / 8, height / 8),
            rect(width * 15 / 16, height * 3 / 4, width / 8, height / 8),
            rect(width / 2, -height / 16, width / 8, height / 8),
        ],
    }
}

fn assert_direct_pixels(
    output: &DirectOutput,
    bytes: &[u8],
    label: &str,
    check_outside: bool,
) -> ProbeResult<()> {
    if bytes.len() != output.expected.len() * 4 {
        return Err(format!("{label}: readback size mismatch").into());
    }
    let mut outside = 0;
    for y in 0..output.height {
        for x in 0..output.width {
            let pixel = (y * output.width + x) as usize;
            let actual = &bytes[pixel * 4..pixel * 4 + 4];
            if check_outside && !output.damage.iter().any(|r| contains_pixel(r, x, y)) {
                outside += 1;
                if actual != &output.before[pixel * 4..pixel * 4 + 4] {
                    return Err(format!(
                        "{label}: OUTSIDE DAMAGE changed at ({x},{y}): was {:?}, now {actual:?}",
                        &output.before[pixel * 4..pixel * 4 + 4]
                    )
                    .into());
                }
            }
            let expected = output.expected[pixel];
            if actual.iter().zip(expected).any(|(&a, e)| a.abs_diff(e) > 2) {
                return Err(format!("{label}: pixel ({x},{y}) got {actual:?}, expected {expected:?}").into());
            }
        }
    }
    if check_outside && outside == 0 {
        return Err(format!("{label}: test did not retain any outside-damage pixels").into());
    }
    println!("PASS {label}: all pixels, {outside} exact unchanged outside-damage pixels");
    Ok(())
}

fn assert_blit_snapshot(output: &DirectOutput, bytes: &[u8]) -> ProbeResult<()> {
    let expected = output
        .scratch_expected
        .as_ref()
        .ok_or("pre-continuation blit model missing")?;
    if bytes.len() != expected.len() * 4 {
        return Err("blit snapshot size mismatch".into());
    }
    for (pixel, (actual, expected)) in bytes.chunks_exact(4).zip(expected).enumerate() {
        if actual.iter().zip(expected).any(|(&a, &e)| a.abs_diff(e) > 2) {
            return Err(format!(
                "PRE-continuation blit snapshot pixel ({},{}) got {actual:?}, expected {expected:?}",
                pixel % output.width as usize,
                pixel / output.width as usize
            )
            .into());
        }
    }
    println!("PASS blit_to PRE-continuation snapshot (not the final resumed target model)");
    Ok(())
}

fn seed_direct_output(
    manager: &mut MultiGpuManager,
    target: &DrmNode,
    output: &mut DirectOutput,
    index: usize,
) -> ProbeResult<()> {
    let device = manager
        .devices_mut()?
        .find(|device| device.node() == target)
        .ok_or("target missing")?;
    let renderer = device.renderer_mut();
    let colors = [[0, 1, 8, 9], [4, 5, 9, 8], [2, 3, 6, 7]][index];
    let rects = quadrants(output.width, output.height);
    let original = output.dmabuf.clone();
    {
        let mut framebuffer = renderer.bind(&mut output.dmabuf)?;
        let prepared = renderer
            .prepare_external_framebuffer_write(&mut framebuffer)?
            .ok_or("GLES did not expose original dma-buf target")?;
        if prepared.dmabuf != original {
            return Err("GLES hook substituted a different allocation".into());
        }
        // No external write occurred: complete the mandatory pair with the acquire itself,
        // so renderer-specific shared-texture synchronization is still correctly published.
        renderer.finish_external_framebuffer_write(&mut framebuffer, &prepared.acquire)?;
        let mut frame = renderer.render(
            &mut framebuffer,
            (output.width, output.height).into(),
            Transform::Normal,
        )?;
        paint(&mut frame, &rects, &colors, None)?;
        output.release = frame.finish()?;
    }
    {
        let mut scratch = renderer.bind(&mut output.scratch)?;
        if renderer
            .prepare_external_framebuffer_write(&mut scratch)?
            .is_some()
        {
            return Err("GLES exposed unsupported renderbuffer as external target".into());
        }
    }
    {
        // Cached/imported texture handles are NOT leased original framebuffer bindings.
        // Bind<Dmabuf> above must expose Some; binding its cached GlesTexture must not.
        let mut texture = renderer.import_dmabuf(&output.dmabuf, None)?;
        output.shared_texture = Some(texture.clone());
        let mut framebuffer = renderer.bind(&mut texture)?;
        if renderer
            .prepare_external_framebuffer_write(&mut framebuffer)?
            .is_some()
        {
            return Err("GLES exposed ordinary texture binding as original dma-buf target".into());
        }
    }
    for (rect, color) in rects.into_iter().zip(colors) {
        model_patch(
            &mut output.expected,
            output.width,
            output.height,
            rect,
            COLORS[color],
        );
    }
    let bytes = capture(renderer, &mut output.dmabuf, output.width, output.height)?;
    assert_direct_pixels(output, &bytes, "direct sentinel initialization", false)?;
    output.before = bytes;
    Ok(())
}

fn capture_shared_texture(renderer: &mut GlesRenderer, output: &mut DirectOutput) -> ProbeResult<Vec<u8>> {
    {
        let texture = output.shared_texture.as_ref().ok_or("shared texture missing")?;
        let mut framebuffer = renderer.bind(&mut output.shared_capture)?;
        let mut frame = renderer.render(
            &mut framebuffer,
            (output.width, output.height).into(),
            Transform::Normal,
        )?;
        // No explicit completion wait on this OTHER context. The shared GlesTexture's
        // TextureSync must observe the write fence published by the paired finish hook.
        frame.render_texture_at(
            texture,
            (0, 0).into(),
            1,
            1.,
            Transform::Normal,
            &[Rectangle::from_size((output.width, output.height).into())],
            &[],
            1.,
        )?;
        let _read_completion = frame.finish()?;
    }
    capture(renderer, &mut output.shared_capture, output.width, output.height)
}

fn draw_direct_output(
    manager: &mut MultiGpuManager,
    source: &DrmNode,
    target: &DrmNode,
    format: Fourcc,
    output: &mut DirectOutput,
    sequence: u32,
    index: usize,
    warmup: bool,
) -> ProbeResult<()> {
    output.scratch_expected = None;
    let damage = if warmup {
        vec![Rectangle::from_size((output.width, output.height).into())]
    } else {
        direct_damage(output.width, output.height, sequence + index as u32)
    };
    let patches: Vec<_> = damage
        .iter()
        .enumerate()
        .map(|(i, &rect)| {
            let color = if i == 0 {
                COLORS[8 + sequence as usize % 2]
            } else {
                COLORS[(index * 3 + sequence as usize + i) % COLORS.len()]
            };
            (rect, color)
        })
        .collect();
    output.damage = damage
        .iter()
        .filter_map(|&r| clipped(r, output.width, output.height))
        .collect();
    for &(rect, color) in &patches {
        model_patch(&mut output.expected, output.width, output.height, rect, color);
    }
    if !warmup && sequence % 6 == 2 {
        // Exercise a target-GLES write followed by a later direct write without toggling
        // manager policy: toggling resets engines, which would mask the following blit
        // tests behind asynchronous initialization fallback instead of the direct route.
        let device = manager
            .devices_mut()?
            .find(|device| device.node() == target)
            .ok_or("target missing")?;
        let renderer = device.renderer_mut();
        let mut framebuffer = renderer.bind(&mut output.dmabuf)?;
        let mut frame = renderer.render(
            &mut framebuffer,
            (output.width, output.height).into(),
            Transform::Normal,
        )?;
        for &(rect, color) in &patches {
            // This stage tests a bounded target-GLES write followed by direct
            // access, not GLES's treatment of offscreen damage geometry. Keep
            // its submitted geometry identical to the pixel oracle's clipped area.
            let Some(rect) = clipped(rect, output.width, output.height) else {
                continue;
            };
            let physical = Rectangle::<i32, Physical>::new(
                (rect.loc.x, rect.loc.y).into(),
                (rect.size.w, rect.size.h).into(),
            );
            frame.clear(rgba_color(color), &[physical])?;
        }
        output.release = frame.finish()?;
        println!("TARGET_GLES_FALLBACK: explicit target draw, not a policy toggle or a Vulkan-route claim");
        return Ok(());
    }
    let blit_to = !warmup && sequence % 6 == 3;
    let blit_from = !warmup && sequence % 6 == 4;
    if blit_from {
        let device = manager
            .devices_mut()?
            .find(|device| device.node() == target)
            .ok_or("target missing")?;
        let renderer = device.renderer_mut();
        let mut framebuffer = renderer.bind(&mut output.scratch)?;
        let mut frame = renderer.render(
            &mut framebuffer,
            (output.width, output.height).into(),
            Transform::Normal,
        )?;
        frame.clear(
            rgba_color(COLORS[9]),
            &[Rectangle::from_size((output.width, output.height).into())],
        )?;
        output.blit_release = Some(frame.finish()?);
    }
    let mut renderer = manager.renderer(source, target, format)?;
    let mut framebuffer = renderer.bind(&mut output.dmabuf)?;
    let mut scratch = renderer.bind(&mut output.scratch)?;
    let mut frame = renderer.render(
        &mut framebuffer,
        (output.width, output.height).into(),
        Transform::Normal,
    )?;
    for &(rect, color) in &patches {
        let rect = Rectangle::<i32, Physical>::new(
            (rect.loc.x, rect.loc.y).into(),
            (rect.size.w, rect.size.h).into(),
        );
        frame.clear(rgba_color(color), &[rect])?;
    }
    if blit_to {
        let full = Rectangle::<i32, Physical>::from_size((output.width, output.height).into());
        output.blit_release = Some(frame.blit_to(&mut scratch, full, full, TextureFilter::Nearest)?);
        // The scratch contains the state at the flush, BEFORE the resumed producer draw.
        output.scratch_expected = Some(output.expected.clone());
        println!("BLIT_TO direct-target: MultiFrame flush then capture PRE-continuation snapshot");
        let rect = Rectangle::<i32, Buffer>::new(
            (output.width * 3 / 4, output.height / 12).into(),
            (output.width / 12, output.height / 10).into(),
        );
        let physical = Rectangle::<i32, Physical>::new(
            (rect.loc.x, rect.loc.y).into(),
            (rect.size.w, rect.size.h).into(),
        );
        let color = if sequence / 6 % 2 == 0 {
            [240, 112, 16, 255]
        } else {
            [16, 176, 224, 255]
        };
        // This really draws on the source AFTER target-GLES blitting switched contexts.
        // MultiFrame must restore its producer context and later flush only this new damage.
        frame.clear(rgba_color(color), &[physical])?;
        model_patch(&mut output.expected, output.width, output.height, rect, color);
        output.damage.push(rect);
        println!("PRODUCER_CONTINUATION after blit_to: distinct source patch {rect:?}");
    }
    if blit_from {
        let rect = Rectangle::<i32, Buffer>::new(
            (output.width / 3, output.height / 2).into(),
            (output.width / 7, output.height / 7).into(),
        );
        let physical = Rectangle::<i32, Physical>::new(
            (rect.loc.x, rect.loc.y).into(),
            (rect.size.w, rect.size.h).into(),
        );
        output.blit_release = Some(frame.blit_from(&scratch, physical, physical, TextureFilter::Nearest)?);
        model_patch(&mut output.expected, output.width, output.height, rect, COLORS[9]);
        output.damage.push(rect);
        println!(
            "BLIT_FROM direct-target: MultiFrame flush then target GLES write; next frame must acquire it"
        );
    }
    output.release = frame.finish()?;
    if blit_from
        && output
            .blit_release
            .as_ref()
            .is_some_and(SyncPoint::contains_fence)
        && !output.release.contains_fence()
    {
        return Err("empty MultiFrame finish discarded the retained target blit fence".into());
    }
    Ok(())
}

fn run_direct_targets(
    manager: &mut MultiGpuManager,
    source: &DrmNode,
    target: &DrmNode,
    args: &Args,
) -> ProbeResult<()> {
    manager.set_vulkan_direct_target_enabled(true);
    let shared_context = {
        let device = manager
            .devices_mut()?
            .find(|device| device.node() == target)
            .ok_or("target missing")?;
        let context = device.renderer().egl_context();
        EGLContext::new_shared(context.display(), context)?
    };
    // The context is newly created in the target renderer's EGL share group and used only
    // on this thread; no unrelated GL objects/state are supplied to GlesRenderer.
    let mut shared_renderer = unsafe { GlesRenderer::new(shared_context) }?;
    let format = Fourcc::from(args.format);
    let base = (args.width as i32, args.height as i32);
    let mixed = [base, (base.0 + 32, base.1 + 24), (base.0 + 16, base.1 + 8)];
    println!(
        "DIRECT-TARGET route proof requires 'submitted direct Vulkan framebuffer transfer' logs. Pixel PASS alone cannot distinguish fallback; no KMS access or scanout activation."
    );
    for (round, sizes) in [[base; 3], mixed, [base; 3]].into_iter().enumerate() {
        if round != 0 {
            println!("INVALIDATE direct-target caches round={round}");
            manager.invalidate_caches()?;
        }
        let mut outputs = Vec::new();
        for (width, height) in sizes {
            let device = manager
                .devices_mut()?
                .find(|device| device.node() == target)
                .ok_or("target missing")?;
            let dmabuf =
                device
                    .allocator()
                    .create_buffer(width as u32, height as u32, format, &[Modifier::Linear])?;
            if dmabuf.format().modifier != Modifier::Linear
                || dmabuf.format().code != format
                || dmabuf.size().w != width
                || dmabuf.size().h != height
                || dmabuf.num_planes() != 1
            {
                return Err(
                    "direct destination is not the requested original single-plane LINEAR descriptor".into(),
                );
            }
            let scratch: GlesRenderbuffer =
                Offscreen::create_buffer(device.renderer_mut(), format, (width, height).into())?;
            let shared_capture: GlesRenderbuffer =
                Offscreen::create_buffer(&mut shared_renderer, Fourcc::Abgr8888, (width, height).into())?;
            outputs.push(DirectOutput {
                dmabuf,
                scratch,
                scratch_expected: None,
                shared_texture: None,
                shared_capture,
                width,
                height,
                expected: vec![[0; 4]; width as usize * height as usize],
                before: Vec::new(),
                damage: Vec::new(),
                release: SyncPoint::signaled(),
                blit_release: None,
            });
        }
        let mut successes = 0;
        let mut last_error = None;
        for attempt in 0..20 {
            match draw_direct_output(manager, source, target, format, &mut outputs[0], attempt, 0, true) {
                Ok(()) => successes += 1,
                Err(err) => {
                    eprintln!("WARMUP direct-target round={round} attempt={attempt}: {err}");
                    last_error = Some(err);
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if successes == 0 {
            return Err(last_error.unwrap_or_else(|| "direct-target warmup failed".into()));
        }
        {
            // Exercise retained-target drop order: after cache invalidation the bound
            // Texture may own the last Arc, so its sync guard must drop before that Arc.
            // Do this BEFORE seeding/shared clones, otherwise they'd keep the texture alive.
            let device = manager
                .devices_mut()?
                .find(|device| device.node() == target)
                .ok_or("target missing")?;
            let renderer = device.renderer_mut();
            let mut framebuffer = renderer.bind(&mut outputs[0].dmabuf)?;
            let prepared = renderer
                .prepare_external_framebuffer_write(&mut framebuffer)?
                .ok_or("bound dma-buf lost its external-write capability")?;
            renderer.finish_external_framebuffer_write(&mut framebuffer, &prepared.acquire)?;
            renderer.invalidate_caches()?;
            drop(framebuffer);
            println!("PASS retained dma-buf target drop after GLES cache invalidation");
        }
        for (index, output) in outputs.iter_mut().enumerate() {
            seed_direct_output(manager, target, output, index)?;
        }
        for sequence in 0..args.frames {
            let stage = match sequence % 6 {
                2 => "explicit_target_gles_fallback",
                3 => "direct_then_blit_to",
                4 => "direct_then_blit_from",
                _ => "direct_then_same_target_capture",
            };
            for (index, output) in outputs.iter_mut().enumerate() {
                println!(
                    "MEASURE direct-target round={round} output={index} sequence={sequence} stage={stage}"
                );
                draw_direct_output(manager, source, target, format, output, sequence, index, false)?;
                println!(
                    "SUBMIT direct-target round={round} output={index} sequence={sequence} exact_damage={:?} release_native={} complete_at_return={}",
                    output.damage,
                    output.release.is_exportable(),
                    output.release.is_reached()
                );
            }
            // Submit all three distinct targets before any CPU readback. Do NOT call wait
            // here: capture must consume the wait queued by the direct integration itself.
            for (index, output) in outputs.iter_mut().enumerate() {
                // Read from a shared context FIRST. A same-context CPU readback here would
                // complete the writer and could hide missing TextureSync publication.
                let shared = capture_shared_texture(&mut shared_renderer, output)?;
                assert_direct_pixels(output, &shared, "direct-target shared TextureSync reader", false)?;
                let device = manager
                    .devices_mut()?
                    .find(|device| device.node() == target)
                    .ok_or("target missing")?;
                let renderer = device.renderer_mut();
                let bytes = capture(renderer, &mut output.dmabuf, output.width, output.height)?;
                assert_direct_pixels(
                    output,
                    &bytes,
                    &format!("direct-target round={round} output={index} sequence={sequence}"),
                    true,
                )?;
                if sequence % 6 == 3 {
                    let scratch = capture(renderer, &mut output.scratch, output.width, output.height)?;
                    assert_blit_snapshot(output, &scratch)?;
                }
                output.before = bytes;
            }
        }
        manager.set_vulkan_direct_target_enabled(true);
    }
    println!(
        "PASS DIRECT-TARGET PIXELS: {:?}, {} measured writes, 3 original targets, exact outside-damage preservation, capture, fallback/GLES, blit_to/from, resize/cache invalidation. Inspect direct-route logs; no scanout or validation-layer claim.",
        args.format,
        args.frames * 3 * 3
    );
    Ok(())
}

fn run_multigpu(
    source_node: DrmNode,
    source_file: File,
    target_node: DrmNode,
    target_file: File,
    args: &Args,
) -> ProbeResult<()> {
    println!(
        "INIT MultiRenderer source={} target={} (render nodes only; no master/KMS)",
        args.source.display(),
        args.target.display()
    );
    println!(
        "ROUTE NOT ASSERTED: pixel PASS does not prove Vulkan. Inspect successful-copy debug logs with RUST_LOG=smithay::backend::renderer::multigpu=debug, including after INVALIDATE."
    );
    let mut backend = GbmGlesBackend::<GlesRenderer, DeviceFd>::default();
    for (node, file) in [(source_node, source_file), (target_node, target_file)] {
        let gbm = GbmDevice::new(DeviceFd::from(OwnedFd::from(file)))?;
        backend.add_node(node, gbm)?;
    }
    let mut manager = GpuManager::new(backend)?;
    if args.direct_target {
        return run_direct_targets(&mut manager, &source_node, &target_node, args);
    }
    let format = Fourcc::from(args.format);
    let base = (args.width as i32, args.height as i32);
    let larger = (base.0 + 32, base.1 + 24);
    // Same-size distinct contents, alternating sizes, then original sizes after another
    // invalidation: exercise pair-wide staging, damage and import/cache identity boundaries.
    for (round, sizes) in [[base, base], [base, larger], [base, base]]
        .into_iter()
        .enumerate()
    {
        if round != 0 {
            println!("INVALIDATE multigpu caches before round={round}");
            manager.invalidate_caches()?;
        }
        let mut outputs = Vec::new();
        {
            let mut renderer = manager.renderer(&source_node, &target_node, format)?;
            for (index, (width, height)) in sizes.into_iter().enumerate() {
                let buffer: GlesRenderbuffer =
                    Offscreen::create_buffer(&mut renderer, Fourcc::Abgr8888, (width, height).into())?;
                outputs.push(MultiOutput {
                    buffer,
                    width,
                    height,
                    colors: if index == 0 { [0, 1, 8, 9] } else { [4, 5, 9, 8] },
                    release: SyncPoint::signaled(),
                    sequence: 0,
                });
            }
        }
        // The actual transfer engine initializes asynchronously on first rendering. Warmup
        // retries are bounded, always paint full damage, and are not counted as test frames.
        // Even successful warmups may use CPU/direct sharing: only route logs prove Vulkan.
        let mut warmup_successes = 0;
        let mut last_error = None;
        for attempt in 0..20 {
            match draw_multi_output(
                &mut manager,
                &source_node,
                &target_node,
                format,
                &mut outputs[0],
                None,
            ) {
                Ok(()) => warmup_successes += 1,
                Err(err) => {
                    eprintln!("WARMUP round={round} attempt={attempt}: {err}");
                    last_error = Some(err);
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        if warmup_successes == 0 {
            return Err(last_error.unwrap_or_else(|| "all MultiRenderer warmups failed".into()));
        }
        println!(
            "WARMUP round={round}: {warmup_successes}/20 frames completed; Vulkan route still requires log inspection"
        );
        let mut previous = None;
        for step in 0..args.frames * 2 {
            let index = step as usize % 2;
            let sequence = step / 2;
            let output = &mut outputs[index];
            let update = (sequence != 0).then(|| (sequence as usize - 1) % 4);
            if let Some(quadrant) = update {
                output.colors[quadrant] =
                    (output.colors[quadrant] + 1 + sequence as usize % 7) % COLORS.len();
            }
            output.sequence = sequence;
            let started = Instant::now();
            draw_multi_output(&mut manager, &source_node, &target_node, format, output, update)?;
            println!(
                "SUBMIT multigpu round={round} output={index} frame={sequence} size={}x{} quadrant={update:?} elapsed_us={} release_native={} complete_at_return={}",
                output.width,
                output.height,
                started.elapsed().as_micros(),
                output.release.is_exportable(),
                output.release.is_reached()
            );
            // Do not CPU-serialize the first pair. Validate the preceding independent
            // output only after this output's MultiFrame::finish has submitted its work.
            if let Some(previous) = previous {
                verify_multi_output(
                    &mut manager,
                    &target_node,
                    &mut outputs[previous],
                    round,
                    previous,
                )?;
            }
            previous = Some(index);
        }
        // Check both once more: changing one output must not corrupt its idle sibling.
        for (index, output) in outputs.iter_mut().enumerate() {
            verify_multi_output(&mut manager, &target_node, output, round, index)?;
        }
    }
    println!(
        "PASS MultiRenderer PIXELS: {:?}, {} measured frames, two outputs, same/mixed sizes, two cache invalidations. Vulkan-route proof NOT automatic: inspect successful-copy debug logs; this is not a Vulkan validation-layer run.",
        args.format,
        args.frames * 2 * 3
    );
    Ok(())
}

#[cfg(test)]
mod direct_tests {
    use super::*;

    #[test]
    fn direct_damage_keeps_holes_and_exceeds_cpu_rect_cap() {
        let (width, height) = (192, 120);
        let l_shape = direct_damage(width, height, 1);
        assert_eq!(l_shape.len(), 2);
        assert!(!l_shape.iter().any(|r| contains_pixel(r, 96, 40)));
        let five = direct_damage(width, height, 2);
        assert_eq!(five.len(), 5);
        for shape in 0..4 {
            let damage: Vec<_> = direct_damage(width, height, shape)
                .into_iter()
                .filter_map(|r| clipped(r, width, height))
                .collect();
            let changed = (0..height)
                .flat_map(|y| (0..width).map(move |x| (x, y)))
                .filter(|&(x, y)| damage.iter().any(|r| contains_pixel(r, x, y)))
                .count();
            assert!(changed > 0 && changed < (width * height) as usize);
        }
    }

    #[test]
    fn model_updates_only_exact_clipped_union() {
        let (width, height) = (192, 120);
        let before = COLORS[9];
        let after = COLORS[8];
        for shape in 0..4 {
            let damage = direct_damage(width, height, shape);
            let mut pixels = vec![before; (width * height) as usize];
            for &rect in &damage {
                model_patch(&mut pixels, width, height, rect, after);
            }
            for y in 0..height {
                for x in 0..width {
                    let changed = damage.iter().any(|r| contains_pixel(r, x, y));
                    assert_eq!(
                        pixels[(y * width + x) as usize],
                        if changed { after } else { before }
                    );
                }
            }
        }
    }
}

fn main() -> ProbeResult<()> {
    let args = Args::parse();
    if args.pool_stress && (args.frames < 64 || !timing::enabled()) {
        return Err("--pool-stress requires --frames >=64 and SMITHAY_FRAME_TIMING=1".into());
    }
    if !cfg!(target_endian = "little") {
        return Err("RGBA byte oracle requires little endian".into());
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .init();
    let (source_node, source_file) = render_node(&args.source)?;
    let (target_node, target_file) = render_node(&args.target)?;
    if source_node == target_node {
        return Err("source and target must be different render nodes".into());
    }
    if args.multigpu {
        return run_multigpu(source_node, source_file, target_node, target_file, &args);
    }
    println!(
        "INIT Vulkan source={} target={} (render nodes only; no master/KMS)",
        args.source.display(),
        args.target.display()
    );
    // One explicitly labelled pair on this recording thread; the worker intentionally
    // does not enter it (wait timing must not be confused with driver allocations).
    let _timing_scope = if args.pool_stress {
        let stream = (1, 1);
        timing::register_stream(
            stream,
            &format!(
                "pool-stress {} -> {}",
                args.source.display(),
                args.target.display()
            ),
        );
        timing::enter(stream)
    } else {
        None
    };
    let mut stress = if args.pool_stress {
        Some(PoolStress::new()?)
    } else {
        None
    };
    let mut bridge = VkBridge::new(source_node)?;
    println!("INIT source GBM/EGL");
    let mut source = Gpu::new(source_file)?;
    println!("INIT target GBM/EGL");
    let mut target = Gpu::new(target_file)?;
    for (w, h) in [(args.width, args.height), (args.width + 32, args.height + 24)] {
        run_size(
            &mut bridge,
            &mut source,
            &mut target,
            &args,
            w as i32,
            h as i32,
            stress.as_mut(),
        )?;
    }
    if let Some(stress) = stress.as_mut() {
        // Teardown legitimately destroys the bounded cache: snapshot/assert BEFORE drop.
        let waited = stress.prove_before_drop()?;
        drop(bridge);
        stress.check_old()?;
        let [created, reused, recycled, destroyed, busy, signals_replaced] = stress.counts;
        println!(
            "PASS POOL-STRESS frames={} old_fences={} worker_waits={waited} created={created} reused={reused} recycled={recycled} destroyed_before_drop={destroyed} busy={busy} signals_replaced={signals_replaced} native_checks={} (old fences valid after engine drop)",
            stress.submitted,
            stress.old.len(),
            stress.native.checks + stress.worker_native.checks
        );
        println!(
            "NATIVE PUBLICATION initialNativeMisses={} main_initial_misses={} worker_initial_misses={} max_observed_latency_us={} main_max_observed_latency_us={} worker_max_observed_latency_us={} deadline_ms=500 (first-publication diagnostic; previously ready fds must stay immediately ready)",
            stress.native.initial_misses + stress.worker_native.initial_misses,
            stress.native.initial_misses,
            stress.worker_native.initial_misses,
            stress
                .native
                .max_observed_latency
                .max(stress.worker_native.max_observed_latency)
                .as_micros(),
            stress.native.max_observed_latency.as_micros(),
            stress.worker_native.max_observed_latency.as_micros(),
        );
    }
    if args.scanout_candidate {
        println!(
            "PASS SCANOUT-requested allocation/copy/readback only; no KMS framebuffer, atomic test, fence import, or display was attempted"
        );
    }
    println!(
        "PASS Vulkan transfer probe: {:?}, {} frames, two sizes, reused buffers and partial damage",
        args.format,
        args.frames * 2
    );
    Ok(())
}
