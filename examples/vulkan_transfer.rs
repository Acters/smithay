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

use std::{
    error::Error,
    fs::{File, OpenOptions},
    os::{
        fd::OwnedFd,
        unix::fs::{FileTypeExt, OpenOptionsExt},
    },
    path::{Path, PathBuf},
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
            Bind, Color32F, ExportMem, Frame, ImportDma, Offscreen, Renderer, TextureMapping,
            gles::{GlesRenderbuffer, GlesRenderer},
            multigpu::{ApiDevice, GpuManager, gbm::GbmGlesBackend, vkbridge::VkBridge},
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

fn verify(
    renderer: &mut GlesRenderer,
    offscreen: &mut GlesRenderbuffer,
    w: i32,
    h: i32,
    colors: &[usize; 4],
    release: &SyncPoint,
    frame: u32,
) -> ProbeResult<()> {
    // This readback is deliberately delayed until AFTER the next Vulkan submission. Its
    // CPU wait therefore cannot accidentally replace that submission's destination-reader
    // dependency. No direct reading of the copied dma-buf is used as the test oracle.
    release.wait()?;
    let target = renderer.bind(offscreen)?;
    let mapping =
        renderer.copy_framebuffer(&target, Rectangle::from_size((w, h).into()), Fourcc::Abgr8888)?;
    if TextureMapping::format(&mapping) != Fourcc::Abgr8888 {
        return Err("readback did not provide requested RGBA8 layout".into());
    }
    let bytes = renderer.map_texture(&mapping)?;
    if bytes.len() != w as usize * h as usize * 4 {
        return Err(format!("unexpected readback length {}", bytes.len()).into());
    }
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

fn run_size(
    bridge: &mut VkBridge,
    source: &mut Gpu,
    target: &mut Gpu,
    args: &Args,
    w: i32,
    h: i32,
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
    let destination_bo = target
        .allocator
        .create_buffer(w as u32, h as u32, format, &[Modifier::Linear])?;
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
    for sequence in 0..args.frames {
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
        let damage = update.map(|index| rects[index]).unwrap_or(full);
        let started = Instant::now();
        let copy = bridge.copy(&src, &dst, &acquire, target_release.as_ref(), &[damage])?;
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
        previous_colors = colors;
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

fn main() -> ProbeResult<()> {
    let args = Args::parse();
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
    let mut bridge = VkBridge::new(source_node)?;
    println!("INIT source GBM/EGL");
    let mut source = Gpu::new(source_file)?;
    println!("INIT target GBM/EGL");
    let mut target = Gpu::new(target_file)?;
    for (w, h) in [(args.width, args.height), (args.width + 32, args.height + 24)] {
        run_size(&mut bridge, &mut source, &mut target, &args, w as i32, h as i32)?;
    }
    println!(
        "PASS Vulkan transfer probe: {:?}, {} frames, two sizes, reused buffers and partial damage",
        args.format,
        args.frames * 2
    );
    Ok(())
}
