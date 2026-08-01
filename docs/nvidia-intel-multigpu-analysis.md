# Cross-GPU Multigpu Bridge: Analysis, Design, and Results

**Hardware:** Intel UHD 630 (i915, internal eDP-1) + NVIDIA GTX 1660 Ti Mobile (nvidia 610.43.03, external DP-1 @ 240 Hz, HDMI-A-1)
**Stack:** CachyOS, kernel 7.1.5, niri 26.04 (fork), smithay ff5fa7d (fork)

This document records the complete investigation: the root-cause of the white screen, the interop probe matrix, the driver-level mechanisms, the throughput measurements, the final bridge architecture, and the forward-looking dynamic-multigpu design.

---

## 1. The Problem

After a full system upgrade (niri 25.11 → 26.04, kernel 7.1.3 → 7.1.5), the internal laptop panel (eDP-1, wired to the Intel iGPU) went pure white while the external monitor (DP-1, wired to the NVIDIA dGPU) worked fine.

### Root cause (bisected)

- niri renders everything on the NVIDIA GPU (`debug { render-drm-device "/dev/dri/renderD129" }`), but eDP-1 lives on the Intel GPU, so frames must be copied across GPUs (smithay's multigpu cpu-copy fallback: NVIDIA renders → `glReadPixels` to RAM → upload to Intel → scanout).
- niri 26.04 commit `9bd6c2ca` ("feat: add 10-bit framebuffer pixel format") prepended 10-bit formats to `SUPPORTED_COLOR_FORMATS`. The panel reports `Max bits per channel: 10-12`, so the surface format became **XB30** (XBGR2101010).
- The cpu-copy's `glReadPixels` uses `(GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV)` (from smithay's `fourcc_to_gl_formats` via `get_transparent`). NVIDIA's GLES rejects `RGBA` reads from an **opaque** (X-format) framebuffer — it mandates the implementation-preferred `(GL_RGB, UNSIGNED_INT_2_10_10_10_REV)`. → `GL_INVALID_OPERATION` → smithay maps it to `Unsupported pixel format: DrmFourcc(XB30)` → every frame fails → white screen.
- PR #4284 (which moved the copy from XR30 to XB30 to fix #4113) only changed *which* 10-bit variant fails: XR30 fails at NVIDIA GBM allocation (`EINVAL`); XB30 fails at the GL readback. Both broken on this stack.

### Why dmabuf sharing was impossible (the reason the cpu-copy path even exists)

smithay's `create_shared_dma_framebuffer` tries to allocate a buffer the target can import. The modifier intersection is empty:

- **NVIDIA** renders only to its own block-linear modifiers (13 variants, `0x0300000...`). It cannot render to `DRM_FORMAT_MOD_LINEAR` (probe: FBO INCOMPLETE even for 8-bit).
- **Intel** imports only LINEAR / X-tiled / Y-tiled / CCS. It cannot import NVIDIA block-linear (`EGL_BAD_MATCH`).
- **NVIDIA EGL** imports only its own block-linear modifiers (not even linear for TEXTURE_2D).

So every cross-GPU frame on this stack bounced through CPU memory (glReadPixels + upload), the path that broke with 10-bit.

---

## 2. The Interop Probe Matrix

All probe-verified on nvidia 610.43.03 + i915. Probes in `~/niri-bisect/`: `xb30_probe.c`, `vkbridge_poc.c`, `fence_diag.c`, `vk_import_test.c`, `vk_export_test.c`, `gbm_matrix.c`, `intel_gbm_nv_vk_poc.c` (user-authored), `intel_tiled_to_nvidia_vk_probe.c` (user-authored).

| Path | Verdict |
|---|---|
| NVIDIA GBM (block-linear) → NVIDIA Vulkan import | ✅ |
| NVIDIA Vulkan → LINEAR → Intel EGL (`TEXTURE_2D`) | ✅ **the bridge** |
| NVIDIA Vulkan → LINEAR → Intel Vulkan (explicit-modifier) | ✅ |
| NVIDIA Vulkan → LINEAR → Intel GBM | ✅ |
| Intel GBM LINEAR → NVIDIA EGL (`GL_TEXTURE_EXTERNAL_OES` only) | ✅ 8-bit + 10-bit (AB30) |
| Intel GBM LINEAR → NVIDIA Vulkan (`TILING_DRM_FORMAT_MODIFIER_EXT` + explicit layout) | ✅ |
| NVIDIA GBM **renderable** LINEAR allocation | ❌ (CPU-access only; `external_only=1` in EGL modifier query — sample-only as external textures) |
| Intel X/Y-tiled → anything NVIDIA (EGL or Vulkan) | ❌ (NVIDIA advertises **no** i915 modifiers; EGL `0x300C`, Vulkan `VK_ERROR_FORMAT_NOT_SUPPORTED`) |

### Key mechanisms

- **cubanismo (NVIDIA engineer)**: foreign dma-buf import works via **nvidia-drm's PRIME helpers** (the kernel's standard dma-buf framework), NOT via the separate resource-manager dma-buf code (which only understands NVIDIA's own buffers and is unrelated). The 2022 "import impossible" analysis quoted the wrong layer. Validated: EGL import works (linear + EXTERNAL_OES); Vulkan import works (`TILING_DRM_FORMAT_MODIFIER_EXT` + explicit plane layout + `vkGetMemoryFdPropertiesKHR` + dedicated alloc).
- **`VK_IMAGE_TILING_LINEAR` ≠ `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT`.** For externally produced dma-bufs, the modifier extension is required to supply the foreign stride/offset. The earlier import failures were entirely due to using OPTIMAL/LINEAR tiling with the modifier *list* struct instead of the *explicit* struct.
- **`external_only=1` in the EGL modifier query** explains the TEXTURE_2D failure: NVIDIA EGL accepts LINEAR dma-bufs only as `GL_TEXTURE_EXTERNAL_OES` (external sampling), never as ordinary TEXTURE_2D (which includes render-target use).
- **Memory placement decides shareability.** NVIDIA GBM's flags-0 LINEAR allocation lands in device VRAM: Intel's PRIME import calls `nv_drm_gem_prime_get_sg_table()` → `-ENOMEM` → reads zeros (probe: NVIDIA-side readback shows correct pixels `BGRA = 0 0 255 0`, Intel-side reads `0 0 0 0`). The Vulkan `EXPORT_MEMORY`-typed linear allocation lands in shareable/system RAM → Intel reads correctly. This is why the destination must be Vulkan-allocated (or Intel-owned), not NVIDIA-GBM-allocated.
- **NVIDIA fence interop is broken cross-API**: Vulkan-exported fence fds are NOT pollable sync_files (fdinfo shows no sync_file marker; poll never signals). EGL-exported fence fds poll fine but can't enter Vulkan (`VK_ERROR_INVALID_EXTERNAL_HANDLE`). So cross-driver fence-based explicit sync is unavailable; the only working sync primitives are CPU waits (EGL client-wait, `vkQueueWaitIdle`) and Vulkan sync-fd semaphores (VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT, which DOES work — proven in the user's `intel_gbm_nv_vk_poc.c`).

---

## 3. Throughput Measurements

User's `multigpu_layout_bench` (1920×1080 clear proxy, 2000 iterations, 7.91 MiB/frame):

```
Intel write proxy: X_TILED     GPU  245.4 µs/frame  (~31.5 GiB/s)
Intel write proxy: LINEAR      GPU 1897.0 µs/frame  (~4.07 GiB/s)
Intel X_TILED -> LINEAR        GPU  951.0 µs/frame  (~8.12 GiB/s)
NVIDIA LINEAR -> optimal       GPU  890.8 µs/frame  (~8.67 GiB/s)
```

### Derived path estimates (GPU timestamps)

| Path | Estimated GPU work |
|---|---|
| Internal Intel tiled render → scanout | **245 µs** |
| Internal Intel LINEAR render → scanout | 1897 µs |
| Intel tiled → LINEAR → NVIDIA (bridge, 2 copies) | 2087 µs (245+951+891) |
| Intel LINEAR → NVIDIA (direct render + 1 copy) | 2788 µs (1897+891) |

**Counterintuitive headline:** keeping Intel tiled *and paying the conversion* (2087 µs) is faster than rendering directly into LINEAR (2788 µs) — LINEAR's ~4 GiB/s write bandwidth makes "save a copy" a net loss.

Frame budgets: 240 Hz = 4167 µs, 144 Hz = 6944 µs.

### What this means for renderer policy

- **Internal-only**: Intel renders tiled (~245 µs), NVIDIA suspends. Best power/performance.
- **External NVIDIA display active**: two choices —
  - *Keep Intel renderer*: internal stays efficient (245 µs), external pays the tiled bridge (2087 µs/frame at 1080p = ~50% of the 240 Hz budget — risky).
  - *Switch renderer to NVIDIA*: external 240 Hz native (zero copies), only the lower-priority internal panel pays one bridge copy (~891 µs = ~13% of the 144 Hz budget — comfortable). **This is the dogfooded setup and the better trade for this hardware.**
- The direct-LINEAR-render idea is a net loss in both modes for this clear-heavy proxy; note real compositor workloads (damage-tracked, shader-bound) differ from the proxy.

---

## 4. The Bridge Architecture (as built)

Pipeline, on-GPU, no CPU pixel movement:

```
NVIDIA renders scene (GLES, block-linear staging buffer, persistent)
        ↓ (worker thread, off critical path)
NVIDIA Vulkan imports staging, vkCmdCopyImage
        ↓
LINEAR destination buffer
        ↓
Intel imports as EGLImage (TEXTURE_2D), blits to scanout surface
```

### Design decisions and why

1. **Pipeline worker thread** (`vkbridge-copy`): the compositor submits `(staging, dst, sync)` jobs and composites the latest completed copy one frame later. No cross-driver fence fds exist, so sync is: CPU wait on the render fence (cheap, usually instant) + `vkQueueWaitIdle` before publishing. The compositor never blocks on the copy.
2. **Destination buffers owned by the TARGET GPU** (Intel GBM, from `intel_gbm_nv_vk_poc.c`'s architecture): no export step from NVIDIA, no `EXPORT_MEMORY` allocation, no placement lottery — Intel owns the memory from the start. The worker imports each Intel BO into NVIDIA Vulkan once (cached per fd, `TILING_DRM_FORMAT_MODIFIER_EXT` + explicit plane layout, TRANSFER_DST) and copies into it.
3. **Init split (tension resolution)**: `vkCreateInstance` deadlocks the NVIDIA ICD once DRM master is held, so the **instance** is created pre-master (early_init at niri start); a full device created at startup **breaks direct scanout** on the NVIDIA output, so **device/queue/pool creation is deferred** to first use (lazy, post-master — which does NOT deadlock). Result: direct scanout for fullscreen clients on the NVIDIA output works *while* the bridge serves the Intel panel. Verified: `XRGB8888` client buffer on the KMS plane for fullscreen es2gears/zen.
4. **Once-per-copy presentation**: completed copies are sequence-tagged; each is presented exactly once. Re-presenting stale copies caused ghost frames when content disappeared.
5. **Completion notification with `pending_completed` gate**: the worker notifies niri's event loop on each completed copy (callback → `queue_redraw` for foreign outputs **only if a new completed copy is newer than the last presented AND no frame is already queued**). This presents the empty-workspace copy after content vanishes (ghost fix) without a `notify → redraw → copy → notify` runaway loop.
6. **Persistent staging buffer** (damage accumulates correctly, matching upstream's damage-tracked semantics); `wait_for_pending_copies` before re-render prevents the copy/render race.
7. **Caching**: src import image (staging is stable), pooled command buffer, memory-type queries, dst import cache (per fd), stable allocator-Dmabuf identities (WeakDmabuf texture-cache hits on the target side).
8. **Empty-damage gate**: no copy submission for empty frames (upstream's skip), but presentation of completed copies continues.

### Performance (dogfooded)

- Internal panel: full 144 fps, 10-bit via dGPU, zero render errors.
- Worker CPU: **idle (~0%) when the internal panel is static/empty**; ~10-25% of one core when the internal animates (proportional to copy rate). Total bridge overhead ~12% (performance profile).
- Direct scanout: fullscreen clients on the 240 Hz external get the native path (client dmabuf on the KMS plane, zero compositing).
- Power profiles: 144 fps survives `balanced`/`power saver` (no single core maxed).

---

## 5. Forward Architecture (the dynamic multigpu design)

### What `zwp_linux_dmabuf_feedback_v1` can and can't do

The protocol lets the compositor tell Wayland clients which device and format/modifier combinations are preferred **per surface**, with tranches that can change over time (window moves between outputs). Tranche targets can be a preferred scanout device (with the `SCANOUT` flag) or a preferred rendering/import device. **Every advertised tranche must still produce a buffer accessible from the feedback's `main_device` (direct import or expensive fallback).**

It does **not** switch niri's compositor renderer, migrate niri's own EGL/Vulkan contexts, output render targets, texture caches, cursor buffers, or renderer-owned intermediates. Clients are advisory; legacy clients may ignore feedback and keep submitting previous buffers (so a fallback import/copy path is always required).

### What niri already has (verified in-tree)

- Per-output **scanout tranches** with a cross-device **Linear-only** restriction (`tty.rs:surface_dmabuf_feedback`: when `surface_render_node != primary_render_node`, scanout formats are limited to Linear — matching the interop matrix exactly).
- Direct scanout machinery (`FrameFlags::ALLOW_PRIMARY_PLANE_SCANOUT`, enabled by default), working on stock niri for fullscreen clients on the NVIDIA output (verified: `XRGB8888` client buffer on the KMS plane).

### Milestone plan and status

| Milestone | Description | Status |
|---|---|---|
| 1. Feedback tranches + direct scanout | NVIDIA scanout tranche for fullscreen games; direct scanout on NVIDIA output | ✅ working (tranches already present; direct scanout verified for fullscreen es2gears/zen) |
| 2. Hot global renderer migration | Switch global renderer on topology change without session restart | not started (large: recreate EGL contexts, renderers, texture caches) |
| 3. Per-output renderers | Intel renders internal, NVIDIA renders external, no copies for composited content on the external | not started (largest lift; the correct end state) |
| 4. Damage-aware bridge copies | Pair each copy with its damage regions so small updates don't pay full-frame cost | partial (regions machinery exists; full-frame copies currently used for correctness) |

### Why per-output renderers matter (the composited-browser case)

Direct scanout only helps fullscreen, undecorated clients. A composited browser on the NVIDIA output needs composition; with a fixed Intel renderer it pays the two-copy bridge (~1.84 ms/frame at 1080p, ~50% of the 240 Hz budget — too risky). Options:

1. Accept two copies (Intel renderer).
2. Render Intel's compositor output directly into LINEAR (slower overall per the benchmark).
3. Dynamically switch the global renderer to NVIDIA (without session restart).
4. **Per-output renderers** (cleanest): NVIDIA renderer composites the NVIDIA output natively; the browser's client buffer imports via LINEAR or a fallback, but the expensive full-output bridge disappears.

### Straddling windows

A `wl_surface` submits one buffer for the whole surface — it cannot be half Intel-tiled and half NVIDIA-block-linear. Three strategies: prefer one GPU for the whole surface (dominant output), recommend a common LINEAR format (costly: LINEAR render is slow), or **composite separately per output** (each output gets a native render target; the source buffer bridges to whichever side doesn't match). In niri's tiling model, windows live on one workspace = one output, so straddling is limited to floating windows and the overview — the everyday topology here is one-window-per-output, exactly where feedback + direct scanout shine.

### Feedback policy with hysteresis

Only switch a surface's preferred device after >60-70% of it is on the other output, it stays for several frames, or it goes fullscreen there — swapchain recreation isn't free. Keep LINEAR as the fallback tranche.

---

## 6. Probe Sources

| File | What it proves |
|---|---|
| `xb30_probe.c` | Modifier import/export matrix across both GPUs (EGL/GBM) |
| `vkbridge_poc.c` | NVIDIA GBM → Vulkan copy → LINEAR → Intel EGL, pixel-verified (red/green) |
| `vk_import_test.c` | Intel GBM LINEAR → NVIDIA Vulkan import+bind (explicit-modifier path), pixel-verified |
| `vk_export_test.c` | NVIDIA optimal → copy → LINEAR → Intel Vulkan/GBM import, pixel-verified |
| `fence_diag.c` | NVIDIA Vulkan fence fds don't signal cross-driver; EGL fence fds poll but can't enter Vulkan |
| `gbm_matrix.c` | NVIDIA GBM allocation flags matrix (renderable = block-linear only; LINEAR = CPU-access only) |
| `intel_gbm_nv_vk_poc.c` | Intel-owned LINEAR BO → NVIDIA Vulkan writes → Intel reads (with sync-fd semaphores) — PASS |
| `intel_tiled_to_nvidia_vk_probe.c` | NVIDIA advertises no i915 tiling modifiers (X/Y/Yf/4/CCS all blocked) |
| `multigpu_layout_bench` | Throughput table in §3 |

---

## 7. Git History (forks)

smithay `vulkan-bridge` branch (key commits, newest first):
- destination buffers owned by the TARGET gpu (Intel GBM)
- split init — instance pre-master, device lazy
- completion notification with pending_completed gate (ghost fix)
- once-per-copy presentation (sequence-tagged)
- cached vulkan objects + stable dmabuf identities
- v3 pipeline — worker-thread copies, persistent staging
- working v1 — preinit before DRM master, threaded init, GpuManager-owned state

niri `vulkan-bridge` branch:
- preinit default-on (NIRI_VKBRIDGE=0 opt-out)
- only preinit for the compositor, not CLI subcommands
- use local smithay fork, enable backend_vulkan
- revert: queue_redraw_all moves (wrong upstream-scheduling assumption)

Preserved for reference: `v2-wip` (fence-fd explicit-sync attempt, proven impossible on this driver).
