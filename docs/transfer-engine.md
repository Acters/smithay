# Same-frame Vulkan transfer engine

This describes the experimental `nvidia-intel-bridge-transfer-engine` branch and its
`nvidia-intel-bridge-initialization` follow-up. The previous implementation and hardware investigation remain recorded in
[nvidia-intel-multigpu-analysis.md](nvidia-intel-multigpu-analysis.md).

## Scope

The bridge is an optional transfer path in `MultiRenderer`, between direct DMA-BUF
sharing and CPU copying. GLES still renders the scene and the target framebuffer.
Vulkan neither schedules presentation nor chooses an older completed frame.

```text
source GLES frame --render fence--> Vulkan copy
    --copy fence--> target GLES wait / damaged-region blit
    --target finish fence--> ordinary caller / DRM compositor
```

`VkBridge::copy()` returns a `SyncPoint` for the submitted copy, not an indication
that the GPU has finished. Native SYNC_FD semaphore import/export connects the GPU
queues where supported. A nonexportable fence uses explicit CPU waiting instead.
Failure after a successful submit cannot hide that submission from the caller: if
native export fails, the returned fence still waits the original Vulkan VkFence.

## Allocation and synchronization ownership

`GpuManager` keeps `TransferState` per source/target device pair. That state has a
single reusable target-owned LINEAR destination, not a modulo-only ring:

- Before writing source staging again, source GLES waits its previous Vulkan-reader
  completion (`source_release`). The same rule applies to intra-frame flushes.
- Before Vulkan writes the destination, it waits the previous target GLES reader
  (`destination_release`).
- The target GLES frame waits the current copy, then reads only the current damage.
  Its finish fence becomes the next destination-reader release dependency.
- Source staging may be shared sequentially between outputs. Pixels outside the
  current damage need not represent this output, so they are not painted to the
  target. The bridge does not silently promote partial source validity to a full
  output image.
- Resizing/replacing staging retires its dependencies and replaces the destination.
  Failed target-frame finish discards the destination rather than recycling an
  allocation without a known reader fence.

A pending Vulkan submission owns its command pool/buffer, original VkFence,
semaphore handles and imported images until actual retirement. On the
`nvidia-intel-bridge-pooled` follow-up, the owner then freezes the submission's logical
completion state and extracts its resource set into a bounded idle pool (maximum
eight sets). Completed SyncPoints keep that immutable outcome and their independent
exported FD, not mutable Vulkan handles. Reset and reuse occur only on owner-thread
checkout; querying or waiting an old fence can never observe a newer use of a reset
VkFence. Concurrent wait/status and resource extraction are protected by the same
state mutex; reaping and pending status queries use nonblocking try-lock behavior.

Exported SYNC_FD payload lifetime is distinct from Vulkan semaphore-handle lifetime.
Reusable wait semaphores use temporary imports; an abandoned import or failed native
signal export is repaired only after safe retirement. DMA-BUFs are retained by
stable allocation identity, not integer FD number. Import/pending/idle caches are
bounded; idle resource sets do not retain old imported images, and input SyncPoints
are not retained as a recursively growing chain of old batches. Ordinary completed
SyncPoint destruction no longer performs per-frame Vulkan resource destruction.

Generic cache invalidation and device re-enumeration retire transfer storage.
Source-manager generation identities are checked when acquiring a cross-manager
renderer, even if re-enumeration/invalidation happened before that acquisition or
Vulkan was disabled in the meantime.

An actual per-fence status/wait returning device loss can establish retirement for
that fence, but a global loss flag is not proof that other pending submissions have
retired. Potentially submitted work (including queue-submit device loss) retains its
ownership until an actual retirement wait or conservative ownership retention on an
unexpected failure. Lost-device resources are not recycled. Device loss invalidates
external memory contents, reports a context-loss error, and requires explicit
invalidation/recreation rather than partial reuse of those pixels.

## Modifier negotiation

Native EGL-renderable does not imply Vulkan-importable. On the tested NVIDIA
stack, unconstrained GBM selected `0x0300000000e08014`, while the relevant Vulkan
format advertised a different modifier set.

`VkBridge::source_modifiers(Fourcc, width, height)` queries explicit single-plane
TRANSFER_SRC/importable combinations for the actual format and extent.
`MultiRenderer` intersects these with source GLES render formats before allocation,
and checks that GBM honored the selection. If device initialization finishes after
CPU staging was allocated, the following render renegotiates that storage.

Both Vulkan imports use actual DMA-BUF modifiers, strides and offsets. Memory type
selection intersects DMA-BUF and image requirements. The engine checks transfer
usage/importability and preserves contents across GENERAL/FOREIGN ownership
boundaries. It rejects unsupported descriptors rather than guessing an OPTIMAL
layout for a foreign allocation.

The target-owned LINEAR allocation remains an intermediate texture. It is NOT
submitted directly to KMS by this implementation.

## Initialization and policy

The initialization follow-up creates Smithay `Instance` / `PhysicalDevice` wrappers
lazily on the existing background initialization thread. The physical device owns
an Arc-backed instance reference; the transfer core owns that physical device until
its logical device, batches and imports retire. Exact render-node matching and the
Vulkan 1.2 capability gate remain in effect. There is no custom global receiver,
one-shot bootstrap, vendor preference or early call from niri's main function.

A real niri session with early preparation disabled created the raw Vulkan instance
after DRM/display startup in 318 ms and had its logical device ready after 417 ms.
Native-fence transfers activated successfully on NVIDIA 610.57.04. The wrapper-based
version passes offscreen engine/MultiRenderer probes, including reconstruction after
invalidation, and a real niri session: late instance creation took 198 ms, the device
was ready after 275 ms, and native-fence transfers activated normally. The user
confirmed that the resulting build looks correct.

Keep initialization off the compositor event-loop thread. During the raw late-init
experiment, an X11 connection caused niri to start xwayland-satellite while instance
creation was in progress. This does not prove the historical hang's root cause, nor
that synchronous main-thread initialization is safe. The test establishes that an
early global hook was unnecessary for this observed late-background path/stack.

`GpuManager::set_vulkan_transfer_enabled(false)` disables Vulkan transfers and
retires their storage. The niri fork connects `NIRI_VKBRIDGE=0` to this policy.
No completion-redraw callback is needed for copied frames.

Allocation/import/unsupported-transfer failures fall back to the upstream CPU
path. A failed pair is retried after invalidation/recreation, not on every frame.
This does not imply every driver/format supports CPU readback; an actual render
error remains an error, never an already-signaled fake successful frame.

## Offscreen hardware probe

Build:

```sh
cargo build --example vulkan_transfer --no-default-features \
  --features backend_gbm,renderer_gl,renderer_multi,backend_vulkan
```

Select the render nodes for the machine explicitly. Example for the test laptop:

```sh
env -u DISPLAY -u WAYLAND_DISPLAY timeout --kill-after=5s 90s \
  target/debug/examples/vulkan_transfer \
  --source /dev/dri/renderD129 --target /dev/dri/renderD128 \
  --format abgr2101010
```

The default mode exercises the engine directly and cannot pass by substituting a
CPU copy. It verifies all target-GLES readback pixels, reused allocations, partial
updates, resize, and non-endpoint colors. Source/destination release dependencies
are provided before the prior output is CPU-read back.

Add `--multigpu` to exercise real `GpuManager` / `MultiRenderer` / `MultiFrame`
rendering with two alternating offscreen outputs and cache invalidation. Enable
`RUST_LOG=smithay::backend::renderer::multigpu=debug` and inspect
`submitted same-frame Vulkan transfer` events: pixel success alone cannot prove
whether this mode selected Vulkan, direct sharing or CPU fallback.

The executable rejects card-node and symlink paths. It uses plain render-node
DeviceFd/GBM/EGL objects, not a DRM master/session/KMS API, and does not restart or
control the running compositor. GPU/driver workloads still run; use bounded runs.

## Recorded validation and remaining gates

On NVIDIA 610.57.04 / Intel Mesa 26.2.2:

- Nine pure transfer tests pass after wrapper migration (eight at the phase-one
  checkpoint); the `renderer_multi`-without-Vulkan build passes.
- Niri workspace check and 216 niri/config/IPC library tests pass.
- Engine ABGR8888 and ABGR2101010 pixel probes pass with negotiated modifiers.
- A 1920x1080 / 1952x1104 ten-bit engine run passes 24 frames.
- MultiRenderer eight- and ten-bit runs each pass 72 measured frames across two
  outputs, same and mixed dimensions, partial damage, midtone colors and two cache
  invalidations. Debug logs confirm native-fence Vulkan submissions before/after
  invalidation.
- Strict Clippy passes for `renderer_multi,backend_vulkan` with default features off.

These are correctness/interoperability probes, not frame-rate benchmarks. Khronos
validation layers were not installed, so no validation-layer success is claimed.
Phase-one and lazy-initialization compositor sessions were adopted with user visual
confirmation. More fault injection, suspend/hotplug tests and direct Vulkan-to-KMS
work remain separate gates.

## Scanout-allocation feasibility (not KMS validation)

On `nvidia-intel-bridge-scanout-probe`, the render-only example accepts
`--scanout-candidate`. Build additionally with
`backend_gbm_has_create_with_modifiers2`; this ensures usage flags are passed
alongside the explicit modifier rather than ignored by the older modifier API.
The destination requests `SCANOUT|RENDERING` and explicit LINEAR. Exported
implicit/non-LINEAR descriptors are rejected.

Both ABGR8888 and ABGR2101010 passed 24 copied/readback frames at 1920x1080 and
1952x1104, including partial updates and midtones. Vulkan populated the candidate
buffer; target GLES only sampled it for verification. No KMS framebuffer, atomic
TEST_ONLY request, native KMS fence import, modeset or actual display was attempted
by these render-node-only tests. Successful allocation/readback does not establish
that a particular KMS plane/mode accepts the candidate.

## Atomic acceptance and direct-target prototype

A separately approved niri diagnostic ran with its existing atomic DRM ownership
on eDP-1, 1920x1080. Both ABGR8888/XBGR8888 and ABGR2101010/XBGR2101010 explicit
LINEAR candidates passed TEST_ONLY: first after copy completion without a fence,
then with an exported native copy fence while `complete_at_test=false`. The helper
never committed or displayed these buffers. Its implementation is retained on
`nvidia-intel-bridge-scanout-probe`, not in the direct-target compositor branch.

`nvidia-intel-bridge-direct-target` adds a default-off direct-write policy:
`GpuManager::set_vulkan_direct_target_enabled(true)`. Niri maps
`NIRI_VK_DIRECT_TARGET=1` to it and requests tested explicit-LINEAR formats only
for foreign atomic output swapchains. Normal/native negotiation is retained, and
failed LINEAR negotiation retries normal formats. Actual framebuffer descriptors
are still checked by the transfer path; requested modifiers are not proof of use.

The renderer's paired `prepare_external_framebuffer_write` /
`finish_external_framebuffer_write` hooks expose the ORIGINAL bound allocation
and its acquire fence, then publish the external completion into renderer-specific
synchronization. GLES supports its Image fallback and the usual Texture binding
only when that texture target came from the original DMA-BUF. Ordinary textures,
renderbuffers and EGL surfaces remain unsupported. The finishing hook does not
modify pixels: it queues the external wait and finishes a zero-draw frame so that
shared TextureSync state is updated and flushed after the external write.
Preparations abandoned before submission are paired with their acquire fence.

The direct transfer writes the acquired DRM swapchain allocation rather than an
intermediate image. It returns the current Vulkan copy fence to the normal DRM
path. The existing swapchain slot owns KMS release/reuse eligibility; the engine's
DMA-BUF reference does not replace that lease. There is no bridge-owned scanout
ring, synthetic presentation element or delayed-frame scheduling.

Only exact clipped original damage is copied. CPU/intermediate-path bounding-box
merges and full-frame thresholds MUST NOT be used here: source staging outside the
actual damage may belong to a different output. Unsupported targets retain the
intermediate path; transient Vulkan resource failures do not permanently poison
the target's capability cache. Direct mode currently requires Normal transform.

Frame-local target completion is retained across flushes and target blits, including
an empty final source frame. Source rendering resumes AFTER target operations so
that another GPU's context is not left current. GLES blits also publish texture
read/write synchronization. These fixes preserve both the final KMS fence and later
capture/shared-context reads. Texture synchronization guards drop before the owned
texture, including when caches were cleared while a framebuffer remained bound.

### Direct-target validation before display adoption

- Sixteen transfer/damage/lifecycle unit tests and two probe-model tests pass.
- The niri workspace passes strict Clippy and 216 library regression tests.
- No-Vulkan feature isolation and strict GLES/multigpu Clippy pass.
- Offscreen direct-bound-target tests pass in 8-bit and 10-bit: 54 measured writes
  per format, three original destinations, sparse/L-shaped/>3-rectangle/clipped
  damage, exact unchanged pixels outside damage, shared-context cached-texture
  reads before same-context capture, explicit GLES writes followed by direct writes,
  blit-to with resumed producer drawing and pre-continuation snapshot, blit-from
  followed by empty finish, and resize/invalidation.
- The same direct-target test also passes 54 measured ten-bit writes at full-HD
  base size (1920x1080 plus the mixed-size round), including shared-context reads
  and post-blit producer continuation.
- Trace events confirm direct Vulkan framebuffer submissions at the measured
  direct stages; pixel success is not being used as a substitute for route proof.

One failed early probe fixture submitted unbounded geometry directly to GLES while
its oracle used clipped damage. The explicit-GLES fallback fixture was corrected
to submit bounded geometry; out-of-bounds inputs remain in the MultiRenderer/direct
clipping cases.

### Approved live adoption

The user separately approved an actual display test with niri `9ee3ff77` and Smithay
`0e2ad6d1`. On eDP-1, the compositor negotiated ABGR2101010 with explicit LINEAR and
logged direct Vulkan writes into the bound framebuffer with native fences. The
internal ~144 Hz, DP ~240 Hz and HDMI ~75 Hz outputs remained active, without observed
render/queue errors. The user confirmed the resulting display looks correct and
asked to keep direct mode enabled. It is enabled in that user's service through
`NIRI_VK_DIRECT_TARGET=1`; the library default remains off and the intermediate route
remains available for unsupported cases.

This establishes the tested live path on this machine, not a universal driver
compatibility or performance claim. No per-output renderer migration, dynamic GPU
selection, long-term soak result, or measured power/latency benefit is implied.

### Pooled-resource validation

Run the retained-fence stress without KMS access:

```sh
SMITHAY_FRAME_TIMING=1 target/debug/examples/vulkan_transfer \
  --pool-stress --frames 128 --source /dev/dri/renderD129 \
  --target /dev/dri/renderD128 --format abgr2101010
```

In both 8-bit and 10-bit runs, 256 submissions used one resource set, reused 255
times, with zero steady resource destruction, pool-busy results or signal-semaphore
replacement. All 256 retained old fences stayed valid across reuse, resize, a
concurrent waiter and engine destruction. Full-resolution direct-target pixel,
shared-context, capture and blit regression tests also pass.

The stress records first native-FD readiness separately: a zero-time first poll can
lag the CPU VkFence observation on this tested driver, also reproduced with the
unpooled reference. A bounded first-publication deadline is enforced without blocking
producer reuse; once POLLIN is observed, every later zero-time poll must remain
ready. This is not a claim that Vulkan permits semaphore signaling after a submit
fence, nor a relaxed old-fence regression check. Native initial-lag diagnostics stay
visible in the test result.

The approved live comparison passed with niri `eca35507` / Smithay `e6912c04` on the
same balanced-profile, direct-target/LINEAR setup. With btop hidden but running,
KMS cadence improved from 140.464/s to 143.946/s; refresh gaps fell from 441 over
125s to 5 over 105s. The closed-monitor pooled run measured 143.991/s with zero
refresh gaps over 110s. Steady windows reused a set per copy without creating or
destroying sets or replacing signal semaphores. Btop-on presentation retirement
fell from 0.814 ms average / 19.817 ms maximum to 0.024 / 0.060 ms. The user confirmed
stable ~144 FPS and chose to keep pooling enabled.

These are sequential machine-specific measurements, not a promise of zero jitter,
a GPU-execution-time measurement or proof of a particular driver lock. The niri
checkout's `docs/vulkan-submission-pool.md` records the full settings, raw-data path,
remaining caveats and rollback.

## Opt-in target-side copy device / reverse bridge

`VulkanCopyDevice::Render` is the default. `Target` chooses the output GPU for
Vulkan, independently of the GLES scene renderer. The role, exact node and source
manager epoch belong to each transfer pair and changing them retires storage.
`vulkan_transfer_target_modifiers` shares that pair's background engine and returns
None while initializing, Some(empty) when unavailable, or exact destination
modifiers. Capability/retry caches are bounded and keyed by format/extent/layout;
recoverable candidate errors do not disable all formats on the pair.

With Target + direct enabled, a valid shared texture is retained as fallback but
the current source is first offered to the direct Vulkan path. Shared GLES reader
completion is tracked before later source writes/foreign ownership. Intermediate
allocations intersect Vulkan destination and target sampling capabilities; native
modifiers are preferred for Target, LINEAR for the existing Render-side route.
Actual descriptors, original framebuffer hooks, external leases and exact damage
remain mandatory. No copy/format-conversion algorithm or pooled-fence contract is
changed by role selection.

Fresh Intel renderD128 -> NVIDIA copy/target renderD129 offscreen tests passed in
ABGR8888/ABGR2101010. Source negotiation selected LINEAR; native target negotiation
selected modifier216172782120099860. Engine/pool stress, SCANOUT-requested full-HD
allocation, and actual MultiRenderer direct-counter/pixel/capture/blit/resize
regressions passed. Ordinary shared-GLES fallback and NVIDIA->Intel tests also pass.
Niri's `docs/intel-nvidia-bridge.md` records the setup. Real NVIDIA KMS transition,
copy-fence presentation, pacing and idle-power behavior are separate live gates;
source-side Intel rendering into LINEAR is not assumed optimal for every workload.
