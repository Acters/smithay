# Same-frame Vulkan transfer engine

This describes the experimental `nvidia-intel-bridge-transfer-engine` branch. The
previous implementation and hardware investigation remain recorded in
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

A Vulkan batch owns its command pool, original VkFence, semaphore handles and
imported images until retirement. Exported SYNC_FD payload lifetime is distinct
from Vulkan semaphore-handle lifetime. DMA-BUFs are retained by stable allocation
identity, not by integer FD number. Import and pending-batch caches are bounded;
input SyncPoints are not retained as a recursively growing chain of old batches.

Generic cache invalidation and device re-enumeration retire transfer storage.
Source-manager generation identities are checked when acquiring a cross-manager
renderer, even if re-enumeration/invalidation happened before that acquisition or
Vulkan was disabled in the meantime.

Known Vulkan device loss retires in-flight access but invalidates external memory
contents. The transfer route reports a context-loss error and requires explicit
invalidation/recreation; it does not fall back to partial reuse of those pixels.

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

The current phase retains asynchronous early instance preparation and lazy logical
device initialization. `VkBridge::new()` now selects the exact requested DRM render
node. The old one-shot preinit handoff and its ordering assumptions still need a
separate investigation; spawning a thread before DRM acquisition does not prove
that instance creation finished before acquisition. Recreating an engine can also
require a fresh instance after the one-shot handoff was consumed.

`GpuManager::set_vulkan_transfer_enabled(false)` disables Vulkan transfers and
retires their storage. The niri fork connects `NIRI_VKBRIDGE=0` to this policy as well
as to preinitialization. No completion-redraw callback is needed for copied frames.

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

- Eight pure transfer tests pass; the `renderer_multi`-without-Vulkan build passes.
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
Compositor-session adoption, initialization-order tests, more fault injection,
suspend/hotplug tests and any direct Vulkan-to-KMS work are separate gates.
