# NVIDIA and Intel probe programs

These programs record the experiments used to develop the NVIDIA-to-Intel multi-GPU bridge. They are hardware-specific diagnostics, not Smithay tests or supported tools. Most assume that `/dev/dri/renderD129` is NVIDIA and `/dev/dri/renderD128` is Intel. Check the node assignments on the test system before running them.

The main investigation and its results are in [`../nvidia-intel-multigpu-analysis.md`](../nvidia-intel-multigpu-analysis.md). Build commands are included in the source headers or the accompanying build script.

## Additional archived experiments

| File | Purpose | Status |
|---|---|---|
| `multigpu_compositor_path_bench_batched_fixed.c` | Compares direct Intel LINEAR composition with Intel tiled composition followed by tiled-to-LINEAR and NVIDIA-native copies. | Corrected batched benchmark. The earlier unbatched and timing-broken variants are intentionally omitted. |
| `nvmod_to_intel.c` | Tests whether Intel can import an NVIDIA-native GBM buffer through GBM and EGL, then queries Intel Vulkan support for its DRM modifier. | Diagnostic probe with test-machine render-node defaults. |
| `vk_wayland_probe.c` | Reports whether each Vulkan physical device has a graphics queue family with Wayland presentation support. | Diagnostic probe. Run it inside a Wayland session. |
| `wayland_drm_syncobj_poc.c` | Creates a Wayland DMA-BUF surface with `linux-drm-syncobj-v1` acquire and release timeline points. | Experimental client. It requires a compositor that advertises the syncobj protocol; use `build_wayland_drm_syncobj_poc.sh` to compile it. |
| `vkbridge_poc2.c` | Tries XR24 and XB30 copies through the early Vulkan bridge design, including an XB30 shader fallback. | Historical experiment. It prints the sampled pixel but does not check an expected value. |
| `vkbridge_thread_test.c` | Moves Vulkan initialization to a worker thread and applies a timeout. | Historical diagnostic for the NVIDIA initialization hang. |

`build_wayland_drm_syncobj_poc.sh` requires `wayland-scanner`, `wayland-protocols`, Wayland client headers, GBM, and libdrm. Run it with `docs/probes/build_wayland_drm_syncobj_poc.sh`; it generates protocol bindings in a temporary directory and writes `wayland_drm_syncobj_poc` to the current directory.

Logs, traces, binaries, machine captures, generated protocol files, and privileged bisect helpers are intentionally excluded.
