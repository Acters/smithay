# Shared bidirectional bridge branch

`bidirectional-vulkan-bridge` contains both winning pooled single-copy paths:

- NVIDIA GLES → NVIDIA Vulkan → Intel-owned LINEAR output.
- Intel GLES LINEAR staging → NVIDIA Vulkan → NVIDIA-native output.

The reverse-route implementation descends from the pooled forward implementation;
no second copy of that code is required. This branch excludes the slower two-stage
source-detile experiment. Both single-copy directions retain their capability checks,
exact damage, old-fence validity, bounded pools and safe fallback behavior.

Use the matching `bidirectional-vulkan-bridge` branch of **Acters/niri**. Its KDL
configuration selects the primary render node and `vulkan-bridge` settings:

```kdl
vulkan-bridge {
    enabled true
    direct-target true
    copy-device "render" // "target" for Intel -> NVIDIA on the tested laptop
}
```

No bridge environment variables are needed with an explicit KDL block. The full
profile examples, defaults, compatibility rules and diagnostic options are in the
niri branch's `docs/bidirectional-vulkan-bridge.md`.

Niri pins this implementation's published revision
`8edc1da00e599358c2c9cafe38db47c6aff2037f` for both Smithay packages. Later commits
on this shared branch may add documentation; update the pin deliberately for future
implementation changes. See `transfer-engine.md` for ownership and validation.
