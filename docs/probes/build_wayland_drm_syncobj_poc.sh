#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$(mktemp -d)"
OUTPUT="${OUTPUT:-$PWD/wayland_drm_syncobj_poc}"
trap 'rm -rf "$BUILD_DIR"' EXIT

WP_DATADIR="$(pkg-config --variable=pkgdatadir wayland-protocols)"
SCANNER="$(pkg-config --variable=wayland_scanner wayland-scanner)"

gen() {
    local xml="$1"
    local stem="$2"
    "$SCANNER" client-header "$xml" "$BUILD_DIR/${stem}-client-protocol.h"
    "$SCANNER" private-code "$xml" "$BUILD_DIR/${stem}-protocol.c"
}

gen "$WP_DATADIR/stable/xdg-shell/xdg-shell.xml" xdg-shell
gen "$WP_DATADIR/unstable/linux-dmabuf/linux-dmabuf-unstable-v1.xml" linux-dmabuf-unstable-v1
gen "$WP_DATADIR/staging/linux-drm-syncobj/linux-drm-syncobj-v1.xml" linux-drm-syncobj-v1

gcc -O1 -Wall -Wextra \
    -I"$BUILD_DIR" \
    -o "$OUTPUT" \
    "$SOURCE_DIR/wayland_drm_syncobj_poc.c" \
    "$BUILD_DIR/xdg-shell-protocol.c" \
    "$BUILD_DIR/linux-dmabuf-unstable-v1-protocol.c" \
    "$BUILD_DIR/linux-drm-syncobj-v1-protocol.c" \
    $(pkg-config --cflags --libs wayland-client gbm libdrm)

echo "Built $OUTPUT"
echo "Run: WAYLAND_DEBUG=1 $OUTPUT"
