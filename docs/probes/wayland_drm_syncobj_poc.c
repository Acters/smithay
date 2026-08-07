/*
 * wayland_drm_syncobj_poc.c
 *
 * Minimal linux-drm-syncobj-v1 client proof of concept.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <gbm.h>
#include <xf86drm.h>
#include <drm_fourcc.h>
#include <wayland-client.h>

#include "xdg-shell-client-protocol.h"
#include "linux-dmabuf-unstable-v1-client-protocol.h"
#include "linux-drm-syncobj-v1-client-protocol.h"

#define WIDTH 320
#define HEIGHT 180

struct app {
    struct wl_display *display;
    struct wl_registry *registry;
    struct wl_compositor *compositor;
    struct xdg_wm_base *wm_base;
    struct zwp_linux_dmabuf_v1 *dmabuf;
    struct wp_linux_drm_syncobj_manager_v1 *sync_manager;
    struct wl_surface *surface;
    struct xdg_surface *xdg_surface;
    struct xdg_toplevel *toplevel;
    struct wp_linux_drm_syncobj_surface_v1 *sync_surface;
    bool configured;
    bool closed;
};

static void split_u64(uint64_t v, uint32_t *hi, uint32_t *lo) {
    *hi = (uint32_t)(v >> 32);
    *lo = (uint32_t)v;
}

static void wm_ping(void *data, struct xdg_wm_base *wm, uint32_t serial) {
    (void)data;
    xdg_wm_base_pong(wm, serial);
}
static const struct xdg_wm_base_listener wm_listener = { .ping = wm_ping };

static void xdg_configure(void *data, struct xdg_surface *surf, uint32_t serial) {
    struct app *app = data;
    xdg_surface_ack_configure(surf, serial);
    app->configured = true;
}
static const struct xdg_surface_listener xdg_listener = { .configure = xdg_configure };

static void top_configure(void *data, struct xdg_toplevel *top, int32_t w, int32_t h, struct wl_array *states) {
    (void)data; (void)top; (void)w; (void)h; (void)states;
}
static void top_close(void *data, struct xdg_toplevel *top) {
    (void)top;
    ((struct app *)data)->closed = true;
}
static void top_bounds(void *data, struct xdg_toplevel *top, int32_t w, int32_t h) {
    (void)data; (void)top; (void)w; (void)h;
}
static void top_caps(void *data, struct xdg_toplevel *top, struct wl_array *caps) {
    (void)data; (void)top; (void)caps;
}
static const struct xdg_toplevel_listener top_listener = {
    .configure = top_configure,
    .close = top_close,
    .configure_bounds = top_bounds,
    .wm_capabilities = top_caps,
};

static void registry_global(void *data, struct wl_registry *reg, uint32_t name,
                            const char *iface, uint32_t version) {
    struct app *app = data;
    if (!strcmp(iface, wl_compositor_interface.name)) {
        app->compositor = wl_registry_bind(reg, name, &wl_compositor_interface,
                                           version < 6 ? version : 6);
    } else if (!strcmp(iface, xdg_wm_base_interface.name)) {
        app->wm_base = wl_registry_bind(reg, name, &xdg_wm_base_interface, 1);
        xdg_wm_base_add_listener(app->wm_base, &wm_listener, app);
    } else if (!strcmp(iface, zwp_linux_dmabuf_v1_interface.name)) {
        app->dmabuf = wl_registry_bind(reg, name, &zwp_linux_dmabuf_v1_interface,
                                       version < 4 ? version : 4);
    } else if (!strcmp(iface, wp_linux_drm_syncobj_manager_v1_interface.name)) {
        app->sync_manager = wl_registry_bind(reg, name,
            &wp_linux_drm_syncobj_manager_v1_interface, 1);
    }
}
static void registry_remove(void *data, struct wl_registry *reg, uint32_t name) {
    (void)data; (void)reg; (void)name;
}
static const struct wl_registry_listener registry_listener = {
    .global = registry_global,
    .global_remove = registry_remove,
};

static int create_syncobj_fd(int drm_fd, uint32_t *handle, int *fd_out) {
    if (drmSyncobjCreate(drm_fd, 0, handle) != 0) {
        fprintf(stderr, "drmSyncobjCreate: %s\n", strerror(errno));
        return -1;
    }
    if (drmSyncobjHandleToFD(drm_fd, *handle, fd_out) != 0) {
        fprintf(stderr, "drmSyncobjHandleToFD: %s\n", strerror(errno));
        drmSyncobjDestroy(drm_fd, *handle);
        return -1;
    }
    return 0;
}

static int wait_release(struct app *app, int drm_fd, uint32_t handle, uint64_t point) {
    for (int i = 0; i < 500 && !app->closed; i++) {
        uint32_t first = 0;
        int ret = drmSyncobjTimelineWait(drm_fd, &handle, &point, 1, 0, 0,
                                         &first);
        if (ret == 0)
            return 0;
        if (errno != ETIME && errno != EINVAL) {
            fprintf(stderr, "drmSyncobjTimelineWait: %s\n", strerror(errno));
            return -1;
        }
        wl_display_dispatch_pending(app->display);
        wl_display_flush(app->display);
        usleep(10000);
    }
    errno = ETIMEDOUT;
    return -1;
}

int main(void) {
    struct app app = {0};
    const char *node = getenv("DRM_NODE");
    if (!node) node = "/dev/dri/renderD128";

    int drm_fd = open(node, O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) {
        fprintf(stderr, "open(%s): %s\n", node, strerror(errno));
        return 1;
    }

    app.display = wl_display_connect(NULL);
    if (!app.display) {
        fprintf(stderr, "wl_display_connect failed\n");
        return 1;
    }
    app.registry = wl_display_get_registry(app.display);
    wl_registry_add_listener(app.registry, &registry_listener, &app);
    wl_display_roundtrip(app.display);

    if (!app.compositor || !app.wm_base || !app.dmabuf || !app.sync_manager) {
        fprintf(stderr, "Missing globals: compositor=%p xdg=%p dmabuf=%p syncobj=%p\n",
                (void *)app.compositor, (void *)app.wm_base,
                (void *)app.dmabuf, (void *)app.sync_manager);
        return 1;
    }

    struct gbm_device *gbm = gbm_create_device(drm_fd);
    uint64_t linear = DRM_FORMAT_MOD_LINEAR;
    struct gbm_bo *bo = gbm_bo_create_with_modifiers2(
        gbm, WIDTH, HEIGHT, GBM_FORMAT_XRGB8888, &linear, 1, 0);
    if (!bo) {
        fprintf(stderr, "GBM LINEAR allocation failed: %s\n", strerror(errno));
        return 1;
    }

    int dma_fd = gbm_bo_get_fd_for_plane(bo, 0);
    uint32_t stride = gbm_bo_get_stride_for_plane(bo, 0);
    uint32_t offset = gbm_bo_get_offset(bo, 0);
    uint64_t modifier = gbm_bo_get_modifier(bo);
    printf("DMA-BUF fd=%d stride=%u offset=%u modifier=0x%016" PRIx64 "\n",
           dma_fd, stride, offset, modifier);

    struct zwp_linux_buffer_params_v1 *params =
        zwp_linux_dmabuf_v1_create_params(app.dmabuf);
    zwp_linux_buffer_params_v1_add(params, dma_fd, 0, offset, stride,
                                   (uint32_t)(modifier >> 32),
                                   (uint32_t)modifier);
    struct wl_buffer *buffer = zwp_linux_buffer_params_v1_create_immed(
        params, WIDTH, HEIGHT, DRM_FORMAT_XRGB8888, 0);
    zwp_linux_buffer_params_v1_destroy(params);
    close(dma_fd);

    uint32_t acquire_handle = 0, release_handle = 0;
    int acquire_fd = -1, release_fd = -1;
    if (create_syncobj_fd(drm_fd, &acquire_handle, &acquire_fd) != 0 ||
        create_syncobj_fd(drm_fd, &release_handle, &release_fd) != 0)
        return 1;

    struct wp_linux_drm_syncobj_timeline_v1 *acquire_timeline =
        wp_linux_drm_syncobj_manager_v1_import_timeline(app.sync_manager, acquire_fd);
    struct wp_linux_drm_syncobj_timeline_v1 *release_timeline =
        wp_linux_drm_syncobj_manager_v1_import_timeline(app.sync_manager, release_fd);
    close(acquire_fd);
    close(release_fd);

    app.surface = wl_compositor_create_surface(app.compositor);
    app.xdg_surface = xdg_wm_base_get_xdg_surface(app.wm_base, app.surface);
    xdg_surface_add_listener(app.xdg_surface, &xdg_listener, &app);
    app.toplevel = xdg_surface_get_toplevel(app.xdg_surface);
    xdg_toplevel_add_listener(app.toplevel, &top_listener, &app);
    xdg_toplevel_set_title(app.toplevel, "linux-drm-syncobj-v1 PoC");
    app.sync_surface = wp_linux_drm_syncobj_manager_v1_get_surface(
        app.sync_manager, app.surface);

    wl_surface_commit(app.surface);
    while (!app.configured && !app.closed) {
        if (wl_display_dispatch(app.display) < 0)
            return 1;
    }

    uint64_t acquire_point = 1, release_point = 1;
    if (drmSyncobjTimelineSignal(drm_fd, &acquire_handle, &acquire_point, 1) != 0) {
        fprintf(stderr, "drmSyncobjTimelineSignal: %s\n", strerror(errno));
        return 1;
    }

    uint32_t ahi, alo, rhi, rlo;
    split_u64(acquire_point, &ahi, &alo);
    split_u64(release_point, &rhi, &rlo);

    wp_linux_drm_syncobj_surface_v1_set_acquire_point(
        app.sync_surface, acquire_timeline, ahi, alo);
    wp_linux_drm_syncobj_surface_v1_set_release_point(
        app.sync_surface, release_timeline, rhi, rlo);
    wl_surface_attach(app.surface, buffer, 0, 0);
    wl_surface_damage_buffer(app.surface, 0, 0, WIDTH, HEIGHT);
    wl_surface_commit(app.surface);
    wl_display_flush(app.display);

    printf("Committed acquire=%" PRIu64 " release=%" PRIu64 "\n",
           acquire_point, release_point);
    printf("Waiting for compositor release...\n");

    if (wait_release(&app, drm_fd, release_handle, release_point) != 0) {
        fprintf(stderr, "Release point not signaled: %s\n", strerror(errno));
        return 2;
    }

    printf("PASS: compositor signaled release point %" PRIu64 "\n", release_point);

    wp_linux_drm_syncobj_surface_v1_destroy(app.sync_surface);
    wp_linux_drm_syncobj_timeline_v1_destroy(acquire_timeline);
    wp_linux_drm_syncobj_timeline_v1_destroy(release_timeline);
    wl_buffer_destroy(buffer);
    xdg_toplevel_destroy(app.toplevel);
    xdg_surface_destroy(app.xdg_surface);
    wl_surface_destroy(app.surface);
    drmSyncobjDestroy(drm_fd, acquire_handle);
    drmSyncobjDestroy(drm_fd, release_handle);
    gbm_bo_destroy(bo);
    gbm_device_destroy(gbm);
    close(drm_fd);
    wl_display_disconnect(app.display);
    return 0;
}
