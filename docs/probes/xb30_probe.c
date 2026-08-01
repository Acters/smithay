// xb30_probe — test 10-bit dmabuf allocation / EGL import / FBO render / readback
// across the Intel (renderD128) and NVIDIA (renderD129) GPUs, replicating the
// smithay multigpu copy path that fails in niri (issue niri-wm/niri#4374).
//
// Build: gcc -O1 -o xb30_probe xb30_probe.c -lgbm -lEGL -lGLESv2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

static PFNEGLQUERYDEVICESEXTPROC        pQueryDevices;
static PFNEGLGETPLATFORMDISPLAYEXTPROC  pGetPlatformDisplay;
static PFNEGLCREATEIMAGEPROC            pCreateImage;
static PFNEGLDESTROYIMAGEPROC           pDestroyImage;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC pImageTargetTex;

struct gpu {
    const char *node;
    int fd;
    struct gbm_device *gbm;
    EGLDisplay dpy;
    EGLContext ctx;
    EGLSurface surf;
    const char *name;
};

static void load_procs(void) {
    pQueryDevices       = (void *)eglGetProcAddress("eglQueryDevicesEXT");
    pGetPlatformDisplay = (void *)eglGetProcAddress("eglGetPlatformDisplayEXT");
    pCreateImage        = (void *)eglGetProcAddress("eglCreateImage");
    pDestroyImage       = (void *)eglGetProcAddress("eglDestroyImage");
    pImageTargetTex     = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");
}

static int gpu_init(struct gpu *g, const char *node, const char *name) {
    g->node = node; g->name = name;
    g->fd = open(node, O_RDWR | O_CLOEXEC);
    if (g->fd < 0) { printf("%s: open %s failed: %s\n", name, node, strerror(errno)); return -1; }
    g->gbm = gbm_create_device(g->fd);
    if (!g->gbm) { printf("%s: gbm_create_device failed\n", name); return -1; }

    // Try EGL_PLATFORM_DEVICE matching this DRM node, fall back to GBM platform.
    g->dpy = EGL_NO_DISPLAY;
    EGLDeviceEXT devs[16]; EGLint ndev = 0;
    if (pQueryDevices && pQueryDevices(16, devs, &ndev)) {
        for (int i = 0; i < ndev; i++) {
            const char *devfile = NULL;
            EGLAttrib a;
            // query EGL_DRM_DEVICE_FILE_EXT (0x3233)
            static PFNEGLQUERYDEVICEATTRIBEXTPROC qDevAttr;
            if (!qDevAttr) qDevAttr = (void *)eglGetProcAddress("eglQueryDeviceAttribEXT");
            if (qDevAttr && qDevAttr(devs[i], 0x3233, &a))
                devfile = (const char *)a;
            if (devfile && strcmp(devfile, node) == 0) {
                g->dpy = pGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devs[i], NULL);
                break;
            }
        }
    }
    if (g->dpy == EGL_NO_DISPLAY)
        g->dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, g->gbm, NULL);
    if (g->dpy == EGL_NO_DISPLAY || !eglInitialize(g->dpy, NULL, NULL)) {
        printf("%s: eglInitialize failed (0x%x)\n", name, eglGetError()); return -1;
    }
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint cfg_attrs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE };
    EGLConfig cfg; EGLint n = 0;
    eglChooseConfig(g->dpy, cfg_attrs, &cfg, 1, &n);
    if (n < 1) { printf("%s: no EGL config\n", name); return -1; }
    if (!strstr(eglQueryString(g->dpy, EGL_EXTENSIONS), "EGL_KHR_surfaceless_context")) {
        printf("%s: no surfaceless_context ext\n", name); return -1;
    }
    g->surf = EGL_NO_SURFACE;
    EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    g->ctx = eglCreateContext(g->dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (!g->ctx || !eglMakeCurrent(g->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g->ctx)) {
        printf("%s: context/makecurrent failed (0x%x)\n", name, eglGetError()); return -1;
    }
    printf("== %s (%s): GL_RENDERER = %s\n", name, node, glGetString(GL_RENDERER));
    return 0;
}

static const char *gle_str(GLenum e) {
    switch (e) {
    case GL_NO_ERROR: return "OK";
    case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
    case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
    default: return "GL_?";
    }
}

static void test_format(struct gpu *render, struct gpu *alloc,
                        uint32_t fourcc, const char *fname) {
    printf("-- render=%-7s alloc=%-7s fmt=%s\n", render->name, alloc->name, fname);
    eglMakeCurrent(render->dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, render->ctx);

    struct gbm_bo *bo = gbm_bo_create(alloc->gbm, 64, 64, fourcc, GBM_BO_USE_RENDERING);
    if (!bo) { printf("   gbm alloc: FAILED (%s)\n", strerror(errno)); return; }
    int bfd = gbm_bo_get_fd(bo);
    uint32_t stride = gbm_bo_get_stride_for_plane(bo, 0);
    uint32_t offset = gbm_bo_get_offset(bo, 0);
    uint64_t mod = gbm_bo_get_modifier(bo);
    int nplanes = gbm_bo_get_plane_count(bo);
    printf("   gbm alloc: OK (planes=%d stride=%u modifier=0x%llx)\n",
           nplanes, stride, (unsigned long long)mod);

    EGLAttrib attrs[32]; int i = 0;
    attrs[i++] = EGL_WIDTH;  attrs[i++] = 64;
    attrs[i++] = EGL_HEIGHT; attrs[i++] = 64;
    attrs[i++] = EGL_LINUX_DRM_FOURCC_EXT; attrs[i++] = (EGLAttrib)fourcc;
    attrs[i++] = EGL_DMA_BUF_PLANE0_FD_EXT;     attrs[i++] = bfd;
    attrs[i++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; attrs[i++] = (EGLAttrib)offset;
    attrs[i++] = EGL_DMA_BUF_PLANE0_PITCH_EXT;  attrs[i++] = (EGLAttrib)stride;
    if (mod != 0x00FFFFFFFFFFFFFFULL /* DRM_FORMAT_MOD_INVALID */) {
        attrs[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; attrs[i++] = (EGLAttrib)(mod & 0xFFFFFFFF);
        attrs[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; attrs[i++] = (EGLAttrib)(mod >> 32);
    }
    attrs[i++] = EGL_NONE;

    eglGetError(); // clear
    EGLImage img = pCreateImage(render->dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    if (img == EGL_NO_IMAGE) {
        printf("   EGL import into %s: FAILED (eglError 0x%x)\n", render->name, eglGetError());
        gbm_bo_destroy(bo);
        return;
    }
    printf("   EGL import into %s: OK\n", render->name);

    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    pImageTargetTex(GL_TEXTURE_2D, img);
    printf("   glEGLImageTargetTexture2DOES: %s (0x%x)\n", gle_str(glGetError()), glGetError());

    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    GLenum fbs = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    printf("   FBO status: %s (0x%x)\n", fbs == GL_FRAMEBUFFER_COMPLETE ? "COMPLETE" : "INCOMPLETE", fbs);

    glGetError();
    glClearColor(1.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    GLenum e = glGetError();
    printf("   render (glClear): %s (0x%x)\n", gle_str(e), e);

    GLint read_fmt = 0, read_type = 0;
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_FORMAT, &read_fmt);
    glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
    glGetError();
    uint32_t px = 0;
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, &px);
    e = glGetError();
    printf("   glReadPixels RGBA/U8: %s (0x%x)\n", gle_str(e), e);
    glReadPixels(0, 0, 1, 1, read_fmt, read_type, &px);
    e = glGetError();
    printf("   glReadPixels impl (0x%x/0x%x): %s (0x%x)\n", read_fmt, read_type, gle_str(e), e);
    // The exact combo smithay's cpu-copy readback uses for 10-bit formats:
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, &px);
    e = glGetError();
    printf("   glReadPixels RGBA/2_10_10_10_REV (smithay combo): %s (0x%x)\n", gle_str(e), e);

    // blit to an 8-bit FBO (exercises the read path like a copy pass would)
    GLuint tex2, fbo2;
    glGenTextures(1, &tex2); glBindTexture(GL_TEXTURE_2D, tex2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &fbo2); glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex2, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo2);
    glGetError();
    glBlitFramebuffer(0, 0, 64, 64, 0, 0, 64, 64, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    e = glGetError();
    printf("   glBlitFramebuffer XB30->RGBA8: %s (0x%x)\n", gle_str(e), e);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo); glDeleteFramebuffers(1, &fbo2);
    glDeleteTextures(1, &tex); glDeleteTextures(1, &tex2);
    pDestroyImage(render->dpy, img);
    gbm_bo_destroy(bo);
}

static void print_import_mods(struct gpu *g, uint32_t fourcc, const char *fname) {
    static PFNEGLQUERYDMABUFFORMATSEXTPROC qFmts;
    static PFNEGLQUERYDMABUFMODIFIERSEXTPROC qMods;
    if (!qFmts) qFmts = (void *)eglGetProcAddress("eglQueryDmaBufFormatsEXT");
    if (!qMods) qMods = (void *)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
    if (!qMods) { printf("   %s import %s: no modifier-query ext\n", g->name, fname); return; }
    EGLuint64KHR mods[32]; EGLBoolean ext_only[32]; EGLint n = 0;
    if (qMods(g->dpy, (EGLint)fourcc, 32, mods, ext_only, &n) && n > 0) {
        printf("   %s import %-4s: %d modifiers:", g->name, fname, n);
        for (int i = 0; i < n && i < 10; i++)
            printf(" 0x%llx%s", (unsigned long long)mods[i], ext_only[i] ? "(ext)" : "");
        if (n > 10) printf(" ...");
        printf("\n");
    } else {
        printf("   %s import %-4s: format NOT importable\n", g->name, fname);
    }
}

int main(void) {
    load_procs();
    struct gpu intel, nvidia;
    if (gpu_init(&intel, "/dev/dri/renderD128", "intel")) return 1;
    if (gpu_init(&nvidia, "/dev/dri/renderD129", "nvidia")) return 1;

    print_import_mods(&intel, GBM_FORMAT_XRGB8888, "XR24"); print_import_mods(&intel, GBM_FORMAT_XBGR2101010, "XB30"); print_import_mods(&intel, GBM_FORMAT_ABGR2101010, "AB30"); print_import_mods(&nvidia, GBM_FORMAT_XRGB8888, "XR24"); print_import_mods(&nvidia, GBM_FORMAT_XBGR2101010, "XB30"); print_import_mods(&nvidia, GBM_FORMAT_ABGR2101010, "AB30");
    uint64_t linear = 0ULL; /* DRM_FORMAT_MOD_LINEAR */
    // reverse direction: Intel renders (linear) -> NVIDIA imports
    struct { uint32_t f; const char *n; } rfmts[] = {
        { GBM_FORMAT_XRGB8888,    "XR24" },
        { GBM_FORMAT_XBGR2101010, "XB30" },
        { GBM_FORMAT_ABGR2101010, "AB30" },
    };
    for (unsigned i = 0; i < sizeof(rfmts)/sizeof(rfmts[0]); i++) {
        printf("== share test: Intel alloc+render %s LINEAR -> NVIDIA import\n", rfmts[i].n);
        struct gbm_bo *bo = gbm_bo_create_with_modifiers(intel.gbm, 64, 64, rfmts[i].f,
                                                         &linear, 1);
        if (!bo) { printf("   Intel gbm alloc LINEAR: FAILED (%s)\n", strerror(errno)); continue; }
        printf("   Intel gbm alloc: OK modifier=0x%llx\n",
               (unsigned long long)gbm_bo_get_modifier(bo));
        eglMakeCurrent(intel.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, intel.ctx);
        int bfd = gbm_bo_get_fd(bo);
        EGLAttrib a[20]; int j = 0;
        a[j++] = EGL_WIDTH; a[j++] = 64; a[j++] = EGL_HEIGHT; a[j++] = 64;
        a[j++] = EGL_LINUX_DRM_FOURCC_EXT; a[j++] = (EGLAttrib)rfmts[i].f;
        a[j++] = EGL_DMA_BUF_PLANE0_FD_EXT; a[j++] = bfd;
        a[j++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; a[j++] = 0;
        a[j++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; a[j++] = (EGLAttrib)gbm_bo_get_stride_for_plane(bo, 0);
        uint64_t mod = gbm_bo_get_modifier(bo);
        a[j++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; a[j++] = (EGLAttrib)(mod & 0xFFFFFFFF);
        a[j++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; a[j++] = (EGLAttrib)(mod >> 32);
        a[j++] = EGL_NONE;
        // render into it on Intel first (self-import, FBO, clear)
        eglGetError();
        EGLImage img = pCreateImage(intel.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
        if (img) {
            GLuint t, f;
            glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
            pImageTargetTex(GL_TEXTURE_2D, t ? img : img);
            glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
            printf("   Intel render into LINEAR: FBO %s, clear %s\n",
                   glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE ? "COMPLETE" : "INCOMPLETE",
                   (glGetError(), glClearColor(0,0,1,1), glClear(GL_COLOR_BUFFER_BIT), gle_str(glGetError())));
            glDeleteFramebuffers(1, &f); glDeleteTextures(1, &t);
            pDestroyImage(intel.dpy, img);
        } else printf("   Intel self-import: FAILED 0x%x\n", eglGetError());
        // now import into NVIDIA
        eglMakeCurrent(nvidia.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, nvidia.ctx);
        eglGetError();
        img = pCreateImage(nvidia.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
        if (!img) printf("   NVIDIA import: FAILED 0x%x\n", eglGetError());
        else {
            GLuint t;
            glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
            pImageTargetTex(GL_TEXTURE_2D, img);
            GLenum e2 = glGetError();
            printf("   NVIDIA import + TEXTURE_2D bind: %s\n", gle_str(e2));
            if (e2 != GL_NO_ERROR) {
                GLuint t2;
                glGenTextures(1, &t2); glBindTexture(GL_TEXTURE_EXTERNAL_OES, t2);
                pImageTargetTex(GL_TEXTURE_EXTERNAL_OES, img);
                printf("   NVIDIA import + EXTERNAL_OES bind: %s\n", gle_str(glGetError()));
                glDeleteTextures(1, &t2);
            }
            glDeleteTextures(1, &t);
            pDestroyImage(nvidia.dpy, img);
        }
        gbm_bo_destroy(bo);
    }

    // smithay's share path: alloc on NVIDIA requesting DRM_FORMAT_MOD_LINEAR,
    // render on NVIDIA, import into Intel.
    struct { uint32_t f; const char *n; } lfmts[] = {
        { GBM_FORMAT_XRGB8888,    "XR24" },
        { GBM_FORMAT_XBGR2101010, "XB30" },
    };
    for (unsigned i = 0; i < sizeof(lfmts)/sizeof(lfmts[0]); i++) {
        printf("== share test: NVIDIA alloc+render %s LINEAR -> Intel import\n", lfmts[i].n);
        struct gbm_bo *bo = gbm_bo_create_with_modifiers(nvidia.gbm, 64, 64, lfmts[i].f,
                                                         &linear, 1);
        if (!bo) { printf("   NVIDIA gbm alloc LINEAR: FAILED (%s)\n", strerror(errno)); continue; }
        printf("   NVIDIA gbm alloc: OK modifier=0x%llx\n",
               (unsigned long long)gbm_bo_get_modifier(bo));
        // render into it on NVIDIA (import own fd as EGLImage, FBO, clear)
        eglMakeCurrent(nvidia.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, nvidia.ctx);
        int bfd = gbm_bo_get_fd(bo);
        EGLAttrib a[20]; int j = 0;
        a[j++] = EGL_WIDTH; a[j++] = 64; a[j++] = EGL_HEIGHT; a[j++] = 64;
        a[j++] = EGL_LINUX_DRM_FOURCC_EXT; a[j++] = (EGLAttrib)lfmts[i].f;
        a[j++] = EGL_DMA_BUF_PLANE0_FD_EXT; a[j++] = bfd;
        a[j++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; a[j++] = 0;
        a[j++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; a[j++] = (EGLAttrib)gbm_bo_get_stride_for_plane(bo, 0);
        uint64_t mod = gbm_bo_get_modifier(bo);
        a[j++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; a[j++] = (EGLAttrib)(mod & 0xFFFFFFFF);
        a[j++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; a[j++] = (EGLAttrib)(mod >> 32);
        a[j++] = EGL_NONE;
        eglGetError();
        EGLImage img = pCreateImage(nvidia.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
        if (!img) { printf("   NVIDIA self-import: FAILED 0x%x\n", eglGetError()); gbm_bo_destroy(bo); continue; }
        GLuint tex, fbo;
        glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
        pImageTargetTex(GL_TEXTURE_2D, img);
        glGenFramebuffers(1, &fbo); glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        printf("   NVIDIA render into LINEAR: FBO %s, clear %s\n",
               glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE ? "COMPLETE" : "INCOMPLETE",
               (glGetError(), glClearColor(0,1,0,1), glClear(GL_COLOR_BUFFER_BIT), gle_str(glGetError())));
        glDeleteFramebuffers(1, &fbo); glDeleteTextures(1, &tex);
        pDestroyImage(nvidia.dpy, img);
        // now import into Intel
        eglMakeCurrent(intel.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, intel.ctx);
        eglGetError();
        img = pCreateImage(intel.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
        if (!img) printf("   Intel import: FAILED 0x%x\n", eglGetError());
        else {
            glGenTextures(1, &tex); glBindTexture(GL_TEXTURE_2D, tex);
            pImageTargetTex(GL_TEXTURE_2D, img);
            printf("   Intel import + texture bind: %s\n", gle_str(glGetError()));
            glDeleteTextures(1, &tex);
            pDestroyImage(intel.dpy, img);
        }
        gbm_bo_destroy(bo);
    }

    struct { uint32_t f; const char *n; } fmts[] = {
        { GBM_FORMAT_XBGR2101010, "XB30" },
        { GBM_FORMAT_ABGR2101010, "AB30" },
        { GBM_FORMAT_XRGB2101010, "XR30" },
        { GBM_FORMAT_XRGB8888,    "XR24 (control)" },
    };
    for (unsigned i = 0; i < sizeof(fmts)/sizeof(fmts[0]); i++) {
        test_format(&nvidia, &nvidia, fmts[i].f, fmts[i].n);
        test_format(&nvidia, &intel,  fmts[i].f, fmts[i].n);
        test_format(&intel,  &nvidia, fmts[i].f, fmts[i].n);
        test_format(&intel,  &intel,  fmts[i].f, fmts[i].n);
    }
    return 0;
}
