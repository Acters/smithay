#define _GNU_SOURCE
// vkbridge_thread_test — worker-thread initialization variant of vkbridge_poc:
//   NVIDIA: GBM buffer -> GLES renders color -> Vulkan imports dmabuf,
//           copies to a LINEAR image, exports as dmabuf
//   Intel:  imports linear dmabuf as EGLImage, samples it, verifies pixels
//
// Historical diagnostic for the Vulkan initialization hang. It starts the
// Vulkan work on another thread and applies a ten-second timeout.
// Build: gcc -O1 -Wall -Wextra -pthread -o vkbridge_thread_test vkbridge_thread_test.c $(pkg-config --cflags --libs gbm egl glesv2 vulkan)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <inttypes.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <vulkan/vulkan.h>
#include <pthread.h>
#include <time.h>

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { \
    printf("   VK FAIL %s:%d -> %d\n", __func__, __LINE__, r); return -1; } } while (0)

static int nv_fd = -1, in_fd_dev = -1;
static struct gbm_device *nv_gbm, *in_gbm;
static EGLDisplay nv_dpy, in_dpy;
static EGLContext nv_ctx;

typedef EGLImage (EGLAPIENTRYP PFNEGLCREATEIMAGEPROC_)(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLAttrib*);
typedef void (EGLAPIENTRYP PFNGLEGLIMAGETARGETTEXTURE2DOESPROC_)(GLenum, GLeglImageOES);
static PFNEGLCREATEIMAGEPROC_ pCreateImage;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC_ pImageTargetTex;

// ---------- EGL/GBM helpers (same as xb30_probe) ----------
static EGLDisplay egl_for_gbm(struct gbm_device *gbm, EGLContext *ctx_out, int make_ctx) {
    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    if (dpy == EGL_NO_DISPLAY || !eglInitialize(dpy, NULL, NULL)) return EGL_NO_DISPLAY;
    eglBindAPI(EGL_OPENGL_ES_API);
    if (make_ctx) {
        EGLint attrs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE };
        EGLConfig cfg; EGLint n = 0;
        eglChooseConfig(dpy, attrs, &cfg, 1, &n);
        EGLint cattrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
        *ctx_out = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, cattrs);
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, *ctx_out);
    }
    return dpy;
}

// GLES renders a solid color into a GBM buffer (self-import, FBO, clear)
static int gles_fill(struct gbm_device *gbm, EGLDisplay dpy, struct gbm_bo *bo,
                     uint32_t fourcc, float r, float g, float b) {
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, nv_ctx);
    int bfd = gbm_bo_get_fd(bo);
    EGLAttrib a[20]; int i = 0;
    a[i++] = EGL_WIDTH; a[i++] = 64; a[i++] = EGL_HEIGHT; a[i++] = 64;
    a[i++] = EGL_LINUX_DRM_FOURCC_EXT; a[i++] = (EGLAttrib)fourcc;
    a[i++] = EGL_DMA_BUF_PLANE0_FD_EXT; a[i++] = bfd;
    a[i++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; a[i++] = 0;
    a[i++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; a[i++] = (EGLAttrib)gbm_bo_get_stride_for_plane(bo, 0);
    uint64_t mod = gbm_bo_get_modifier(bo);
    a[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; a[i++] = (EGLAttrib)(mod & 0xFFFFFFFF);
    a[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; a[i++] = (EGLAttrib)(mod >> 32);
    a[i++] = EGL_NONE;
    EGLImage img = pCreateImage(dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
    if (!img) { printf("   gles_fill: import failed 0x%x\n", eglGetError()); return -1; }
    GLuint t, f;
    glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    pImageTargetTex(GL_TEXTURE_2D, img);
    glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        printf("   gles_fill: FBO incomplete\n"); return -1;
    }
    glClearColor(r, g, b, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glDeleteFramebuffers(1, &f); glDeleteTextures(1, &t);
    return 0;
}

// ---------- Vulkan state ----------
static VkInstance inst;
static VkPhysicalDevice phd;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfamily;
static VkCommandPool cmdpool;

static int vk_init(uint32_t want_major, uint32_t want_minor) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vkbridge_poc", .apiVersion = VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app };
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));

    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, devs));
    phd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceDrmPropertiesEXT drm = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT };
        VkPhysicalDeviceProperties2 p2 = { .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &drm };
        vkGetPhysicalDeviceProperties2(devs[i], &p2);
        printf("   vk phd %u: %s (render %" PRId64 ":%" PRId64 ")\n", i,
               p2.properties.deviceName, drm.renderMajor, drm.renderMinor);
        if (drm.hasRender && drm.renderMajor == want_major && drm.renderMinor == want_minor)
            phd = devs[i];
    }
    free(devs);
    if (!phd) { printf("   VK: no matching physical device\n"); return -1; }

    // find a queue family with GRAPHICS|TRANSFER
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phd, &qn, NULL);
    VkQueueFamilyProperties *qp = malloc(qn * sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(phd, &qn, qp);
    qfamily = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++)
        if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfamily = i; break; }
    free(qp);
    if (qfamily == UINT32_MAX) { printf("   VK: no graphics queue\n"); return -1; }

    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = qfamily, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = {
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    VkDeviceCreateInfo dci = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts };
    VK_CHECK(vkCreateDevice(phd, &dci, NULL, &dev));
    vkGetDeviceQueue(dev, qfamily, 0, &queue);

    VkCommandPoolCreateInfo cpci = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = qfamily, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT };
    VK_CHECK(vkCreateCommandPool(dev, &cpci, NULL, &cmdpool));
    printf("   VK device up (NVIDIA)\n");
    return 0;
}

static uint32_t mem_type_for(uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}

static int run_cmd_and_wait(VkCommandBuffer cmd) {
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd };
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    return 0;
}

// ---------- the bridge ----------
static int bridge_copy(int src_fd, uint32_t src_stride, uint64_t src_mod,
                       VkFormat vkfmt, uint32_t fourcc,
                       int *out_fd, uint32_t *out_stride) {
    // 1. create destination image: LINEAR, exportable, same format
    VkExternalMemoryImageCreateInfo ext_img = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &ext_img,
        .imageType = VK_IMAGE_TYPE_2D, .format = vkfmt,
        .extent = { 64, 64, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage dst;
    VK_CHECK(vkCreateImage(dev, &ici, NULL, &dst));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, dst, &req);
    VkExportMemoryAllocateInfo exp_alloc = {
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkMemoryAllocateInfo ai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &exp_alloc, .allocationSize = req.size,
        .memoryTypeIndex = mem_type_for(req.memoryTypeBits, 0) };
    if (ai.memoryTypeIndex == UINT32_MAX) { printf("   no memtype for dst\n"); return -1; }
    VkDeviceMemory dst_mem;
    VK_CHECK(vkAllocateMemory(dev, &ai, NULL, &dst_mem));
    VK_CHECK(vkBindImageMemory(dev, dst, dst_mem, 0));

    // 2. create source image wrapping the imported GBM dmabuf
    VkImageDrmFormatModifierListCreateInfoEXT src_mods = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
        .drmFormatModifierCount = 1, .pDrmFormatModifiers = &src_mod,
    };
    VkExternalMemoryImageCreateInfo src_ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .pNext = &src_mods,
    };
    VkImageCreateInfo sici = ici;
    sici.pNext = &src_ext;
    sici.tiling = VK_IMAGE_TILING_OPTIMAL;
    sici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage src;
    VK_CHECK(vkCreateImage(dev, &sici, NULL, &src));
    VkMemoryRequirements sreq;
    vkGetImageMemoryRequirements(dev, src, &sreq);
    int dup_fd = dup(src_fd);
    VkImportMemoryFdInfoKHR imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = dup_fd,
    };
    VkMemoryAllocateInfo sai = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imp, .allocationSize = sreq.size,
        .memoryTypeIndex = mem_type_for(sreq.memoryTypeBits, 0) };
    if (sai.memoryTypeIndex == UINT32_MAX) { printf("   no memtype for src import\n"); return -1; }
    VkDeviceMemory src_mem;
    VkResult r = vkAllocateMemory(dev, &sai, NULL, &src_mem);
    if (r != VK_SUCCESS) { printf("   src import alloc failed %d\n", r); return -1; }
    VK_CHECK(vkBindImageMemory(dev, src, src_mem, 0));
    printf("   VK: src dmabuf imported, dst LINEAR image created\n");

    // 3. copy src -> dst
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdpool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 };
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));
    VkCommandBufferBeginInfo bi = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    VkImageMemoryBarrier barriers[2] = {
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL, .dstQueueFamilyIndex = qfamily,
          .image = src, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL, .dstQueueFamilyIndex = qfamily,
          .image = dst, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 2, barriers);
    VkImageCopy region = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { 64, 64, 1 },
    };
    vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    if (run_cmd_and_wait(cmd)) return -1;
    printf("   VK: copy done (optimal -> LINEAR)\n");

    // 4. export dst as dmabuf
    VkMemoryGetFdInfoKHR fd_info = { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
        .memory = dst_mem, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT };
    PFN_vkGetMemoryFdKHR pGetMemoryFd = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdKHR");
    VK_CHECK(pGetMemoryFd(dev, &fd_info, out_fd));
    VkImageSubresource sub = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT };
    VkSubresourceLayout layout;
    vkGetImageSubresourceLayout(dev, dst, &sub, &layout);
    *out_stride = (uint32_t)layout.rowPitch;
    printf("   VK: exported LINEAR dmabuf fd=%d stride=%u\n", *out_fd, *out_stride);
    (void)fourcc; (void)src_stride;
    return 0;
}

// ---------- Intel verify ----------
static int intel_check(int fd, uint32_t stride, uint32_t fourcc) {
    EGLAttrib a[20]; int i = 0;
    a[i++] = EGL_WIDTH; a[i++] = 64; a[i++] = EGL_HEIGHT; a[i++] = 64;
    a[i++] = EGL_LINUX_DRM_FOURCC_EXT; a[i++] = (EGLAttrib)fourcc;
    a[i++] = EGL_DMA_BUF_PLANE0_FD_EXT; a[i++] = fd;
    a[i++] = EGL_DMA_BUF_PLANE0_OFFSET_EXT; a[i++] = 0;
    a[i++] = EGL_DMA_BUF_PLANE0_PITCH_EXT; a[i++] = (EGLAttrib)stride;
    a[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT; a[i++] = 0;
    a[i++] = EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT; a[i++] = 0;
    a[i++] = EGL_NONE;
    eglMakeCurrent(in_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLint cattrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLConfig cfg; EGLint n = 0;
    EGLint attrs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE };
    eglChooseConfig(in_dpy, attrs, &cfg, 1, &n);
    EGLContext ctx = eglCreateContext(in_dpy, cfg, EGL_NO_CONTEXT, cattrs);
    eglMakeCurrent(in_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);

    EGLImage img = pCreateImage(in_dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, a);
    if (!img) { printf("   INTEL: import failed 0x%x\n", eglGetError()); return -1; }
    GLuint t, f;
    glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    pImageTargetTex(GL_TEXTURE_2D, img);
    // sample via draw into RGBA8 FBO, then read center pixel
    GLuint t2, f2;
    glGenTextures(1, &t2); glBindTexture(GL_TEXTURE_2D, t2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glGenFramebuffers(1, &f2); glBindFramebuffer(GL_FRAMEBUFFER, f2);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t2, 0);
    // simple path: attach imported texture to FBO and read via blit? Mesa XB30 FBO quirk
    // -> instead draw with it as texture would need a shader; use glCopyTexSubImage2D
    glBindTexture(GL_TEXTURE_2D, t2);
    glBindTexture(GL_TEXTURE_2D, t);
    uint8_t px[4] = {0};
    // read the imported texture's first pixel by attaching to FBO if possible
    glGenFramebuffers(1, &f); glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE) {
        glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    } else {
        // Mesa XB30 quirk: imported tex not FBO-attachable. Sample via shader instead.
        glBindFramebuffer(GL_FRAMEBUFFER, f2);
        const char *vs_src = "attribute vec2 p; varying vec2 uv;"
            "void main(){ uv = p*0.5+0.5; gl_Position = vec4(p,0.,1.); }";
        const char *fs_src = "precision mediump float; varying vec2 uv;"
            "uniform sampler2D tex; void main(){ gl_FragColor = texture2D(tex, uv); }";
        GLuint vs = glCreateShader(GL_VERTEX_SHADER), fs = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(vs, 1, &vs_src, NULL); glCompileShader(vs);
        glShaderSource(fs, 1, &fs_src, NULL); glCompileShader(fs);
        GLint ok = 0; char infolog[512];
        glGetShaderiv(vs, GL_COMPILE_STATUS, &ok);
        if (!ok) { glGetShaderInfoLog(vs, 512, NULL, infolog); printf("   vs: %s\n", infolog); }
        glGetShaderiv(fs, GL_COMPILE_STATUS, &ok);
        if (!ok) { glGetShaderInfoLog(fs, 512, NULL, infolog); printf("   fs: %s\n", infolog); }
        GLuint prog = glCreateProgram();
        glAttachShader(prog, vs); glAttachShader(prog, fs);
        glBindAttribLocation(prog, 0, "p");
        glLinkProgram(prog);
        glGetProgramiv(prog, GL_LINK_STATUS, &ok);
        if (!ok) { glGetProgramInfoLog(prog, 512, NULL, infolog); printf("   link: %s\n", infolog); }
        glUseProgram(prog);
        glViewport(0, 0, 64, 64);
        printf("   shader err after use: 0x%x\n", glGetError());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glUniform1i(glGetUniformLocation(prog, "tex"), 0);
        float quad[] = { -1,-1, 1,-1, -1,1, 1,1 };
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, quad);
        glEnableVertexAttribArray(0);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        printf("   shader err after draw: 0x%x fbo=0x%x\n", glGetError(), glCheckFramebufferStatus(GL_FRAMEBUFFER));
        glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    }
    printf("   INTEL: center pixel RGBA = %u %u %u %u\n", px[0], px[1], px[2], px[3]);
    return 0;
}

static void *vk_init_trampoline(void *arg) {
    (void)arg;
    long r = (long)vk_init(226, 129);
    printf("   [thread] vk_init done: %ld\n", r);
    return (void *)r;
}

int main(void) {
    pCreateImage = (void *)eglGetProcAddress("eglCreateImage");
    pImageTargetTex = (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    nv_fd = open("/dev/dri/renderD129", O_RDWR | O_CLOEXEC);
    in_fd_dev = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    nv_gbm = gbm_create_device(nv_fd);
    in_gbm = gbm_create_device(in_fd_dev);
    nv_dpy = egl_for_gbm(nv_gbm, &nv_ctx, 1);
    in_dpy = egl_for_gbm(in_gbm, &(EGLContext){0}, 0);
    if (nv_dpy == EGL_NO_DISPLAY || in_dpy == EGL_NO_DISPLAY) { printf("EGL init failed\n"); return 1; }
    printf("   NVIDIA GL: %s\n", glGetString(GL_RENDERER));

    printf("   spawning vk_init on worker thread...\n");
    pthread_t th;
    int rc = pthread_create(&th, NULL, (void *(*)(void *))vk_init_trampoline, NULL);
    if (rc) { printf("   pthread_create failed\n"); return 1; }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 10;
    void *ret = NULL;
    int jrc = pthread_timedjoin_np(th, &ret, &ts);
    if (jrc == 0) {
        printf("   vk_init thread returned: %ld\n", (long)ret);
        if ((long)ret != 0) return 1;
    } else {
        printf("   vk_init thread TIMED OUT after 10s (err=%d) - still stuck\n", jrc);
        pthread_detach(th);
        return 1;
    }

    // 8-bit first: XR24 = DRM XRGB8888 <-> VK_FORMAT_B8G8R8A8_UNORM
    struct { uint32_t fourcc; VkFormat vk; const char *name; float r, g, b; } tests[] = {
        { GBM_FORMAT_XRGB8888, VK_FORMAT_B8G8R8A8_UNORM, "XR24", 1.f, 0.f, 0.f },
        { GBM_FORMAT_XBGR2101010, VK_FORMAT_A2B10G10R10_UNORM_PACK32, "XB30", 0.f, 1.f, 0.f },
    };
    for (int t = 0; t < 2; t++) {
        printf("== bridge test %s\n", tests[t].name);
        struct gbm_bo *bo = gbm_bo_create(nv_gbm, 64, 64, tests[t].fourcc,
                                          GBM_BO_USE_RENDERING);
        if (!bo) { printf("   gbm alloc failed\n"); continue; }
        printf("   gbm src: modifier=0x%llx stride=%u\n",
               (unsigned long long)gbm_bo_get_modifier(bo), gbm_bo_get_stride_for_plane(bo, 0));
        if (gles_fill(nv_gbm, nv_dpy, bo, tests[t].fourcc, tests[t].r, tests[t].g, tests[t].b)) continue;
        printf("   GLES rendered solid color into src\n");
        int out_fd = -1; uint32_t out_stride = 0;
        if (bridge_copy(gbm_bo_get_fd(bo), gbm_bo_get_stride_for_plane(bo, 0),
                        gbm_bo_get_modifier(bo), tests[t].vk, tests[t].fourcc,
                        &out_fd, &out_stride) == 0) {
            intel_check(out_fd, out_stride, tests[t].fourcc);
        }
        gbm_bo_destroy(bo);
    }
    return 0;
}
