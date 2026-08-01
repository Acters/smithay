// vk_export_test — NVIDIA produces (GBM block-linear + GLES fill, Vulkan copy to
// LINEAR, export) and Intel VULKAN imports it (explicit-modifier path), then
// verifies pixels via a host-visible readback. The full "promising reverse path".
// Also tests Intel GBM-level import (gbm_bo_import) of an NVIDIA GBM LINEAR bo.
// Build: gcc -O1 -o vk_export_test vk_export_test.c -lgbm -lEGL -lGLESv2 -lvulkan
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
#include <vulkan/vulkan.h>

// ---------------- EGL/GBM helpers (from earlier probes) ----------------
typedef EGLImage (EGLAPIENTRYP PCREATEIMG)(EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLAttrib*);
typedef void (EGLAPIENTRYP PIMGTARGET)(GLenum, GLeglImageOES);
static PCREATEIMG pCreateImage;
static PIMGTARGET pImageTargetTex;

static int gles_fill(EGLDisplay dpy, EGLContext ctx, struct gbm_bo *bo, uint32_t fourcc, float r, float g, float b) {
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
    glClearColor(r, g, b, 1.f); glClear(GL_COLOR_BUFFER_BIT); glFinish();
    return 0;
}

// ---------------- Vulkan context with pluggable GPU ----------------
typedef struct {
    VkInstance inst;
    VkPhysicalDevice phd;
    VkDevice dev;
    VkQueue queue;
    VkCommandPool pool;
} VkCtx;

static int vk_init_ctx(VkCtx *c, uint32_t want_vendor, const char *tag) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vk_export_test", .apiVersion = VK_API_VERSION_1_3 };
    vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app }, NULL, &c->inst);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(c->inst, &n, NULL);
    VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
    vkEnumeratePhysicalDevices(c->inst, &n, devs);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == want_vendor) c->phd = devs[i];
    }
    free(devs);
    if (!c->phd) { printf("   %s: no phd\n", tag); return -1; }
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = {
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };
    if (vkCreateDevice(c->phd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 5, .ppEnabledExtensionNames = exts }, NULL, &c->dev) != VK_SUCCESS)
        return -1;
    vkGetDeviceQueue(c->dev, 0, 0, &c->queue);
    vkCreateCommandPool(c->dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &c->pool);
    printf("   VK device up (%s)\n", tag);
    return 0;
}

static uint32_t mem_type(VkCtx *c, uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(c->phd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    return UINT32_MAX;
}

// ---------------- globals ----------------
static struct gbm_device *nv_gbm, *in_gbm;
static EGLDisplay nv_dpy;
static EGLContext nv_ctx;

int main(void) {
    pCreateImage = (PCREATEIMG)eglGetProcAddress("eglCreateImage");
    pImageTargetTex = (PIMGTARGET)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    int nfd = open("/dev/dri/renderD129", O_RDWR | O_CLOEXEC);
    int ifd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    nv_gbm = gbm_create_device(nfd);
    in_gbm = gbm_create_device(ifd);
    nv_dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, nv_gbm, NULL);
    eglInitialize(nv_dpy, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLConfig cfg; EGLint nc = 0;
    eglChooseConfig(nv_dpy, (EGLint[]){ EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE }, &cfg, 1, &nc);
    nv_ctx = eglCreateContext(nv_dpy, cfg, EGL_NO_CONTEXT,
        (EGLint[]){ EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE });
    eglMakeCurrent(nv_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, nv_ctx);

    VkCtx nv = {0}, in = {0};
    if (vk_init_ctx(&nv, 0x10de, "nvidia") || vk_init_ctx(&in, 0x8086, "intel")) return 1;

    uint32_t fourcc = GBM_FORMAT_XRGB8888;

    // ================= test 1: NVIDIA GBM LINEAR -> Intel GBM import =================
    printf("== test 1: NVIDIA GBM LINEAR -> Intel GBM import\n");
    uint64_t linear = 0ULL;
    struct gbm_bo *gbo = gbm_bo_create_with_modifiers2(nv_gbm, 64, 64, fourcc, &linear, 1,
                                                       GBM_BO_USE_RENDERING);
    if (!gbo) {
        printf("   NVIDIA GBM LINEAR alloc: FAILED\n");
    } else {
        printf("   NVIDIA GBM alloc: modifier=0x%llx stride=%u\n",
               (unsigned long long)gbm_bo_get_modifier(gbo), gbm_bo_get_stride_for_plane(gbo, 0));
        struct gbm_import_fd_modifier_data idata = {
            .width = 64, .height = 64, .format = fourcc,
            .num_fds = 1, .fds = { gbm_bo_get_fd(gbo) },
            .strides = { gbm_bo_get_stride_for_plane(gbo, 0) },
            .offsets = { 0 },
            .modifier = gbm_bo_get_modifier(gbo),
        };
        struct gbm_bo *ibo = gbm_bo_import(in_gbm, GBM_BO_IMPORT_FD_MODIFIER, &idata,
                                           GBM_BO_USE_RENDERING);
        printf("   Intel GBM import: %s\n", ibo ? "OK" : "FAILED");
    }

    // ================= test 2: NVIDIA produce (optimal->copy->LINEAR export) -> Intel Vulkan =================
    printf("== test 2: NVIDIA produce -> Intel Vulkan import + pixel verify\n");
    // 2a. NVIDIA GBM block-linear src, GLES fills red
    struct gbm_bo *src = gbm_bo_create(nv_gbm, 64, 64, fourcc, GBM_BO_USE_RENDERING);
    printf("   NVIDIA GBM src: modifier=0x%llx\n", (unsigned long long)gbm_bo_get_modifier(src));
    if (gles_fill(nv_dpy, nv_ctx, src, fourcc, 1.f, 0.f, 0.f)) return 1;
    printf("   GLES rendered red (NVIDIA)\n");

    // 2b. NVIDIA Vulkan: import src (explicit modifier), create LINEAR dst (exportable), copy
    //     (mirrors the bridge path)
    uint64_t src_mod = gbm_bo_get_modifier(src);
    uint32_t src_stride = gbm_bo_get_stride_for_plane(src, 0);
    VkSubresourceLayout src_layout = {
        .offset = 0, .size = 0, .rowPitch = src_stride, .arrayPitch = 0, .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT src_mod_create = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = src_mod,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &src_layout,
    };
    VkExternalMemoryImageCreateInfo src_ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &src_mod_create,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo src_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &src_ext,
        .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { 64, 64, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage src_img;
    VkResult r = vkCreateImage(nv.dev, &src_ci, NULL, &src_img);
    printf("   NVIDIA import image (block-linear src): %d\n", r);
    VkMemoryRequirements sreq;
    vkGetImageMemoryRequirements(nv.dev, src_img, &sreq);
    int dup_fd = dup(gbm_bo_get_fd(src));
    VkImportMemoryFdInfoKHR imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = dup_fd,
    };
    VkMemoryAllocateInfo sai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imp, .allocationSize = sreq.size,
        .memoryTypeIndex = mem_type(&nv, sreq.memoryTypeBits, 0),
    };
    VkDeviceMemory smem;
    r = vkAllocateMemory(nv.dev, &sai, NULL, &smem);
    printf("   NVIDIA import alloc: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    vkBindImageMemory(nv.dev, src_img, smem, 0);

    // dst: the flags-0 LINEAR GBM BO, imported into vulkan as TRANSFER_DST
    uint64_t linear2 = 0ULL;
    struct gbm_bo *dst_bo = gbm_bo_create_with_modifiers2(nv_gbm, 64, 64, fourcc,
                                                          &linear2, 1, 0);
    printf("   flags-0 LINEAR dst BO alloc: %s\n", dst_bo ? "OK" : "FAILED");
    if (!dst_bo) return 1;
    VkSubresourceLayout dst_layout = {
        .offset = 0, .size = 0,
        .rowPitch = gbm_bo_get_stride_for_plane(dst_bo, 0), .arrayPitch = 0, .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT dst_mod_create = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = 0ULL,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &dst_layout,
    };
    VkExternalMemoryImageCreateInfo dst_ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &dst_mod_create,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo dst_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &dst_ext,
        .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { 64, 64, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                 VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage dst_img;
    r = vkCreateImage(nv.dev, &dst_ci, NULL, &dst_img);
    printf("   dst BO import image create: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    VkMemoryRequirements dreq;
    vkGetImageMemoryRequirements(nv.dev, dst_img, &dreq);
    int ddup = dup(gbm_bo_get_fd(dst_bo));
    VkImportMemoryFdInfoKHR dimp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = ddup,
    };
    VkMemoryAllocateInfo dai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dimp, .allocationSize = dreq.size,
        .memoryTypeIndex = mem_type(&nv, dreq.memoryTypeBits, 0),
    };
    VkDeviceMemory dmem;
    r = vkAllocateMemory(nv.dev, &dai, NULL, &dmem);
    printf("   dst BO import alloc: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    vkBindImageMemory(nv.dev, dst_img, dmem, 0);

    // copy src -> dst
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(nv.dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = nv.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cmd);
    vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
    VkImageMemoryBarrier barriers[2] = {
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = 0,
          .image = src_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
        { .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
          .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
          .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = 0,
          .image = dst_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } },
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 2, barriers);
    VkImageCopy region = {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .extent = { 64, 64, 1 },
    };
    vkCmdCopyImage(cmd, src_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   dst_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    vkEndCommandBuffer(cmd);
    vkQueueSubmit(nv.queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmd }, VK_NULL_HANDLE);
    vkQueueWaitIdle(nv.queue);
    printf("   NVIDIA copy done (optimal -> LINEAR)\n");

    // NVIDIA-side readback of dst BO to localize the failure
    {
        VkBuffer rb;
        vkCreateBuffer(nv.dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 64 * 64 * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE }, NULL, &rb);
        VkMemoryRequirements rreq;
        vkGetBufferMemoryRequirements(nv.dev, rb, &rreq);
        uint32_t rmt = mem_type(&nv, rreq.memoryTypeBits,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkDeviceMemory rmem;
        vkAllocateMemory(nv.dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = rreq.size, .memoryTypeIndex = rmt }, NULL, &rmem);
        vkBindBufferMemory(nv.dev, rb, rmem, 0);
        VkCommandBuffer rc;
        vkAllocateCommandBuffers(nv.dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = nv.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &rc);
        vkBeginCommandBuffer(rc, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
        VkImageMemoryBarrier rbar = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = dst_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(rc, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &rbar);
        VkBufferImageCopy rregion = {
            .bufferOffset = 0, .bufferRowLength = 64, .bufferImageHeight = 64,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { 64, 64, 1 },
        };
        vkCmdCopyImageToBuffer(rc, dst_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1, &rregion);
        vkEndCommandBuffer(rc);
        vkQueueSubmit(nv.queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &rc }, VK_NULL_HANDLE);
        vkQueueWaitIdle(nv.queue);
        void *rmapped;
        vkMapMemory(nv.dev, rmem, 0, 64 * 64 * 4, 0, &rmapped);
        uint8_t *rpx = rmapped;
        printf("   NVIDIA-side readback of dst BO: BGRA = %u %u %u %u\n", rpx[0], rpx[1], rpx[2], rpx[3]);
        vkUnmapMemory(nv.dev, rmem);
    }

    // export
    int bridge_sync_fd = -1;

    // Release the imported GBM-backed image from NVIDIA Vulkan to a
    // different driver/device. vkQueueWaitIdle alone does not transfer ownership.
    {
        VkExportSemaphoreCreateInfo sem_export = {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        };
        VkSemaphoreCreateInfo sem_ci = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &sem_export,
        };
        VkSemaphore nv_release_sem = VK_NULL_HANDLE;
        VkResult sem_r = vkCreateSemaphore(nv.dev, &sem_ci, NULL, &nv_release_sem);
        if (sem_r != VK_SUCCESS) {
            printf("   NVIDIA sync semaphore create failed: %d\n", sem_r);
            return 1;
        }

        VkCommandBuffer rel_cmd;
        vkAllocateCommandBuffers(nv.dev,
            &(VkCommandBufferAllocateInfo){
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = nv.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            }, &rel_cmd);
        vkBeginCommandBuffer(rel_cmd,
            &(VkCommandBufferBeginInfo){
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            });

        VkImageMemoryBarrier rel = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = 0,
            // The NVIDIA-side verification readback left dst_img in this layout.
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = 0,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT,
            .image = dst_img,
            .subresourceRange = {
                VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1
            },
        };

        vkCmdPipelineBarrier(rel_cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0, 0, NULL, 0, NULL, 1, &rel);
        vkEndCommandBuffer(rel_cmd);

        VkResult rel_r = vkQueueSubmit(nv.queue, 1,
            &(VkSubmitInfo){
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &rel_cmd,
                .signalSemaphoreCount = 1,
                .pSignalSemaphores = &nv_release_sem,
            }, VK_NULL_HANDLE);
        if (rel_r != VK_SUCCESS) {
            printf("   NVIDIA release submit failed: %d\n", rel_r);
            return 1;
        }
        PFN_vkGetSemaphoreFdKHR pGetSemaphoreFd =
            (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(
                nv.dev, "vkGetSemaphoreFdKHR");
        if (!pGetSemaphoreFd) {
            printf("   NVIDIA vkGetSemaphoreFdKHR unavailable\n");
            return 1;
        }

        sem_r = pGetSemaphoreFd(
            nv.dev,
            &(VkSemaphoreGetFdInfoKHR){
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
                .semaphore = nv_release_sem,
                .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            },
            &bridge_sync_fd);
        printf("   NVIDIA exported release SYNC_FD: result=%d fd=%d\n",
               sem_r, bridge_sync_fd);
        if (sem_r != VK_SUCCESS || bridge_sync_fd < 0)
            return 1;

        vkQueueWaitIdle(nv.queue);
        printf("   NVIDIA released dst BO to FOREIGN\n");
    }


    /*
     * TRUE semaphore-only isolation test.
     *
     * Create a fresh Intel VkDevice which never imports or binds the DMA-BUF.
     * Import a dup() of NVIDIA's release SYNC_FD and wait on it with an empty
     * command buffer.  If this fails, SYNC_FD interoperability itself fails.
     * If it succeeds but the later image submission fails, the DMA-BUF mapping
     * or use is the problem.
     */
    {
        VkCtx sync_in = {0};
        if (vk_init_ctx(&sync_in, 0x8086, "intel-sync-only")) {
            printf("   Fresh Intel sync-only device creation failed\n");
            return 1;
        }

        VkSemaphore sync_only_sem = VK_NULL_HANDLE;
        r = vkCreateSemaphore(
            sync_in.dev,
            &(VkSemaphoreCreateInfo) {
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            },
            NULL,
            &sync_only_sem);
        printf("   Fresh Intel sync-only semaphore create: %d\n", r);
        if (r != VK_SUCCESS)
            return 1;

        PFN_vkImportSemaphoreFdKHR sync_only_import =
            (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(
                sync_in.dev, "vkImportSemaphoreFdKHR");
        if (!sync_only_import) {
            printf("   Fresh Intel vkImportSemaphoreFdKHR unavailable\n");
            return 1;
        }

        int sync_only_fd = dup(bridge_sync_fd);
        if (sync_only_fd < 0) {
            perror("dup bridge_sync_fd");
            return 1;
        }

        r = sync_only_import(
            sync_in.dev,
            &(VkImportSemaphoreFdInfoKHR) {
                .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
                .semaphore = sync_only_sem,
                .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
                .handleType =
                    VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
                .fd = sync_only_fd,
            });
        printf("   Fresh Intel sync-only FD import: %d\n", r);
        if (r != VK_SUCCESS) {
            close(sync_only_fd);
            return 1;
        }
        /* Ownership of sync_only_fd transferred on successful import. */

        VkCommandBuffer sync_only_cmd = VK_NULL_HANDLE;
        r = vkAllocateCommandBuffers(
            sync_in.dev,
            &(VkCommandBufferAllocateInfo) {
                .sType =
                    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                .commandPool = sync_in.pool,
                .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                .commandBufferCount = 1,
            },
            &sync_only_cmd);
        if (r != VK_SUCCESS) {
            printf("   Fresh Intel sync-only command allocation: %d\n", r);
            return 1;
        }

        r = vkBeginCommandBuffer(
            sync_only_cmd,
            &(VkCommandBufferBeginInfo) {
                .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            });
        if (r != VK_SUCCESS) {
            printf("   Fresh Intel sync-only command begin: %d\n", r);
            return 1;
        }

        r = vkEndCommandBuffer(sync_only_cmd);
        if (r != VK_SUCCESS) {
            printf("   Fresh Intel sync-only command end: %d\n", r);
            return 1;
        }

        VkPipelineStageFlags sync_only_stage =
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkSubmitInfo sync_only_submit = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &sync_only_sem,
            .pWaitDstStageMask = &sync_only_stage,
            .commandBufferCount = 1,
            .pCommandBuffers = &sync_only_cmd,
        };

        r = vkQueueSubmit(
            sync_in.queue, 1, &sync_only_submit, VK_NULL_HANDLE);
        printf("   Fresh Intel sync-only submit: %d\n", r);
        if (r != VK_SUCCESS)
            return 1;

        r = vkQueueWaitIdle(sync_in.queue);
        printf("   Fresh Intel sync-only wait idle: %d\n", r);
        if (r != VK_SUCCESS)
            return 1;
    }

    // the GBM BO's own fd is the export — no vkGetMemoryFdKHR needed
    int out_fd = dup(gbm_bo_get_fd(dst_bo));
    uint32_t out_stride = gbm_bo_get_stride_for_plane(dst_bo, 0);
    printf("   exported GBM BO fd=%d stride=%u\n", out_fd, out_stride);

    // 2b2. Intel GBM import of the same export (gbm_bo_import)
    {
        struct gbm_import_fd_modifier_data idata = {
            .width = 64, .height = 64, .format = fourcc,
            .num_fds = 1, .fds = { dup(out_fd) },
            .strides = { (int)out_stride },
            .offsets = { 0 },
            .modifier = 0ULL,
        };
        struct gbm_bo *ibo = gbm_bo_import(in_gbm, GBM_BO_IMPORT_FD_MODIFIER, &idata, 0);
        printf("   Intel GBM import (flags=0): %s\n", ibo ? "OK" : "FAILED");
        struct gbm_import_fd_modifier_data idata2 = idata;
        idata2.fds[0] = dup(out_fd);
        struct gbm_bo *ibo2 = gbm_bo_import(in_gbm, GBM_BO_IMPORT_FD_MODIFIER, &idata2,
                                            GBM_BO_USE_RENDERING);
        printf("   Intel GBM import (RENDERING): %s\n", ibo2 ? "OK" : "FAILED");
    }

    // 2c. Intel Vulkan: import with explicit modifier (LINEAR) + plane layout, bind, verify
    VkSubresourceLayout in_layout = {
        .offset = 0, .size = 0,
        .rowPitch = out_stride, .arrayPitch = 0, .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT in_mod_create = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = 0ULL, // LINEAR
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &in_layout,
    };
    VkExternalMemoryImageCreateInfo in_ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &in_mod_create,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo in_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &in_ext,
        .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { 64, 64, 1 }, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage in_img;
    r = vkCreateImage(in.dev, &in_ci, NULL, &in_img);
    printf("   Intel Vulkan import image: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    VkMemoryRequirements ireq;
    vkGetImageMemoryRequirements(in.dev, in_img, &ireq);
    int in_dup = dup(out_fd);
    VkImportMemoryFdInfoKHR iimp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, .fd = in_dup,
    };

    PFN_vkGetMemoryFdPropertiesKHR pGetMemoryFdProperties =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(
            in.dev, "vkGetMemoryFdPropertiesKHR");
    if (!pGetMemoryFdProperties) {
        printf("   Intel vkGetMemoryFdPropertiesKHR unavailable\n");
        return 1;
    }

    VkMemoryFdPropertiesKHR intel_fd_props = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    };
    r = pGetMemoryFdProperties(
        in.dev,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        in_dup,
        &intel_fd_props);
    printf("   Intel fd properties query: %d\n", r);
    if (r != VK_SUCCESS) {
        close(in_dup);
        return 1;
    }

    uint32_t intel_compatible_types =
        ireq.memoryTypeBits & intel_fd_props.memoryTypeBits;

    printf("   Intel FD/image memory types: image=0x%08x fd=0x%08x intersection=0x%08x\n",
           ireq.memoryTypeBits,
           intel_fd_props.memoryTypeBits,
           intel_compatible_types);

    if (intel_compatible_types == 0) {
        printf("   Intel import has no bind-compatible memory type\n");
        close(in_dup);
        return 1;
    }

    uint32_t intel_memory_type =
        mem_type(&in, intel_compatible_types, 0);
    printf("   Intel selected imported memory type: %u\n",
           intel_memory_type);
    if (intel_memory_type == UINT32_MAX) {
        close(in_dup);
        return 1;
    }

    VkMemoryAllocateInfo iai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &iimp, .allocationSize = ireq.size,
        .memoryTypeIndex = intel_memory_type,
    };
    VkDeviceMemory imem;
    r = vkAllocateMemory(in.dev, &iai, NULL, &imem);
    printf("   Intel Vulkan import alloc: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    r = vkBindImageMemory(in.dev, in_img, imem, 0);
    printf("   Intel Vulkan bind: %d\n", r);
    if (r != VK_SUCCESS) return 1;

    VkSemaphore intel_wait_sem = VK_NULL_HANDLE;
    r = vkCreateSemaphore(
        in.dev,
        &(VkSemaphoreCreateInfo){
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        },
        NULL,
        &intel_wait_sem);
    if (r != VK_SUCCESS) {
        printf("   Intel sync semaphore create failed: %d\n", r);
        return 1;
    }

    PFN_vkImportSemaphoreFdKHR pImportSemaphoreFd =
        (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(
            in.dev, "vkImportSemaphoreFdKHR");
    if (!pImportSemaphoreFd) {
        printf("   Intel vkImportSemaphoreFdKHR unavailable\n");
        return 1;
    }

    r = pImportSemaphoreFd(
        in.dev,
        &(VkImportSemaphoreFdInfoKHR){
            .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            .semaphore = intel_wait_sem,
            .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            .fd = bridge_sync_fd,
        });
    printf("   Intel imported release SYNC_FD: %d\n", r);
    if (r != VK_SUCCESS) return 1;
    // fd ownership transferred to Vulkan on successful import.
    bridge_sync_fd = -1;

    // verify: copy imported image -> host buffer on Intel, read pixel
    VkBuffer buf;
    vkCreateBuffer(in.dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = 64 * 64 * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE }, NULL, &buf);
    VkMemoryRequirements breq;
    vkGetBufferMemoryRequirements(in.dev, buf, &breq);
    uint32_t bmt = mem_type(&in, breq.memoryTypeBits,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory bmem;
    vkAllocateMemory(in.dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = breq.size, .memoryTypeIndex = bmt }, NULL, &bmem);
    vkBindBufferMemory(in.dev, buf, bmem, 0);
    VkCommandBuffer icmd;
    vkAllocateCommandBuffers(in.dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = in.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &icmd);
    vkBeginCommandBuffer(icmd, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
    VkImageMemoryBarrier ibar = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT, .dstQueueFamilyIndex = 0,
        .image = in_img, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    vkCmdPipelineBarrier(icmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &ibar);
    VkBufferImageCopy bregion = {
        .bufferOffset = 0, .bufferRowLength = 64, .bufferImageHeight = 64,
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent = { 64, 64, 1 },
    };
    vkCmdCopyImageToBuffer(icmd, in_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &bregion);
    vkEndCommandBuffer(icmd);
        /*
     * Split the consumer operation into two submissions:
     *
     * 1. Wait on the imported NVIDIA SYNC_FD without referencing the
     *    imported image.
     * 2. Submit the command buffer that acquires and reads the imported image,
     *    without an external-semaphore wait.
     *
     * This distinguishes semaphore interoperability failure from failure to
     * map/use the NVIDIA-allocated DMA-BUF on Intel.
     */
    VkCommandBuffer wait_cmd = VK_NULL_HANDLE;
    r = vkAllocateCommandBuffers(
        in.dev,
        &(VkCommandBufferAllocateInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = in.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        },
        &wait_cmd);
    if (r != VK_SUCCESS) {
        printf("   Intel semaphore-only command allocation: %d\n", r);
        return 1;
    }

    r = vkBeginCommandBuffer(
        wait_cmd,
        &(VkCommandBufferBeginInfo) {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        });
    if (r != VK_SUCCESS) {
        printf("   Intel semaphore-only command begin: %d\n", r);
        return 1;
    }

    r = vkEndCommandBuffer(wait_cmd);
    if (r != VK_SUCCESS) {
        printf("   Intel semaphore-only command end: %d\n", r);
        return 1;
    }

    VkPipelineStageFlags intel_wait_stage =
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

    VkSubmitInfo wait_submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &intel_wait_sem,
        .pWaitDstStageMask = &intel_wait_stage,
        .commandBufferCount = 1,
        .pCommandBuffers = &wait_cmd,
    };

    r = vkQueueSubmit(in.queue, 1, &wait_submit, VK_NULL_HANDLE);
    printf("   Intel semaphore-only submit: %d\n", r);
    if (r != VK_SUCCESS)
        return 1;

    r = vkQueueWaitIdle(in.queue);
    printf("   Intel semaphore-only wait idle: %d\n", r);
    if (r != VK_SUCCESS)
        return 1;

    VkSubmitInfo image_submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &icmd,
    };

    r = vkQueueSubmit(in.queue, 1, &image_submit, VK_NULL_HANDLE);
    printf("   Intel imported-image submit: %d\n", r);
    if (r != VK_SUCCESS)
        return 1;

    r = vkQueueWaitIdle(in.queue);
    printf("   Intel imported-image wait idle: %d\n", r);
    if (r != VK_SUCCESS)
        return 1;
    void *mapped;
    vkMapMemory(in.dev, bmem, 0, 64 * 64 * 4, 0, &mapped);
    uint8_t *px = mapped;
    printf("   pixel(32,32) BGRA = %u %u %u %u %s\n", px[0], px[1], px[2], px[3],
           (px[2] > 200 && px[0] < 60) ? "(REVERSE PATH WORKS!)" : "(wrong data)");
    vkUnmapMemory(in.dev, bmem);
    return 0;
}
