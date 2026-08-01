// vk_import_test — properly-structured foreign dmabuf import into NVIDIA Vulkan:
//   Intel GBM (LINEAR) -> Vulkan import via VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
//   + explicit plane layout -> vkGetMemoryFdPropertiesKHR -> dedicated alloc ->
//   bind -> vkCmdCopyImageToBuffer -> verify pixels on CPU.
// Build: gcc -O1 -o vk_import_test vk_import_test.c -lgbm -lEGL -lGLESv2 -lvulkan
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

static struct gbm_device *in_gbm;
static EGLDisplay in_dpy;
static EGLContext in_ctx;
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

static VkInstance inst;
static VkPhysicalDevice phd;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;

static int vk_init(void) {
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vk_import_test", .apiVersion = VK_API_VERSION_1_3 };
    vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app }, NULL, &inst);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
    vkEnumeratePhysicalDevices(inst, &n, devs);
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        printf("   phd %u: %s\n", i, p.deviceName);
        if (p.vendorID == 0x10de) phd = devs[i];
    }
    free(devs);
    if (!phd) return -1;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = {
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    if (vkCreateDevice(phd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 3, .ppEnabledExtensionNames = exts }, NULL, &dev) != VK_SUCCESS)
        return -1;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &pool);
    printf("   VK device up (NVIDIA)\n");
    return 0;
}

int main(void) {
    pCreateImage = (PCREATEIMG)eglGetProcAddress("eglCreateImage");
    pImageTargetTex = (PIMGTARGET)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    // 1. Intel GBM LINEAR buffer, filled red via Intel GLES
    int ifd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    in_gbm = gbm_create_device(ifd);
    in_dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, in_gbm, NULL);
    eglInitialize(in_dpy, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLConfig cfg; EGLint nc = 0;
    eglChooseConfig(in_dpy, (EGLint[]){ EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE }, &cfg, 1, &nc);
    in_ctx = eglCreateContext(in_dpy, cfg, EGL_NO_CONTEXT,
        (EGLint[]){ EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE });
    eglMakeCurrent(in_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, in_ctx);
    printf("   Intel GL: %s\n", glGetString(GL_RENDERER));

    uint32_t fourcc = GBM_FORMAT_ARGB8888;
    uint64_t linear = 0ULL;
    struct gbm_bo *bo = gbm_bo_create_with_modifiers(in_gbm, 64, 64, fourcc, &linear, 1);
    if (!bo) { printf("intel alloc failed\n"); return 1; }
    uint32_t plane_count = gbm_bo_get_plane_count(bo);
    uint64_t modifier = gbm_bo_get_modifier(bo);
    int fd = gbm_bo_get_fd(bo);
    uint32_t stride = gbm_bo_get_stride_for_plane(bo, 0);
    uint32_t offset = gbm_bo_get_offset(bo, 0);
    printf("   gbm: planes=%u modifier=0x%llx fd=%d stride=%u offset=%u\n",
           plane_count, (unsigned long long)modifier, fd, stride, offset);
    if (gles_fill(in_dpy, in_ctx, bo, fourcc, 1.f, 0.f, 0.f)) return 1;
    printf("   Intel rendered red into src\n");

    if (vk_init()) { printf("vk init failed\n"); return 1; }

    // 2. create image with VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT + explicit plane layout
    VkSubresourceLayout plane_layout = {
        .offset = offset, .size = 0, .rowPitch = stride, .arrayPitch = 0, .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod_create = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = modifier,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &plane_layout,
    };
    VkExternalMemoryImageCreateInfo ext_create = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &mod_create,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo img_create = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext_create,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { 64, 64, 1 },
        .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage image;
    VkResult r = vkCreateImage(dev, &img_create, NULL, &image);
    printf("   vkCreateImage (DRM_MODIFIER tiling): %d\n", r);
    if (r != VK_SUCCESS) return 1;

    // 3. image memory requirements + fd properties intersection
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, image, &req);
    printf("   image memoryTypeBits=0x%x size=%zu\n", req.memoryTypeBits, (size_t)req.size);

    PFN_vkGetMemoryFdPropertiesKHR pFdProps =
        (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev, "vkGetMemoryFdPropertiesKHR");
    VkMemoryFdPropertiesKHR fd_props = { .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    r = pFdProps(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fd_props);
    printf("   vkGetMemoryFdPropertiesKHR: %d memoryTypeBits=0x%x\n", r, fd_props.memoryTypeBits);
    uint32_t compat = req.memoryTypeBits & fd_props.memoryTypeBits;
    printf("   intersection: 0x%x %s\n", compat, compat ? "(bind-compatible type exists)" : "(NONE)");

    // 4. import + bind
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phd, &mp);
    uint32_t mt = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount && compat; i++)
        if (compat & (1u << i)) { mt = i; break; }
    int dup_fd = dup(fd);
    VkMemoryDedicatedAllocateInfo ded = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .image = image,
    };
    VkImportMemoryFdInfoKHR imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .pNext = &ded,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = dup_fd,
    };
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &imp,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    VkDeviceMemory mem;
    r = vkAllocateMemory(dev, &ai, NULL, &mem);
    printf("   vkAllocateMemory (import, type %u): %d\n", mt, r);
    if (r == VK_SUCCESS) {
        r = vkBindImageMemory(dev, image, mem, 0);
        printf("   vkBindImageMemory: %d %s\n", r, r == VK_SUCCESS ? "SUCCESS" : "FAIL");
    }

    // 5. verify pixels: copy image -> host-visible buffer, map, check
    if (r == VK_SUCCESS) {
        VkBuffer buf;
        vkCreateBuffer(dev, &(VkBufferCreateInfo){ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 64 * 64 * 4, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE }, NULL, &buf);
        VkMemoryRequirements breq;
        vkGetBufferMemoryRequirements(dev, buf, &breq);
        uint32_t bmt = UINT32_MAX;
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((breq.memoryTypeBits & (1u << i)) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) { bmt = i; break; }
        VkDeviceMemory bmem;
        vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = breq.size, .memoryTypeIndex = bmt }, NULL, &bmem);
        vkBindBufferMemory(dev, buf, bmem, 0);

        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cmd);
        vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL, .dstQueueFamilyIndex = 0,
            .image = image, .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, NULL, 0, NULL, 1, &barrier);
        VkBufferImageCopy region = {
            .bufferOffset = 0, .bufferRowLength = 64, .bufferImageHeight = 64,
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { 64, 64, 1 },
        };
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &region);
        vkEndCommandBuffer(cmd);
        vkQueueSubmit(queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1, .pCommandBuffers = &cmd }, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        void *mapped;
        vkMapMemory(dev, bmem, 0, 64 * 64 * 4, 0, &mapped);
        uint8_t *px = mapped;
        printf("   pixel(32,32) BGRA = %u %u %u %u %s\n", px[0], px[1], px[2], px[3],
               (px[2] > 200 && px[0] < 60) ? "(IMPORT WORKS!)" : "(wrong data)");
        vkUnmapMemory(dev, bmem);
    }
    return 0;
}
