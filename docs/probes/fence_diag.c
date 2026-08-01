// fence_diag — what IS the fd that NVIDIA Vulkan exports via vkGetFenceFdKHR,
// does it signal, and can Mesa EGL import it? Also: what fd does NVIDIA EGL
// export via eglDupNativeFenceFDANDROID, and why does Vulkan reject it?
// Build: gcc -O1 -o fence_diag fence_diag.c -lgbm -lEGL -lGLESv2 -lvulkan
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <stdint.h>
#include <poll.h>
#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <vulkan/vulkan.h>

static void fdinfo(int fd, const char *label) {
    char path[64], buf[1024];
    snprintf(path, sizeof path, "/proc/self/fdinfo/%d", fd);
    int f = open(path, O_RDONLY);
    if (f < 0) { printf("   %s: fdinfo unavailable (%s)\n", label, strerror(errno)); return; }
    int n = read(f, buf, sizeof buf - 1);
    close(f);
    buf[n > 0 ? n : 0] = 0;
    printf("   %s (fd %d):\n", label, fd);
    char *line = strtok(buf, "\n");
    while (line) { printf("      %s\n", line); line = strtok(NULL, "\n"); }
}

static int poll_fd(int fd, int timeout_ms, const char *label) {
    struct pollfd p = { .fd = fd, .events = POLLIN };
    int r = poll(&p, 1, timeout_ms);
    printf("   poll(%s, %dms) -> %d (%s)\n", label, timeout_ms, r,
           r > 0 ? "SIGNALED" : r == 0 ? "timeout" : strerror(errno));
    return r;
}

int main(void) {
    // ---- vulkan on NVIDIA (renderD129)
    VkInstance inst;
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "fence_diag", .apiVersion = VK_API_VERSION_1_3 };
    vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app }, NULL, &inst);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice *devs = malloc(n * sizeof(*devs));
    vkEnumeratePhysicalDevices(inst, &n, devs);
    VkPhysicalDevice phd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        printf("   phd %u: %s (vendor 0x%04x)\n", i, p.deviceName, p.vendorID);
        if (p.vendorID == 0x10de) phd = devs[i];
    }
    if (!phd) { printf("no NVIDIA phd\n"); return 1; }
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = 0, .queueCount = 1, .pQueuePriorities = &prio };
    const char *exts[] = { "VK_KHR_external_fence", "VK_KHR_external_fence_fd",
                           "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd" };
    VkDevice dev;
    VkResult r = vkCreateDevice(phd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 4, .ppEnabledExtensionNames = exts }, NULL, &dev);
    printf("== vkCreateDevice: %d\n", r);
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    VkCommandPool pool;
    vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = 0, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &pool);
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1 }, &cmd);

    // ---- A) exportable fence, trivial submit, export fd
    printf("== A) NVIDIA vulkan fence fd export\n");
    VkExportFenceCreateInfo expf = { .sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT };
    VkFence fence;
    r = vkCreateFence(dev, &(VkFenceCreateInfo){ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = &expf }, NULL, &fence);
    printf("   create exportable fence: %d\n", r);
    vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT });
    vkEndCommandBuffer(cmd);
    VkCommandBuffer cmds[1] = { cmd };
    r = vkQueueSubmit(queue, 1, &(VkSubmitInfo){ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = cmds }, fence);
    printf("   submit: %d\n", r);
    PFN_vkGetFenceFdKHR pGetFd = (PFN_vkGetFenceFdKHR)vkGetDeviceProcAddr(dev, "vkGetFenceFdKHR");
    int ffd = -1;
    r = pGetFd(dev, &(VkFenceGetFdInfoKHR){ .sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
        .fence = fence, .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT }, &ffd);
    printf("   get_fence_fd: %d (fd=%d)\n", r, ffd);
    fdinfo(ffd, "NVIDIA vulkan exported fence");
    printf("   vkGetFenceStatus: %d\n", vkGetFenceStatus(dev, fence));
    poll_fd(ffd, 1000, "nv-vk-fence");
    printf("   vkGetFenceStatus after poll: %d\n", vkGetFenceStatus(dev, fence));

    // ---- B) NVIDIA EGL native fence export
    printf("== B) NVIDIA EGL native fence export\n");
    int gfd = open("/dev/dri/renderD129", O_RDWR | O_CLOEXEC);
    struct gbm_device *gbm = gbm_create_device(gfd);
    EGLDisplay dpy = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm, NULL);
    eglInitialize(dpy, NULL, NULL);
    eglBindAPI(EGL_OPENGL_ES_API);
    EGLConfig cfg; EGLint nc = 0;
    eglChooseConfig(dpy, (EGLint[]){ EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR, EGL_NONE }, &cfg, 1, &nc);
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT,
        (EGLint[]){ EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE });
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx);
    printf("   EGL: %s\n", glGetString(GL_RENDERER));
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT); // no FBO bound; just to have work
    EGLSyncKHR sync = ((PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR"))(dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, NULL);
    printf("   eglCreateSyncKHR(NATIVE_FENCE): %p err=0x%x\n", (void*)sync, eglGetError());
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC pDup =
        (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress("eglDupNativeFenceFDANDROID");
    int efd = pDup ? pDup(dpy, sync) : -2;
    printf("   eglDupNativeFenceFDANDROID -> fd %d\n", efd);
    if (efd >= 0) {
        fdinfo(efd, "NVIDIA EGL exported fence");
        poll_fd(efd, 1000, "nv-egl-fence");
        // C) try importing that fd into NVIDIA Vulkan as a semaphore
        printf("== C) import NVIDIA EGL fence fd into NVIDIA vulkan\n");
        VkSemaphore sem;
        r = vkCreateSemaphore(dev, &(VkSemaphoreCreateInfo){ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }, NULL, &sem);
        printf("   create semaphore: %d\n", r);
        PFN_vkImportSemaphoreFdKHR pImp = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(dev, "vkImportSemaphoreFdKHR");
        int dupfd = dup(efd);
        r = pImp(dev, &(VkImportSemaphoreFdInfoKHR){ .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            .semaphore = sem, .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT, .fd = dupfd });
        printf("   import_semaphore_fd: %d\n", r);
    }
    return 0;
}
