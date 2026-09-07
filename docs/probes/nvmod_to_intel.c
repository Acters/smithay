/*
 * Probe whether Intel can import an NVIDIA-native GBM buffer through GBM or
 * EGL, and query Intel Vulkan support for the buffer's DRM modifier.
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o nvmod_to_intel nvmod_to_intel.c \
 *       $(pkg-config --cflags --libs gbm egl vulkan)
 *
 * The render-node defaults below match the machine used for the investigation.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

#include <gbm.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <vulkan/vulkan.h>

#define NVIDIA_NODE "/dev/dri/renderD129"
#define INTEL_NODE  "/dev/dri/renderD128"
#define WIDTH  64
#define HEIGHT 64

typedef EGLBoolean
(EGLAPIENTRYP QueryModifiers)(
    EGLDisplay,
    EGLint,
    EGLint,
    EGLuint64KHR *,
    EGLBoolean *,
    EGLint *);

typedef EGLImageKHR
(EGLAPIENTRYP CreateImage)(
    EGLDisplay,
    EGLContext,
    EGLenum,
    EGLClientBuffer,
    const EGLint *);

typedef EGLBoolean
(EGLAPIENTRYP DestroyImage)(
    EGLDisplay,
    EGLImageKHR);

static VkPhysicalDevice find_intel(VkInstance instance)
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, NULL);

    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    vkEnumeratePhysicalDevices(instance, &count, devices);

    VkPhysicalDevice intel = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);

        if (props.vendorID == 0x8086) {
            intel = devices[i];
            printf("   Intel Vulkan device: %s\n", props.deviceName);
            break;
        }
    }

    free(devices);
    return intel;
}

static int has_device_extension(VkPhysicalDevice device, const char *wanted)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL);

    VkExtensionProperties *exts = calloc(count, sizeof(*exts));
    vkEnumerateDeviceExtensionProperties(device, NULL, &count, exts);

    int found = 0;
    for (uint32_t i = 0; i < count; i++) {
        if (!strcmp(exts[i].extensionName, wanted)) {
            found = 1;
            break;
        }
    }

    free(exts);
    return found;
}

static void test_intel_gbm(
    struct gbm_device *intel_gbm,
    struct gbm_bo *nvidia_bo,
    uint32_t fourcc,
    uint64_t modifier,
    uint32_t stride,
    uint32_t offset)
{
    printf("\n== Intel GBM import ==\n");

    const struct {
        const char *name;
        uint32_t flags;
    } tests[] = {
        { "flags=0",   0 },
        { "RENDERING", GBM_BO_USE_RENDERING },
        { "SCANOUT",   GBM_BO_USE_SCANOUT },
    };

    for (unsigned i = 0; i < sizeof(tests) / sizeof(tests[0]); i++) {
        int dma_fd = gbm_bo_get_fd_for_plane(nvidia_bo, 0);

        struct gbm_import_fd_modifier_data data = {
            .width = WIDTH,
            .height = HEIGHT,
            .format = fourcc,
            .num_fds = 1,
            .fds = { dma_fd },
            .strides = { stride },
            .offsets = { offset },
            .modifier = modifier,
        };

        errno = 0;

        struct gbm_bo *imported =
            gbm_bo_import(
                intel_gbm,
                GBM_BO_IMPORT_FD_MODIFIER,
                &data,
                tests[i].flags);

        printf("   %-10s: %s",
               tests[i].name,
               imported ? "OK" : "FAIL");

        if (!imported)
            printf(" errno=%d (%s)", errno, strerror(errno));

        printf("\n");

        if (imported)
            gbm_bo_destroy(imported);

        close(dma_fd);
    }
}

static void test_intel_egl(
    struct gbm_device *intel_gbm,
    struct gbm_bo *nvidia_bo,
    uint32_t fourcc,
    uint64_t modifier,
    uint32_t stride,
    uint32_t offset)
{
    printf("\n== Intel EGL import ==\n");

    EGLDisplay display =
        eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, intel_gbm, NULL);

    if (display == EGL_NO_DISPLAY ||
        !eglInitialize(display, NULL, NULL)) {
        printf("   Intel EGL initialization failed: 0x%x\n",
               eglGetError());
        return;
    }

    QueryModifiers query_modifiers =
        (QueryModifiers)eglGetProcAddress(
            "eglQueryDmaBufModifiersEXT");

    if (!query_modifiers) {
        printf("   eglQueryDmaBufModifiersEXT unavailable\n");
    } else {
        EGLint count = 0;

        if (!query_modifiers(
                display,
                (EGLint)fourcc,
                0,
                NULL,
                NULL,
                &count)) {
            printf("   modifier count query failed: 0x%x\n",
                   eglGetError());
        } else {
            EGLuint64KHR *mods =
                calloc((size_t)count, sizeof(*mods));
            EGLBoolean *external_only =
                calloc((size_t)count, sizeof(*external_only));

            EGLint returned = 0;
            int found = 0;

            if (query_modifiers(
                    display,
                    (EGLint)fourcc,
                    count,
                    mods,
                    external_only,
                    &returned)) {
                for (EGLint i = 0; i < returned; i++) {
                    if ((uint64_t)mods[i] == modifier) {
                        printf(
                            "   modifier advertised: YES "
                            "external_only=%d\n",
                            external_only[i]);
                        found = 1;
                    }
                }

                if (!found)
                    printf("   modifier advertised: NO\n");
            } else {
                printf("   modifier list query failed: 0x%x\n",
                       eglGetError());
            }

            free(mods);
            free(external_only);
        }
    }

    CreateImage create_image =
        (CreateImage)eglGetProcAddress("eglCreateImageKHR");
    DestroyImage destroy_image =
        (DestroyImage)eglGetProcAddress("eglDestroyImageKHR");

    if (!create_image) {
        printf("   eglCreateImageKHR unavailable\n");
        eglTerminate(display);
        return;
    }

    int dma_fd = gbm_bo_get_fd_for_plane(nvidia_bo, 0);

    EGLint attrs[] = {
        EGL_WIDTH, WIDTH,
        EGL_HEIGHT, HEIGHT,
        EGL_LINUX_DRM_FOURCC_EXT, (EGLint)fourcc,

        EGL_DMA_BUF_PLANE0_FD_EXT, dma_fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT, (EGLint)offset,
        EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,

        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
        (EGLint)(modifier & 0xffffffffu),

        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
        (EGLint)(modifier >> 32),

        EGL_NONE
    };

    EGLImageKHR image =
        create_image(
            display,
            EGL_NO_CONTEXT,
            EGL_LINUX_DMA_BUF_EXT,
            NULL,
            attrs);

    if (image == EGL_NO_IMAGE_KHR) {
        printf("   direct EGLImage import: FAIL error=0x%x\n",
               eglGetError());
    } else {
        printf("   direct EGLImage import: OK\n");

        if (destroy_image)
            destroy_image(display, image);
    }

    close(dma_fd);
    eglTerminate(display);
}

static void test_intel_vulkan(uint64_t modifier)
{
    printf("\n== Intel Vulkan modifier support ==\n");

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "nvmod-to-intel",
        .apiVersion = VK_API_VERSION_1_1,
    };

    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };

    VkInstance instance = VK_NULL_HANDLE;
    VkResult result =
        vkCreateInstance(&create_info, NULL, &instance);

    if (result != VK_SUCCESS) {
        printf("   vkCreateInstance: %d\n", result);
        return;
    }

    VkPhysicalDevice intel = find_intel(instance);

    if (intel == VK_NULL_HANDLE) {
        printf("   Intel Vulkan device not found\n");
        vkDestroyInstance(instance, NULL);
        return;
    }

    if (!has_device_extension(
            intel,
            VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME)) {
        printf("   VK_EXT_image_drm_format_modifier missing\n");
        vkDestroyInstance(instance, NULL);
        return;
    }

    /*
     * First ask for every modifier Intel exposes for BGRA8.
     */
    VkDrmFormatModifierPropertiesListEXT modifier_list = {
        .sType =
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
    };

    VkFormatProperties2 format_properties = {
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &modifier_list,
    };

    vkGetPhysicalDeviceFormatProperties2(
        intel,
        VK_FORMAT_B8G8R8A8_UNORM,
        &format_properties);

    VkDrmFormatModifierPropertiesEXT *properties =
        calloc(
            modifier_list.drmFormatModifierCount,
            sizeof(*properties));

    modifier_list.pDrmFormatModifierProperties = properties;

    vkGetPhysicalDeviceFormatProperties2(
        intel,
        VK_FORMAT_B8G8R8A8_UNORM,
        &format_properties);

    int found = 0;

    for (uint32_t i = 0;
         i < modifier_list.drmFormatModifierCount;
         i++) {
        if (properties[i].drmFormatModifier == modifier) {
            printf(
                "   modifier advertised: YES "
                "planes=%u features=0x%x\n",
                properties[i].drmFormatModifierPlaneCount,
                properties[i].drmFormatModifierTilingFeatures);

            found = 1;
            break;
        }
    }

    if (!found)
        printf("   modifier advertised: NO\n");

    free(properties);

    /*
     * Ask about the exact sampled-image + DMA-BUF combination.
     */
    VkPhysicalDeviceExternalImageFormatInfo external_info = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
        .handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };

    VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifier_info = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
        .pNext = &external_info,
        .drmFormatModifier = modifier,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VkPhysicalDeviceImageFormatInfo2 image_info = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext = &modifier_info,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .type = VK_IMAGE_TYPE_2D,
        .tiling =
            VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
    };

    VkExternalImageFormatProperties external_properties = {
        .sType =
            VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    };

    VkImageFormatProperties2 image_properties = {
        .sType =
            VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = &external_properties,
    };

    result =
        vkGetPhysicalDeviceImageFormatProperties2(
            intel,
            &image_info,
            &image_properties);

    printf("   exact sampled DMA-BUF query: %d", result);

    if (result == VK_SUCCESS) {
        printf(
            " externalMemoryFeatures=0x%x",
            external_properties
                .externalMemoryProperties
                .externalMemoryFeatures);
    }

    printf("\n");

    vkDestroyInstance(instance, NULL);
}

int main(void)
{
    int nvidia_fd =
        open(NVIDIA_NODE, O_RDWR | O_CLOEXEC);
    int intel_fd =
        open(INTEL_NODE, O_RDWR | O_CLOEXEC);

    if (nvidia_fd < 0 || intel_fd < 0) {
        perror("open render node");
        return 1;
    }

    struct gbm_device *nvidia_gbm =
        gbm_create_device(nvidia_fd);
    struct gbm_device *intel_gbm =
        gbm_create_device(intel_fd);

    if (!nvidia_gbm || !intel_gbm) {
        fprintf(stderr, "gbm_create_device failed\n");
        return 1;
    }

    uint32_t fourcc = GBM_FORMAT_XRGB8888;

    struct gbm_bo *nvidia_bo =
        gbm_bo_create(
            nvidia_gbm,
            WIDTH,
            HEIGHT,
            fourcc,
            GBM_BO_USE_RENDERING);

    if (!nvidia_bo) {
        fprintf(
            stderr,
            "NVIDIA native BO allocation failed: %s\n",
            strerror(errno));
        return 1;
    }

    uint32_t planes =
        gbm_bo_get_plane_count(nvidia_bo);
    uint64_t modifier =
        gbm_bo_get_modifier(nvidia_bo);
    uint32_t stride =
        gbm_bo_get_stride_for_plane(nvidia_bo, 0);
    uint32_t offset =
        gbm_bo_get_offset(nvidia_bo, 0);

    printf(
        "NVIDIA native BO:\n"
        "   fourcc=XR24\n"
        "   modifier=0x%016" PRIx64 "\n"
        "   planes=%u stride=%u offset=%u\n",
        modifier,
        planes,
        stride,
        offset);

    if (planes != 1) {
        printf("Probe currently handles one memory plane only.\n");
        return 1;
    }

    test_intel_gbm(
        intel_gbm,
        nvidia_bo,
        fourcc,
        modifier,
        stride,
        offset);

    test_intel_egl(
        intel_gbm,
        nvidia_bo,
        fourcc,
        modifier,
        stride,
        offset);

    test_intel_vulkan(modifier);

    gbm_bo_destroy(nvidia_bo);
    gbm_device_destroy(intel_gbm);
    gbm_device_destroy(nvidia_gbm);
    close(intel_fd);
    close(nvidia_fd);

    return 0;
}
