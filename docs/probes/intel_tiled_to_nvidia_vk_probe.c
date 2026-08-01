/*
 * intel_tiled_to_nvidia_vk_probe.c
 *
 * Probe Intel GBM XR24 modifier allocations and NVIDIA Vulkan DMA-BUF import:
 *
 *   - DRM_FORMAT_MOD_LINEAR
 *   - I915_FORMAT_MOD_X_TILED
 *   - I915_FORMAT_MOD_Y_TILED
 *   - I915_FORMAT_MOD_Yf_TILED, when present in installed headers
 *   - I915_FORMAT_MOD_4_TILED, when present in installed headers
 *
 * For every Intel allocation that succeeds, test NVIDIA Vulkan for:
 *
 *   1. modifier advertised for VK_FORMAT_B8G8R8A8_UNORM
 *   2. exact DMA-BUF import query as TRANSFER_SRC
 *   3. exact DMA-BUF import query as TRANSFER_DST
 *   4. actual vkCreateImage + vkAllocateMemory + vkBindImageMemory
 *
 * This tests importability and bindability. It does not yet verify pixels.
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o intel_tiled_to_nvidia_vk_probe \
 *       intel_tiled_to_nvidia_vk_probe.c \
 *       $(pkg-config --cflags --libs gbm vulkan)
 *
 * Run:
 *   ./intel_tiled_to_nvidia_vk_probe
 *
 * Defaults:
 *   Intel GBM node: /dev/dri/renderD128
 *
 * Override:
 *   INTEL_DRM_NODE=/dev/dri/renderD128 ./intel_tiled_to_nvidia_vk_probe
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
#include <drm_fourcc.h>
#include <vulkan/vulkan.h>

#define WIDTH  64u
#define HEIGHT 64u

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    PFN_vkGetMemoryFdPropertiesKHR get_fd_props;
} NvVk;

typedef struct {
    const char *name;
    uint64_t modifier;
} ModifierCase;

static bool has_extension(VkPhysicalDevice physical, const char *name)
{
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(
            physical, NULL, &count, NULL) != VK_SUCCESS) {
        return false;
    }

    VkExtensionProperties *exts = calloc(count, sizeof(*exts));
    if (!exts)
        return false;

    bool found = false;
    if (vkEnumerateDeviceExtensionProperties(
            physical, NULL, &count, exts) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; i++) {
            if (strcmp(exts[i].extensionName, name) == 0) {
                found = true;
                break;
            }
        }
    }

    free(exts);
    return found;
}

static uint32_t choose_memory_type(
    VkPhysicalDevice physical,
    uint32_t allowed_bits)
{
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);

    for (uint32_t i = 0; i < memory.memoryTypeCount; i++) {
        if (allowed_bits & (1u << i))
            return i;
    }

    return UINT32_MAX;
}

static int init_nvidia_vulkan(NvVk *vk)
{
    memset(vk, 0, sizeof(*vk));

    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "intel-tiled-to-nvidia-probe",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };

    VkResult result =
        vkCreateInstance(&instance_info, NULL, &vk->instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance: %d\n", result);
        return 1;
    }

    uint32_t count = 0;
    result = vkEnumeratePhysicalDevices(vk->instance, &count, NULL);
    if (result != VK_SUCCESS || count == 0) {
        fprintf(stderr, "vkEnumeratePhysicalDevices: %d\n", result);
        return 1;
    }

    VkPhysicalDevice *physical = calloc(count, sizeof(*physical));
    if (!physical)
        return 1;

    result = vkEnumeratePhysicalDevices(
        vk->instance, &count, physical);
    if (result != VK_SUCCESS) {
        free(physical);
        return 1;
    }

    for (uint32_t i = 0; i < count; i++) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physical[i], &properties);

        if (properties.vendorID == 0x10de) {
            vk->physical = physical[i];
            printf("NVIDIA Vulkan device: %s\n",
                   properties.deviceName);
            break;
        }
    }

    free(physical);

    if (vk->physical == VK_NULL_HANDLE) {
        fprintf(stderr, "NVIDIA Vulkan device not found\n");
        return 1;
    }

    const char *extensions[] = {
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };

    for (size_t i = 0;
         i < sizeof(extensions) / sizeof(extensions[0]);
         i++) {
        if (!has_extension(vk->physical, extensions[i])) {
            fprintf(stderr, "NVIDIA is missing %s\n",
                    extensions[i]);
            return 1;
        }
    }

    uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        vk->physical, &queue_count, NULL);

    VkQueueFamilyProperties *queues =
        calloc(queue_count, sizeof(*queues));
    if (!queues)
        return 1;

    vkGetPhysicalDeviceQueueFamilyProperties(
        vk->physical, &queue_count, queues);

    uint32_t queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < queue_count; i++) {
        if (queues[i].queueFlags &
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)) {
            queue_family = i;
            break;
        }
    }
    free(queues);

    if (queue_family == UINT32_MAX) {
        fprintf(stderr, "NVIDIA queue family not found\n");
        return 1;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount =
            (uint32_t)(sizeof(extensions) / sizeof(extensions[0])),
        .ppEnabledExtensionNames = extensions,
    };

    result = vkCreateDevice(
        vk->physical, &device_info, NULL, &vk->device);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateDevice: %d\n", result);
        return 1;
    }

    vk->get_fd_props =
        (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(
            vk->device, "vkGetMemoryFdPropertiesKHR");

    if (!vk->get_fd_props) {
        fprintf(stderr,
                "vkGetMemoryFdPropertiesKHR unavailable\n");
        return 1;
    }

    return 0;
}

static bool modifier_is_advertised(
    NvVk *vk,
    uint64_t wanted,
    VkFormatFeatureFlags *out_features,
    uint32_t *out_planes)
{
    VkDrmFormatModifierPropertiesListEXT list = {
        .sType =
            VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
    };
    VkFormatProperties2 format = {
        .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
        .pNext = &list,
    };

    vkGetPhysicalDeviceFormatProperties2(
        vk->physical,
        VK_FORMAT_B8G8R8A8_UNORM,
        &format);

    VkDrmFormatModifierPropertiesEXT *properties =
        calloc(list.drmFormatModifierCount, sizeof(*properties));
    if (!properties)
        return false;

    list.pDrmFormatModifierProperties = properties;

    vkGetPhysicalDeviceFormatProperties2(
        vk->physical,
        VK_FORMAT_B8G8R8A8_UNORM,
        &format);

    bool found = false;
    for (uint32_t i = 0;
         i < list.drmFormatModifierCount;
         i++) {
        if (properties[i].drmFormatModifier == wanted) {
            *out_features =
                properties[i].drmFormatModifierTilingFeatures;
            *out_planes =
                properties[i].drmFormatModifierPlaneCount;
            found = true;
            break;
        }
    }

    free(properties);
    return found;
}

static VkResult exact_query(
    NvVk *vk,
    uint64_t modifier,
    VkImageUsageFlags usage,
    VkExternalMemoryFeatureFlags *out_external)
{
    VkPhysicalDeviceExternalImageFormatInfo external = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
        .handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT drm = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
        .pNext = &external,
        .drmFormatModifier = modifier,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkPhysicalDeviceImageFormatInfo2 input = {
        .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
        .pNext = &drm,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .type = VK_IMAGE_TYPE_2D,
        .tiling =
            VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = usage,
    };
    VkExternalImageFormatProperties external_properties = {
        .sType =
            VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
    };
    VkImageFormatProperties2 output = {
        .sType =
            VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
        .pNext = &external_properties,
    };

    VkResult result =
        vkGetPhysicalDeviceImageFormatProperties2(
            vk->physical, &input, &output);

    if (result == VK_SUCCESS) {
        *out_external =
            external_properties.externalMemoryProperties
                .externalMemoryFeatures;
    } else {
        *out_external = 0;
    }

    return result;
}

static VkResult actual_import(
    NvVk *vk,
    int dma_fd,
    uint64_t modifier,
    uint32_t stride,
    uint32_t offset,
    uint32_t plane_count,
    VkImageUsageFlags usage)
{
    if (plane_count != 1)
        return VK_ERROR_FORMAT_NOT_SUPPORTED;

    VkSubresourceLayout plane = {
        .offset = offset,
        .size = 0,
        .rowPitch = stride,
        .arrayPitch = 0,
        .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT drm = {
        .sType =
            VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = modifier,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &plane,
    };
    VkExternalMemoryImageCreateInfo external = {
        .sType =
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &drm,
        .handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &external,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = { WIDTH, HEIGHT, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling =
            VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkImage image = VK_NULL_HANDLE;
    VkResult result =
        vkCreateImage(vk->device, &image_info, NULL, &image);
    if (result != VK_SUCCESS)
        return result;

    VkMemoryRequirements requirements;
    vkGetImageMemoryRequirements(
        vk->device, image, &requirements);

    int import_fd = dup(dma_fd);
    if (import_fd < 0) {
        vkDestroyImage(vk->device, image, NULL);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }

    VkMemoryFdPropertiesKHR fd_properties = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
    };
    result = vk->get_fd_props(
        vk->device,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        import_fd,
        &fd_properties);

    if (result != VK_SUCCESS) {
        close(import_fd);
        vkDestroyImage(vk->device, image, NULL);
        return result;
    }

    uint32_t compatible =
        requirements.memoryTypeBits &
        fd_properties.memoryTypeBits;

    printf("      memory bits image=0x%08x fd=0x%08x both=0x%08x\n",
           requirements.memoryTypeBits,
           fd_properties.memoryTypeBits,
           compatible);

    uint32_t memory_type =
        choose_memory_type(vk->physical, compatible);

    if (memory_type == UINT32_MAX) {
        close(import_fd);
        vkDestroyImage(vk->device, image, NULL);
        return VK_ERROR_INVALID_EXTERNAL_HANDLE;
    }

    VkImportMemoryFdInfoKHR import = {
        .sType =
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = import_fd,
    };
    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &import,
        .image = image,
    };
    VkMemoryAllocateInfo allocation = {
        .sType =
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = requirements.size,
        .memoryTypeIndex = memory_type,
    };

    VkDeviceMemory memory = VK_NULL_HANDLE;
    result = vkAllocateMemory(
        vk->device, &allocation, NULL, &memory);

    if (result == VK_SUCCESS) {
        result = vkBindImageMemory(
            vk->device, image, memory, 0);
    }

    if (memory != VK_NULL_HANDLE)
        vkFreeMemory(vk->device, memory, NULL);
    vkDestroyImage(vk->device, image, NULL);

    return result;
}

static void probe_one(
    struct gbm_device *intel_gbm,
    NvVk *nv,
    const ModifierCase *test)
{
    printf("\n== %s: 0x%016" PRIx64 " ==\n",
           test->name, test->modifier);

    errno = 0;
    struct gbm_bo *bo =
        gbm_bo_create_with_modifiers2(
            intel_gbm,
            WIDTH, HEIGHT,
            GBM_FORMAT_XRGB8888,
            &test->modifier,
            1,
            0);

    if (!bo) {
        printf("   Intel GBM allocation: FAIL errno=%d (%s)\n",
               errno, strerror(errno));
        return;
    }

    uint64_t actual_modifier =
        gbm_bo_get_modifier(bo);
    uint32_t planes =
        gbm_bo_get_plane_count(bo);
    uint32_t stride =
        gbm_bo_get_stride_for_plane(bo, 0);
    uint32_t offset =
        gbm_bo_get_offset(bo, 0);
    int fd =
        gbm_bo_get_fd_for_plane(bo, 0);

    printf("   Intel GBM allocation: OK\n");
    printf("   actual modifier=0x%016" PRIx64
           " planes=%u stride=%u offset=%u fd=%d\n",
           actual_modifier, planes, stride, offset, fd);

    VkFormatFeatureFlags features = 0;
    uint32_t advertised_planes = 0;
    bool advertised =
        modifier_is_advertised(
            nv, actual_modifier,
            &features, &advertised_planes);

    printf("   NVIDIA modifier advertised: %s",
           advertised ? "YES" : "NO");
    if (advertised) {
        printf(" planes=%u features=0x%08x",
               advertised_planes, features);
    }
    printf("\n");

    const struct {
        const char *name;
        VkImageUsageFlags usage;
    } usages[] = {
        {
            "TRANSFER_SRC",
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        },
        {
            "TRANSFER_DST",
            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        },
        {
            "TRANSFER_SRC|TRANSFER_DST",
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        },
        {
            "SAMPLED|TRANSFER_SRC",
            VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        },
    };

    for (size_t i = 0;
         i < sizeof(usages) / sizeof(usages[0]);
         i++) {
        VkExternalMemoryFeatureFlags external = 0;
        VkResult query =
            exact_query(
                nv, actual_modifier,
                usages[i].usage,
                &external);

        printf("   %-25s exact query=%d external=0x%x\n",
               usages[i].name, query, external);

        if (query == VK_SUCCESS &&
            (external &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)) {
            VkResult imported =
                actual_import(
                    nv, fd,
                    actual_modifier,
                    stride, offset,
                    planes,
                    usages[i].usage);

            printf("   %-25s actual create/alloc/bind=%d\n",
                   usages[i].name, imported);
        }
    }

    close(fd);
    gbm_bo_destroy(bo);
}

int main(void)
{
    const char *intel_node =
        getenv("INTEL_DRM_NODE");
    if (!intel_node)
        intel_node = "/dev/dri/renderD128";

    printf("Intel GBM node: %s\n", intel_node);

    int intel_fd =
        open(intel_node, O_RDWR | O_CLOEXEC);
    if (intel_fd < 0) {
        fprintf(stderr, "open(%s): %s\n",
                intel_node, strerror(errno));
        return 1;
    }

    struct gbm_device *intel_gbm =
        gbm_create_device(intel_fd);
    if (!intel_gbm) {
        fprintf(stderr,
                "gbm_create_device(Intel) failed\n");
        return 1;
    }

    NvVk nv;
    if (init_nvidia_vulkan(&nv))
        return 1;

    ModifierCase tests[8];
    size_t count = 0;

    tests[count++] = (ModifierCase) {
        "LINEAR",
        DRM_FORMAT_MOD_LINEAR,
    };

#ifdef I915_FORMAT_MOD_X_TILED
    tests[count++] = (ModifierCase) {
        "I915_X_TILED",
        I915_FORMAT_MOD_X_TILED,
    };
#endif

#ifdef I915_FORMAT_MOD_Y_TILED
    tests[count++] = (ModifierCase) {
        "I915_Y_TILED",
        I915_FORMAT_MOD_Y_TILED,
    };
#endif

#ifdef I915_FORMAT_MOD_Yf_TILED
    tests[count++] = (ModifierCase) {
        "I915_Yf_TILED",
        I915_FORMAT_MOD_Yf_TILED,
    };
#endif

#ifdef I915_FORMAT_MOD_4_TILED
    tests[count++] = (ModifierCase) {
        "I915_4_TILED",
        I915_FORMAT_MOD_4_TILED,
    };
#endif

#ifdef I915_FORMAT_MOD_4_TILED_DG2_RC_CCS
    tests[count++] = (ModifierCase) {
        "I915_4_TILED_DG2_RC_CCS",
        I915_FORMAT_MOD_4_TILED_DG2_RC_CCS,
    };
#endif

    printf("Modifier cases compiled in: %zu\n", count);

    for (size_t i = 0; i < count; i++)
        probe_one(intel_gbm, &nv, &tests[i]);

    return 0;
}
