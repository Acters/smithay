/*
 * multigpu_compositor_path_bench_batched_fixed.c
 *
 * Microbenchmark for an Intel iGPU + NVIDIA dGPU hybrid compositor path.
 *
 * Compositor-like full-frame benchmark for two Intel-renderer paths:
 *
 * Path A:
 *   Intel composites directly into an Intel LINEAR bridge image
 *     -> NVIDIA copies LINEAR into an NVIDIA-native image.
 *
 * Path B:
 *   Intel composites into an Intel X-tiled image
 *     -> Intel copies X-tiled into LINEAR
 *     -> NVIDIA copies LINEAR into an NVIDIA-native image.
 *
 * The composition workload uses many scaled blits from several source images
 * into overlapping full-frame rectangles. This exercises texture reads,
 * scaling and destination writes without requiring external shader compilers.
 * It is more compositor-like than a clear-only proxy, but it does not model
 * alpha blending or fragment shaders.
 *
 * Reports GPU timestamp and submit+wait wall time for each component, then
 * derives:
 *   - serialized end-to-end latency
 *   - ideal triple-buffered steady-state throughput bottleneck
 *   - remaining frame budget at 60, 144 and 240 Hz
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o multigpu_compositor_path_bench_batched_fixed \
 *       multigpu_compositor_path_bench_batched_fixed.c \
 *       $(pkg-config --cflags --libs gbm vulkan libdrm)
 *
 * Run (defaults 1920x1080, 300 iterations):
 *   ./multigpu_compositor_path_bench_batched_fixed
 *   ./multigpu_compositor_path_bench_batched_fixed 1920 1080 1000
 *
 * Environment:
 *   INTEL_DRM_NODE=/dev/dri/renderD128
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <gbm.h>
#include <drm_fourcc.h>
#include <vulkan/vulkan.h>

#define VK_OK(call) do { \
    VkResult _r = (call); \
    if (_r != VK_SUCCESS) { \
        fprintf(stderr, "%s failed: %d at %s:%d\n", #call, _r, __FILE__, __LINE__); \
        exit(1); \
    } \
} while (0)

typedef struct {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice dev;
    VkQueue queue;
    uint32_t qfam;
    VkCommandPool pool;
    float timestamp_period;
    PFN_vkGetMemoryFdPropertiesKHR get_fd_props;
} Ctx;

typedef struct {
    struct gbm_bo *bo;
    int fd;
    uint64_t modifier;
    uint32_t stride;
    uint32_t offset;
} GbmImage;

typedef struct {
    VkImage image;
    VkDeviceMemory memory;
} VkImageMem;

typedef struct {
    double gpu_ms_total;
    double gpu_us_frame;
    double wall_ms_total;
    double wall_us_frame;
    double gib_s;
} BenchResult;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static bool has_ext(VkPhysicalDevice p, const char *name) {
    uint32_t n = 0;
    VK_OK(vkEnumerateDeviceExtensionProperties(p, NULL, &n, NULL));
    VkExtensionProperties *exts = calloc(n, sizeof(*exts));
    if (!exts) return false;
    VK_OK(vkEnumerateDeviceExtensionProperties(p, NULL, &n, exts));
    bool found = false;
    for (uint32_t i = 0; i < n; i++) {
        if (!strcmp(exts[i].extensionName, name)) { found = true; break; }
    }
    free(exts);
    return found;
}

static uint32_t mem_type(VkPhysicalDevice p, uint32_t bits,
                         VkMemoryPropertyFlags required) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(p, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & required) == required)
            return i;
    }
    return UINT32_MAX;
}

static void init_ctx(Ctx *c, uint32_t vendor, const char *tag) {
    memset(c, 0, sizeof(*c));
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "multigpu-compositor-path-bench",
        .apiVersion = VK_API_VERSION_1_2,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VK_OK(vkCreateInstance(&ici, NULL, &c->instance));

    uint32_t n = 0;
    VK_OK(vkEnumeratePhysicalDevices(c->instance, &n, NULL));
    VkPhysicalDevice *devs = calloc(n, sizeof(*devs));
    VK_OK(vkEnumeratePhysicalDevices(c->instance, &n, devs));
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.vendorID == vendor) {
            c->phys = devs[i];
            c->timestamp_period = p.limits.timestampPeriod;
            printf("%s Vulkan: %s (timestampPeriod %.3f ns)\n",
                   tag, p.deviceName, c->timestamp_period);
            break;
        }
    }
    free(devs);
    if (!c->phys) { fprintf(stderr, "No %s Vulkan device\n", tag); exit(1); }

    const char *exts[] = {
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    };
    for (size_t i = 0; i < sizeof(exts)/sizeof(exts[0]); i++) {
        if (!has_ext(c->phys, exts[i])) {
            fprintf(stderr, "%s missing on %s\n", exts[i], tag);
            exit(1);
        }
    }

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &qn, NULL);
    VkQueueFamilyProperties *qp = calloc(qn, sizeof(*qp));
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &qn, qp);
    c->qfam = UINT32_MAX;
    for (uint32_t i = 0; i < qn; i++) {
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            qp[i].timestampValidBits > 0) {
            c->qfam = i;
            break;
        }
    }
    free(qp);
    if (c->qfam == UINT32_MAX) { fprintf(stderr, "No timestamp graphics queue\n"); exit(1); }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = c->qfam,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = 3,
        .ppEnabledExtensionNames = exts,
    };
    VK_OK(vkCreateDevice(c->phys, &dci, NULL, &c->dev));
    vkGetDeviceQueue(c->dev, c->qfam, 0, &c->queue);
    VkCommandPoolCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = c->qfam,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    VK_OK(vkCreateCommandPool(c->dev, &pci, NULL, &c->pool));
    c->get_fd_props = (PFN_vkGetMemoryFdPropertiesKHR)
        vkGetDeviceProcAddr(c->dev, "vkGetMemoryFdPropertiesKHR");
    if (!c->get_fd_props) { fprintf(stderr, "vkGetMemoryFdPropertiesKHR missing\n"); exit(1); }
}

static GbmImage make_gbm(struct gbm_device *g, uint32_t w, uint32_t h,
                         uint64_t modifier, const char *name) {
    GbmImage out = {0};
    errno = 0;
    out.bo = gbm_bo_create_with_modifiers2(g, w, h, GBM_FORMAT_XRGB8888,
                                            &modifier, 1, 0);
    if (!out.bo) {
        fprintf(stderr, "%s GBM allocation failed errno=%d (%s)\n",
                name, errno, strerror(errno));
        exit(1);
    }
    if (gbm_bo_get_plane_count(out.bo) != 1) {
        fprintf(stderr, "%s is not single-plane\n", name); exit(1);
    }
    out.modifier = gbm_bo_get_modifier(out.bo);
    out.stride = gbm_bo_get_stride_for_plane(out.bo, 0);
    out.offset = gbm_bo_get_offset(out.bo, 0);
    out.fd = gbm_bo_get_fd_for_plane(out.bo, 0);
    printf("%s BO: modifier=0x%016" PRIx64 " stride=%u offset=%u\n",
           name, out.modifier, out.stride, out.offset);
    return out;
}

static VkImageMem import_image(Ctx *c, const GbmImage *g,
                               uint32_t w, uint32_t h,
                               VkImageUsageFlags usage, const char *tag) {
    VkSubresourceLayout layout = {
        .offset = g->offset, .size = 0, .rowPitch = g->stride,
        .arrayPitch = 0, .depthPitch = 0,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT mod = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = g->modifier,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &layout,
    };
    VkExternalMemoryImageCreateInfo ext = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &mod,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImageMem out = {0};
    VkResult r = vkCreateImage(c->dev, &ci, NULL, &out.image);
    if (r != VK_SUCCESS) { fprintf(stderr, "%s vkCreateImage=%d\n", tag, r); exit(1); }
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->dev, out.image, &req);
    int fd = dup(g->fd);
    VkMemoryFdPropertiesKHR fp = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    r = c->get_fd_props(c->dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &fp);
    if (r != VK_SUCCESS) { fprintf(stderr, "%s fd props=%d\n", tag, r); exit(1); }
    uint32_t both = req.memoryTypeBits & fp.memoryTypeBits;
    uint32_t mt = mem_type(c->phys, both, 0);
    printf("%s memory bits image=0x%08x fd=0x%08x both=0x%08x type=%u\n",
           tag, req.memoryTypeBits, fp.memoryTypeBits, both, mt);
    if (!both || mt == UINT32_MAX) exit(1);
    VkImportMemoryFdInfoKHR imp = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };
    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &imp, .image = out.image,
    };
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = req.size,
        .memoryTypeIndex = mt,
    };
    VK_OK(vkAllocateMemory(c->dev, &ai, NULL, &out.memory));
    VK_OK(vkBindImageMemory(c->dev, out.image, out.memory, 0));
    return out;
}

static VkImageMem make_optimal(Ctx *c, uint32_t w, uint32_t h,
                               VkImageUsageFlags usage) {
    VkImageMem out = {0};
    VkImageCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = {w,h,1}, .mipLevels=1, .arrayLayers=1,
        .samples=VK_SAMPLE_COUNT_1_BIT,
        .tiling=VK_IMAGE_TILING_OPTIMAL,
        .usage=usage, .sharingMode=VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout=VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VK_OK(vkCreateImage(c->dev, &ci, NULL, &out.image));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(c->dev, out.image, &req);
    uint32_t mt = mem_type(c->phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt == UINT32_MAX) mt = mem_type(c->phys, req.memoryTypeBits, 0);
    VkMemoryAllocateInfo ai = {.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize=req.size,.memoryTypeIndex=mt};
    VK_OK(vkAllocateMemory(c->dev,&ai,NULL,&out.memory));
    VK_OK(vkBindImageMemory(c->dev,out.image,out.memory,0));
    return out;
}

static VkCommandBuffer alloc_cmd(Ctx *c) {
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai = {
        .sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool=c->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount=1};
    VK_OK(vkAllocateCommandBuffers(c->dev,&ai,&cmd));
    return cmd;
}

static void barrier(VkCommandBuffer cmd, VkImage image,
                    VkImageLayout old_l, VkImageLayout new_l,
                    VkAccessFlags src, VkAccessFlags dst) {
    VkImageMemoryBarrier b = {
        .sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask=src,.dstAccessMask=dst,
        .oldLayout=old_l,.newLayout=new_l,
        .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,
        .image=image,
        .subresourceRange={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1}};
    vkCmdPipelineBarrier(cmd,
        src ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&b);
}

static BenchResult finish_bench(Ctx *c, VkCommandBuffer cmd, VkQueryPool qp,
                                uint32_t iterations, uint64_t bytes_per_iter) {
    VK_OK(vkEndCommandBuffer(cmd));
    double t0 = now_ms();
    VkSubmitInfo si = {.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount=1,.pCommandBuffers=&cmd};
    VK_OK(vkQueueSubmit(c->queue,1,&si,VK_NULL_HANDLE));
    VK_OK(vkQueueWaitIdle(c->queue));
    double t1 = now_ms();
    uint64_t q[2] = {0};
    VK_OK(vkGetQueryPoolResults(c->dev, qp, 0, 2, sizeof(q), q, sizeof(uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    double gpu_ns = (double)(q[1]-q[0]) * c->timestamp_period;
    BenchResult r = {0};
    r.gpu_ms_total = gpu_ns / 1e6;
    r.gpu_us_frame = gpu_ns / 1e3 / iterations;
    r.wall_ms_total = t1-t0;
    r.wall_us_frame = (t1-t0)*1000.0/iterations;
    double seconds = gpu_ns / 1e9;
    r.gib_s = seconds > 0 ? ((double)bytes_per_iter * iterations / (1024.0*1024.0*1024.0)) / seconds : 0;
    return r;
}

static BenchResult bench_clear(Ctx *c, VkImage image, uint32_t iterations,
                               uint64_t bytes, const char *name) {
    VkQueryPool qp; VkQueryPoolCreateInfo qi={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2};
    VK_OK(vkCreateQueryPool(c->dev,&qi,NULL,&qp));
    VkCommandBuffer cmd=alloc_cmd(c);
    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_OK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdResetQueryPool(cmd,qp,0,2);
    barrier(cmd,image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,qp,0);
    VkImageSubresourceRange range={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    for(uint32_t i=0;i<iterations;i++){
        VkClearColorValue color={.uint32={i,0x11223344u,0,0xffffffffu}};
        vkCmdClearColorImage(cmd,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&color,1,&range);
    }
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,qp,1);
    BenchResult r=finish_bench(c,cmd,qp,iterations,bytes);
    printf("%-30s GPU %9.3f us/frame  wall %9.3f us/frame  effective %6.2f GiB/s\n",
           name,r.gpu_us_frame,r.wall_us_frame,r.gib_s);
    return r;
}

static BenchResult bench_copy(Ctx *c,
                              VkImage src, VkImageLayout src_old_layout,
                              VkAccessFlags src_old_access,
                              VkImage dst, VkImageLayout dst_old_layout,
                              VkAccessFlags dst_old_access,
                              uint32_t w,uint32_t h,uint32_t iterations,
                              uint64_t bytes,const char *name) {
    VkQueryPool qp; VkQueryPoolCreateInfo qi={.sType=VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType=VK_QUERY_TYPE_TIMESTAMP,.queryCount=2};
    VK_OK(vkCreateQueryPool(c->dev,&qi,NULL,&qp));
    VkCommandBuffer cmd=alloc_cmd(c);
    VkCommandBufferBeginInfo bi={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_OK(vkBeginCommandBuffer(cmd,&bi));
    vkCmdResetQueryPool(cmd,qp,0,2);
    barrier(cmd,src,src_old_layout,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            src_old_access,VK_ACCESS_TRANSFER_READ_BIT);
    barrier(cmd,dst,dst_old_layout,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            dst_old_access,VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageCopy region={.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        .dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.extent={w,h,1}};
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,qp,0);
    for(uint32_t i=0;i<iterations;i++)
        vkCmdCopyImage(cmd,src,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      dst,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);
    vkCmdWriteTimestamp(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,qp,1);
    BenchResult r=finish_bench(c,cmd,qp,iterations,bytes);
    printf("%-30s GPU %9.3f us/frame  wall %9.3f us/frame  effective %6.2f GiB/s\n",
           name,r.gpu_us_frame,r.wall_us_frame,r.gib_s);
    return r;
}

static void init_source_images(Ctx *c, VkImageMem *sources, uint32_t count,
                               uint32_t size) {
    VkCommandBuffer cmd = alloc_cmd(c);
    VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_OK(vkBeginCommandBuffer(cmd, &bi));

    static const float colors[][4] = {
        {0.95f, 0.15f, 0.12f, 1.0f},
        {0.10f, 0.70f, 0.95f, 1.0f},
        {0.20f, 0.85f, 0.30f, 1.0f},
        {0.95f, 0.75f, 0.10f, 1.0f},
        {0.70f, 0.20f, 0.90f, 1.0f},
        {0.10f, 0.85f, 0.75f, 1.0f},
        {0.95f, 0.35f, 0.60f, 1.0f},
        {0.55f, 0.55f, 0.60f, 1.0f},
    };

    VkImageSubresourceRange range = {
        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
        .baseMipLevel = 0, .levelCount = 1,
        .baseArrayLayer = 0, .layerCount = 1,
    };

    for (uint32_t i = 0; i < count; i++) {
        barrier(cmd, sources[i].image,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT);

        VkClearColorValue color = {
            .float32 = {
                colors[i % (sizeof(colors) / sizeof(colors[0]))][0],
                colors[i % (sizeof(colors) / sizeof(colors[0]))][1],
                colors[i % (sizeof(colors) / sizeof(colors[0]))][2],
                1.0f
            }
        };
        vkCmdClearColorImage(cmd, sources[i].image,
                             VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &color, 1, &range);

        barrier(cmd, sources[i].image,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT);
    }

    VK_OK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    VK_OK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
    VK_OK(vkQueueWaitIdle(c->queue));
    (void)size;
}

static BenchResult bench_compositor_blits(
        Ctx *c,
        VkImage *sources,
        uint32_t source_count,
        VkImage dst,
        uint32_t w,
        uint32_t h,
        uint32_t iterations,
        uint32_t layers,
        uint64_t bytes,
        const char *name) {
    /*
     * Do not put thousands of compositor frames into one command buffer.
     * LINEAR rendering can be several times slower than tiled rendering, and
     * a multi-second command buffer can trigger the Intel GPU hang checker.
     *
     * Submit a small number of frames per batch and accumulate GPU timestamps.
     */
    const uint32_t frames_per_batch = 4;
    uint32_t completed = 0;
    double total_gpu_ns = 0.0;
    double total_wall_us = 0.0;

    VkImageLayout current_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    while (completed < iterations) {
        uint32_t batch_frames = iterations - completed;
        if (batch_frames > frames_per_batch)
            batch_frames = frames_per_batch;

        VkQueryPool qp;
        VkQueryPoolCreateInfo qi = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
        };
        VK_OK(vkCreateQueryPool(c->dev, &qi, NULL, &qp));

        VkCommandBuffer cmd = alloc_cmd(c);
        VkCommandBufferBeginInfo bi = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_OK(vkBeginCommandBuffer(cmd, &bi));
        vkCmdResetQueryPool(cmd, qp, 0, 2);

        barrier(cmd, dst,
                current_layout,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                current_layout == VK_IMAGE_LAYOUT_UNDEFINED
                    ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_ACCESS_TRANSFER_WRITE_BIT);
        current_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        VkImageSubresourceRange range = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0, .levelCount = 1,
            .baseArrayLayer = 0, .layerCount = 1,
        };

        vkCmdWriteTimestamp(
            cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, qp, 0);

        for (uint32_t local_frame = 0;
             local_frame < batch_frames;
             local_frame++) {
            uint32_t frame = completed + local_frame;

            VkClearColorValue background = {
                .float32 = {
                    0.015f + 0.002f * (float)(frame & 7),
                    0.020f,
                    0.028f,
                    1.0f
                }
            };
            vkCmdClearColorImage(
                cmd, dst,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                &background, 1, &range);

            for (uint32_t layer = 0; layer < layers; layer++) {
                uint32_t src_index = layer % source_count;

                uint32_t min_w = w / 5u;
                uint32_t min_h = h / 5u;
                uint32_t span_w = w > min_w ? w - min_w : 1u;
                uint32_t span_h = h > min_h ? h - min_h : 1u;

                uint32_t rw =
                    min_w + ((layer * 97u + frame * 11u) % span_w);
                uint32_t rh =
                    min_h + ((layer * 53u + frame * 7u) % span_h);
                if (rw > w) rw = w;
                if (rh > h) rh = h;

                uint32_t x_max = w > rw ? w - rw : 0u;
                uint32_t y_max = h > rh ? h - rh : 0u;
                uint32_t x = x_max
                    ? ((layer * 131u + frame * 17u) % (x_max + 1u))
                    : 0u;
                uint32_t y = y_max
                    ? ((layer * 71u + frame * 13u) % (y_max + 1u))
                    : 0u;

                VkImageBlit blit = {
                    .srcSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                    .srcOffsets = {
                        {0, 0, 0},
                        {256, 256, 1},
                    },
                    .dstSubresource = {
                        .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                        .mipLevel = 0,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                    },
                    .dstOffsets = {
                        {(int32_t)x, (int32_t)y, 0},
                        {(int32_t)(x + rw), (int32_t)(y + rh), 1},
                    },
                };

                vkCmdBlitImage(
                    cmd,
                    sources[src_index],
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    dst,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &blit,
                    VK_FILTER_LINEAR);
            }
        }

        vkCmdWriteTimestamp(
            cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, qp, 1);
        VK_OK(vkEndCommandBuffer(cmd));

        VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };

        double wall_start_ms = now_ms();
        VK_OK(vkQueueSubmit(c->queue, 1, &si, VK_NULL_HANDLE));
        VK_OK(vkQueueWaitIdle(c->queue));
        double wall_end_ms = now_ms();

        uint64_t q[2] = {0, 0};
        VK_OK(vkGetQueryPoolResults(
            c->dev, qp, 0, 2,
            sizeof(q), q, sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT |
            VK_QUERY_RESULT_WAIT_BIT));

        total_gpu_ns +=
            (double)(q[1] - q[0]) *
            (double)c->timestamp_period;
        total_wall_us += (wall_end_ms - wall_start_ms) * 1000.0;

        vkFreeCommandBuffers(
            c->dev, c->pool, 1, &cmd);
        vkDestroyQueryPool(
            c->dev, qp, NULL);

        completed += batch_frames;
    }

    BenchResult result = {
        .gpu_us_frame =
            total_gpu_ns / 1000.0 / (double)iterations,
        .wall_us_frame =
            total_wall_us / (double)iterations,
        .gib_s =
            ((double)bytes / (1024.0 * 1024.0 * 1024.0)) /
            (total_gpu_ns / 1.0e9 / (double)iterations),
    };

    printf("%-34s GPU %9.3f us/frame  wall %9.3f us/frame\n",
           name, result.gpu_us_frame, result.wall_us_frame);

    return result;
}

static void print_budget(const char *path_name,
                         double serialized_us,
                         double throughput_stage_us) {
    const double rates[] = {60.0, 144.0, 240.0};

    printf("\n%s\n", path_name);
    printf("   serialized GPU latency:             %9.3f us\n",
           serialized_us);
    printf("   ideal pipelined bottleneck stage:   %9.3f us\n",
           throughput_stage_us);
    printf("   ideal pipelined ceiling:            %9.1f fps\n",
           throughput_stage_us > 0.0
               ? 1000000.0 / throughput_stage_us
               : 0.0);

    for (size_t i = 0; i < sizeof(rates) / sizeof(rates[0]); i++) {
        double budget = 1000000.0 / rates[i];
        printf("   %3.0f Hz serialized remainder:       %9.3f us"
               "  (%5.1f%% budget used)\n",
               rates[i],
               budget - serialized_us,
               serialized_us * 100.0 / budget);
    }
}

int main(int argc,char **argv){
    uint32_t w=1920,h=1080,iters=300;
    if(argc>1)w=(uint32_t)strtoul(argv[1],NULL,10);
    if(argc>2)h=(uint32_t)strtoul(argv[2],NULL,10);
    if(argc>3)iters=(uint32_t)strtoul(argv[3],NULL,10);
    if(!w||!h||!iters){fprintf(stderr,"usage: %s [width height iterations]\n",argv[0]);return 1;}
    printf("Benchmark: %ux%u, %u iterations, %.2f MiB/frame\n",
           w,h,iters,(double)w*h*4/(1024.0*1024.0));

    const char *node=getenv("INTEL_DRM_NODE"); if(!node)node="/dev/dri/renderD128";
    int fd=open(node,O_RDWR|O_CLOEXEC); if(fd<0){perror("open Intel node");return 1;}
    struct gbm_device *gbm=gbm_create_device(fd); if(!gbm){fprintf(stderr,"gbm_create_device failed\n");return 1;}

    GbmImage linear=make_gbm(gbm,w,h,DRM_FORMAT_MOD_LINEAR,"Intel LINEAR");
    GbmImage tiled=make_gbm(gbm,w,h,I915_FORMAT_MOD_X_TILED,"Intel X_TILED");

    Ctx intel,nv; init_ctx(&intel,0x8086,"Intel"); init_ctx(&nv,0x10de,"NVIDIA");
    VkImageUsageFlags shared_usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageMem i_linear=import_image(&intel,&linear,w,h,shared_usage,"Intel LINEAR import");
    VkImageMem i_tiled=import_image(&intel,&tiled,w,h,shared_usage,"Intel X_TILED import");
    VkImageMem nv_linear=import_image(&nv,&linear,w,h,VK_IMAGE_USAGE_TRANSFER_SRC_BIT,"NVIDIA LINEAR import");
    VkImageMem nv_native=make_optimal(&nv,w,h,VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    uint64_t bytes=(uint64_t)w*h*4;

    const uint32_t source_count = 8;
    const uint32_t layers = 48;
    VkImageMem source_mem[source_count];
    VkImage source_images[source_count];

    for (uint32_t i = 0; i < source_count; i++) {
        source_mem[i] = make_optimal(
            &intel, 256, 256,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        source_images[i] = source_mem[i].image;
    }
    init_source_images(&intel, source_mem, source_count, 256);

    printf("\n== Full-damage compositor-like workload ==\n");
    printf("Sources: %u synthetic surfaces; layers/frame: %u\n",
           source_count, layers);

    BenchResult tiled_composite =
        bench_compositor_blits(
            &intel,
            source_images,
            source_count,
            i_tiled.image,
            w, h, iters, layers, bytes,
            "Intel composition into X_TILED");

    BenchResult linear_composite =
        bench_compositor_blits(
            &intel,
            source_images,
            source_count,
            i_linear.image,
            w, h, iters, layers, bytes,
            "Intel composition into LINEAR");

    printf("\n== Bridge copy components ==\n");
    BenchResult tiled_to_linear =
        bench_copy(
            &intel,
            i_tiled.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            i_linear.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            w, h, iters, bytes,
            "Intel X_TILED -> LINEAR");

    BenchResult linear_to_nv =
        bench_copy(
            &nv,
            nv_linear.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT,
            nv_native.image,
            VK_IMAGE_LAYOUT_UNDEFINED,
            0,
            w, h, iters, bytes,
            "NVIDIA LINEAR -> native");

    double path_a_intel = linear_composite.gpu_us_frame;
    double path_a_nv = linear_to_nv.gpu_us_frame;
    double path_a_latency = path_a_intel + path_a_nv;
    double path_a_pipeline =
        path_a_intel > path_a_nv ? path_a_intel : path_a_nv;

    double path_b_intel =
        tiled_composite.gpu_us_frame +
        tiled_to_linear.gpu_us_frame;
    double path_b_nv = linear_to_nv.gpu_us_frame;
    double path_b_latency = path_b_intel + path_b_nv;
    double path_b_pipeline =
        path_b_intel > path_b_nv ? path_b_intel : path_b_nv;

    printf("\n== Path comparison ==\n");
    printf("Path A: Intel composes directly into LINEAR, "
           "then NVIDIA copies native\n");
    printf("   Intel stage:   %9.3f us\n", path_a_intel);
    printf("   NVIDIA stage:  %9.3f us\n", path_a_nv);

    printf("Path B: Intel composes into X_TILED, copies to LINEAR, "
           "then NVIDIA copies native\n");
    printf("   Intel stage:   %9.3f us\n", path_b_intel);
    printf("      composition %9.3f us\n",
           tiled_composite.gpu_us_frame);
    printf("      tiled copy  %9.3f us\n",
           tiled_to_linear.gpu_us_frame);
    printf("   NVIDIA stage:  %9.3f us\n", path_b_nv);

    print_budget(
        "Path A — direct LINEAR composition",
        path_a_latency,
        path_a_pipeline);

    print_budget(
        "Path B — X_TILED composition + Intel LINEAR copy",
        path_b_latency,
        path_b_pipeline);

    printf("\nWinner by serialized GPU latency: %s (difference %.3f us/frame)\n",
           path_a_latency < path_b_latency ? "Path A" : "Path B",
           fabs(path_a_latency - path_b_latency));

    printf("Winner by ideal pipelined throughput: %s "
           "(bottleneck difference %.3f us/frame)\n",
           path_a_pipeline < path_b_pipeline ? "Path A" : "Path B",
           fabs(path_a_pipeline - path_b_pipeline));

    printf("\nCAVEATS:\n");
    printf("  * This uses scaled blits of many synthetic surfaces, not shader-based\n");
    printf("    alpha blending. It exercises source reads, scaling and target writes.\n");
    printf("  * Pipelined figures are ideal max-stage estimates; they exclude external\n");
    printf("    semaphore creation/import, KMS commits, scheduling jitter and vblank.\n");
    printf("  * Run with the display manager stopped and repeat at least 1000 frames.\n");

    return 0;
}
