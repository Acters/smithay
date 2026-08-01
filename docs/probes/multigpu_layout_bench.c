/*
 * multigpu_layout_bench.c
 *
 * Microbenchmark for an Intel iGPU + NVIDIA dGPU hybrid compositor path.
 *
 * Measures at configurable resolution:
 *   A. Intel transfer-clear into Intel X-tiled BO (tiled write proxy)
 *   B. Intel transfer-clear into Intel LINEAR BO (linear write proxy)
 *   C. Intel X-tiled -> Intel LINEAR vkCmdCopyImage
 *   D. NVIDIA import Intel LINEAR -> NVIDIA optimal vkCmdCopyImage
 *
 * Reports GPU timestamp time and submit+wait CPU wall time. It also prints:
 *   internal-only direct LINEAR penalty versus X-tiled
 *   external path estimate:
 *      direct LINEAR -> NVIDIA       = D
 *      X-tiled -> LINEAR -> NVIDIA   = C + D
 *
 * This is a layout/copy microbenchmark, not a full compositor shader benchmark.
 * vkCmdClearColorImage is used as a repeatable GPU-write proxy.
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o multigpu_layout_bench \
 *       multigpu_layout_bench.c $(pkg-config --cflags --libs gbm vulkan libdrm)
 *
 * Run (defaults 1920x1080, 300 iterations):
 *   ./multigpu_layout_bench
 *   ./multigpu_layout_bench 1920 1080 1000
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
        .pApplicationName = "multigpu-layout-bench",
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

static BenchResult bench_copy(Ctx *c, VkImage src, VkImage dst,
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
    barrier(cmd,src,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,0,VK_ACCESS_TRANSFER_READ_BIT);
    barrier(cmd,dst,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,0,VK_ACCESS_TRANSFER_WRITE_BIT);
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
    printf("\n== Component measurements ==\n");
    BenchResult tiled_write=bench_clear(&intel,i_tiled.image,iters,bytes,"Intel write proxy: X_TILED");
    BenchResult linear_write=bench_clear(&intel,i_linear.image,iters,bytes,"Intel write proxy: LINEAR");
    BenchResult tiled_to_linear=bench_copy(&intel,i_tiled.image,i_linear.image,w,h,iters,bytes,"Intel X_TILED -> LINEAR");
    BenchResult linear_to_nv=bench_copy(&nv,nv_linear.image,nv_native.image,w,h,iters,bytes,"NVIDIA LINEAR -> optimal");

    printf("\n== Derived path estimates (GPU timestamps) ==\n");
    double linear_penalty = tiled_write.gpu_us_frame > 0 ?
        linear_write.gpu_us_frame / tiled_write.gpu_us_frame : NAN;
    printf("Internal-only direct LINEAR write proxy: %.3fx X_TILED time\n",linear_penalty);
    printf("Internal-only tiled render + scanout:       %.3f us/frame write proxy\n",tiled_write.gpu_us_frame);
    printf("Internal-only direct LINEAR render proxy:  %.3f us/frame write proxy\n",linear_write.gpu_us_frame);
    printf("External, direct LINEAR -> NVIDIA:         %.3f us/frame copy\n",linear_to_nv.gpu_us_frame);
    printf("External, tiled -> LINEAR -> NVIDIA:       %.3f us/frame copies\n",
           tiled_to_linear.gpu_us_frame+linear_to_nv.gpu_us_frame);
    printf("Extra cost of keeping Intel tiled:         %.3f us/frame\n",tiled_to_linear.gpu_us_frame);
    printf("240 Hz frame budget:                       4166.667 us\n");
    printf("144 Hz frame budget:                       6944.444 us\n");

    printf("\nNOTE: clear timing is a memory-write/layout proxy, not a full compositor render.\n");
    printf("For final policy decisions, repeat with a real compositor shader workload and a pipelined buffer ring.\n");
    return 0;
}
