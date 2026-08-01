/*
 * NVIDIA GLES -> EGL native fence -> NVIDIA Vulkan copy -> Intel Vulkan verify
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o intel_gbm_nv_vk_poc \
 *       gles_fence_vk_bridge_poc.c -lgbm -lEGL -lGLESv2 -lvulkan
 *
 * Run:
 *   ./intel_gbm_nv_vk_poc
 *
 * Override nodes with INTEL_DRM_NODE / NVIDIA_DRM_NODE if needed.
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
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <vulkan/vulkan.h>

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0ULL
#endif
#define W 64u
#define H 64u

#define VKCHK(x) do { VkResult _r=(x); if (_r!=VK_SUCCESS) { \
    fprintf(stderr,"%s -> %d at %s:%d\n",#x,_r,__FILE__,__LINE__); return 1; } } while(0)

typedef struct {
    VkInstance inst; VkPhysicalDevice phys; VkDevice dev; VkQueue q;
    uint32_t qfam; VkCommandPool pool;
    PFN_vkGetMemoryFdPropertiesKHR get_fd_props;
    PFN_vkGetSemaphoreFdKHR get_sem_fd;
    PFN_vkImportSemaphoreFdKHR import_sem_fd;
} Ctx;

static uint32_t mem_type(VkPhysicalDevice p, uint32_t bits, VkMemoryPropertyFlags req) {
    VkPhysicalDeviceMemoryProperties m; vkGetPhysicalDeviceMemoryProperties(p,&m);
    for (uint32_t i=0;i<m.memoryTypeCount;i++)
        if ((bits&(1u<<i)) && (m.memoryTypes[i].propertyFlags&req)==req) return i;
    return UINT32_MAX;
}

static bool has_ext(VkPhysicalDevice p, const char *name) {
    uint32_t n=0; vkEnumerateDeviceExtensionProperties(p,NULL,&n,NULL);
    VkExtensionProperties *e=calloc(n,sizeof(*e)); if(!e) return false;
    vkEnumerateDeviceExtensionProperties(p,NULL,&n,e);
    bool ok=false; for(uint32_t i=0;i<n;i++) if(!strcmp(e[i].extensionName,name)){ok=true;break;}
    free(e); return ok;
}

static int ctx_init(Ctx *c, uint32_t vendor, const char *tag) {
    memset(c,0,sizeof(*c));
    VkApplicationInfo ai={.sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,.pApplicationName="intel-gbm-nv-vk",.apiVersion=VK_API_VERSION_1_2};
    VkInstanceCreateInfo ici={.sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,.pApplicationInfo=&ai};
    VKCHK(vkCreateInstance(&ici,NULL,&c->inst));
    uint32_t n=0; VKCHK(vkEnumeratePhysicalDevices(c->inst,&n,NULL));
    VkPhysicalDevice *d=calloc(n,sizeof(*d)); VKCHK(vkEnumeratePhysicalDevices(c->inst,&n,d));
    for(uint32_t i=0;i<n;i++){ VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(d[i],&p);
        if(p.vendorID==vendor){c->phys=d[i]; printf("   %s Vulkan: %s\n",tag,p.deviceName); break;}}
    free(d); if(!c->phys){fprintf(stderr,"No %s Vulkan device\n",tag);return 1;}
    const char *exts[]={
        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
    };
    for(size_t i=0;i<sizeof(exts)/sizeof(exts[0]);i++) if(!has_ext(c->phys,exts[i])){fprintf(stderr,"%s missing on %s\n",exts[i],tag);return 1;}
    uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(c->phys,&qn,NULL);
    VkQueueFamilyProperties *qp=calloc(qn,sizeof(*qp)); vkGetPhysicalDeviceQueueFamilyProperties(c->phys,&qn,qp);
    c->qfam=UINT32_MAX; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_GRAPHICS_BIT){c->qfam=i;break;} free(qp);
    if(c->qfam==UINT32_MAX){fprintf(stderr,"No graphics queue on %s\n",tag);return 1;}
    float prio=1.f; VkDeviceQueueCreateInfo qci={.sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,.queueFamilyIndex=c->qfam,.queueCount=1,.pQueuePriorities=&prio};
    VkDeviceCreateInfo dci={.sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,.queueCreateInfoCount=1,.pQueueCreateInfos=&qci,
        .enabledExtensionCount=(uint32_t)(sizeof(exts)/sizeof(exts[0])),.ppEnabledExtensionNames=exts};
    VKCHK(vkCreateDevice(c->phys,&dci,NULL,&c->dev)); vkGetDeviceQueue(c->dev,c->qfam,0,&c->q);
    VkCommandPoolCreateInfo pci={.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,.queueFamilyIndex=c->qfam,.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VKCHK(vkCreateCommandPool(c->dev,&pci,NULL,&c->pool));
    c->get_fd_props=(PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(c->dev,"vkGetMemoryFdPropertiesKHR");
    c->get_sem_fd=(PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(c->dev,"vkGetSemaphoreFdKHR");
    c->import_sem_fd=(PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(c->dev,"vkImportSemaphoreFdKHR");
    if(!c->get_fd_props||!c->get_sem_fd||!c->import_sem_fd){fprintf(stderr,"Missing fd functions on %s\n",tag);return 1;}
    return 0;
}

static VkCommandBuffer begin_cmd(Ctx *c) {
    VkCommandBuffer cmd=VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo a={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,.commandPool=c->pool,.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,.commandBufferCount=1};
    if(vkAllocateCommandBuffers(c->dev,&a,&cmd)!=VK_SUCCESS) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo b={.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if(vkBeginCommandBuffer(cmd,&b)!=VK_SUCCESS) return VK_NULL_HANDLE; return cmd;
}

static int imported_type(Ctx *c,int fd,uint32_t img_bits,const char *tag,uint32_t *out) {
    VkMemoryFdPropertiesKHR fp={.sType=VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
    VkResult r=c->get_fd_props(c->dev,VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,fd,&fp);
    printf("   %s fd props: %d\n",tag,r); if(r!=VK_SUCCESS)return 1;
    uint32_t both=img_bits&fp.memoryTypeBits;
    printf("   %s mem bits image=0x%08x fd=0x%08x both=0x%08x\n",tag,img_bits,fp.memoryTypeBits,both);
    if(!both)return 1; *out=mem_type(c->phys,both,0); printf("   %s memory type=%u\n",tag,*out); return *out==UINT32_MAX;
}


typedef EGLImageKHR (EGLAPIENTRYP PFN_CREATE_IMAGE_KHR)(
    EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLint *);
typedef void (EGLAPIENTRYP PFN_IMAGE_TARGET_TEXTURE)(GLenum, GLeglImageOES);

static int egl_has_extension(EGLDisplay dpy, const char *name) {
    const char *exts = eglQueryString(dpy, EGL_EXTENSIONS);
    if (!exts) return 0;
    size_t n = strlen(name);
    const char *p = exts;
    while ((p = strstr(p, name))) {
        if ((p == exts || p[-1] == ' ') && (p[n] == '\0' || p[n] == ' '))
            return 1;
        p += n;
    }
    return 0;
}

static int gles_render_red_and_export_fence(
    EGLDisplay dpy, struct gbm_bo *bo, uint32_t fourcc)
{
    PFN_CREATE_IMAGE_KHR create_image =
        (PFN_CREATE_IMAGE_KHR)eglGetProcAddress("eglCreateImageKHR");
    PFNEGLDESTROYIMAGEKHRPROC destroy_image =
        (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
    PFN_IMAGE_TARGET_TEXTURE image_target =
        (PFN_IMAGE_TARGET_TEXTURE)eglGetProcAddress(
            "glEGLImageTargetTexture2DOES");
    PFNEGLCREATESYNCKHRPROC create_sync =
        (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
    PFNEGLDESTROYSYNCKHRPROC destroy_sync =
        (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC dup_fence =
        (PFNEGLDUPNATIVEFENCEFDANDROIDPROC)eglGetProcAddress(
            "eglDupNativeFenceFDANDROID");

    if (!create_image || !destroy_image || !image_target ||
        !create_sync || !destroy_sync || !dup_fence) {
        fprintf(stderr, "Missing EGL image/native-fence entry points\n");
        return -1;
    }
    if (!egl_has_extension(dpy, "EGL_ANDROID_native_fence_sync")) {
        fprintf(stderr, "EGL_ANDROID_native_fence_sync is not advertised\n");
        return -1;
    }

    int fd = gbm_bo_get_fd_for_plane(bo, 0);
    if (fd < 0) {
        perror("gbm_bo_get_fd_for_plane(source)");
        return -1;
    }
    uint64_t mod = gbm_bo_get_modifier(bo);
    EGLint attrs[] = {
        EGL_WIDTH, (EGLint)W,
        EGL_HEIGHT, (EGLint)H,
        EGL_LINUX_DRM_FOURCC_EXT, (EGLint)fourcc,
        EGL_DMA_BUF_PLANE0_FD_EXT, fd,
        EGL_DMA_BUF_PLANE0_OFFSET_EXT,
            (EGLint)gbm_bo_get_offset(bo, 0),
        EGL_DMA_BUF_PLANE0_PITCH_EXT,
            (EGLint)gbm_bo_get_stride_for_plane(bo, 0),
        EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
            (EGLint)(mod & 0xffffffffu),
        EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
            (EGLint)(mod >> 32),
        EGL_NONE,
    };
    EGLImageKHR image = create_image(
        dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attrs);
    close(fd);
    if (image == EGL_NO_IMAGE_KHR) {
        fprintf(stderr, "eglCreateImageKHR(source) failed: 0x%x\n",
                eglGetError());
        return -1;
    }

    GLuint tex = 0, fbo = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    image_target(GL_TEXTURE_2D, image);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(
        GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "NVIDIA GLES FBO incomplete: 0x%x\n", status);
        return -1;
    }

    glViewport(0, 0, W, H);
    glClearColor(1.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    const EGLint sync_attrs[] = {
        EGL_SYNC_NATIVE_FENCE_FD_ANDROID,
        EGL_NO_NATIVE_FENCE_FD_ANDROID,
        EGL_NONE,
    };
    EGLSyncKHR sync = create_sync(
        dpy, EGL_SYNC_NATIVE_FENCE_ANDROID, sync_attrs);
    if (sync == EGL_NO_SYNC_KHR) {
        fprintf(stderr, "eglCreateSyncKHR(native fence) failed: 0x%x\n",
                eglGetError());
        return -1;
    }

    /* Flush only: deliberately do not glFinish or CPU-wait. */
    glFlush();
    int fence_fd = dup_fence(dpy, sync);
    if (fence_fd == EGL_NO_NATIVE_FENCE_FD_ANDROID) {
        fprintf(stderr, "eglDupNativeFenceFDANDROID failed: 0x%x\n",
                eglGetError());
        return -1;
    }

    destroy_sync(dpy, sync);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    destroy_image(dpy, image);

    printf("   GLES rendered red; exported native fence FD=%d (no CPU wait)\n",
           fence_fd);
    return fence_fd;
}

static int query_import(Ctx *c,VkImageUsageFlags usage,const char *tag) {
    VkPhysicalDeviceExternalImageFormatInfo e={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkPhysicalDeviceImageDrmFormatModifierInfoEXT m={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,.pNext=&e,.drmFormatModifier=DRM_FORMAT_MOD_LINEAR,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VkPhysicalDeviceImageFormatInfo2 i={.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,.pNext=&m,.format=VK_FORMAT_B8G8R8A8_UNORM,.type=VK_IMAGE_TYPE_2D,.tiling=VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,.usage=usage};
    VkExternalImageFormatProperties ep={.sType=VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
    VkImageFormatProperties2 op={.sType=VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,.pNext=&ep};
    VkResult r=vkGetPhysicalDeviceImageFormatProperties2(c->phys,&i,&op);
    printf("   %s exact query: %d features=0x%x\n",tag,r,r==VK_SUCCESS?ep.externalMemoryProperties.externalMemoryFeatures:0);
    return r!=VK_SUCCESS || !(ep.externalMemoryProperties.externalMemoryFeatures&VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);
}

int main(void) {
    const char *inode=getenv("INTEL_DRM_NODE"); if(!inode) inode="/dev/dri/renderD128";
    const char *nnode=getenv("NVIDIA_DRM_NODE"); if(!nnode) nnode="/dev/dri/renderD129";
    printf("Intel node: %s\nNVIDIA node: %s\n",inode,nnode);
    int ifd=open(inode,O_RDWR|O_CLOEXEC), nfd=open(nnode,O_RDWR|O_CLOEXEC);
    if(ifd<0||nfd<0){perror("open render node");return 1;}
    struct gbm_device *gbm=gbm_create_device(ifd);
    struct gbm_device *nv_gbm=gbm_create_device(nfd);
    if(!gbm||!nv_gbm){fprintf(stderr,"gbm_create_device failed\n");return 1;}

    EGLDisplay nv_dpy=eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR,nv_gbm,NULL);
    if(nv_dpy==EGL_NO_DISPLAY||!eglInitialize(nv_dpy,NULL,NULL)){
        fprintf(stderr,"NVIDIA eglInitialize failed: 0x%x\n",eglGetError());return 1;}
    if(!eglBindAPI(EGL_OPENGL_ES_API)){fprintf(stderr,"eglBindAPI failed\n");return 1;}
    EGLConfig cfg=0; EGLint ncfg=0;
    const EGLint cfg_attrs[]={EGL_RENDERABLE_TYPE,EGL_OPENGL_ES3_BIT_KHR,EGL_NONE};
    if(!eglChooseConfig(nv_dpy,cfg_attrs,&cfg,1,&ncfg)||ncfg<1){
        fprintf(stderr,"eglChooseConfig failed: 0x%x\n",eglGetError());return 1;}
    const EGLint ctx_attrs[]={EGL_CONTEXT_CLIENT_VERSION,3,EGL_NONE};
    EGLContext nv_ctx=eglCreateContext(nv_dpy,cfg,EGL_NO_CONTEXT,ctx_attrs);
    if(nv_ctx==EGL_NO_CONTEXT||!eglMakeCurrent(nv_dpy,EGL_NO_SURFACE,EGL_NO_SURFACE,nv_ctx)){
        fprintf(stderr,"NVIDIA EGL context/current failed: 0x%x\n",eglGetError());return 1;}
    uint64_t linear=DRM_FORMAT_MOD_LINEAR;
    struct gbm_bo *bo=gbm_bo_create_with_modifiers2(gbm,W,H,GBM_FORMAT_XRGB8888,&linear,1,0);
    if(!bo){fprintf(stderr,"Intel GBM LINEAR alloc failed errno=%d (%s)\n",errno,strerror(errno));return 1;}
    uint64_t mod=gbm_bo_get_modifier(bo); uint32_t stride=gbm_bo_get_stride_for_plane(bo,0),off=gbm_bo_get_offset(bo,0);
    int dmafd=gbm_bo_get_fd_for_plane(bo,0);
    printf("Intel BO: mod=0x%016"PRIx64" stride=%u offset=%u fd=%d\n",mod,stride,off,dmafd);
    if(mod!=DRM_FORMAT_MOD_LINEAR||gbm_bo_get_plane_count(bo)!=1)return 1;

    Ctx nv,in; if(ctx_init(&nv,0x10de,"NVIDIA")||ctx_init(&in,0x8086,"Intel"))return 1;
    if(query_import(&nv,VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,"NVIDIA import") ||
       query_import(&in,VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,"Intel import")) return 1;

    /* NVIDIA GLES native/block-linear source + native fence. */
    struct gbm_bo *src_bo = gbm_bo_create(
        nv_gbm, W, H, GBM_FORMAT_XRGB8888, GBM_BO_USE_RENDERING);
    if (!src_bo) {
        fprintf(stderr, "NVIDIA GBM renderable source allocation failed\n");
        return 1;
    }
    uint64_t src_mod = gbm_bo_get_modifier(src_bo);
    uint32_t src_stride = gbm_bo_get_stride_for_plane(src_bo, 0);
    uint32_t src_off = gbm_bo_get_offset(src_bo, 0);
    printf("NVIDIA GLES source: mod=0x%016" PRIx64
           " stride=%u offset=%u\n", src_mod, src_stride, src_off);

    int render_fence_fd = gles_render_red_and_export_fence(
        nv_dpy, src_bo, GBM_FORMAT_XRGB8888);
    if (render_fence_fd < 0)
        return 1;

    VkSubresourceLayout src_pl = {
        .offset = src_off, .size = 0, .rowPitch = src_stride,
    };
    VkImageDrmFormatModifierExplicitCreateInfoEXT src_mc = {
        .sType =
            VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
        .drmFormatModifier = src_mod,
        .drmFormatModifierPlaneCount = 1,
        .pPlaneLayouts = &src_pl,
    };
    VkExternalMemoryImageCreateInfo src_ex = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &src_mc,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    VkImageCreateInfo src_ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &src_ex,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_B8G8R8A8_UNORM,
        .extent = {W,H,1}, .mipLevels = 1, .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkImage src_img;
    VKCHK(vkCreateImage(nv.dev, &src_ci, NULL, &src_img));
    VkMemoryRequirements src_req;
    vkGetImageMemoryRequirements(nv.dev, src_img, &src_req);
    int src_fd = gbm_bo_get_fd_for_plane(src_bo, 0);
    uint32_t src_type;
    if (imported_type(&nv, src_fd, src_req.memoryTypeBits,
                      "NVIDIA GLES source import", &src_type))
        return 1;
    VkImportMemoryFdInfoKHR src_im = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = src_fd,
    };
    VkMemoryDedicatedAllocateInfo src_di = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &src_im,
        .image = src_img,
    };
    VkMemoryAllocateInfo src_ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &src_di,
        .allocationSize = src_req.size,
        .memoryTypeIndex = src_type,
    };
    VkDeviceMemory src_mem;
    VKCHK(vkAllocateMemory(nv.dev, &src_ai, NULL, &src_mem));
    VKCHK(vkBindImageMemory(nv.dev, src_img, src_mem, 0));

    VkSemaphore render_sem;
    VkSemaphoreCreateInfo render_sci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    VKCHK(vkCreateSemaphore(nv.dev, &render_sci, NULL, &render_sem));
    VkImportSemaphoreFdInfoKHR render_import = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
        .semaphore = render_sem,
        .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
        .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
        .fd = render_fence_fd,
    };
    VKCHK(nv.import_sem_fd(nv.dev, &render_import));
    render_fence_fd = -1;
    printf("   NVIDIA Vulkan imported the EGL render fence\n");

    /* Shared Intel BO imported on NVIDIA */
    VkSubresourceLayout pl={.offset=off,.size=0,.rowPitch=stride};
    VkImageDrmFormatModifierExplicitCreateInfoEXT mc={.sType=VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,.drmFormatModifier=DRM_FORMAT_MOD_LINEAR,.drmFormatModifierPlaneCount=1,.pPlaneLayouts=&pl};
    VkExternalMemoryImageCreateInfo ex={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,.pNext=&mc,.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo sci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.pNext=&ex,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_B8G8R8A8_UNORM,
        .extent={W,H,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
        .usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage nshared; VKCHK(vkCreateImage(nv.dev,&sci,NULL,&nshared)); VkMemoryRequirements sr; vkGetImageMemoryRequirements(nv.dev,nshared,&sr);
    int nimpfd=dup(dmafd); uint32_t nit; if(imported_type(&nv,nimpfd,sr.memoryTypeBits,"NVIDIA import",&nit))return 1;
    VkImportMemoryFdInfoKHR im={.sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,.fd=nimpfd};
    VkMemoryDedicatedAllocateInfo di={.sType=VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,.pNext=&im,.image=nshared};
    VkMemoryAllocateInfo sai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&di,.allocationSize=sr.size,.memoryTypeIndex=nit};
    VkDeviceMemory smem; VKCHK(vkAllocateMemory(nv.dev,&sai,NULL,&smem)); VKCHK(vkBindImageMemory(nv.dev,nshared,smem,0));
    printf("   NVIDIA bound Intel-owned BO\n");

    VkExportSemaphoreCreateInfo se={.sType=VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,.handleTypes=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VkSemaphoreCreateInfo sc={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,.pNext=&se}; VkSemaphore nsem; VKCHK(vkCreateSemaphore(nv.dev,&sc,NULL,&nsem));
    VkCommandBuffer nc=begin_cmd(&nv); if(!nc)return 1;
    VkImageSubresourceRange range={VK_IMAGE_ASPECT_COLOR_BIT,0,1,0,1};
    VkImageMemoryBarrier b[2]={
        {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
         .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
         .srcQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,
         .dstQueueFamilyIndex=nv.qfam,
         .image=src_img,.subresourceRange=range},
        {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
         .dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,
         .oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,
         .newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,
         .dstQueueFamilyIndex=nv.qfam,
         .image=nshared,.subresourceRange=range}};
    vkCmdPipelineBarrier(nc,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,2,b);
    VkImageCopy cp={.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},
        .dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.extent={W,H,1}};
    vkCmdCopyImage(nc,src_img,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        nshared,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&cp);
    VkImageMemoryBarrier rel={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex=nv.qfam,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,.image=nshared,.subresourceRange=range};
    vkCmdPipelineBarrier(nc,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,NULL,0,NULL,1,&rel);
    VKCHK(vkEndCommandBuffer(nc));
    VkPipelineStageFlags render_wait_stage=VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo nsub={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount=1,.pWaitSemaphores=&render_sem,
        .pWaitDstStageMask=&render_wait_stage,
        .commandBufferCount=1,.pCommandBuffers=&nc,
        .signalSemaphoreCount=1,.pSignalSemaphores=&nsem};
    VKCHK(vkQueueSubmit(nv.q,1,&nsub,VK_NULL_HANDLE));
    printf("   Vulkan copy submitted with GPU wait on EGL native fence\n");
    int syncfd=-1; VkSemaphoreGetFdInfoKHR gfi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,.semaphore=nsem,.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VKCHK(nv.get_sem_fd(nv.dev,&gfi,&syncfd)); printf("   NVIDIA copied GLES frame into Intel BO; completion SYNC_FD=%d\n",syncfd);

    /* Intel imports its own BO into Vulkan */
    VkSubresourceLayout ipl={.offset=off,.size=0,.rowPitch=stride};
    VkImageDrmFormatModifierExplicitCreateInfoEXT imc={.sType=VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,.drmFormatModifier=DRM_FORMAT_MOD_LINEAR,.drmFormatModifierPlaneCount=1,.pPlaneLayouts=&ipl};
    VkExternalMemoryImageCreateInfo iex={.sType=VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,.pNext=&imc,.handleTypes=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
    VkImageCreateInfo ici={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.pNext=&iex,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_B8G8R8A8_UNORM,.extent={W,H,1},.mipLevels=1,.arrayLayers=1,
        .samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,.usage=VK_IMAGE_USAGE_TRANSFER_SRC_BIT|VK_IMAGE_USAGE_SAMPLED_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage iimg; VKCHK(vkCreateImage(in.dev,&ici,NULL,&iimg)); VkMemoryRequirements ir; vkGetImageMemoryRequirements(in.dev,iimg,&ir);
    int iimpfd=dup(dmafd); uint32_t iit; if(imported_type(&in,iimpfd,ir.memoryTypeBits,"Intel import",&iit))return 1;
    VkImportMemoryFdInfoKHR iim={.sType=VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,.handleType=VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,.fd=iimpfd};
    VkMemoryDedicatedAllocateInfo idi={.sType=VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,.pNext=&iim,.image=iimg};
    VkMemoryAllocateInfo iai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.pNext=&idi,.allocationSize=ir.size,.memoryTypeIndex=iit};
    VkDeviceMemory imem; VKCHK(vkAllocateMemory(in.dev,&iai,NULL,&imem)); VKCHK(vkBindImageMemory(in.dev,iimg,imem,0));

    VkSemaphore isem; VkSemaphoreCreateInfo isci={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}; VKCHK(vkCreateSemaphore(in.dev,&isci,NULL,&isem));
    VkImportSemaphoreFdInfoKHR isi={.sType=VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,.semaphore=isem,.flags=VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
        .handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,.fd=syncfd}; VKCHK(in.import_sem_fd(in.dev,&isi)); syncfd=-1;

    VkBuffer rb; VkBufferCreateInfo rbi={.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,.size=W*H*4,.usage=VK_BUFFER_USAGE_TRANSFER_DST_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(in.dev,&rbi,NULL,&rb)); VkMemoryRequirements rr; vkGetBufferMemoryRequirements(in.dev,rb,&rr);
    uint32_t rt=mem_type(in.phys,rr.memoryTypeBits,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); if(rt==UINT32_MAX)return 1;
    VkMemoryAllocateInfo rai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=rr.size,.memoryTypeIndex=rt}; VkDeviceMemory rmem;
    VKCHK(vkAllocateMemory(in.dev,&rai,NULL,&rmem)); VKCHK(vkBindBufferMemory(in.dev,rb,rmem,0));
    VkCommandBuffer ic=begin_cmd(&in); if(!ic)return 1;
    VkImageMemoryBarrier acq={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_GENERAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,.dstQueueFamilyIndex=in.qfam,.image=iimg,.subresourceRange=range};
    vkCmdPipelineBarrier(ic,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&acq);
    VkImageMemoryBarrier tos={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout=VK_IMAGE_LAYOUT_GENERAL,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=iimg,.subresourceRange=range};
    vkCmdPipelineBarrier(ic,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&tos);
    VkBufferImageCopy reg={.bufferRowLength=W,.bufferImageHeight=H,.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.imageExtent={W,H,1}};
    vkCmdCopyImageToBuffer(ic,iimg,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,rb,1,&reg); VKCHK(vkEndCommandBuffer(ic));
    VkPipelineStageFlags ws=VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; VkSubmitInfo isub={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.waitSemaphoreCount=1,.pWaitSemaphores=&isem,.pWaitDstStageMask=&ws,.commandBufferCount=1,.pCommandBuffers=&ic};
    VKCHK(vkQueueSubmit(in.q,1,&isub,VK_NULL_HANDLE)); VKCHK(vkQueueWaitIdle(in.q));
    void *map=NULL; VKCHK(vkMapMemory(in.dev,rmem,0,VK_WHOLE_SIZE,0,&map)); uint8_t *px=map; size_t o=((H/2)*W+(W/2))*4;
    printf("   center BGRA = %u %u %u %u\n",px[o],px[o+1],px[o+2],px[o+3]);
    bool ok=px[o+2]>200&&px[o+1]<40&&px[o]<40; printf("%s\n",ok?"PASS: GLES -> EGL native fence -> NVIDIA Vulkan copy -> Intel Vulkan, with no CPU render wait.":"FAIL: wrong pixel data.");
    return ok?0:2;
}
