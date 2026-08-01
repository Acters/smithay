/*
 * Intel GBM LINEAR -> NVIDIA Vulkan write -> Intel Vulkan verify
 *
 * Build:
 *   gcc -O1 -Wall -Wextra -o intel_gbm_nv_vk_poc \
 *       intel_gbm_nv_vk_poc.c -lgbm -lvulkan
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
    if(ifd<0||nfd<0){perror("open render node");return 1;} close(nfd);
    struct gbm_device *gbm=gbm_create_device(ifd); if(!gbm){fprintf(stderr,"Intel gbm_create_device failed\n");return 1;}
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

    /* NVIDIA native image */
    VkImage native; VkImageCreateInfo nci={.sType=VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,.imageType=VK_IMAGE_TYPE_2D,.format=VK_FORMAT_B8G8R8A8_UNORM,
        .extent={W,H,1},.mipLevels=1,.arrayLayers=1,.samples=VK_SAMPLE_COUNT_1_BIT,.tiling=VK_IMAGE_TILING_OPTIMAL,
        .usage=VK_IMAGE_USAGE_TRANSFER_DST_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,.sharingMode=VK_SHARING_MODE_EXCLUSIVE,.initialLayout=VK_IMAGE_LAYOUT_UNDEFINED};
    VKCHK(vkCreateImage(nv.dev,&nci,NULL,&native)); VkMemoryRequirements nr; vkGetImageMemoryRequirements(nv.dev,native,&nr);
    uint32_t nt=mem_type(nv.phys,nr.memoryTypeBits,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); if(nt==UINT32_MAX)nt=mem_type(nv.phys,nr.memoryTypeBits,0);
    VkDeviceMemory nmem; VkMemoryAllocateInfo nai={.sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,.allocationSize=nr.size,.memoryTypeIndex=nt};
    VKCHK(vkAllocateMemory(nv.dev,&nai,NULL,&nmem)); VKCHK(vkBindImageMemory(nv.dev,native,nmem,0));

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
        {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=native,.subresourceRange=range},
        {.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.dstAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.oldLayout=VK_IMAGE_LAYOUT_UNDEFINED,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
         .srcQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,.dstQueueFamilyIndex=nv.qfam,.image=nshared,.subresourceRange=range}};
    vkCmdPipelineBarrier(nc,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,2,b);
    VkClearColorValue red={.float32={1,0,0,1}}; vkCmdClearColorImage(nc,native,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,&red,1,&range);
    VkImageMemoryBarrier ns={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT,
        .oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,.srcQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED,.image=native,.subresourceRange=range};
    vkCmdPipelineBarrier(nc,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,0,NULL,0,NULL,1,&ns);
    VkImageCopy cp={.srcSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.dstSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1},.extent={W,H,1}};
    vkCmdCopyImage(nc,native,VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,nshared,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&cp);
    VkImageMemoryBarrier rel={.sType=VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT,.oldLayout=VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,.newLayout=VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex=nv.qfam,.dstQueueFamilyIndex=VK_QUEUE_FAMILY_FOREIGN_EXT,.image=nshared,.subresourceRange=range};
    vkCmdPipelineBarrier(nc,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,0,0,NULL,0,NULL,1,&rel);
    VKCHK(vkEndCommandBuffer(nc)); VkSubmitInfo nsub={.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,.commandBufferCount=1,.pCommandBuffers=&nc,.signalSemaphoreCount=1,.pSignalSemaphores=&nsem};
    VKCHK(vkQueueSubmit(nv.q,1,&nsub,VK_NULL_HANDLE));
    int syncfd=-1; VkSemaphoreGetFdInfoKHR gfi={.sType=VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,.semaphore=nsem,.handleType=VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
    VKCHK(nv.get_sem_fd(nv.dev,&gfi,&syncfd)); printf("   NVIDIA wrote BO; SYNC_FD=%d\n",syncfd);

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
    bool ok=px[o+2]>200&&px[o+1]<40&&px[o]<40; printf("%s\n",ok?"PASS: Intel-owned BO written by NVIDIA and read by Intel.":"FAIL: wrong pixel data.");
    return ok?0:2;
}
