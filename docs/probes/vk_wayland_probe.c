// vk_wayland_probe — check per-GPU Wayland presentation support (smithay#2112)
// Build: gcc -O1 -o vk_wayland_probe vk_wayland_probe.c -lvulkan -lwayland-client
#define VK_USE_PLATFORM_WAYLAND_KHR
#include <vulkan/vulkan.h>
#include <wayland-client.h>
#include <stdio.h>

int main(void) {
    struct wl_display *dpy = wl_display_connect(NULL);
    if (!dpy) { fprintf(stderr, "cannot connect to wayland display\n"); return 1; }

    const char *exts[] = { "VK_KHR_surface", "VK_KHR_wayland_surface" };
    VkApplicationInfo app = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1 };
    VkInstanceCreateInfo ci = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app, .enabledExtensionCount = 2, .ppEnabledExtensionNames = exts };
    VkInstance inst;
    if (vkCreateInstance(&ci, NULL, &inst) != VK_SUCCESS) { printf("vkCreateInstance failed\n"); return 1; }

    PFN_vkGetPhysicalDeviceWaylandPresentationSupportKHR fp =
        (void *)vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceWaylandPresentationSupportKHR");
    if (!fp) { printf("no vkGetPhysicalDeviceWaylandPresentationSupportKHR\n"); return 1; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice devs[8];
    if (n > 8) n = 8;
    vkEnumeratePhysicalDevices(inst, &n, devs);

    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, NULL);
        VkQueueFamilyProperties q[16];
        if (qn > 16) qn = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &qn, q);
        int present = 0;
        for (uint32_t f = 0; f < qn; f++) {
            if ((q[f].queueFlags & VK_QUEUE_GRAPHICS_BIT) && fp(devs[i], f, dpy)) { present = 1; break; }
        }
        printf("GPU%u: %s [%04x:%04x] -> wayland presentation: %s\n",
               i, p.deviceName, p.vendorID, p.deviceID,
               present ? "SUPPORTED" : "NOT SUPPORTED");
    }
    return 0;
}
