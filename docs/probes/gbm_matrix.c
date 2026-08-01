#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <gbm.h>

int main(void) {
    int fd = open("/dev/dri/renderD129", O_RDWR | O_CLOEXEC);
    struct gbm_device *g = gbm_create_device(fd);
    struct { const char *name; uint32_t flags; } tests[] = {
        { "0", 0 },
        { "LINEAR", GBM_BO_USE_LINEAR },
        { "RENDERING", GBM_BO_USE_RENDERING },
        { "RENDERING|LINEAR", GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR },
        { "COPYOUT", GBM_BO_USE_RENDERING },
        { "COPYOUT|LINEAR", GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR },
    };
    uint64_t linear = 0ULL;
    for (int i = 0; i < 6; i++) {
        errno = 0;
        struct gbm_bo *bo = gbm_bo_create_with_modifiers2(g, 64, 64, GBM_FORMAT_XRGB8888,
                                                          &linear, 1, tests[i].flags);
        printf("with_modifiers2 LINEAR %-20s -> %s (errno=%d, mod=0x%llx)\n", tests[i].name,
               bo ? "OK" : "FAIL", errno,
               bo ? (unsigned long long)gbm_bo_get_modifier(bo) : 0ULL);
        if (bo) gbm_bo_destroy(bo);
        errno = 0;
        bo = gbm_bo_create(g, 64, 64, GBM_FORMAT_XRGB8888, tests[i].flags);
        printf("create            %-20s -> %s (errno=%d, mod=0x%llx)\n", tests[i].name,
               bo ? "OK" : "FAIL", errno,
               bo ? (unsigned long long)gbm_bo_get_modifier(bo) : 0ULL);
        if (bo) gbm_bo_destroy(bo);
    }
    return 0;
}
