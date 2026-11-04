/* Container-only uname shim. Lux reports a RHEL 5.14 kernel, and RCCL turns
 * cuMem off for any Linux older than 6.8 even when amdgpu supports it.
 * Preload this and set LUX_FAKE_KERNEL=6.8.0-lux so RCCL enables cuMem and
 * the copy-engine collectives run. The host utsname is left unchanged.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/utsname.h>

int uname(struct utsname *buf) {
    static int (*real_uname)(struct utsname *) = NULL;
    const char *fake;
    int rc;

    if (real_uname == NULL) {
        real_uname = (int (*)(struct utsname *))dlsym(RTLD_NEXT, "uname");
    }
    rc = real_uname(buf);
    if (rc != 0 || buf == NULL) {
        return rc;
    }
    fake = getenv("LUX_FAKE_KERNEL");
    if (fake != NULL && fake[0] != '\0') {
        memset(buf->release, 0, sizeof(buf->release));
        strncpy(buf->release, fake, sizeof(buf->release) - 1);
    }
    return rc;
}
