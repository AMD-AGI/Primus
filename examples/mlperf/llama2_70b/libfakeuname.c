/*******************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE for license information.
 ******************************************************************************/

/* Container-only uname shim. RCCL turns cuMem off when uname reports a
 * kernel older than 6.8, even when amdgpu supports cuMem. Some hosts report
 * an older kernel than the driver supports (a RHEL 5.14 uname is one
 * example). Preload this and set FAKE_UNAME_RELEASE=6.8.0 so RCCL enables
 * cuMem and the copy-engine collectives run. LUX_FAKE_KERNEL is still
 * accepted. The host utsname is left unchanged.
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
    fake = getenv("FAKE_UNAME_RELEASE");
    if (fake == NULL || fake[0] == '\0') {
        fake = getenv("LUX_FAKE_KERNEL");
    }
    if (fake != NULL && fake[0] != '\0') {
        memset(buf->release, 0, sizeof(buf->release));
        strncpy(buf->release, fake, sizeof(buf->release) - 1);
    }
    return rc;
}
