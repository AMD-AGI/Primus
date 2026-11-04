#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Global hook: keep HIP memory pools off virtual-memory mappings on old amdgpu drivers.
#
# Trigger:
#   The host amdgpu DRM interface is older than 3.64 and HIP_MEM_POOL_USE_VM is unset.
#
# ROCm 10.x userspace waits on a GEM_VA timeline syncobj after every virtual-memory map
# (ROCm/rocm-systems#9820). Drivers before DRM 3.64 ignore the timeline fields, so the wait
# never returns. hipMallocAsync pools map memory this way by default, and Primus Turbo's MoE
# permute allocates with hipMallocAsync, so every MoE model hangs on its first step. This hook
# emits env.HIP_MEM_POOL_USE_VM=0 on such hosts; newer drivers keep the HIP default.
#

import ctypes
import fcntl
import glob
import os
import sys

MIN_TIMELINE_DRM_VERSION = (3, 64)


class _DrmVersion(ctypes.Structure):
    _fields_ = [
        ("version_major", ctypes.c_int),
        ("version_minor", ctypes.c_int),
        ("version_patchlevel", ctypes.c_int),
        ("name_len", ctypes.c_size_t),
        ("name", ctypes.c_void_p),
        ("date_len", ctypes.c_size_t),
        ("date", ctypes.c_void_p),
        ("desc_len", ctypes.c_size_t),
        ("desc", ctypes.c_void_p),
    ]


# _IOWR('d', 0x00, struct drm_version)
DRM_IOCTL_VERSION = (3 << 30) | (ctypes.sizeof(_DrmVersion) << 16) | (ord("d") << 8) | 0x00


def amdgpu_drm_version():
    """(major, minor) of the first amdgpu render node, or None if none is readable."""
    for node in sorted(glob.glob("/dev/dri/renderD*")):
        try:
            fd = os.open(node, os.O_RDWR)
        except OSError:
            continue
        name = ctypes.create_string_buffer(32)
        version = _DrmVersion(name_len=len(name) - 1, name=ctypes.addressof(name))
        try:
            fcntl.ioctl(fd, DRM_IOCTL_VERSION, version, True)
        except Exception:  # noqa: BLE001 - a failed probe must never abort the launch
            continue
        finally:
            os.close(fd)
        if name.value == b"amdgpu":
            return version.version_major, version.version_minor
    return None


def main():
    version = amdgpu_drm_version()
    if version is None:
        print("[INFO] [vm_timeline] No readable amdgpu render node; skipping the DRM version check.")
        return 0

    version_str = "%d.%d" % version
    min_str = "%d.%d" % MIN_TIMELINE_DRM_VERSION
    if version >= MIN_TIMELINE_DRM_VERSION:
        print(f"[INFO] [vm_timeline] amdgpu DRM {version_str} supports GEM_VA timelines.")
        return 0

    user_value = os.environ.get("HIP_MEM_POOL_USE_VM")
    if user_value is not None:
        level = "INFO" if user_value == "0" else "WARN"
        print(
            f"[{level}] [vm_timeline] amdgpu DRM {version_str} < {min_str}; keeping user-set "
            f"HIP_MEM_POOL_USE_VM={user_value} (any value but 0 hangs hipMallocAsync, e.g. in MoE)."
        )
        return 0

    print(
        f"[INFO] [vm_timeline] amdgpu DRM {version_str} < {min_str}: the host driver never signals the "
        "GEM_VA timeline ROCm waits on, so HIP memory pools must not use virtual memory."
    )
    print("env.HIP_MEM_POOL_USE_VM=0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
