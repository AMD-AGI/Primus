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


def _visible_render_nodes():
    """Render nodes of the GPUs in ROCR_VISIBLE_DEVICES, or all of them if it is unset.

    Opening a wedged GPU's render node can block in the kernel, so stay off GPUs this job
    does not use.
    """
    nodes = sorted(glob.glob("/dev/dri/renderD*"))
    visible = os.environ.get("ROCR_VISIBLE_DEVICES", "").strip()
    if not visible or not all(v.strip().isdigit() for v in visible.split(",")):
        return nodes
    domains = []
    kfd_nodes = glob.glob("/sys/class/kfd/kfd/topology/nodes/*")
    for kfd in sorted(kfd_nodes, key=lambda p: int(os.path.basename(p))):
        try:
            with open(os.path.join(kfd, "gpu_id")) as f:
                if f.read().strip() == "0":
                    continue
            with open(os.path.join(kfd, "properties")) as f:
                props = dict(line.split()[:2] for line in f if len(line.split()) >= 2)
            domains.append(int(props["domain"]))
        except (OSError, KeyError, ValueError):
            return nodes
    wanted = set()
    for v in visible.split(","):
        if int(v) >= len(domains):
            return nodes
        wanted.add(domains[int(v)])
    picked = []
    for node in nodes:
        dev = os.path.realpath(f"/sys/class/drm/{os.path.basename(node)}/device")
        pci = os.path.basename(dev)
        if ":" in pci and int(pci.split(":")[0], 16) in wanted:
            picked.append(node)
    return picked or nodes


def amdgpu_drm_version():
    """(major, minor) of the first visible amdgpu render node, or None if none is readable."""
    for node in _visible_render_nodes():
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
