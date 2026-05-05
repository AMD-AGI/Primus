###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tier 1 -- C. Host limits (ulimit -l, /dev/shm, NUMA, CPU governor)."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from ..shell_utils import _read_text


def _collect_host_limits(*, ulimit_l_min_gb: float, shm_min_gb: float) -> Dict[str, Any]:
    """Capture training-relevant kernel/process limits and tunables and
    return hard-failure reasons for the ones that block training under load.

    Hard fail today (cause node FAIL):

    * ``ulimit -l`` (RLIMIT_MEMLOCK) is not unlimited and below
      ``ulimit_l_min_gb`` -- RDMA pin failures look like NCCL hangs.
    * ``/dev/shm`` total size below ``shm_min_gb`` -- NCCL shared-memory
      transport falls back or fails.

    Soft (collected for drift detection only):

    * NUMA node count, CPU count, CPU governor, kernel/OS version. The
      aggregator flags drift across the cluster but does not FAIL nodes
      individually for these.
    """
    out: Dict[str, Any] = {}

    # Resource limits.
    try:
        import resource  # type: ignore

        soft_l, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        out["memlock_soft_bytes"] = -1 if soft_l == resource.RLIM_INFINITY else int(soft_l)
        soft_n, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        out["nofile_soft"] = int(soft_n)
        soft_p, _ = resource.getrlimit(resource.RLIMIT_NPROC)
        out["nproc_soft"] = -1 if soft_p == resource.RLIM_INFINITY else int(soft_p)
    except Exception as e:
        out["resource_error"] = str(e)

    # /dev/shm size + free.
    try:
        st = os.statvfs("/dev/shm")
        out["shm_size_bytes"] = int(st.f_blocks) * int(st.f_frsize)
        out["shm_avail_bytes"] = int(st.f_bavail) * int(st.f_frsize)
    except Exception as e:
        out["shm_error"] = str(e)

    # NUMA topology.
    try:
        nodes = [
            n for n in os.listdir("/sys/devices/system/node") if n.startswith("node") and n[4:].isdigit()
        ]
        out["numa_nodes"] = len(nodes)
    except Exception:
        out["numa_nodes"] = None

    # CPU count + governor.
    try:
        out["cpu_count"] = os.cpu_count()
    except Exception:
        out["cpu_count"] = None
    out["cpu_governor"] = _read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") or None

    # Hard checks.
    fail_reasons: List[str] = []
    memlock = out.get("memlock_soft_bytes")
    if memlock is not None and memlock != -1 and ulimit_l_min_gb > 0:
        if memlock < ulimit_l_min_gb * (1 << 30):
            fail_reasons.append(
                f"ulimit -l (memlock) = {memlock // (1 << 20)} MiB; "
                f"required: unlimited or >= {ulimit_l_min_gb} GiB. "
                "RDMA pin will fail under load."
            )
    shm = out.get("shm_size_bytes")
    if shm is not None and shm_min_gb > 0:
        if shm < shm_min_gb * (1 << 30):
            fail_reasons.append(
                f"/dev/shm size = {shm / (1 << 30):.2f} GiB; "
                f"required: >= {shm_min_gb} GiB. NCCL shared-mem may fail."
            )
    out["fail_reasons"] = fail_reasons

    return out
