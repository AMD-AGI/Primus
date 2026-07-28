###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .sysfs_probe import sysfs_probe
from .utils import ProbeResult, run_cmd, which

logger = logging.getLogger(__name__)


def _normalize_gfx_arch(raw: str) -> str:
    """
    Normalize ROCm arch strings.
    Examples:
      - "gfx950:sramecc+:xnack-" -> "gfx950"
      - "gfx942" -> "gfx942"
    """
    s = (raw or "").strip()
    if ":" in s:
        s = s.split(":", 1)[0]
    return s


def _probe_amdgpu_version() -> Optional[str]:
    # Best-effort: prefer sysfs module version, then modinfo.
    try:
        path = "/sys/module/amdgpu/version"
        with open(path, "r", encoding="utf-8") as f:
            v = f.read().strip()
            return v or None
    except Exception:
        pass

    if which("modinfo") is None:
        return None
    rc, out, _err = run_cmd(["modinfo", "amdgpu"], timeout_s=5)
    if rc != 0 or not out:
        return None
    for ln in out.splitlines():
        if ln.lower().startswith("version:"):
            return ln.split(":", 1)[1].strip() or None
    return None


def _probe_rocm_version() -> Optional[str]:
    # Best-effort: prefer /opt/rocm .info version files, then rocminfo output.
    for path in ("/opt/rocm/.info/version", "/opt/rocm/.info/rocm_version"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass

    if which("rocminfo") is None:
        return None
    rc, out, _err = run_cmd(["rocminfo"], timeout_s=10)
    if rc != 0 or not out:
        return None
    for ln in out.splitlines():
        if "rocm version" in ln.lower():
            # Common format: "ROCm version: 7.1.0"
            parts = ln.split(":")
            if len(parts) >= 2:
                return parts[-1].strip() or None
    return None


def _probe_with_torch() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
    except Exception as e:
        return {"ok": False, "error": f"torch import failed: {e}", "devices": []}

    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    backend = "unknown"
    if getattr(torch.version, "hip", None):
        backend = "rocm"
    elif getattr(torch.version, "cuda", None):
        backend = "cuda"

    devices: List[Dict[str, Any]] = []
    if available:
        for i in range(count):
            d: Dict[str, Any] = {"index": i}
            try:
                p = torch.cuda.get_device_properties(i)
                d["name"] = getattr(p, "name", None)
                # ROCm-only (best-effort): expose gfx arch if torch provides it.
                # On ROCm builds, this is often available as `gcnArchName` and looks like "gfx942".
                for attr in ("gcnArchName", "gcn_arch_name", "gcnArch"):
                    if hasattr(p, attr):
                        val = getattr(p, attr)
                        if val:
                            raw = str(val)
                            d["arch_raw"] = raw
                            d["arch"] = _normalize_gfx_arch(raw)
                            break
                # bytes
                d["total_memory"] = getattr(p, "total_memory", None)
            except Exception as e:
                d["error"] = str(e)
            devices.append(d)

    return {
        "ok": True,
        "backend": backend,
        "cuda_is_available": available,
        "device_count": count,
        "devices": devices,
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "torch_hip_version": getattr(getattr(torch, "version", None), "hip", None),
    }


def _probe_amd_smi() -> Optional[Dict[str, Any]]:
    """Best-effort amd-smi JSON probe for process occupancy detection only."""
    if which("amd-smi") is None:
        return None
    try:
        rc, out, err = run_cmd(["amd-smi", "list", "--json"], timeout_s=10)
        if rc == 0 and out:
            return {"rc": rc, "json": json.loads(out), "err": err}
    except Exception as e:
        logger.debug("amd-smi probe failed: %s", e)
    return None


_PROBE_CACHE: Optional[ProbeResult] = None


def probe_gpus() -> ProbeResult:
    """
    Best-effort GPU probe for identity + memory + occupancy.

    Returns a normalized structure; individual fields may be missing depending
    on the environment/tooling availability.

    Results are cached since the probe is called multiple times per rank
    (from gpu_basic, gpu_topology, and gpu_perf) and the output is static
    during a single preflight run.

    Primary GPU enumeration uses sysfs (KFD topology), which is safe to call
    from any rank (no subprocesses, no /dev/shm mutex).  amd-smi is kept as
    an optional probe on LOCAL_RANK == 0 for process-occupancy data only;
    its failure never crashes the run.
    """
    global _PROBE_CACHE
    if _PROBE_CACHE is not None:
        return _PROBE_CACHE

    from primus.tools.preflight.global_vars import LOCAL_RANK

    torch_info = _probe_with_torch()
    tooling: Dict[str, Any] = {"torch": torch_info}

    # sysfs probe: safe from every rank, no subprocess calls.
    sysfs_result = sysfs_probe()
    if sysfs_result.ok:
        tooling["sysfs"] = {
            "gpu_count": sysfs_result.gpu_count,
            "gpus": [
                {
                    "index": i,
                    "pci_bdf": g.pci_bdf_str,
                    "unique_id": hex(g.unique_id) if g.unique_id else None,
                    "numa_node": g.numa_node,
                }
                for i, g in enumerate(sysfs_result.gpus)
            ],
            "link_count": len(sysfs_result.links),
        }
    else:
        logger.debug("sysfs probe unavailable: %s", sysfs_result.error)

    # amd-smi JSON: subprocess-based, LOCAL_RANK 0 only, best-effort.
    # Used solely for process-occupancy detection in gpu_basic.py.
    if LOCAL_RANK == 0:
        amd = _probe_amd_smi()
        if amd is not None:
            tooling["amd-smi"] = amd

    backend = torch_info.get("backend") if torch_info.get("ok") else "unknown"
    devices = torch_info.get("devices", []) if torch_info.get("ok") else []

    tooling["amdgpu_version"] = _probe_amdgpu_version()
    tooling["rocm_version"] = _probe_rocm_version()

    ok = (
        bool(torch_info.get("ok"))
        and bool(torch_info.get("cuda_is_available"))
        and int(torch_info.get("device_count", 0)) > 0
    )
    _PROBE_CACHE = ProbeResult(ok=ok, backend=str(backend), devices=list(devices), tooling=tooling)
    return _PROBE_CACHE
