###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .utils import ProbeResult, run_cmd, which


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
    if which("amd-smi") is None:
        return None

    # Prefer JSON if available (newer amd-smi); fall back to `list` output.
    rc, out, err = run_cmd(["amd-smi", "list", "--json"], timeout_s=10)
    if rc == 0 and out:
        try:
            return {"rc": rc, "json": json.loads(out), "err": err}
        except Exception:
            # Fall through to non-json.
            pass

    rc, out, err = run_cmd(["amd-smi", "list"], timeout_s=10)
    return {"rc": rc, "out": out, "err": err}


def _probe_rocm_smi() -> Optional[Dict[str, Any]]:
    if which("rocm-smi") is None:
        return None
    rc, out, err = run_cmd(["rocm-smi", "-a"], timeout_s=10)
    return {"rc": rc, "out": out, "err": err}


def probe_gpus() -> ProbeResult:
    """
    Best-effort GPU probe for identity + memory + occupancy.

    Returns a normalized structure; individual fields may be missing depending
    on the environment/tooling availability.
    """
    torch_info = _probe_with_torch()
    tooling: Dict[str, Any] = {"torch": torch_info}

    amd = _probe_amd_smi()
    if amd is not None:
        tooling["amd-smi"] = amd
    rocmsmi = _probe_rocm_smi()
    if rocmsmi is not None:
        tooling["rocm-smi"] = rocmsmi

    backend = torch_info.get("backend") if torch_info.get("ok") else "unknown"
    devices = torch_info.get("devices", []) if torch_info.get("ok") else []

    # Version metadata (best-effort)
    tooling["amdgpu_version"] = _probe_amdgpu_version()
    tooling["rocm_version"] = _probe_rocm_version()

    ok = (
        bool(torch_info.get("ok"))
        and bool(torch_info.get("cuda_is_available"))
        and int(torch_info.get("device_count", 0)) > 0
    )
    return ProbeResult(ok=ok, backend=str(backend), devices=list(devices), tooling=tooling)
