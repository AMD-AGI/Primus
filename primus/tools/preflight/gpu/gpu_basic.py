###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .gpu_probe import probe_gpus
from .utils import Finding, ProbeResult, default_min_free_mem_gb


def _bytes_to_gb(b: Optional[int]) -> Optional[float]:
    if b is None:
        return None
    try:
        return round(float(b) / (1024**3), 2)
    except Exception:
        return None


def _torch_free_mem_gb(index: int) -> Optional[float]:
    # torch.cuda.mem_get_info exists on newer torch; best-effort only.
    try:
        import torch  # type: ignore

        if hasattr(torch.cuda, "mem_get_info"):
            free_b, _total_b = torch.cuda.mem_get_info(index)
            return _bytes_to_gb(int(free_b))
    except Exception:
        return None
    return None


def _infer_arch_from_name(name: str) -> Optional[str]:
    # Best-effort heuristics.
    # ROCm: gfx942, gfx90a, etc. might appear in name strings depending on build.
    lname = name.lower()
    for token in ("gfx942", "gfx950", "gfx90a", "gfx1100", "gfx1101", "gfx1030"):
        if token in lname:
            return token
    return None


def run_gpu_basic_checks(
    *,
    expect_gpu_count: Optional[int] = None,
    expect_arch: Optional[str] = None,
    expect_memory_gb: Optional[int] = None,
    min_free_memory_gb: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Level: basic

    Purpose: Verify GPU availability and identity.

    Fail conditions (as requested):
      - GPU count mismatch
      - Architecture mismatch
      - Any GPU occupied by foreign process (best-effort)
      - Free memory below threshold
    """
    findings: List[Finding] = []
    probe: ProbeResult = probe_gpus()

    torch_info = probe.tooling.get("torch", {})
    findings.append(
        Finding(
            "info",
            "GPU enumeration",
            {
                "backend": probe.backend,
                "cuda_is_available": torch_info.get("cuda_is_available"),
                "device_count": torch_info.get("device_count"),
                "amdgpu_version": probe.tooling.get("amdgpu_version"),
                "rocm_version": probe.tooling.get("rocm_version"),
            },
        )
    )

    if not probe.ok:
        findings.append(
            Finding("fail", "No GPUs detected (torch.cuda.is_available/device_count)", {"torch": torch_info})
        )
        return {"ok": False, "probe": probe, "findings": findings}

    count = int(torch_info.get("device_count", 0))
    if expect_gpu_count is not None and count != expect_gpu_count:
        findings.append(
            Finding("fail", "GPU count mismatch", {"expected": expect_gpu_count, "actual": count})
        )

    # Per-device identity & memory.
    min_free = default_min_free_mem_gb() if min_free_memory_gb is None else int(min_free_memory_gb)
    per_dev: List[Dict[str, Any]] = []
    for d in probe.devices:
        idx = int(d.get("index", -1))
        name = str(d.get("name", ""))
        total_gb = _bytes_to_gb(d.get("total_memory"))
        # Prefer explicit arch from probe (e.g., torch ROCm gcnArchName), then fallback to name heuristics.
        arch = d.get("arch") or _infer_arch_from_name(name)
        free_gb = _torch_free_mem_gb(idx)
        per = {
            "index": idx,
            "name": name,
            "arch": arch,
            "total_memory_gb": total_gb,
            "free_memory_gb": free_gb,
        }

        if expect_arch is not None:
            # Accept match either in inferred arch token or name substring.
            if (arch and expect_arch != arch) and (expect_arch not in name):
                findings.append(
                    Finding(
                        "fail",
                        "GPU architecture mismatch",
                        {"expected": expect_arch, "actual": arch or name, "gpu": idx},
                    )
                )

        if expect_memory_gb is not None and total_gb is not None:
            if float(total_gb) < float(expect_memory_gb):
                findings.append(
                    Finding(
                        "fail",
                        "GPU total memory below expectation",
                        {"expected_gb": expect_memory_gb, "actual_gb": total_gb, "gpu": idx},
                    )
                )

        if free_gb is not None and free_gb < float(min_free):
            findings.append(
                Finding(
                    "fail",
                    "GPU free memory below threshold",
                    {"min_free_gb": min_free, "actual_free_gb": free_gb, "gpu": idx},
                )
            )

        per_dev.append(per)

    findings.append(Finding("info", "GPU identity", {"devices": per_dev}))

    # Occupancy (other PIDs): best-effort via amd-smi JSON if available.
    amd = probe.tooling.get("amd-smi", {})
    if isinstance(amd, dict) and "json" in amd:
        # We do not hardcode schema; best-effort scan for process lists.
        foreign: List[Dict[str, Any]] = []
        try:
            payload = amd["json"]
            # Scan nested dicts/lists for "Processes" arrays.
            stack = [payload]
            while stack:
                cur = stack.pop()
                if isinstance(cur, dict):
                    for k, v in cur.items():
                        if k.lower() in ("processes", "process") and isinstance(v, list):
                            for proc in v:
                                if isinstance(proc, dict):
                                    foreign.append(proc)
                        else:
                            stack.append(v)
                elif isinstance(cur, list):
                    stack.extend(cur)
        except Exception:
            foreign = []

        if foreign:
            # Treat as FAIL per requirement (foreign process occupancy).
            findings.append(
                Finding("fail", "GPU occupied by other processes (amd-smi)", {"processes": foreign})
            )
        else:
            findings.append(Finding("info", "GPU occupancy (amd-smi)", {"processes": []}))
    else:
        findings.append(Finding("info", "GPU occupancy", {"note": "amd-smi JSON not available; skipped"}))

    # ROCm runtime availability: best-effort presence via tooling.
    if probe.backend == "rocm":
        if "amd-smi" in probe.tooling or "rocm-smi" in probe.tooling:
            findings.append(
                Finding("info", "ROCm runtime/tooling detected", {"tooling": list(probe.tooling.keys())})
            )
        else:
            findings.append(Finding("warn", "ROCm tooling not found (amd-smi/rocm-smi)", {}))

    ok = not any(f.level == "fail" for f in findings)
    return {"ok": ok, "probe": probe, "findings": findings}
