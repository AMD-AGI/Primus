###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Finding:
    # "info" | "warn" | "fail"
    level: str
    message: str
    details: Dict[str, Any]


@dataclass
class ProbeResult:
    """
    Normalized GPU probe results (best-effort across ROCm/NVIDIA/CPU-only).
    """

    ok: bool
    backend: str  # "rocm" | "cuda" | "unknown"
    devices: List[Dict[str, Any]]
    tooling: Dict[str, Any]


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def run_cmd(cmd: Sequence[str], timeout_s: int = 5) -> Tuple[int, str, str]:
    p = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()


def which(name: str) -> Optional[str]:
    # Minimal reimplementation to avoid importing shutil everywhere.
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, name)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def default_min_free_mem_gb() -> int:
    # Basic-level FAIL threshold (free memory must be >= this).
    return env_int("PRIMUS_PREFLIGHT_MIN_FREE_MEM_GB", 1)


def default_min_tflops() -> float:
    # Full-level WARN threshold (perf sanity).
    return env_float("PRIMUS_PREFLIGHT_MIN_TFLOPS", 10.0)
