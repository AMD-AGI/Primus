###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from .gpu_probe import probe_gpus
from .utils import Finding, ProbeResult, default_min_tflops


def _gemm_tflops_ms(m: int, n: int, k: int, ms: float) -> float:
    # 2*m*n*k ops (FMA) per matmul.
    flops = 2.0 * float(m) * float(n) * float(k)
    return (flops / (ms / 1000.0)) / 1e12


def _single_gpu_gemm_sanity(dtype: str = "fp16") -> List[Finding]:
    """
    Run GEMM sanity on each rank's GPU and report individual results.
    Results are aggregated (min/max/avg) in preflight_check.py.
    """
    findings: List[Finding] = []
    try:
        import torch  # type: ignore
    except Exception as e:
        return [Finding("warn", "torch not available; skip perf sanity", {"error": str(e)})]

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return [Finding("warn", "No GPUs detected; skip perf sanity", {})]

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # Small-ish GEMM; keep it quick.
    m = n = k = 1024
    torch_dtype = torch.float16
    if dtype.lower() in ("bf16", "bfloat16"):
        torch_dtype = torch.bfloat16

    try:
        torch.cuda.set_device(local_rank)
        a = torch.randn((m, k), device="cuda", dtype=torch_dtype)
        b = torch.randn((k, n), device="cuda", dtype=torch_dtype)

        # Warmup
        for _ in range(3):
            _ = a @ b
        torch.cuda.synchronize()

        # Timed
        iters = 5
        t0 = time.time()
        for _ in range(iters):
            _ = a @ b
        torch.cuda.synchronize()
        t1 = time.time()

        ms = ((t1 - t0) * 1000.0) / float(iters)
        tflops = _gemm_tflops_ms(m, n, k, ms)
        findings.append(
            Finding(
                "info",
                "Single-GPU GEMM sanity",
                {
                    "dtype": str(torch_dtype),
                    "m": m,
                    "n": n,
                    "k": k,
                    "ms": round(ms, 3),
                    "tflops": round(tflops, 2),
                    "rank": rank,
                    "local_rank": local_rank,
                },
            )
        )

        min_tflops = default_min_tflops()
        if tflops < min_tflops:
            findings.append(
                Finding(
                    "warn",
                    "GEMM TFLOPS below threshold (possible perf regression)",
                    {"measured_tflops": round(tflops, 2), "min_tflops": min_tflops, "rank": rank},
                )
            )
    except Exception as e:
        findings.append(Finding("warn", "GEMM sanity failed (warn-only)", {"error": str(e), "rank": rank}))

    return findings


def _memory_alloc_sanity() -> List[Finding]:
    """
    Run memory allocation sanity on each rank's GPU.
    """
    findings: List[Finding] = []
    try:
        import torch  # type: ignore
    except Exception as e:
        return [Finding("warn", "torch not available; skip memory sanity", {"error": str(e)})]

    if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
        return [Finding("warn", "No GPUs detected; skip memory sanity", {})]

    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        # Use a smaller, safer allocation by default (~128MB).
        n = 64 * 1024 * 1024  # elements (~128MB)
        x = torch.empty((n,), device="cuda", dtype=torch.float16)
        del x
        torch.cuda.empty_cache()
        findings.append(
            Finding("info", "Memory alloc/free sanity", {"bytes_approx": int(n) * 2, "rank": rank})
        )
    except Exception as e:
        findings.append(Finding("warn", "Memory alloc/free sanity failed (warn-only)", {"error": str(e)}))

    return findings


def run_gpu_full_checks(*, perf_sanity: bool = False) -> Dict[str, Any]:
    """
    Level: full

    Purpose: Detect obvious performance regressions.

    These checks should never hard-fail by default (WARN only).
    """
    findings: List[Finding] = []
    probe: ProbeResult = probe_gpus()
    if not probe.ok:
        findings.append(Finding("warn", "No GPUs detected; skipping full checks", {}))
        return {"ok": True, "probe": probe, "findings": findings}

    # Single-GPU compute sanity - runs on every rank
    findings.extend(_single_gpu_gemm_sanity(dtype="fp16"))
    findings.extend(_memory_alloc_sanity())

    # Communication sanity is intentionally omitted here by default (requires multi-process).
    if perf_sanity:
        findings.append(
            Finding(
                "warn",
                "Communication sanity not implemented in lightweight single-process mode",
                {"note": "Requires multi-process (torch.distributed) to be meaningful."},
            )
        )

    return {"ok": True, "probe": probe, "findings": findings}
