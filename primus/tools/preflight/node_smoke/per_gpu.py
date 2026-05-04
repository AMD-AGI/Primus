###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-GPU subprocess body (Tier 1 + optional Tier 2 perf).

This module is the body of the ``_per_gpu`` subcommand: it runs every
GPU-local check on a single device index and returns a dict result.
The orchestrator spawns one subprocess per GPU with a hard timeout so a
stuck driver call cannot wedge the parent.

Kept intact (no internal split) -- every code path returns a complete
verdict dict, and inlining each stage is currently easier to follow
than splitting the function up.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from .shell_utils import _read_text, _resolve_gpu_bdf


def _per_gpu_body(
    gpu: int,
    *,
    tier2_perf: bool,
    gemm_tflops_min: float,
    hbm_gbs_min: float,
    hbm_busy_threshold_bytes: int = 2 * (1 << 30),
) -> Dict[str, Any]:
    """Run all per-GPU tests for a single GPU and return a dict result.

    Tier 1 (always): set_device, **pre-touch HBM-busy check** (FAIL if more
    than ``hbm_busy_threshold_bytes`` already in use before our test
    allocates anything), allocate 256 MB, tiny GEMM 2048x2048 bf16 with
    finite-value check.

    Tier 2 (when ``tier2_perf`` is True): GEMM 8192x8192 bf16 TFLOPS
    measurement against ``gemm_tflops_min``, and HBM device-to-device
    copy bandwidth against ``hbm_gbs_min``. Each metric below threshold
    yields FAIL.

    Pre-touch HBM check: ``torch.cuda.mem_get_info`` is called BEFORE we
    allocate anything on this GPU, so the "used" reading reflects only
    foreign / leaked allocations. The post-test reading is also captured
    (under ``low_level.hbm_free_bytes``) for completeness, but the FAIL
    rule uses only the pre-touch number to avoid being polluted by our
    own caching-allocator footprint.
    """
    t0 = time.time()
    details: Dict[str, Any] = {}

    try:
        import torch  # type: ignore
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"torch import failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    if not torch.cuda.is_available():
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": "torch.cuda.is_available() is False",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }
    if gpu >= torch.cuda.device_count():
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": (
                f"gpu index {gpu} >= visible device_count {torch.cuda.device_count()}"
            ),
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- set_device ---
    try:
        torch.cuda.set_device(gpu)
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"set_device({gpu}) raised: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- pre-touch HBM-busy check (BEFORE we allocate anything) ---
    # Captured here, NOT in the low_level block at the end, because by
    # then PyTorch's caching allocator has already taken pages we won't
    # truly release on empty_cache(). The pre-touch reading is the only
    # honest answer to "is someone else holding this GPU?".
    try:
        free_b, total_b = torch.cuda.mem_get_info(gpu)
        used_b = max(0, int(total_b) - int(free_b))
        details["hbm_pre_touch_total_bytes"] = int(total_b)
        details["hbm_pre_touch_free_bytes"] = int(free_b)
        details["hbm_pre_touch_used_bytes"] = used_b
        details["hbm_pre_touch_used_gib"] = round(used_b / (1 << 30), 3)
        if used_b > hbm_busy_threshold_bytes:
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": (
                    f"pre-touch HBM busy: {round(used_b / (1 << 30), 2)} GiB "
                    f"already in use (threshold "
                    f"{round(hbm_busy_threshold_bytes / (1 << 30), 2)} GiB) "
                    f"-- likely leaked process from a previous job; "
                    f"see node-level gpu_processes section to identify the PID"
                ),
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }
    except Exception as e:
        details["hbm_pre_touch_error"] = f"mem_get_info failed: {e}"

    # --- 256 MB tensor alloc + simple write + sync ---
    try:
        nbytes_alloc = 256 * 1024 * 1024
        n_elem = nbytes_alloc // 2  # bf16
        x = torch.empty(n_elem, dtype=torch.bfloat16, device=f"cuda:{gpu}")
        x.fill_(1.0)
        torch.cuda.synchronize()
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"256MB bf16 alloc/fill/sync failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- tiny GEMM 2048x2048 bf16, finite-value check ---
    try:
        m = n = k = 2048
        a = torch.randn((m, k), dtype=torch.bfloat16, device=f"cuda:{gpu}")
        b = torch.randn((k, n), dtype=torch.bfloat16, device=f"cuda:{gpu}")
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        if not torch.isfinite(c).all().item():
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": "tiny GEMM produced non-finite values (possible HW corruption)",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        return {
            "gpu": gpu,
            "status": "FAIL",
            "reason": f"tiny GEMM 2048x2048 failed: {e}",
            "duration_sec": round(time.time() - t0, 3),
            "details": details,
        }

    # --- D-1 light: PCIe link + HBM (sysfs + torch only, fast & no shell-out) ---
    # Captured into details.low_level so the aggregator can flag drift across
    # the cluster (e.g. a single GPU running at Gen3 x8 because the slot
    # needs reseating, or a GPU that exposes only half of its HBM).
    # Each sub-capture is independent: a missing/unparseable PCIe BDF must
    # not cost us the HBM size, and vice-versa.
    low: Dict[str, Any] = {}
    props = None
    try:
        props = torch.cuda.get_device_properties(gpu)
    except Exception as e:
        low["error"] = f"get_device_properties failed: {e}"

    if props is not None:
        # PCIe link details (sysfs)
        try:
            bdf = _resolve_gpu_bdf(props)
            if bdf:
                low["pci_bdf"] = bdf
                sysdir = f"/sys/bus/pci/devices/{bdf}"
                speed = _read_text(f"{sysdir}/current_link_speed")
                width = _read_text(f"{sysdir}/current_link_width")
                low["pcie_link_speed_raw"] = speed or None
                low["pcie_link_width"] = int(width) if width.isdigit() else None
                # speed is e.g. "32.0 GT/s PCIe" -> 32.0
                try:
                    low["pcie_link_speed_gts"] = (
                        float(speed.split()[0]) if speed else None
                    )
                except Exception:
                    low["pcie_link_speed_gts"] = None
            else:
                low["pcie_error"] = (
                    f"could not resolve PCIe BDF (pci_bus_id="
                    f"{getattr(props, 'pci_bus_id', None)!r})"
                )
        except Exception as e:
            low["pcie_error"] = f"PCIe sysfs capture failed: {e}"

        # HBM total/free (torch). Independent of BDF resolution.
        try:
            free_b, total_b = torch.cuda.mem_get_info(gpu)
            low["hbm_total_bytes"] = int(total_b)
            low["hbm_free_bytes"] = int(free_b)
            low["hbm_total_gib"] = round(total_b / (1 << 30), 2)
        except Exception as e:
            low["hbm_error"] = f"mem_get_info failed: {e}"
            tm = int(getattr(props, "total_memory", 0) or 0)
            if tm:
                low["hbm_total_bytes"] = tm
                low["hbm_total_gib"] = round(tm / (1 << 30), 2)
    if low:
        details["low_level"] = low

    # --- Tier 2 perf sanity (optional) ---
    if tier2_perf:
        # GEMM TFLOPS. Warmup/iter counts mirror the preflight `--quick` preset
        # (`square_gemm.py` with WARMUP=5, ITERATION=20) so smoke and preflight
        # report comparable steady-state numbers.
        try:
            tflops = _measure_gemm_tflops(gpu, size=8192, warmup=5, iters=20)
            details["gemm_tflops"] = round(tflops, 2)
            if tflops < gemm_tflops_min:
                return {
                    "gpu": gpu,
                    "status": "FAIL",
                    "reason": (
                        f"GEMM TFLOPS {tflops:.0f} < threshold {gemm_tflops_min:.0f}"
                    ),
                    "duration_sec": round(time.time() - t0, 3),
                    "details": details,
                }
        except Exception as e:
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": f"GEMM TFLOPS measurement failed: {e}",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }

        # HBM device-to-device copy bandwidth. HBM is fast enough that we
        # need a healthy number of timed iterations for stable timing.
        try:
            gbs = _measure_hbm_gbs(gpu, size_bytes=512 * 1024 * 1024, warmup=10, iters=20)
            details["hbm_gbs"] = round(gbs, 1)
            if gbs < hbm_gbs_min:
                return {
                    "gpu": gpu,
                    "status": "FAIL",
                    "reason": f"HBM GB/s {gbs:.0f} < threshold {hbm_gbs_min:.0f}",
                    "duration_sec": round(time.time() - t0, 3),
                    "details": details,
                }
        except Exception as e:
            return {
                "gpu": gpu,
                "status": "FAIL",
                "reason": f"HBM bandwidth measurement failed: {e}",
                "duration_sec": round(time.time() - t0, 3),
                "details": details,
            }

    return {
        "gpu": gpu,
        "status": "PASS",
        "reason": "",
        "duration_sec": round(time.time() - t0, 3),
        "details": details,
    }


def _measure_gemm_tflops(gpu: int, *, size: int, warmup: int, iters: int) -> float:
    """Measure GEMM TFLOPS for square ``size x size`` bf16 matmul on ``cuda:gpu``."""
    import torch  # type: ignore

    torch.cuda.set_device(gpu)
    a = torch.randn((size, size), dtype=torch.bfloat16, device=f"cuda:{gpu}")
    b = torch.randn((size, size), dtype=torch.bfloat16, device=f"cuda:{gpu}")
    for _ in range(warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    end.record()
    end.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0 / iters
    flops = 2.0 * size * size * size
    return (flops / elapsed_s) / 1e12


def _measure_hbm_gbs(
    gpu: int, *, size_bytes: int, warmup: int, iters: int
) -> float:
    """Measure local HBM bandwidth via device-to-device ``copy_``.

    Each iteration: 1 read of ``src`` + 1 write to ``dst`` = ``2 * size_bytes``
    of HBM traffic. Uses ``torch.cuda.Event`` for accurate GPU-side timing.
    """
    import torch  # type: ignore

    torch.cuda.set_device(gpu)
    n = size_bytes // 2  # bf16 = 2 bytes/element
    src = torch.empty(n, dtype=torch.bfloat16, device=f"cuda:{gpu}")
    dst = torch.empty_like(src)
    src.fill_(1.0)
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    end.synchronize()

    elapsed_s = start.elapsed_time(end) / 1000.0 / iters
    return (2.0 * size_bytes / elapsed_s) / 1e9
