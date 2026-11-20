###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""STREAM memory-bandwidth tests (copy / scale / add / triad) on one GPU.

Follows the STREAM conventions: each kernel's bandwidth is computed from its
best trial, and counts only the bytes the kernel must move (reads + writes,
no write-allocate traffic).  Arrays default to 1 GiB each so the working set
(3 GiB) is many times the MI300-class 256 MB Infinity Cache, and the result
is HBM bandwidth rather than cache bandwidth.

Why bandwidth and not memory clock
----------------------------------
Achieved bandwidth = MCLK x bytes-per-MCLK-cycle x efficiency, and the
efficiency of a streaming kernel (refresh, bank conflicts, read/write
turnaround) is not known independently of the clock.  So one measurement
cannot separate the two.  What it does bound is the *lowest* MCLK that could
deliver it (efficiency = 100 %).  The live MCLK sampled while the timed
kernels run (amdsmi or sysfs) is reported next to it, together with the
efficiency at that clock.

MCLK idles low and ramps up over a few hundred ms once memory traffic starts,
so the kernels run until it settles before any trial is timed.

From STREAM to the projection's HBM bandwidth
---------------------------------------------
STREAM is the ceiling; a training layer's memory-bound kernels (norms, RoPE,
SwiGLU, residual adds, layout copies) sustain only part of it.  rocprofv3
FETCH_SIZE / WRITE_SIZE counters on a Llama-3-8B layer (fwd+bwd, MI325X)
put them at 72 % of STREAM in aggregate (3.41 of 4.76 TB/s), so the workload
bandwidth is ``STREAM x STREAM_WORKLOAD_DERATE``.

The profilers price memory traffic as ``peak x fraction`` with fractions
fitted against the spec peak, anchored at ``_ACTIVATION_BW_FRACTION`` (0.566)
for exactly these streaming kernels.  The measured 3.41 TB/s is 0.569 of the
MI325X 6.0 TB/s spec, so the fitted model already agrees with the counters.
To keep every fitted fraction valid the workload bandwidth is handed to the
model as a peak-equivalent, ``workload / _ACTIVATION_BW_FRACTION``; on MI325X
that reproduces the spec peak to within 1 %.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from primus.core.projection.hardware_calibration.device_info import run_until_settled
from primus.core.projection.module_profilers.moe_mlp import _ACTIVATION_BW_FRACTION

STREAM_WORKLOAD_DERATE = 0.72
STREAM_DERATE_ENV = "PRIMUS_HBM_STREAM_DERATE"

# (name, bytes moved per element in units of the element size)
STREAM_KERNELS = (("copy", 2), ("scale", 2), ("add", 3), ("triad", 3))

DEFAULT_ARRAY_MIB = 1024
_MIN_ARRAY_MIB = 256
_SCALAR = 3.0
# MCLK takes ~0.6 s to ramp from idle to its loaded level on MI300-class parts.
DEFAULT_MIN_SETTLE_S = 0.75


def _allocate_arrays(num_elements: int, device):
    import torch

    a = torch.full((num_elements,), 1.0, dtype=torch.float32, device=device)
    b = torch.full((num_elements,), 2.0, dtype=torch.float32, device=device)
    c = torch.zeros(num_elements, dtype=torch.float32, device=device)
    return a, b, c


def run_stream(
    device,
    array_mib: int = DEFAULT_ARRAY_MIB,
    trials: int = 50,
    reader=None,
    sampler=None,
    min_settle_s: float = DEFAULT_MIN_SETTLE_S,
) -> Dict:
    """Run the four STREAM kernels; returns per-kernel best / mean GB/s.

    Halves the array size on OOM down to 256 MiB per array.  When a
    :class:`ClockSampler` is given, the timed trials run inside its window.
    """
    import contextlib

    import torch

    elem_bytes = 4
    arrays = None
    while arrays is None:
        num_elements = array_mib * 1024**2 // elem_bytes
        try:
            arrays = _allocate_arrays(num_elements, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if array_mib // 2 < _MIN_ARRAY_MIB:
                raise
            array_mib //= 2
    a, b, c = arrays

    kernels = {
        "copy": lambda: c.copy_(a),
        "scale": lambda: torch.mul(c, _SCALAR, out=b),
        "add": lambda: torch.add(a, b, out=c),
        "triad": lambda: torch.add(b, c, alpha=_SCALAR, out=a),
    }
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def step():
        for fn in kernels.values():
            fn()
        torch.cuda.synchronize(device)

    settle = run_until_settled(step, reader, "mclk", min_s=min_settle_s)

    times_ms: Dict[str, list] = {name: [] for name, _ in STREAM_KERNELS}
    with sampler.window() if sampler is not None else contextlib.nullcontext():
        for _ in range(trials):
            for name, _ in STREAM_KERNELS:
                start.record()
                kernels[name]()
                end.record()
                end.synchronize()
                times_ms[name].append(start.elapsed_time(end))
    a = b = c = arrays = None
    torch.cuda.empty_cache()

    array_bytes = num_elements * elem_bytes
    results: Dict[str, Dict[str, float]] = {}
    for name, arrays_touched in STREAM_KERNELS:
        moved = arrays_touched * array_bytes
        best_ms = min(times_ms[name])
        mean_ms = sum(times_ms[name]) / len(times_ms[name])
        results[name] = {
            "bytes_moved": float(moved),
            "best_ms": best_ms,
            "best_gbps": moved / (best_ms / 1e3) / 1e9,
            "mean_gbps": moved / (mean_ms / 1e3) / 1e9,
        }
    return {"array_mib": array_mib, "dtype": "fp32", "trials": trials, "settle": settle, "kernels": results}


def stream_workload_derate() -> float:
    """``PRIMUS_HBM_STREAM_DERATE`` if set, else :data:`STREAM_WORKLOAD_DERATE`."""
    env = os.getenv(STREAM_DERATE_ENV)
    derate = float(env) if env else STREAM_WORKLOAD_DERATE
    if not 0.0 < derate <= 1.0:
        raise ValueError(f"{STREAM_DERATE_ENV} must be in (0, 1], got {derate}")
    return derate


def derive_model_hbm_bandwidth(sustained_gbps: float, derate: Optional[float] = None) -> Dict[str, float]:
    """Turn a STREAM result into the workload bandwidth and the model's peak-equivalent."""
    if derate is None:
        derate = stream_workload_derate()
    workload = sustained_gbps * derate
    return {
        "stream_derate": derate,
        "workload_bandwidth_gbps": workload,
        "model_reference_fraction": _ACTIVATION_BW_FRACTION,
        "model_hbm_bandwidth_gbps": workload / _ACTIVATION_BW_FRACTION,
    }


def analyze_memory_clock(
    sustained_gbps: float,
    peak_gbps: Optional[float],
    max_mclk_mhz: Optional[int],
    observed_mclk_mhz: Optional[float],
) -> Dict[str, Optional[float]]:
    """Relate measured bandwidth to the memory clock.

    ``bytes_per_mclk_cycle`` is the spec peak divided by the top MCLK level
    (e.g. MI325X: 6000 GB/s / 1500 MHz = 4000 B/cycle), which keeps the
    relation consistent with the profile peak without hardcoding HBM bus
    widths or data rates per part.
    """
    out: Dict[str, Optional[float]] = {
        "peak_bandwidth_gbps": peak_gbps,
        "efficiency_vs_peak": None,
        "bytes_per_mclk_cycle": None,
        "min_mclk_mhz_at_full_efficiency": None,
        "observed_mclk_mhz": observed_mclk_mhz,
        "efficiency_at_observed_mclk": None,
    }
    if peak_gbps:
        out["efficiency_vs_peak"] = sustained_gbps / peak_gbps
    if peak_gbps and max_mclk_mhz:
        bytes_per_cycle = peak_gbps * 1e9 / (max_mclk_mhz * 1e6)
        out["bytes_per_mclk_cycle"] = bytes_per_cycle
        out["min_mclk_mhz_at_full_efficiency"] = sustained_gbps * 1e9 / bytes_per_cycle / 1e6
        if observed_mclk_mhz:
            out["efficiency_at_observed_mclk"] = (
                sustained_gbps * 1e9 / (bytes_per_cycle * observed_mclk_mhz * 1e6)
            )
    return out
