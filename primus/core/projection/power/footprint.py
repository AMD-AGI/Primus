###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Workload footprint: what the GPU is doing during an iteration.

Two pieces:

* :class:`OpRecorder` wraps the GEMM / SDPA simulation backends for one
  simulate-mode run and accumulates FLOPs, bytes and time of every matrix op.
  From those it derives scale-free ratios (achieved matrix utilization, HBM
  utilization and the compute-bound share of the time), so it does not matter
  how often a profiler re-simulates the same op.
* :func:`build_iteration_phases` splits one projected iteration into power
  phases from the projection's own time breakdown (pipeline span and bubble,
  exposed gradient reduce, optimizer step, overlapped communication).
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from primus.core.projection.power.model import Phase

_BYTES_PER_ELEM = {"fp8": 1, "bf16": 2, "fp16": 2, "fp32": 4, "tf32": 4}

# Used when no simulate-mode footprint is available (benchmark mode without a
# recorder).  Typical of dense transformer training; documented assumptions,
# not measurements.
DEFAULT_MATRIX_UTILIZATION = 0.55
DEFAULT_MATRIX_HBM_UTILIZATION = 0.25
DEFAULT_MATRIX_CLOCK_SENSITIVITY = 0.85
# Share of the busy compute span spent outside GEMM / attention kernels
# (norms, activations, permutes, optimizer-free elementwise work).
DEFAULT_NON_MATRIX_FRACTION = 0.15
NON_MATRIX_HBM_UTILIZATION = 0.55
NON_MATRIX_CLOCK_SENSITIVITY = 0.2
COMM_KERNEL_BUSY = 0.3
COMM_HBM_UTILIZATION = 0.2
OPTIMIZER_HBM_UTILIZATION = 0.7
OPTIMIZER_CLOCK_SENSITIVITY = 0.1


@dataclass
class WorkloadFootprint:
    matrix_utilization: float = DEFAULT_MATRIX_UTILIZATION
    matrix_hbm_utilization: float = DEFAULT_MATRIX_HBM_UTILIZATION
    matrix_clock_sensitivity: float = DEFAULT_MATRIX_CLOCK_SENSITIVITY
    non_matrix_fraction: float = DEFAULT_NON_MATRIX_FRACTION
    num_ops: int = 0
    source: str = "default"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WorkloadFootprint":
        if not data:
            return cls()
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


class OpRecorder:
    """Records matrix ops issued to the simulation backends.

    Installs instance-level wrappers on ``simulate_gemm`` / ``simulate_sdpa``
    (so a backend's own composite helpers, which call ``self.simulate_gemm``,
    are seen too).  Only the outermost call is recorded: SDPA kernels priced
    from GEMM tiles are not double counted.
    """

    def __init__(self, peak_tflops: Optional[float], hbm_gbps: Optional[float]):
        self.peak_flops_per_ms = (peak_tflops or 0.0) * 1e9
        self.hbm_bytes_per_ms = (hbm_gbps or 0.0) * 1e6
        self._depth = threading.local()
        self.flops = 0.0
        self.bytes = 0.0
        self.time_ms = 0.0
        self.compute_bound_ms = 0.0
        self.num_ops = 0
        self._installed: List[tuple] = []

    # -- installation ----------------------------------------------------
    def install(self, gemm_backend: Any = None, sdpa_backend: Any = None) -> "OpRecorder":
        if gemm_backend is not None and hasattr(gemm_backend, "simulate_gemm"):
            self._wrap(gemm_backend, "simulate_gemm", self._gemm_cost)
        if sdpa_backend is not None and hasattr(sdpa_backend, "simulate_sdpa"):
            self._wrap(sdpa_backend, "simulate_sdpa", self._sdpa_cost)
        return self

    def uninstall(self) -> None:
        for obj, attr, own in reversed(self._installed):
            if own is not None:
                setattr(obj, attr, own)
            else:
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
        self._installed.clear()

    def _wrap(self, obj: Any, attr: str, cost_fn) -> None:
        inner = getattr(obj, attr)
        own = getattr(obj, "__dict__", {}).get(attr)
        recorder = self

        def wrapped(*args, **kwargs):
            depth = getattr(recorder._depth, "n", 0)
            recorder._depth.n = depth + 1
            try:
                result = inner(*args, **kwargs)
            finally:
                recorder._depth.n = depth
            if depth == 0:
                try:
                    flops, nbytes = cost_fn(result, *args, **kwargs)
                    t = float(getattr(result, "forward_time_ms", 0.0) or 0.0) + float(
                        getattr(result, "backward_time_ms", 0.0) or 0.0
                    )
                    recorder.record(flops, nbytes, t)
                except Exception:
                    pass  # footprint is best-effort; never break the projection
            return result

        setattr(obj, attr, wrapped)
        self._installed.append((obj, attr, own))

    # -- cost models -----------------------------------------------------
    @staticmethod
    def _gemm_cost(result, m, n, k, dtype="bf16", trans_a=False, trans_b=False, batch=1, **_):
        bpe = _BYTES_PER_ELEM.get(str(dtype).lower(), 2)
        flops = 2.0 * m * n * k * batch
        nbytes = float(batch) * (m * k + k * n + m * n) * bpe
        return flops, nbytes

    @staticmethod
    def _sdpa_cost(result, *args, **kwargs):
        md = getattr(result, "metadata", None) or {}
        if "fwd_flops" in md:
            flops = float(md.get("fwd_flops", 0.0)) + float(md.get("bwd_flops", 0.0))
            nbytes = float(md.get("fwd_bytes", 0.0)) + float(md.get("bwd_bytes", 0.0))
            return flops, nbytes
        names = ("batch_size", "num_heads", "seq_len", "head_dim")
        vals = dict(zip(names, args))
        vals.update({k: v for k, v in kwargs.items() if k in names})
        b, h, s, d = (int(vals.get(x, 0) or 0) for x in names)
        causal = 0.5 if kwargs.get("causal", True) else 1.0
        s_kv = int(kwargs.get("seq_len_kv") or s)
        fwd = 4.0 * b * h * s * s_kv * d * causal
        bpe = _BYTES_PER_ELEM.get(str(kwargs.get("dtype", "bf16")).lower(), 2)
        nbytes = 4.0 * b * h * s * d * bpe * 3.0
        return fwd * 3.5, nbytes

    # -- accumulation ----------------------------------------------------
    def record(self, flops: float, nbytes: float, time_ms: float) -> None:
        if time_ms <= 0:
            return
        self.flops += flops
        self.bytes += nbytes
        self.time_ms += time_ms
        self.num_ops += 1
        if self.peak_flops_per_ms > 0 and self.hbm_bytes_per_ms > 0:
            t_compute = flops / self.peak_flops_per_ms
            t_memory = nbytes / self.hbm_bytes_per_ms
            # Roofline: the share of the op that would shrink with a faster clock.
            share = 1.0 if t_compute >= t_memory else (t_compute / t_memory if t_memory > 0 else 1.0)
            self.compute_bound_ms += time_ms * share
        else:
            self.compute_bound_ms += time_ms

    def footprint(self, non_matrix_fraction: float = DEFAULT_NON_MATRIX_FRACTION) -> WorkloadFootprint:
        if self.time_ms <= 0:
            return WorkloadFootprint(non_matrix_fraction=non_matrix_fraction)
        util = self.flops / (self.time_ms * self.peak_flops_per_ms) if self.peak_flops_per_ms > 0 else None
        hbm = self.bytes / (self.time_ms * self.hbm_bytes_per_ms) if self.hbm_bytes_per_ms > 0 else None
        return WorkloadFootprint(
            matrix_utilization=min(1.0, util) if util is not None else DEFAULT_MATRIX_UTILIZATION,
            matrix_hbm_utilization=min(1.0, hbm) if hbm is not None else DEFAULT_MATRIX_HBM_UTILIZATION,
            matrix_clock_sensitivity=min(1.0, self.compute_bound_ms / self.time_ms),
            non_matrix_fraction=non_matrix_fraction,
            num_ops=self.num_ops,
            source="simulation",
        )


@dataclass
class IterationBreakdown:
    """Per-GPU time breakdown of one iteration at the peak clock (ms)."""

    iteration_ms: float
    compute_span_ms: float
    bubble_ms: float = 0.0
    exposed_comm_ms: float = 0.0
    optimizer_ms: float = 0.0
    overlapped_comm_ms: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def build_iteration_phases(breakdown: IterationBreakdown, footprint: WorkloadFootprint) -> List[Phase]:
    busy_compute = max(0.0, breakdown.compute_span_ms - breakdown.bubble_ms)
    non_matrix = busy_compute * min(max(footprint.non_matrix_fraction, 0.0), 1.0)
    matrix = busy_compute - non_matrix
    # Communication hidden under compute keeps the links busy during compute.
    link_during_compute = min(1.0, breakdown.overlapped_comm_ms / busy_compute) if busy_compute > 0 else 0.0
    return [
        Phase(
            name="matrix",
            time_ms=matrix,
            busy=1.0,
            matrix=footprint.matrix_utilization,
            hbm=footprint.matrix_hbm_utilization,
            link=link_during_compute,
            clock_sensitivity=footprint.matrix_clock_sensitivity,
        ),
        Phase(
            name="vector",
            time_ms=non_matrix,
            busy=1.0,
            matrix=0.0,
            hbm=NON_MATRIX_HBM_UTILIZATION,
            link=link_during_compute,
            clock_sensitivity=NON_MATRIX_CLOCK_SENSITIVITY,
        ),
        Phase(
            name="comm",
            time_ms=breakdown.exposed_comm_ms,
            busy=COMM_KERNEL_BUSY,
            hbm=COMM_HBM_UTILIZATION,
            link=1.0,
            clock_sensitivity=0.0,
        ),
        Phase(
            name="optimizer",
            time_ms=breakdown.optimizer_ms,
            busy=1.0,
            hbm=OPTIMIZER_HBM_UTILIZATION,
            clock_sensitivity=OPTIMIZER_CLOCK_SENSITIVITY,
        ),
        Phase(name="bubble", time_ms=breakdown.bubble_ms, busy=0.0, clock_sensitivity=0.0),
    ]
