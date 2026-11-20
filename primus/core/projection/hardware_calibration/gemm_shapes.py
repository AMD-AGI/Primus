###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Extract the GEMM shapes a workload actually runs.

The profiler tree already knows every GEMM a layer issues -- it prices them
one by one in simulate mode.  Rather than re-deriving those shapes here (and
drifting from the simulator for MLA, MoE, shared experts, ...), the tree is
run once against a backend that records each ``simulate_gemm`` call instead
of pricing it.
"""

from __future__ import annotations

import contextlib
import io
from collections import Counter
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Tuple

from primus.core.projection.simulation_backends.base import (
    GEMMSimulationBackend,
    SimulationResult,
)

# A GEMM this small is launch-bound: its time says nothing about the clock.
_MIN_CALIBRATION_GFLOP = 1.0
# Keep the calibration tensors well clear of whatever the bench host has free.
_MAX_CALIBRATION_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class GemmShape:
    """``C[batch, M, N] = A[batch, M, K] @ B[batch, K, N]`` computed in *dtype*."""

    m: int
    n: int
    k: int
    batch: int = 1
    dtype: str = "bf16"

    @property
    def flops(self) -> float:
        return 2.0 * self.m * self.n * self.k * self.batch

    @property
    def nbytes(self) -> int:
        """A, B and C footprint; the output is bf16 for sub-16-bit inputs."""
        in_bytes = {"fp32": 4, "bf16": 2, "fp16": 2}.get(self.dtype, 1)
        out_bytes = max(in_bytes, 2)
        return self.batch * (in_bytes * (self.m * self.k + self.k * self.n) + out_bytes * self.m * self.n)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


# Used when the workload's shapes cannot be extracted.
REFERENCE_GEMM = GemmShape(8192, 8192, 8192)


class RecordingGEMMBackend(GEMMSimulationBackend):
    """A GEMM backend that records shapes and returns a nominal time.

    The returned time is small but non-zero, so profiler code that divides by
    a simulated time keeps working.
    """

    _NOMINAL_MS = 1e-3

    def __init__(self):
        self.shapes: Counter = Counter()
        self.dtypes: Counter = Counter()

    def name(self) -> str:
        return "gemm-shape-recorder"

    def is_available(self) -> bool:
        return True

    @property
    def hbm_bandwidth_gbps(self) -> Optional[float]:
        return 5300.0

    def vector_flops(self) -> Optional[float]:
        return 1.0e14

    def simulate_gemm(self, m, n, k, dtype="bf16", trans_a=False, trans_b=False, batch=1) -> SimulationResult:
        shape = GemmShape(int(m), int(n), int(k), max(int(batch), 1), str(dtype))
        if shape.m > 0 and shape.n > 0 and shape.k > 0:
            self.shapes[shape] += 1
            self.dtypes[str(dtype)] += 1
        return SimulationResult(forward_time_ms=self._NOMINAL_MS, metadata={"backend": self.name()})


def collect_workload_gemms(training_config) -> Tuple[Dict[GemmShape, int], Dict[str, int]]:
    """Return ``({shape: occurrences}, {dtype: occurrences})`` for this rank's layers.

    Occurrences count each GEMM once per representative layer type (dense /
    MoE), forward and backward, which is what the selection weights by.
    """
    from primus.core.projection.module_profilers.language_model import (
        build_profiler,
        get_language_model_profiler_spec,
    )

    recorder = RecordingGEMMBackend()
    runtime = training_config.runtime_config
    with contextlib.redirect_stdout(io.StringIO()):
        profiler = build_profiler(get_language_model_profiler_spec(training_config))
        profiler.set_simulation_backends(recorder, None)
        profiler.run_layer_benchmark(
            model=None,
            batch_size=runtime.micro_batch_size,
            seq_len=runtime.sequence_length,
        )
    return dict(recorder.shapes), dict(recorder.dtypes)


def select_calibration_gemms(
    shape_counts: Dict[GemmShape, int],
    max_gemms: int = 4,
    min_gflop: float = _MIN_CALIBRATION_GFLOP,
    max_bytes: int = _MAX_CALIBRATION_BYTES,
) -> List[Tuple[GemmShape, int]]:
    """Pick the GEMMs that dominate the workload's FLOPs and are clock-bound.

    Ranked by total FLOPs (``flops x occurrences``); shapes too small to be
    compute-bound or too large to allocate beside the bench are skipped.
    Falls back to an 8192^3 reference GEMM, in the workload's dominant dtype,
    when nothing qualifies.
    """
    candidates = [
        (shape, count)
        for shape, count in shape_counts.items()
        if shape.flops >= min_gflop * 1e9 and shape.nbytes <= max_bytes
    ]
    candidates.sort(key=lambda item: (item[0].flops * item[1], item[0].flops), reverse=True)
    selected = candidates[: max(max_gemms, 1)]
    if not selected:
        selected = [(replace(REFERENCE_GEMM, dtype=dominant_dtype(shape_counts)), 1)]
    return selected


def dominant_dtype(shape_counts: Dict[GemmShape, int]) -> str:
    """The dtype carrying most of the workload's GEMM FLOPs (bf16 if none)."""
    flops: Counter = Counter()
    for shape, count in shape_counts.items():
        flops[shape.dtype] += shape.flops * count
    return flops.most_common(1)[0][0] if flops else "bf16"
