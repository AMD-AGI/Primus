###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DLRM / HSTU backward GEMMs are priced as explicit dgrad + wgrad GEMMs.

Uses a stub GEMM backend whose time depends on the shape, so a flat
``backward = 2 x forward`` would not reproduce the expected values.
"""

from primus.core.projection.module_profilers.dlrm import DLRMProfiler
from primus.core.projection.module_profilers.hstu import HSTULayerProfiler
from primus.core.projection.simulation_backends.base import SimulationResult


class _ShapeGemm:
    """Time = M + 10 N + 100 K (ms); records every call."""

    def __init__(self):
        self.calls = []

    def simulate_gemm(self, m, n, k, dtype="bf16", **kw):
        self.calls.append((m, n, k))
        return SimulationResult(forward_time_ms=float(m + 10 * n + 100 * k))


def _t(m, n, k):
    return float(m + 10 * n + 100 * k)


def test_hstu_gemm_backward_is_dgrad_plus_wgrad():
    prof = object.__new__(HSTULayerProfiler)
    prof._gemm_backend = _ShapeGemm()
    fwd, bwd = prof._gemm_fwd_bwd(4096, 2048, 512, "bf16")
    assert prof._gemm_backend.calls == [(4096, 2048, 512), (4096, 512, 2048), (512, 2048, 4096)]
    assert fwd == _t(4096, 2048, 512)
    assert bwd == _t(4096, 512, 2048) + _t(512, 2048, 4096)
    assert bwd != 2.0 * fwd


def test_dlrm_mlp_backward_is_dgrad_plus_wgrad():
    prof = object.__new__(DLRMProfiler)
    prof._gemm_backend = _ShapeGemm()
    layers = [(512, 256), (256, 1)]
    fwd, bwd = prof._mlp_step_ms(layers, 1024, "bf16")
    assert prof._gemm_backend.calls == [
        (1024, 256, 512),
        (1024, 512, 256),
        (512, 256, 1024),
        (1024, 1, 256),
        (1024, 256, 1),
        (256, 1, 1024),
    ]
    assert fwd == _t(1024, 256, 512) + _t(1024, 1, 256)
    assert bwd == _t(1024, 512, 256) + _t(512, 256, 1024) + _t(1024, 256, 1) + _t(256, 1, 1024)
