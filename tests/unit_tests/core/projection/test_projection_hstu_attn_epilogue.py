###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""HSTU attention epilogue: fixed Gelem/s rate vs vector-throughput pricing.

The FAv3 tile matmuls are stubbed so the tests run without a GEMM backend.
"""

import pytest

from primus.core.projection.simulation_backends import hstu_attention_simulator as has
from primus.core.projection.simulation_backends.base import SimulationResult

B, H, L, D = 8, 4, 1000, 128
SCORE_ELEMS = B * H * L * L * 0.5
MATMUL_FWD, MATMUL_BWD = 2.0, 5.0


@pytest.fixture(autouse=True)
def _stub_tile_matmuls(monkeypatch):
    monkeypatch.setattr(has.SDPASimulator, "__init__", lambda self, **kw: setattr(self, "_mode", "stub"))
    monkeypatch.setattr(
        has.SDPASimulator,
        "simulate_sdpa",
        lambda self, **kw: SimulationResult(
            forward_time_ms=MATMUL_FWD, backward_time_ms=MATMUL_BWD, metadata={"fwd_flops": 1.0}
        ),
    )


def _run(**kw):
    sim = has.HSTUAttentionSimulator(gemm_backend="stub", **kw)
    return sim.simulate_sdpa(batch_size=B, num_heads=H, seq_len=L, head_dim=D, causal=True, head_dim_v=D)


def test_fixed_rate_epilogue_is_arch_independent():
    r = _run(epilogue_gelem_fwd=1000.0, epilogue_gelem_bwd=500.0)
    assert r.forward_time_ms == pytest.approx(MATMUL_FWD + SCORE_ELEMS / 1000e6)
    assert r.backward_time_ms == pytest.approx(MATMUL_BWD + SCORE_ELEMS / 500e6)


def test_vector_epilogue_scales_with_vector_throughput():
    vf = 50e12
    r1 = _run(epilogue_flops_per_elem_fwd=30.0, epilogue_flops_per_elem_bwd=25.0, vector_flops=vf)
    assert r1.forward_time_ms == pytest.approx(MATMUL_FWD + SCORE_ELEMS * 30.0 / vf * 1e3)
    assert r1.backward_time_ms == pytest.approx(MATMUL_BWD + SCORE_ELEMS * 25.0 / vf * 1e3)
    r2 = _run(epilogue_flops_per_elem_fwd=30.0, epilogue_flops_per_elem_bwd=25.0, vector_flops=2 * vf)
    assert r2.forward_time_ms - MATMUL_FWD == pytest.approx((r1.forward_time_ms - MATMUL_FWD) / 2)


def test_vector_epilogue_falls_back_without_vector_flops():
    r = _run(
        epilogue_gelem_fwd=1000.0,
        epilogue_gelem_bwd=500.0,
        epilogue_flops_per_elem_fwd=30.0,
        epilogue_flops_per_elem_bwd=25.0,
        vector_flops=None,
    )
    assert r.forward_time_ms == pytest.approx(MATMUL_FWD + SCORE_ELEMS / 1000e6)
