###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Unit tests for the memory-projection extrapolation primitives
(``core/projection/memory_projection/extrapolation.py``).

The module projects per-rank peak memory at a target cluster from a small
bench measurement plus analytical models.  It is pure arithmetic (the only
side effect is the ``env_overrides`` context manager touching ``os.environ``),
so we exercise it with lightweight fake profilers / training-configs:

  * dataclass ``.total`` rollups;
  * ``env_overrides`` set/restore semantics;
  * ``estimate_comm_buffer_bytes`` (floor / growth / ceiling);
  * ``estimate_deepep_buffer_bytes`` (disabled / enabled / missing fields);
  * ``extract_bench_measurement`` dict → dataclass extraction;
  * ``extrapolate_per_rank_peak`` residual decomposition + point/upper math.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("primus.core.projection.memory_projection.extrapolation")

from primus.core.projection.memory_projection.extrapolation import (  # noqa: E402
    _COMM_BASE_BYTES,
    _COMM_CEILING_BYTES,
    ActivationBreakdown,
    AnalyticalBreakdown,
    BenchMeasurement,
    StaticBreakdown,
    env_overrides,
    estimate_comm_buffer_bytes,
    estimate_deepep_buffer_bytes,
    extract_bench_measurement,
    extrapolate_per_rank_peak,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dataclass rollups
# ─────────────────────────────────────────────────────────────────────────────


def test_static_breakdown_total():
    s = StaticBreakdown(params_bytes=10, grads_bytes=20, optimizer_bytes=30)
    assert s.total == 60


def test_activation_breakdown_total():
    a = ActivationBreakdown(
        transformer_layers_bytes=100,
        embedding_bytes=10,
        output_bytes=5,
        loss_bytes=1,
    )
    assert a.total == 116


def test_analytical_breakdown_total():
    ab = AnalyticalBreakdown(
        static=StaticBreakdown(params_bytes=10, grads_bytes=10, optimizer_bytes=10),
        activations=ActivationBreakdown(transformer_layers_bytes=100),
        deepep_buffers_bytes=7,
        comm_buffers_bytes=3,
    )
    assert ab.total == 30 + 100 + 7 + 3


# ─────────────────────────────────────────────────────────────────────────────
# env_overrides
# ─────────────────────────────────────────────────────────────────────────────


def test_env_overrides_sets_and_restores():
    os.environ["PRIMUS_TEST_KEEP"] = "orig"
    os.environ.pop("PRIMUS_TEST_NEW", None)
    with env_overrides(PRIMUS_TEST_KEEP="changed", PRIMUS_TEST_NEW=5):
        assert os.environ["PRIMUS_TEST_KEEP"] == "changed"
        assert os.environ["PRIMUS_TEST_NEW"] == "5"  # coerced to str
    # Restored afterwards.
    assert os.environ["PRIMUS_TEST_KEEP"] == "orig"
    assert "PRIMUS_TEST_NEW" not in os.environ
    os.environ.pop("PRIMUS_TEST_KEEP", None)


def test_env_overrides_removes_var_when_value_none():
    os.environ["PRIMUS_TEST_DROP"] = "present"
    with env_overrides(PRIMUS_TEST_DROP=None):
        assert "PRIMUS_TEST_DROP" not in os.environ
    assert os.environ["PRIMUS_TEST_DROP"] == "present"
    os.environ.pop("PRIMUS_TEST_DROP", None)


# ─────────────────────────────────────────────────────────────────────────────
# estimate_comm_buffer_bytes
# ─────────────────────────────────────────────────────────────────────────────


def test_comm_buffer_zero_for_single_rank():
    assert estimate_comm_buffer_bytes(1) == 0
    assert estimate_comm_buffer_bytes(0) == 0


def test_comm_buffer_grows_with_world_size():
    small = estimate_comm_buffer_bytes(8)
    big = estimate_comm_buffer_bytes(64)
    assert big > small
    assert small >= _COMM_BASE_BYTES  # base + growth


def test_comm_buffer_respects_ceiling():
    assert estimate_comm_buffer_bytes(10_000_000) == _COMM_CEILING_BYTES


# ─────────────────────────────────────────────────────────────────────────────
# estimate_deepep_buffer_bytes
# ─────────────────────────────────────────────────────────────────────────────


def _deepep_tc(*, enabled=True, hidden=128, topk=2, mbs=2, seq=512, tp=1, cp=1):
    return SimpleNamespace(
        model_parallel_config=SimpleNamespace(
            use_turbo_deepep=enabled,
            tensor_model_parallel_size=tp,
            context_model_parallel_size=cp,
        ),
        model_config=SimpleNamespace(
            use_turbo_deepep=enabled,
            hidden_size=hidden,
            moe_router_topk=topk,
        ),
        runtime_config=SimpleNamespace(micro_batch_size=mbs, sequence_length=seq),
    )


def test_deepep_zero_when_disabled():
    assert estimate_deepep_buffer_bytes(_deepep_tc(enabled=False)) == 0


def test_deepep_zero_when_configs_missing():
    assert estimate_deepep_buffer_bytes(SimpleNamespace()) == 0


def test_deepep_zero_when_hidden_or_topk_invalid():
    assert estimate_deepep_buffer_bytes(_deepep_tc(hidden=0)) == 0
    assert estimate_deepep_buffer_bytes(_deepep_tc(topk=0)) == 0
    assert estimate_deepep_buffer_bytes(_deepep_tc(seq=0)) == 0


def test_deepep_closed_form_value():
    # tokens_per_rank = mbs*seq // (tp*cp) = 2*512 // 1 = 1024
    # num_max_dispatch = int(1024 * topk(2) * OVERPROVISION(2.0)) = 4096
    # per_buffer = 4096 * hidden(128) * BF16(2) = 1_048_576
    # total = 2 * per_buffer = 2_097_152
    got = estimate_deepep_buffer_bytes(_deepep_tc(hidden=128, topk=2, mbs=2, seq=512, tp=1, cp=1))
    assert got == 2 * (4096 * 128 * 2)


def test_deepep_scales_down_with_tp_cp():
    # Doubling TP halves tokens_per_rank → halves the buffer.
    base = estimate_deepep_buffer_bytes(_deepep_tc(tp=1))
    with_tp = estimate_deepep_buffer_bytes(_deepep_tc(tp=2))
    assert with_tp == base // 2


# ─────────────────────────────────────────────────────────────────────────────
# extract_bench_measurement
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_bench_measurement_picks_one_per_layer_type():
    profiling_results = {
        "_memory_benchmark": {
            "global_peak_allocated_bytes": 1000,
            "global_peak_reserved_bytes": 1500,
        },
        0: {"type": "dense", "activation_memory_bytes": 111},
        1: {"type": "moe", "activation_memory_bytes": 222},
        2: {"type": "moe", "activation_memory_bytes": 999},  # duplicate type ignored
        3: {"type": "other", "activation_memory_bytes": 5},  # non dense/moe ignored
        "embedding": {"activation_memory_bytes": 7},
        "output": {"activation_memory_bytes": 9},
    }
    bm = extract_bench_measurement(profiling_results)
    assert bm.global_peak_allocated_bytes == 1000
    assert bm.global_peak_reserved_bytes == 1500
    assert bm.per_layer_activation_bytes == {"dense": 111, "moe": 222}
    assert bm.embedding_activation_bytes == 7
    assert bm.output_activation_bytes == 9


def test_extract_bench_measurement_empty():
    bm = extract_bench_measurement({})
    assert bm.global_peak_allocated_bytes == 0
    assert bm.per_layer_activation_bytes == {}


# ─────────────────────────────────────────────────────────────────────────────
# extrapolate_per_rank_peak — residual decomposition + final math
# ─────────────────────────────────────────────────────────────────────────────


class _FakeProfiler:
    """Minimal stand-in for LanguageModelProfiler.

    ``extrapolate_per_rank_peak`` only calls ``estimated_num_params``,
    ``get_dp_size`` and ``estimated_activation_memory`` on the point-estimate
    path when there is no per-layer correction to apply (empty bench
    per-layer dict), so that is all we implement here.
    """

    def __init__(self, n_params: int, dp_size: int, act_bytes: int):
        self._n = n_params
        self._dp = dp_size
        self._act = act_bytes

    def estimated_num_params(self, rank: int = 0) -> int:
        return self._n

    def get_dp_size(self) -> int:
        return self._dp

    def estimated_activation_memory(self, batch_size: int, seq_len: int) -> int:
        return self._act


def _sum_breakdown_total(d: dict) -> int:
    static = d["static"]
    acts = d["activations"]
    return (
        static["params_bytes"]
        + static["grads_bytes"]
        + static["optimizer_bytes"]
        + acts["transformer_layers_bytes"]
        + acts["embedding_bytes"]
        + acts["output_bytes"]
        + acts["loss_bytes"]
        + d["deepep_buffers_bytes"]
        + d["comm_buffers_bytes"]
    )


def test_extrapolate_per_rank_peak_decomposition():
    bench_profiler = _FakeProfiler(n_params=1000, dp_size=1, act_bytes=100)
    target_profiler = _FakeProfiler(n_params=2000, dp_size=2, act_bytes=500)

    # DeepEP disabled on both → no deepep term to reason about.
    tc = SimpleNamespace(
        model_parallel_config=SimpleNamespace(use_turbo_deepep=False),
        model_config=SimpleNamespace(use_turbo_deepep=False),
    )

    bench = BenchMeasurement(
        global_peak_allocated_bytes=20000,
        global_peak_reserved_bytes=30000,
        # Empty per-layer → no activation correction (applied_correction False).
        per_layer_activation_bytes={},
    )

    proj = extrapolate_per_rank_peak(
        bench_profiler=bench_profiler,
        target_profiler=target_profiler,
        bench=bench,
        bench_training_config=tc,
        target_training_config=tc,
        bench_nnodes=1,
        bench_gpus_per_node=1,  # bench world = 1 → comm = 0
        target_nnodes=1,
        target_gpus_per_node=2,
        batch_size=1,
        seq_len=4096,
        safety_margin=0.05,
    )

    bd = proj.breakdown
    # No correction was applied.
    assert bd["activation_correction"]["applied_correction"] is False

    # analytical_at_bench = static(2000+2000+10000) + act(100) + deepep(0) + comm(0) = 14100
    # framework_overhead = reserved - allocated - comm_at_bench(0) = 30000-20000 = 10000
    assert bd["framework_overhead_bytes"] == 10000
    # live_tensor_excess = allocated(20000) - analytical_bench.total(14100) = 5900
    assert bd["live_tensor_excess_bytes"] == 5900
    assert bd["residual_total_bytes"] == 15900

    # Point/upper math anchored on the corrected target breakdown total.
    target_total = _sum_breakdown_total(bd["analytical_at_target_corrected"])
    assert proj.point_estimate_bytes == target_total + 15900
    assert proj.upper_bound_bytes == target_total + int(15900 * 1.05)
    # Point estimate must never exceed the upper bound.
    assert proj.point_estimate_bytes <= proj.upper_bound_bytes


def test_extrapolate_clamps_live_tensor_excess_to_zero():
    # Analytical model already over-predicts at bench (huge params) → the
    # live-tensor excess term must clamp to 0 rather than go negative.
    bench_profiler = _FakeProfiler(n_params=10_000_000, dp_size=1, act_bytes=0)
    target_profiler = _FakeProfiler(n_params=10_000_000, dp_size=1, act_bytes=0)
    tc = SimpleNamespace(
        model_parallel_config=SimpleNamespace(use_turbo_deepep=False),
        model_config=SimpleNamespace(use_turbo_deepep=False),
    )
    bench = BenchMeasurement(
        global_peak_allocated_bytes=1000,
        global_peak_reserved_bytes=1200,
        per_layer_activation_bytes={},
    )
    proj = extrapolate_per_rank_peak(
        bench_profiler=bench_profiler,
        target_profiler=target_profiler,
        bench=bench,
        bench_training_config=tc,
        target_training_config=tc,
        bench_nnodes=1,
        bench_gpus_per_node=1,
        target_nnodes=1,
        target_gpus_per_node=1,
        batch_size=1,
        seq_len=128,
        safety_margin=0.05,
    )
    assert proj.breakdown["live_tensor_excess_bytes"] == 0
