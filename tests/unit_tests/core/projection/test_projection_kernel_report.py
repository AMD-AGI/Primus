###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import json
from types import SimpleNamespace

import pytest

from primus.core.projection import kernel_report
from primus.core.projection.simulation_backends.base import (
    GEMMSimulationBackend,
    SDPASimulationBackend,
    SimulationResult,
)


class _FlopGEMM(GEMMSimulationBackend):
    """1 ms per 1e9 multiply-adds, so every shape has a distinct, checkable time."""

    def name(self):
        return "flop"

    def is_available(self):
        return True

    def simulate_gemm(self, m, n, k, dtype="bf16", trans_a=False, trans_b=False, batch=1):
        return SimulationResult(forward_time_ms=m * n * k * batch / 1e9)


class _FixedSDPA(SDPASimulationBackend):
    def name(self):
        return "fixed"

    def is_available(self):
        return True

    def simulate_sdpa(
        self,
        batch_size,
        num_heads,
        seq_len,
        head_dim,
        causal=True,
        dtype="bf16",
        seq_len_kv=None,
        num_heads_kv=None,
        head_dim_v=None,
    ):
        return SimulationResult(forward_time_ms=1.0, backward_time_ms=2.5)


@pytest.fixture
def ledger():
    led = kernel_report.Ledger()
    kernel_report.activate(led)
    yield led
    kernel_report.activate(None)


def _config(num_layers=4, pp=1, vpp=1, moe_pattern=None, gbs=16, mbs=2, recompute=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            num_layers=num_layers, moe_pattern=moe_pattern or [1] * num_layers, compress_ratios=None
        ),
        model_parallel_config=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=vpp,
            context_model_parallel_size=1,
            expert_model_parallel_size=8,
            decoder_first_pipeline_num_layers=None,
            decoder_last_pipeline_num_layers=None,
            pipeline_model_parallel_layout=None,
            recompute_granularity="full" if recompute else None,
            recompute_method="block" if recompute else None,
            recompute_num_layers=recompute or 0,
        ),
        runtime_config=SimpleNamespace(global_batch_size=gbs, micro_batch_size=mbs, sequence_length=4096),
    )


def test_term_is_a_pass_through_when_not_recording():
    assert kernel_report.active() is None
    assert kernel_report.term("x", "fwd", 1.25) == 1.25
    assert kernel_report.extras("moe", "fwd") == 0.0
    gemm, sdpa = _FlopGEMM(), _FixedSDPA()
    assert kernel_report.wrap_backends(gemm, sdpa) == (gemm, sdpa)


def test_most_specific_calibration_and_speedup_apply():
    led = kernel_report.Ledger(
        calibration={"factors": [{"op": "g", "factor": 2.0}, {"op": "g", "phase": "wgrad", "factor": 3.0}]},
        speedups={"speedups": [{"op": "g", "dims": {"m": 8}, "speedup": 4.0}]},
    )
    assert led.adjust("moe", "g", "fwd", 1.0, {"m": 4})[0] == 2.0
    assert led.adjust("moe", "g", "wgrad", 1.0, {"m": 4})[0] == 3.0
    assert led.adjust("moe", "g", "wgrad", 1.0, {"m": 8})[0] == 0.75
    assert led.adjust("moe", "other", "fwd", 1.0, {})[0] == 1.0


def test_a_measured_shape_applies_to_any_op_of_its_kind_and_dtype():
    entry = {"op": "*", "kind": "gemm", "dtype": "fp8", "dims": {"m": 8, "n": 16, "k": 32}}
    led = kernel_report.Ledger(speedups={"speedups": [dict(entry, baseline_us=30.0, tuned_us=20.0)]})
    dims = {"m": 8, "n": 16, "k": 32, "dtype": "fp8"}
    assert led.adjust("moe", "attn_o", "fwd", 3.0, dims, kind="gemm")[0] == pytest.approx(2.0)
    assert led.adjust("moe", "attn_o", "fwd", 3.0, dims, kind="grouped_gemm")[0] == 3.0
    assert led.adjust("moe", "attn_o", "fwd", 3.0, dict(dims, dtype="bf16"), kind="gemm")[0] == 3.0


def test_per_step_extras_scale_with_the_iteration_terms(ledger):
    ledger.calibration = {
        "extras": [{"scope": "iteration", "phase": "step", "op": "param_gather", "share": 2.0}]
    }
    kernel_report.term("optimizer", "step", 1.5, scope_name="iteration")
    assert kernel_report.extras("iteration", "step") == pytest.approx(3.0)


def test_shared_gemm_helpers_are_labelled_by_projection_and_phase(ledger):
    gemm, _ = kernel_report.wrap_backends(_FlopGEMM(), None)
    with kernel_report.scope("dense"):
        mlp = gemm.simulate_mlp_gemms(1024, 512, 2048, swiglu=True)
        with kernel_report.module("moe_shared"):
            gemm.simulate_mlp_gemms(1024, 512, 256)
        gemm.simulate_attention_gemms(1024, 512, 8, 64, 2)
    labels = [(r.op, r.phase) for r in ledger.rows]
    assert labels[:9] == [
        ("mlp_gate", "fwd"),
        ("mlp_up", "fwd"),
        ("mlp_down", "fwd"),
        ("mlp_gate", "dgrad"),
        ("mlp_gate", "wgrad"),
        ("mlp_up", "dgrad"),
        ("mlp_up", "wgrad"),
        ("mlp_down", "dgrad"),
        ("mlp_down", "wgrad"),
    ]
    assert {op for op, _ in labels[9:15]} == {"moe_shared_up", "moe_shared_down"}
    assert {op for op, _ in labels[15:]} == {"attn_q", "attn_k", "attn_v", "attn_o"}
    assert all(r.scope == "dense" for r in ledger.rows)
    wgrad = ledger.rows[4]
    assert wgrad.dims == {"m": 512, "n": 2048, "k": 1024, "batch": 1, "dtype": "bf16", "layout": "TN"}
    assert mlp.forward_time_ms == pytest.approx(sum(r.ms for r in ledger.rows[:3]))


def test_sdpa_is_recorded_per_direction_and_adjusted(ledger):
    ledger.speedups = {"speedups": [{"op": "attn_sdpa", "phase": "bwd", "speedup": 2.0}]}
    _, sdpa = kernel_report.wrap_backends(None, _FixedSDPA())
    with kernel_report.scope("moe"):
        res = sdpa.simulate_sdpa(2, 8, 4096, 64, num_heads_kv=1)
    assert (res.forward_time_ms, res.backward_time_ms) == (1.0, 1.25)
    assert [(r.op, r.phase, r.kind) for r in ledger.rows] == [
        ("attn_sdpa", "fwd", "attention"),
        ("attn_sdpa", "bwd", "attention"),
    ]
    assert ledger.rows[0].dims["kv_heads"] == 1


def test_extras_are_a_share_of_the_modelled_time(ledger):
    ledger.calibration = {
        "factors": [{"op": "a", "factor": 3.0}],
        "extras": [{"scope": "moe", "phase": "bwd", "op": "fp8_cast", "share": 0.5}],
    }
    with kernel_report.scope("moe"):
        assert kernel_report.term("a", "dgrad", 2.0) == 6.0
        kernel_report.term("a", "wgrad", 2.0)
        kernel_report.term("a", "fwd", 100.0)
    assert kernel_report.extras("moe", "bwd") == pytest.approx(2.0)
    assert ledger.rows[-1].op == "fp8_cast" and ledger.rows[-1].src == "calibration"


def test_report_counts_calls_per_stage(ledger):
    with kernel_report.scope("moe"):
        start = kernel_report.mark()
        kernel_report.term("expert", "fwd", 1.0, kind="gemm")
        kernel_report.repeat_since(start, 4)
        kernel_report.term("router", "bwd", 0.5)
    with kernel_report.scope("embedding"):
        kernel_report.term("embedding", "fwd", 0.1)
    with kernel_report.scope("output"):
        kernel_report.term("lm_head", "fwd", 2.0)
    kernel_report.term("optimizer", "step", 3.0, scope_name="iteration")

    doc = kernel_report.build_report(ledger, _config(num_layers=4), num_gpus=8)
    calls = {(r["stage"], r["op"]): r["calls"] for r in doc["kernels"]}
    # 4 layers x 1 microbatch (gbs 16 / (mbs 2 x dp 8)), the expert loop 4 times per layer
    assert calls == {
        (0, "expert"): 16,
        (0, "router"): 4,
        (0, "embedding"): 1,
        (0, "lm_head"): 1,
        (0, "optimizer"): 1,
    }
    assert doc["deployment"]["microbatches"] == 1
    json.dumps(doc)

    doc = kernel_report.build_report(ledger, _config(num_layers=4, pp=2, recompute=1), num_gpus=8)
    calls = {(r["stage"], r["op"]): r["calls"] for r in doc["kernels"]}
    # dp 4 -> 2 microbatches; 2 layers per stage, 1 of them re-runs its forward
    assert calls[(0, "expert")] == (2 + 1) * 2 * 4
    assert calls[(1, "router")] == 2 * 2
    assert (0, "embedding") in calls and (1, "embedding") not in calls
    assert (1, "lm_head") in calls and (0, "lm_head") not in calls
    assert calls[(0, "optimizer")] == calls[(1, "optimizer")] == 1


def test_documents_are_checked_for_their_schema(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps({"schema": "something.else"}))
    with pytest.raises(ValueError):
        kernel_report.Ledger.from_paths(calibration=str(path))
