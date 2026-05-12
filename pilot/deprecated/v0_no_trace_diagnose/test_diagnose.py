"""Unit tests for ``pilot.tools.diagnose`` (rule engine).

Six fixtures exercising one rule path each:

1. ``test_R0_oom_failure`` — R0 routes to MEMORY_BOUND with REPLAN, no candidate scan.
2. ``test_R0_hang_status`` — R0 routes to COMM_BOUND with HANG extended class.
3. ``test_R1_mem_tight`` — R1 fires when peak mem_pct > 0.92.
4. ``test_R3b_profilerless_comm_bound`` — the case our auto-heuristic missed:
   low TFLOPs vs peak + alltoall plan -> COMM_BOUND.
5. ``test_R4_compute_headroom`` — high TFLOPs/peak ratio -> COMPUTE_BOUND.
6. ``test_axes_respect_hard_constraints`` — engine refuses to emit a candidate
   that would violate the deepep mutex pair.
"""

from __future__ import annotations

from typing import Any

import pytest

from pilot.tools import diagnose as engine


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    status: str = "completed",
    iter_ms: list[float] | None = None,
    tflops: list[float] | None = None,
    oom: bool = False,
    nccl: bool = False,
    cuda: bool = False,
    python: bool = False,
    hang: bool = False,
    loss_finite: bool = True,
    silent_for_s: float | None = None,
) -> dict[str, Any]:
    iter_ms = iter_ms or [400.0, 410.0, 405.0, 408.0, 406.0]
    tflops = tflops or [200.0, 205.0, 202.0, 204.0, 203.0]
    return {
        "schema_version": "1.0",
        "run_id": "test_run",
        "snapshot_at": "2026-05-09T00:00:00+00:00",
        "status": status,
        "process": {"alive": False, "pid": 123, "exit_code": 0, "wallclock_s": 60.0},
        "progress": {
            "current_iter": len(iter_ms),
            "total_iters": len(iter_ms),
            "pct": 100.0,
            "iters_per_min": None,
            "last_iter_at": "2026-05-09T00:00:00+00:00",
            "silent_for_s": silent_for_s,
        },
        "metrics": {
            "latest": {
                "loss": 1.23,
                "iter_time_ms": iter_ms[-1],
                "tflops": tflops[-1],
                "consumed_samples": None,
                "learning_rate": None,
                "grad_norm": None,
            },
            "history": {
                "iters": list(range(1, len(iter_ms) + 1)),
                "loss": [1.23] * len(iter_ms),
                "iter_time_ms": iter_ms,
                "tflops": tflops,
            },
            "loss_finite": loss_finite,
        },
        "symptoms": {
            "hang_suspected": hang,
            "hang_threshold_s": 120.0,
            "oom_detected": oom,
            "nccl_error": nccl,
            "cuda_error": cuda,
            "python_error": python,
            "loss_nan_or_inf": not loss_finite,
            "evidence": [],
        },
        "log": {"ref": "/nonexistent/train.log", "bytes": 0, "lines": 0, "tailed_bytes": 0},
    }


def _cluster_profile(*, peak_bf16: float = 1200.0, peak_fp8: float | None = 2400.0,
                     gpus_per_node: int = 8, nodes_total: int = 1) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "cluster_id": "test",
        "collected_at": "2026-05-08T00:00:00+00:00",
        "nodes_total": nodes_total,
        "gpus_per_node": gpus_per_node,
        "compute": {
            "peak_tflops_bf16": peak_bf16,
            "peak_tflops_fp8": peak_fp8,
            "hbm_bandwidth_gbs": 5300.0,
            "hbm_capacity_gb": 192.0,
        },
    }


def _plan(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "context_parallel_size": 1,
        "micro_batch_size": 4,
        "global_batch_size": 32,
        "seq_length": 4096,
        "fp8_format": "e4m3",
    }
    base.update(overrides)
    return {"modules": {"pre_trainer": {"overrides": base}}}


# ---------------------------------------------------------------------------
# 1. R0: OOM failure
# ---------------------------------------------------------------------------


def test_R0_oom_failure() -> None:
    snap = _snapshot(status="failed", oom=True)
    report = engine.run(snap, _cluster_profile(), _plan(micro_batch_size=8))

    assert report["bottleneck"] == "MEMORY_BOUND"
    assert report["confidence"] >= 0.9
    assert report["meta"]["rule_id"] == "R0_OOM"
    assert report["suggested_transition"]["to"] == "OPTIMIZE_LOOP.REPLAN"
    assert report["suggested_transition"]["counts_against_budget"] is False
    assert report["candidate_axes"] == []
    assert any("OOM" in e for e in report["evidence"])


# ---------------------------------------------------------------------------
# 2. R0: Hang
# ---------------------------------------------------------------------------


def test_R0_hang_status() -> None:
    snap = _snapshot(status="hung", hang=True, silent_for_s=400.0)
    report = engine.run(snap, _cluster_profile(), _plan(expert_model_parallel_size=8))

    assert report["bottleneck"] == "COMM_BOUND"
    assert report["meta"]["bottleneck_extended"] == "HANG"
    assert report["meta"]["rule_id"] == "R0_HANG"
    assert report["confidence"] == 0.90
    assert report["suggested_transition"]["to"] == "OPTIMIZE_LOOP.REPLAN"
    assert report["suggested_transition"]["counts_against_budget"] is False


# ---------------------------------------------------------------------------
# 3. R1: Memory tight (peak mem > 0.92)
# ---------------------------------------------------------------------------


def test_R1_mem_tight(tmp_path: Any) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "[2026-05-09 00:00:00.000000] [INFO] iteration 1/10 ... usage_ratio: 95.5 %\n"
        "[2026-05-09 00:00:01.000000] [INFO] iteration 2/10 ... usage_ratio: 95.7 %\n"
    )
    snap = _snapshot()
    snap["log"]["ref"] = str(log_path)

    report = engine.run(snap, _cluster_profile(), _plan())

    assert report["bottleneck"] == "MEMORY_BOUND"
    assert report["meta"]["rule_id"] == "R1_MEM_TIGHT"
    assert report["meta"]["mem_pct"] == pytest.approx(0.957, abs=1e-3)
    assert any(a["axis"] == "PYTORCH_HIP_ALLOC_CONF" for a in report["candidate_axes"])


# ---------------------------------------------------------------------------
# 4. R3b: profiler-less COMM detection (the case our auto-heuristic missed)
# ---------------------------------------------------------------------------


def test_R3b_profilerless_comm_bound() -> None:
    """Mirrors the deepseek_v2_lite baseline pattern: ~20% of measured peak,
    EP=8 single-node alltoall heavy. The legacy heuristic returned COMPUTE.
    The engine MUST classify this as COMM_BOUND."""

    snap = _snapshot(
        iter_ms=[32000.0, 32100.0, 32050.0, 32030.0, 32040.0, 32045.0, 32035.0],
        tflops=[239.0, 240.0, 239.5, 240.5, 239.8, 240.1, 239.9],
    )
    profile = _cluster_profile(peak_bf16=1219.2, peak_fp8=None)  # FP8 unmeasured -> falls back to BF16
    plan = _plan(
        expert_model_parallel_size=8,
        num_experts=64,
        use_turbo_deepep=True,
        turbo_deepep_use_comm_stream=False,
        turbo_deepep_num_cu=64,
        moe_router_force_load_balancing=True,
    )

    report = engine.run(snap, profile, plan)

    assert report["bottleneck"] == "COMM_BOUND"
    assert "R3b" in report["meta"]["rule_id"]
    assert report["meta"]["tflops_pct_of_peak"] == pytest.approx(239.9 / 1219.2, abs=1e-3)
    assert any(a["axis"] == "turbo_deepep_use_comm_stream" for a in report["candidate_axes"])
    assert any(a["axis"] == "turbo_deepep_num_cu" for a in report["candidate_axes"])
    assert report["suggested_strategy"] in ("Per-Plan", "Successive_Halving")


# ---------------------------------------------------------------------------
# 5. R4: COMPUTE_BOUND when util is high
# ---------------------------------------------------------------------------


def test_R4_compute_headroom() -> None:
    snap = _snapshot(
        iter_ms=[100.0, 100.0, 100.0, 100.0, 100.0],
        tflops=[800.0, 810.0, 805.0, 808.0, 802.0],
    )
    report = engine.run(snap, _cluster_profile(peak_bf16=1200.0, peak_fp8=None), _plan(micro_batch_size=2))

    assert report["bottleneck"] == "COMPUTE_BOUND"
    assert report["meta"]["rule_id"].startswith("R4_COMPUTE_BOUND_HIGH")
    assert report["meta"]["tflops_pct_of_peak"] >= 0.55
    assert any(a["axis"] == "micro_batch_size" for a in report["candidate_axes"])


# ---------------------------------------------------------------------------
# 6. Hard-constraint enforcement: deepep + shared expert overlap mutex
# ---------------------------------------------------------------------------


def test_axes_respect_hard_constraints() -> None:
    """When deepEP is on, the engine must NOT emit
    moe_shared_expert_overlap=true as a candidate (deepep_x_sharedovrlp_mutex)."""

    snap = _snapshot(
        iter_ms=[32000.0] * 7,
        tflops=[240.0] * 7,
    )
    plan = _plan(
        expert_model_parallel_size=8,
        num_experts=64,
        use_turbo_deepep=True,
        moe_shared_expert_overlap=False,
    )

    report = engine.run(snap, _cluster_profile(peak_bf16=1219.2, peak_fp8=None), plan)

    for axis in report["candidate_axes"]:
        if axis["axis"] == "moe_shared_expert_overlap":
            assert True not in axis.get("candidates", []), \
                "engine emitted a candidate that violates deepep_x_sharedovrlp_mutex"


# ---------------------------------------------------------------------------
# Bonus: re-entry trigger when ClusterProfile is stale
# ---------------------------------------------------------------------------


def test_reentry_stale_profile() -> None:
    snap = _snapshot()
    profile = _cluster_profile()
    profile["collected_at"] = "2025-01-01T00:00:00+00:00"  # very old

    report = engine.run(snap, profile, _plan())

    assert report["suggested_transition"]["to"] == "PREFLIGHT"
    assert "stale" in report["suggested_transition"]["reason"].lower()
    assert report["suggested_transition"]["counts_against_budget"] is False
