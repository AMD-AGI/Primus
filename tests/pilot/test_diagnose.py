"""Tests for `pilot.tools.diagnose` v2 (trace-driven rule engine).

Each test builds a minimal `trace_analysis` dict that triggers exactly
one decision rule, then asserts on the resulting `DiagnosisReport`. This
locks the rule semantics for future refactors. Optional inputs
(`snapshot`, `plan`, `champion_snapshot`) are added only where a rule
requires them.
"""

from __future__ import annotations

from typing import Any

from pilot.tools import diagnose as diag


def _trace_with(
    *,
    iter_ms: float = 30_000.0,
    ratios: dict[str, float] | None = None,
    cost_breakdown: dict[str, float] | None = None,
    kernels_full: list[dict[str, Any]] | None = None,
    pipeline_streams: int = 2,
    phases_ms: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal `trace_analysis` doc covering exactly the fields
    the diagnose engine reads."""
    base_ratios = {
        "comm_ratio": 0.0,
        "comm_collective_ratio": 0.0,
        "comm_moe_dispatch_ratio": 0.0,
        "comm_p2p_ratio": 0.0,
        "compute_ratio": 0.0,
        "compute_gemm_ratio": 0.0,
        "compute_attention_ratio": 0.0,
        "compute_fp8_prep_ratio": 0.0,
        "compute_norm_act_ratio": 0.0,
        "compute_optim_ratio": 0.0,
        "compute_other_ratio": 0.0,
        "memcpy_ratio": 0.0,
        "bubble_ratio": 0.0,
        "schedule_overhead_ratio": 0.0,
    }
    base_ratios.update(ratios or {})
    base_cb = {
        "iter_wallclock_ms": iter_ms,
        "pure_compute_pct": 0.0,
        "pure_comm_pct": 0.0,
        "overlap_pct": 0.0,
        "bubble_pct": 0.0,
        "longest_serialized_comm_ms": 0.0,
    }
    base_cb.update(cost_breakdown or {})
    pl_streams = [
        {"stream": s, "wall_ms": iter_ms, "glyphs": "G" * 80, "kernels_top": []}
        for s in range(pipeline_streams)
    ]
    return {
        "iter_wallclock_ms": iter_ms,
        "ratios": base_ratios,
        "cost_breakdown": base_cb,
        "phases": {"source": "autograd_engine", **(phases_ms or {})},
        "kernels_full": kernels_full or [],
        "pipeline_timeline": {
            "stream_count_total": pipeline_streams,
            "streams": pl_streams,
        },
        "warnings": warnings or [],
    }


# ---------------------------------------------------------------------------
# R4 — Compute bound (the pds_lite_trace_baseline shape)
# ---------------------------------------------------------------------------


def test_R4_A_compute_bound_gemm_dominant() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92, "comm_ratio": 0.07},
        cost_breakdown={"pure_compute_pct": 0.92, "pure_comm_pct": 0.07,
                        "overlap_pct": 0.001, "bubble_pct": 0.006},
    )
    report = diag.run(trace_analysis=trace)
    assert report["bottleneck"] == "COMPUTE_BOUND"
    assert report["meta"]["rule_id"] == "R4-A"
    assert report["confidence"] == 0.85
    assert "compute_gemm_ratio" in report["evidence"][0]


def test_R4_DEFAULT_low_confidence_when_nothing_dominates() -> None:
    trace = _trace_with(
        ratios={"compute_ratio": 0.40, "comm_ratio": 0.15},  # below all thresholds
        cost_breakdown={"pure_compute_pct": 0.40, "pure_comm_pct": 0.15,
                        "overlap_pct": 0.0, "bubble_pct": 0.05},
    )
    report = diag.run(trace_analysis=trace)
    assert report["bottleneck"] == "COMPUTE_BOUND"
    assert report["meta"]["rule_id"] == "R4-DEFAULT"
    assert report["confidence"] <= 0.55


# ---------------------------------------------------------------------------
# R3 — Communication bound (trio of sub-signals)
# ---------------------------------------------------------------------------


def test_R3_strong_when_comm_ratio_very_high() -> None:
    trace = _trace_with(
        ratios={"comm_ratio": 0.45, "compute_ratio": 0.50},
        cost_breakdown={"pure_comm_pct": 0.30, "pure_compute_pct": 0.50,
                        "overlap_pct": 0.10, "bubble_pct": 0.10,
                        "longest_serialized_comm_ms": 100.0},
    )
    report = diag.run(trace_analysis=trace)
    assert report["bottleneck"] == "COMM_BOUND"
    assert report["meta"]["rule_id"] == "R3-A-strong"
    assert report["confidence"] == 0.90


def test_R3_two_signals() -> None:
    trace = _trace_with(
        ratios={"comm_ratio": 0.22, "compute_ratio": 0.70},
        cost_breakdown={"pure_comm_pct": 0.15, "pure_compute_pct": 0.65,
                        "overlap_pct": 0.05, "bubble_pct": 0.05,
                        "longest_serialized_comm_ms": 1.0},
    )
    report = diag.run(trace_analysis=trace)
    assert report["bottleneck"] == "COMM_BOUND"
    assert report["meta"]["rule_id"] in ("R3-AB", "R3-AB ", "R3-ABC")
    assert report["confidence"] >= 0.80


# ---------------------------------------------------------------------------
# R2 — Pipeline bubble (requires plan.pp > 1)
# ---------------------------------------------------------------------------


def test_R2_pipeline_bubble_requires_pp_gt_1() -> None:
    trace = _trace_with(
        ratios={"compute_ratio": 0.70, "compute_gemm_ratio": 0.50},
        cost_breakdown={"pure_compute_pct": 0.70, "pure_comm_pct": 0.05,
                        "overlap_pct": 0.0, "bubble_pct": 0.20},
    )
    plan = {
        "modules": {
            "pre_trainer": {"overrides": {"pipeline_model_parallel_size": 4,
                                           "micro_batch_size": 1}}
        }
    }
    report = diag.run(trace_analysis=trace, plan=plan)
    assert report["bottleneck"] == "PIPELINE_BOUND"
    assert report["meta"]["rule_id"] == "R2-A"
    # Should also recommend VPP as a structural axis
    assert any(a["axis"] == "virtual_pipeline_model_parallel_size"
               for a in report["candidate_axes"])


def test_R2_skipped_when_pp_eq_1() -> None:
    trace = _trace_with(
        ratios={"compute_ratio": 0.70, "compute_gemm_ratio": 0.50},
        cost_breakdown={"pure_compute_pct": 0.70, "pure_comm_pct": 0.05,
                        "overlap_pct": 0.0, "bubble_pct": 0.20},
    )
    report = diag.run(trace_analysis=trace)  # no plan → pp defaults to 1
    assert report["bottleneck"] != "PIPELINE_BOUND"


# ---------------------------------------------------------------------------
# R1 — Memory bound (requires snapshot)
# ---------------------------------------------------------------------------


def test_R1_OOM_short_circuit() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92},
        cost_breakdown={"pure_compute_pct": 0.92, "bubble_pct": 0.006},
    )
    snapshot = {"status": "failed", "symptoms": {"oom_detected": True}}
    report = diag.run(trace_analysis=trace, snapshot=snapshot)
    assert report["bottleneck"] == "MEMORY_BOUND"
    assert report["meta"]["rule_id"] == "R0-OOM"
    assert report["suggested_transition"]["to"] == "OPTIMIZE_LOOP.REPLAN"


def test_R1_MEM_TIGHT_with_snapshot() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92},
        cost_breakdown={"pure_compute_pct": 0.92, "bubble_pct": 0.006},
    )
    snapshot = {
        "status": "completed",
        "symptoms": {},
        "metrics": {"history": {"mem_pct": [0.85, 0.93, 0.94]}},
    }
    report = diag.run(trace_analysis=trace, snapshot=snapshot)
    assert report["bottleneck"] == "MEMORY_BOUND"
    assert report["meta"]["rule_id"] == "R1-MEM"


# ---------------------------------------------------------------------------
# R5 — Regression overlay
# ---------------------------------------------------------------------------


def test_R5_regression_overlay() -> None:
    trace = _trace_with(
        iter_ms=15_000.0,
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92},
        cost_breakdown={"pure_compute_pct": 0.92, "bubble_pct": 0.006,
                        "iter_wallclock_ms": 15_000.0},
    )
    champion = {"metrics": {"history": {"iter_time_ms": [10_000, 10_000, 10_500, 10_400]}}}
    report = diag.run(trace_analysis=trace, champion_snapshot=champion)
    # Schema-visible bottleneck is COMPUTE (R4-A), but extended is REGRESSION
    assert report["bottleneck"] == "COMPUTE_BOUND"
    assert report["meta"]["bottleneck_extended"] == "REGRESSION"
    assert report["confidence"] == 0.95
    assert report["suggested_transition"]["to"] == "OPTIMIZE_LOOP.REPLAN"


# ---------------------------------------------------------------------------
# Axis production: MoE-shared-stream signal
# ---------------------------------------------------------------------------


def test_axis_turbo_deepep_use_comm_stream_when_shared() -> None:
    """The hallmark pds_lite case: MoE EP comm and GEMM live on stream 0."""
    trace = _trace_with(
        ratios={"comm_moe_dispatch_ratio": 0.07, "compute_gemm_ratio": 0.80,
                "compute_ratio": 0.92},
        cost_breakdown={"pure_compute_pct": 0.92, "pure_comm_pct": 0.07,
                        "overlap_pct": 0.001, "bubble_pct": 0.006},
        pipeline_streams=2,
        kernels_full=[
            {"name": "deep_ep_dispatch", "name_short": "deep_ep_dispatch",
             "bucket": "comm_moe_dispatch", "wall_ms": 1500.0, "wall_pct": 0.05,
             "self_ms": 1500.0, "calls": 100, "p50_ms": 15.0, "p99_ms": 20.0,
             "streams": [0]},
            {"name": "ck_tile_kernel", "name_short": "ck_tile_kernel",
             "bucket": "compute_gemm_fp8_grouped", "wall_ms": 24000.0, "wall_pct": 0.80,
             "self_ms": 24000.0, "calls": 1000, "p50_ms": 24.0, "p99_ms": 40.0,
             "streams": [0]},
        ],
    )
    report = diag.run(trace_analysis=trace)
    axes = {a["axis"]: a for a in report["candidate_axes"]}
    assert "turbo_deepep_use_comm_stream" in axes
    assert axes["turbo_deepep_use_comm_stream"]["candidates"] == [True]
    assert any(e["flag"] == "turbo_deepep_use_comm_stream" for e in report["env_suspect"])


def test_axis_NOT_turbo_when_already_separate_streams() -> None:
    trace = _trace_with(
        ratios={"comm_moe_dispatch_ratio": 0.07, "compute_gemm_ratio": 0.80,
                "compute_ratio": 0.92},
        cost_breakdown={"pure_compute_pct": 0.92, "pure_comm_pct": 0.07,
                        "overlap_pct": 0.06, "bubble_pct": 0.006},  # high overlap
        pipeline_streams=4,  # multiple streams
        kernels_full=[
            {"name": "deep_ep_dispatch", "name_short": "deep_ep_dispatch",
             "bucket": "comm_moe_dispatch", "wall_ms": 1500.0, "wall_pct": 0.05,
             "self_ms": 1500.0, "calls": 100, "p50_ms": 15.0, "p99_ms": 20.0,
             "streams": [14]},  # comm stream
            {"name": "ck_tile_kernel", "name_short": "ck_tile_kernel",
             "bucket": "compute_gemm_fp8_grouped", "wall_ms": 24000.0, "wall_pct": 0.80,
             "self_ms": 24000.0, "calls": 1000, "p50_ms": 24.0, "p99_ms": 40.0,
             "streams": [0]},
        ],
    )
    report = diag.run(trace_analysis=trace)
    axes = {a["axis"]: a for a in report["candidate_axes"]}
    assert "turbo_deepep_use_comm_stream" not in axes


# ---------------------------------------------------------------------------
# skipped_axes: structural axis without plan
# ---------------------------------------------------------------------------


def test_skipped_axes_when_no_plan() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92, "comm_ratio": 0.05},
        cost_breakdown={"pure_compute_pct": 0.92, "bubble_pct": 0.006},
    )
    report = diag.run(trace_analysis=trace)
    skipped = report["meta"].get("skipped_axes", [])
    assert any("micro_batch_size" in s for s in skipped)


def test_emits_micro_batch_size_when_plan_provided() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92, "comm_ratio": 0.05},
        cost_breakdown={"pure_compute_pct": 0.92, "bubble_pct": 0.006},
    )
    plan = {"modules": {"pre_trainer": {"overrides": {"micro_batch_size": 8}}}}
    report = diag.run(trace_analysis=trace, plan=plan)
    axes = {a["axis"]: a for a in report["candidate_axes"]}
    assert "micro_batch_size" in axes
    # current=8 → candidates=[10, 12]
    assert axes["micro_batch_size"]["candidates"] == [10, 12]


# ---------------------------------------------------------------------------
# Strategy mapping
# ---------------------------------------------------------------------------


def test_strategy_per_plan_when_only_weakly_local() -> None:
    trace = _trace_with(
        ratios={"compute_gemm_ratio": 0.80, "compute_ratio": 0.92,
                "comm_moe_dispatch_ratio": 0.07},
        cost_breakdown={"pure_compute_pct": 0.92, "pure_comm_pct": 0.07,
                        "overlap_pct": 0.001, "bubble_pct": 0.006},
        pipeline_streams=2,
        kernels_full=[
            {"name": "deep_ep_dispatch", "name_short": "deep_ep_dispatch",
             "bucket": "comm_moe_dispatch", "wall_ms": 1500.0, "wall_pct": 0.05,
             "self_ms": 1500.0, "calls": 100, "p50_ms": 15.0, "p99_ms": 20.0,
             "streams": [0]},
            {"name": "ck_tile_kernel", "name_short": "ck_tile_kernel",
             "bucket": "compute_gemm_fp8_grouped", "wall_ms": 24000.0, "wall_pct": 0.80,
             "self_ms": 24000.0, "calls": 1000, "p50_ms": 24.0, "p99_ms": 40.0,
             "streams": [0]},
        ],
    )
    report = diag.run(trace_analysis=trace)
    assert report["suggested_strategy"] == "Per-Plan"
