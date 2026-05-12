"""Tests for `pilot.tools.tune_single.replan` engine path.

Runs the full `diagnose -> replan` pipeline against the real
`pds_lite_trace_baseline` trace_analysis.json checked into pilot/state, so
this acts as a regression test for the trace-driven CandidatePool shape:

- engine path is taken (source == "engine")
- the trace-recommended axes (turbo_deepep_use_comm_stream / turbo_deepep_num_cu /
  gradient_accumulation_fusion) all appear in the resulting candidates
- env_suspect flags that aren't shadowed by an axis become env-only candidates
- legacy fallback still triggers when no trace is available
- env_overrides are returned as a separate dict (not leaked into trainer overrides)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from pilot.tools import tune_single

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BASELINE_DIR = _REPO_ROOT / "pilot" / "state" / "runs" / "pds_lite_trace_baseline"


def _load_baseline_trace() -> dict[str, Any]:
    return json.loads((_BASELINE_DIR / "trace_analysis.json").read_text())


def _load_baseline_plan() -> dict[str, Any]:
    return yaml.safe_load((_BASELINE_DIR / "plan.effective.yaml").read_text())


def _baseline_snapshot() -> dict[str, Any]:
    """Synthetic RunSnapshot consistent with the baseline trace (32s/iter,
    no failure, no OOM). Real submit.run produces a richer one but the
    engine only needs status + symptoms + iter timing for R0..R5."""
    return {
        "run_id": "pds_lite_trace_baseline",
        "status": "completed",
        "symptoms": {
            "oom_detected": False,
            "hang_suspected": False,
            "nccl_error": False,
            "cuda_error": False,
            "python_error": False,
            "loss_nan_or_inf": False,
        },
        "metrics": {
            "loss_finite": True,
            "history": {
                "iter_time_ms": [32_000.0] * 8,
                "tflops": [200.0] * 8,
                "loss": [3.0, 2.8, 2.6],
            },
            "latest": {"iter_time_ms": 32_000.0, "tflops": 200.0, "loss": 2.6},
        },
        "progress": {"iters": 10},
    }


def _cluster_single_8gpu() -> dict[str, Any]:
    """Mirrors mi355-gpu-8 (single-mode, 8 GPUs)."""
    return {
        "mode": "single",
        "single": {"max_local_gpus": 8},
    }


# ---------------------------------------------------------------------------
# Engine path: real baseline trace -> trace-driven candidate pool
# ---------------------------------------------------------------------------


def test_engine_replan_consumes_real_baseline_trace() -> None:
    trace = _load_baseline_trace()
    plan = _load_baseline_plan()
    snapshot = _baseline_snapshot()

    # 1) diagnose: must take the engine path
    diag = tune_single.diagnose(
        snapshot,
        trace_analysis=trace,
        plan=plan,
    )
    assert diag.get("source") == "engine", \
        f"expected engine path; got {diag.get('source')!r}"
    er = diag["meta"]["engine_report"]
    assert er["bottleneck"] == "COMPUTE_BOUND"
    assert er["confidence"] >= 0.7

    # 2) replan: must take the engine path too, and produce candidates
    pool = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=5,
        train_iters=20,
    )
    assert pool["source"] == "engine"
    assert pool["status"] == "ready"
    assert pool["candidates"], "expected at least one candidate from the engine path"

    # 3) the trace-recommended axes the engine emitted MUST be reachable
    axis_seen = {c["axis_meta"].get("axis") for c in pool["candidates"]}
    expected = {"turbo_deepep_use_comm_stream", "turbo_deepep_num_cu", "gradient_accumulation_fusion"}
    missing = expected - axis_seen
    assert not missing, (
        f"engine recommended {expected}; missing in candidate pool: {missing}; "
        f"got: {axis_seen}"
    )

    # 4) `turbo_deepep_use_comm_stream=true` must end up in trainer overrides,
    #    NOT env_overrides (it's a YAML knob, not an env var).
    comm_stream = next(
        c for c in pool["candidates"]
        if c["axis_meta"].get("axis") == "turbo_deepep_use_comm_stream"
    )
    assert comm_stream["overrides"].get("turbo_deepep_use_comm_stream") is True
    assert "turbo_deepep_use_comm_stream" not in (comm_stream.get("env_overrides") or {})

    # 5) every candidate's `train_iters` must be set to the requested 20
    #    (engine-pool runs are short by design).
    for c in pool["candidates"]:
        assert c["overrides"].get("train_iters") == 20

    # 6) priority must be monotonically non-increasing (sort guarantee).
    priorities = [c["priority"] for c in pool["candidates"]]
    assert priorities == sorted(priorities, reverse=True), priorities


def test_env_suspect_only_flags_become_env_candidates() -> None:
    """If diagnose emits an env_suspect that isn't already covered by a
    trainer-override axis, replan should produce an env-only candidate."""
    # Synthesize a diagnosis that names ONLY env-side flags (and no axes
    # that shadow them), so we hit the env-only branch deterministically.
    trace = _load_baseline_trace()
    plan = _load_baseline_plan()
    snapshot = _baseline_snapshot()
    diag = tune_single.diagnose(snapshot, trace_analysis=trace, plan=plan)
    er = diag["meta"]["engine_report"]
    # Inject an env_suspect that is NOT in candidate_axes (RCCL_MSCCL_ENABLE
    # is purely env-channel).
    er = {**er, "env_suspect": [
        *er.get("env_suspect", []),
        {"flag": "RCCL_MSCCL_ENABLE", "reason": "test injection",
         "hint": "try true; algorithm pick can change collective shape"},
    ]}
    diag["meta"]["engine_report"] = er

    pool = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=10,  # be generous so env-only candidate doesn't get pruned
        train_iters=20,
    )
    msccl_cand = next(
        (c for c in pool["candidates"] if c["axis_meta"].get("axis") == "RCCL_MSCCL_ENABLE"),
        None,
    )
    assert msccl_cand is not None, "RCCL_MSCCL_ENABLE env_suspect did not become a candidate"
    assert msccl_cand["axis_meta"]["channel"] == "env"
    assert msccl_cand["env_overrides"].get("RCCL_MSCCL_ENABLE") == "1"
    assert "RCCL_MSCCL_ENABLE" not in msccl_cand["overrides"]


# ---------------------------------------------------------------------------
# Legacy fallback: no engine_report -> legacy axis-name path still works
# ---------------------------------------------------------------------------


def test_legacy_fallback_when_no_trace() -> None:
    """When diagnose is called WITHOUT trace_analysis, replan must fall
    back to the axis-name path so SMOKE / BASELINE failure routing keeps
    working in environments without a profiler."""
    snapshot = _baseline_snapshot()
    diag = tune_single.diagnose(snapshot)  # no trace
    assert diag["source"] == "legacy"
    assert "meta" not in diag or "engine_report" not in (diag.get("meta") or {})

    plan = _load_baseline_plan()
    pool = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=2,
        train_iters=20,
    )
    assert pool["source"] == "legacy"
    assert pool["status"] == "ready"
    assert pool["candidates"]
    for c in pool["candidates"]:
        assert c["axis_meta"]["source"] == "legacy.axis_name"
        assert c.get("env_overrides") == {}


# ---------------------------------------------------------------------------
# Priority and cost proxy semantics
# ---------------------------------------------------------------------------


def test_replan_drops_exhausted_candidates_when_plan_graph_provided() -> None:
    """When a PlanGraph is passed in, candidates that hit
    `exhausted_neighborhoods` (per plan_graph.md §5) must be dropped from
    the resulting CandidatePool and surfaced in `selection.rejected`."""
    from pilot.tools import plan_graph as pg

    diag = {
        "bottleneck": "COMPUTE",
        "candidate_axes": [],
        "meta": {
            "engine_report": {
                "schema_version": "1.0",
                "bottleneck": "COMPUTE_BOUND",
                "confidence": 0.85,
                "candidate_axes": [
                    {
                        "axis": "turbo_deepep_use_comm_stream",
                        "type": "weakly_local",
                        "candidates": [True],
                        "expected_gain_band_pct": [3, 8],
                        "rationale": "weakly_local probe",
                    },
                    {
                        "axis": "turbo_deepep_num_cu",
                        "type": "weakly_local",
                        "candidates": [80],
                        "expected_gain_band_pct": [1, 5],
                        "rationale": "weakly_local probe",
                    },
                ],
                "env_suspect": [],
                "meta": {"rule_id": "TEST"},
            },
        },
    }

    graph = pg.new(session_id="t", root_id="baseline", root_tps=100.0)
    # Mark `turbo_deepep_use_comm_stream=True` as already exhausted around baseline.
    graph = pg.mark_exhausted(
        graph,
        around="baseline",
        axis="turbo_deepep_use_comm_stream",
        value=True,
        axis_type="weakly_local",
    )

    plan = _load_baseline_plan()
    pool = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=5,
        train_iters=20,
        plan_graph=graph,
        parent_plan_id="baseline",
    )

    seen = {c["axis_meta"]["axis"] for c in pool["candidates"]}
    assert "turbo_deepep_use_comm_stream" not in seen, \
        f"exhausted axis leaked into pool: {seen}"
    assert "turbo_deepep_num_cu" in seen, \
        f"non-exhausted axis should still appear: {seen}"

    rejected_axes = {r.get("axis") for r in pool["selection"]["rejected"]}
    assert "turbo_deepep_use_comm_stream" in rejected_axes
    # selection.strategy is the single-node v1 default per execution_strategy.md §6.
    assert pool["selection"]["strategy"] == "Per-Plan"
    assert pool["derived_from"]["primary"] == "baseline"


def test_replan_applies_novelty_and_stability_bonuses() -> None:
    """When PlanGraph reports the parent has been champion ≥2 rounds and the
    axis hasn't been used as a sibling, both bonuses (1.20 × 1.10) should
    appear in `axis_meta` and lift the priority above the no-graph baseline.
    """
    from pilot.tools import plan_graph as pg

    diag = {
        "bottleneck": "COMPUTE",
        "candidate_axes": [],
        "meta": {
            "engine_report": {
                "schema_version": "1.0",
                "bottleneck": "COMPUTE_BOUND",
                "confidence": 0.85,
                "candidate_axes": [
                    {
                        "axis": "turbo_deepep_use_comm_stream",
                        "type": "weakly_local",
                        "candidates": [True],
                        "expected_gain_band_pct": [3, 8],
                        "rationale": "fresh axis",
                    },
                ],
                "env_suspect": [],
                "meta": {"rule_id": "TEST"},
            },
        },
    }
    plan = _load_baseline_plan()

    # First call WITHOUT plan_graph: baseline priority, no bonuses.
    pool_no_graph = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=5,
        train_iters=20,
    )
    no_graph_cand = next(
        c for c in pool_no_graph["candidates"]
        if c["axis_meta"].get("axis") == "turbo_deepep_use_comm_stream"
    )

    # Now build a graph where baseline has been champion in rounds 0 and 2
    # (stability bonus active), and no sibling has covered the axis yet
    # (novelty bonus active).
    graph = pg.new(session_id="t", root_id="baseline", root_tps=100.0)
    graph = pg.add_node(graph, plan_id="r1c1", parent="baseline", round_id=1,
                       derived_axis={"axis": "tp", "value": 2, "type": "structural"})
    graph = pg.record_result(graph, plan_id="r1c1", status="completed", tps=105.0)
    graph = pg.promote(graph, plan_id="r1c1", round_id=1)
    graph = pg.promote(graph, plan_id="baseline", round_id=2)  # reclaim
    # baseline.champion_at == [0, 2] → stability bonus active

    pool_with_graph = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=3,
        max_candidates=5,
        train_iters=20,
        plan_graph=graph,
        parent_plan_id="baseline",
    )
    with_graph_cand = next(
        c for c in pool_with_graph["candidates"]
        if c["axis_meta"].get("axis") == "turbo_deepep_use_comm_stream"
    )
    assert with_graph_cand["axis_meta"]["novelty_bonus"] == pytest.approx(pg.DEFAULT_NOVELTY_BONUS)
    assert with_graph_cand["axis_meta"]["stability_bonus"] == pytest.approx(pg.DEFAULT_STABILITY_BONUS)
    # Combined factor 1.20 × 1.10 = 1.32 → priority must be strictly higher.
    assert with_graph_cand["priority"] == pytest.approx(
        no_graph_cand["priority"] * pg.DEFAULT_NOVELTY_BONUS * pg.DEFAULT_STABILITY_BONUS,
        rel=1e-3,
    )


def test_priority_uses_gain_mid_and_cost_proxy() -> None:
    """A weakly_local axis with a high expected_gain_band should outrank
    a structural axis with a low band, even at the same confidence."""
    diag = {
        "bottleneck": "COMPUTE",
        "candidate_axes": [],
        "meta": {
            "engine_report": {
                "schema_version": "1.0",
                "bottleneck": "COMPUTE_BOUND",
                "confidence": 0.85,
                "candidate_axes": [
                    {
                        "axis": "turbo_deepep_use_comm_stream",
                        "type": "weakly_local",
                        "candidates": [True],
                        "expected_gain_band_pct": [3, 8],   # mid 5.5%
                        "rationale": "weakly_local high band",
                    },
                    {
                        "axis": "micro_batch_size",
                        "type": "structural",
                        "candidates": [16],
                        "expected_gain_band_pct": [2, 4],   # mid 3%
                        "rationale": "structural lower band",
                    },
                ],
                "env_suspect": [],
                "meta": {"rule_id": "TEST"},
            },
        },
    }
    plan = _load_baseline_plan()
    pool = tune_single.replan(
        base_plan=plan,
        cluster=_cluster_single_8gpu(),
        diagnosis=diag,
        round_id=1,
        max_candidates=5,
        train_iters=20,
    )
    # weakly_local 5.5% / 1.0 = 0.055 * 0.85 = 0.04675
    # structural   3.0% / 2.0 = 0.015 * 0.85 = 0.01275
    weakly = next(c for c in pool["candidates"]
                  if c["axis_meta"]["axis"] == "turbo_deepep_use_comm_stream")
    structural = next(c for c in pool["candidates"]
                      if c["axis_meta"]["axis"] == "micro_batch_size")
    assert weakly["priority"] > structural["priority"], \
        f"expected weakly_local to outrank structural; got {weakly['priority']=} {structural['priority']=}"
