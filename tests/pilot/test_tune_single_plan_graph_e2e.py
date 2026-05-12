"""End-to-end PlanGraph integration test for ``tune_single.run_session``.

Stubs out the heavy I/O surface (`preflight_check`, `submit.run`,
`observe.snapshot`, `_maybe_analyze_trace`, `report.build`) and drives the
full loop in-process so we can verify:

  - PlanGraph is constructed at BASELINE exit with the baseline as root.
  - Each candidate becomes a node, transitions running → completed | dead.
  - `mark_exhausted` runs after every candidate so a re-emit is dropped.
  - Settle.promoted=True triggers a `promote` on the graph; champion changes.
  - Settle.promoted=False shelves the round's completed children.
  - `state/plan_graphs/<session>.yaml` is persisted and re-loadable.
  - Periodic Exploration counter increments on non-explore rounds.

This complements the unit tests in ``test_plan_graph.py`` (which prove the
engine's correctness in isolation) by verifying the wire-up inside the
tune_single main loop.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import yaml

from pilot.tools import plan_graph as pg
from pilot.tools import tune_single


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FAKE_CFG = SimpleNamespace(mode="single", single={"max_local_gpus": 8})
_FAKE_LAUNCH_PLAN = SimpleNamespace()


def _fake_snapshot(iter_time_ms: float, status: str = "completed") -> dict[str, Any]:
    """Build a RunSnapshot the tune_single scoring math accepts."""
    return {
        "run_id": "fake",
        "status": status,
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
                "iter_time_ms": [iter_time_ms] * 8,
                "tflops": [200.0] * 8,
                "loss": [3.0, 2.8, 2.6, 2.5],
            },
            "latest": {"iter_time_ms": iter_time_ms, "tflops": 200.0, "loss": 2.5},
        },
        "progress": {"iters": 8},
    }


def _fake_trace_compute_bound() -> dict[str, Any]:
    """Minimal trace_analysis stub that the diagnose engine accepts."""
    # We do NOT go through the real engine — instead the test monkeypatches
    # `tune_single.diagnose` to return a synthetic engine_report that the
    # engine-path replan can consume. (Going through the real engine here
    # would require a full rocprof trace, which is overkill for this test.)
    return {"_synthetic": True}


def _fake_diagnose_factory(rotate_values: bool = False):
    """Returns a `diagnose` replacement that always emits a COMPUTE_BOUND
    engine_report with two weakly_local candidate axes.

    When ``rotate_values=True``, the numeric `turbo_deepep_num_cu` value
    rotates through a sequence with non-overlapping weakly_local 25% radii
    (so later rounds aren't auto-rejected by exhausted_neighborhoods).
    """
    state = {"i": 0}
    # When rotate_values is on: emit a single weakly_local axis whose
    # numeric value rotates through non-overlapping 25% radii so every
    # round generates a fresh candidate that survives both
    # exhausted_neighborhoods and overrides-dedup.
    # When off (default): emit the original two-axis report used by the
    # promote/marginal/dead tests.
    num_cu_seq = [80, 160, 320, 640]

    def fake_diagnose(snapshot, **kwargs):
        if rotate_values:
            i = state["i"] % len(num_cu_seq)
            state["i"] += 1
            num_cu = num_cu_seq[i]
            axes = [{
                "axis": "turbo_deepep_num_cu",
                "type": "weakly_local",
                "candidates": [num_cu],
                "expected_gain_band_pct": [10, 20],
                "rationale": "promote num_cu",
            }]
        else:
            axes = [
                {
                    "axis": "turbo_deepep_num_cu",
                    "type": "weakly_local",
                    "candidates": [80],
                    "expected_gain_band_pct": [10, 20],
                    "rationale": "promote num_cu",
                },
                {
                    "axis": "turbo_deepep_use_comm_stream",
                    "type": "weakly_local",
                    "candidates": [True],
                    "expected_gain_band_pct": [8, 16],
                    "rationale": "try comm_stream",
                },
            ]
        return {
            "status": "success",
            "bottleneck": "COMPUTE",
            "candidate_axes": [],
            "measurement": tune_single.summarize_snapshot(snapshot),
            "failure": None,
            "source": "engine",
            "meta": {
                "engine_report": {
                    "schema_version": "1.0",
                    "bottleneck": "COMPUTE_BOUND",
                    "confidence": 0.85,
                    "candidate_axes": axes,
                    "env_suspect": [],
                    "meta": {"rule_id": "TEST"},
                },
            },
        }
    return fake_diagnose


# ---------------------------------------------------------------------------
# E2E: PlanGraph persists across rounds, dedupes exhausted candidates,
#       promotes on real gain, shelves on marginal gain
# ---------------------------------------------------------------------------


def test_run_session_builds_plan_graph_and_promotes_on_real_gain(tmp_path: Path) -> None:
    # Scenario:
    #   round 0 BASELINE   → 100 ms/iter
    #   round 1 candidate1 → 90  ms/iter  (gain +11% over baseline; promote)
    #   round 1 candidate2 → 95  ms/iter  (shelved; lost to candidate1)
    #
    # Expected:
    #   - PlanGraph YAML written under tmp_path/plan_graphs/.../yaml
    #   - champion changes after round 1
    #   - candidate1 has status=completed and is now champion
    #   - candidate2 is shelved
    #   - both candidates appear in `frontier_excluding_dead` (champion + shelved)
    #   - exhausted_neighborhoods includes both (axis, value) pairs

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({
        "modules": {
            "pre_trainer": {
                "overrides": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "expert_model_parallel_size": 1,
                    "micro_batch_size": 1,
                    "global_batch_size": 8,
                    "seq_length": 128,
                    "train_iters": 4,
                }
            }
        }
    }))
    cluster_yaml = tmp_path / "cluster.yaml"
    cluster_yaml.write_text(yaml.safe_dump({
        "schema_version": "1.0",
        "cluster_id": "fake",
        "mode": "single",
        "single": {"max_local_gpus": 8},
    }))

    # Fake _run_one returning deterministic timings per candidate.
    # The engine emits 2 axes + 1 control rerun → 3 candidates in top-K.
    iter_time_by_call = iter([
        100.0,  # SMOKE
        100.0,  # BASELINE
        90.0,   # round 1 candidate 1 (10 ms faster → +11% gain)
        95.0,   # round 1 candidate 2 (5  ms faster → +5%  gain)
        100.0,  # round 1 control rerun (priority=0.05 but still gets run)
    ])

    def fake_run_one(*, cfg, launch_plan, plan_path, run_id, overrides, train_iters,
                    timeout_s, log_dir, env_overrides=None, analyze_trace=True):
        iter_time = next(iter_time_by_call)
        snap = _fake_snapshot(iter_time)
        measurement = tune_single.summarize_snapshot(snap)
        measurement["status"] = "completed"
        return {
            "id": run_id,
            "run_id": run_id,
            "overrides": {**overrides, "train_iters": train_iters},
            "env_overrides": env_overrides or {},
            "submit": {"status": "completed"},
            "snapshot": snap,
            "measurement": measurement,
            "trace_analysis": _fake_trace_compute_bound(),
        }

    fake_load = lambda path: {
        "mode": "single",
        "single": {"max_local_gpus": 8},
        "cluster_id": "fake",
    }
    fake_load_cluster_obj = SimpleNamespace(
        __dict__={"mode": "single", "single": {"max_local_gpus": 8}, "cluster_id": "fake"},
        mode="single",
    )

    with mock.patch.object(tune_single, "preflight_check", return_value=(_FAKE_CFG, _FAKE_LAUNCH_PLAN)), \
         mock.patch.object(tune_single, "load_cluster_config", return_value=fake_load_cluster_obj), \
         mock.patch.object(tune_single, "_latest_cluster_profile", return_value=None), \
         mock.patch.object(tune_single, "_run_one", side_effect=fake_run_one), \
         mock.patch.object(tune_single, "diagnose", side_effect=_fake_diagnose_factory()), \
         mock.patch.object(tune_single.report, "build",
                           return_value={"artifacts": [{"ref": str(tmp_path / "rep.yaml")}]}):
        result = tune_single.run_session(
            cluster_config=str(cluster_yaml),
            plan_path=str(plan_yaml),
            session_id="e2e_promote",
            rounds=1,
            candidates_per_round=3,
            smoke_iters=4,
            train_iters=4,
            timeout_s=60,
            root=str(tmp_path / "state"),
            log_dir=str(tmp_path / "state" / "runs"),
        )

    plan_graph_path = Path(result["plan_graph_ref"])
    assert plan_graph_path.exists(), f"PlanGraph not persisted: {plan_graph_path}"
    graph = pg.load(plan_graph_path)

    assert graph["session_id"] == "e2e_promote"
    # Baseline must be a non-champion node (it was demoted by the promote).
    baseline_nodes = [
        nid for nid, n in graph["nodes"].items()
        if n.get("parent") is None
    ]
    assert len(baseline_nodes) == 1
    baseline_id = baseline_nodes[0]

    # The new champion must be one of the two candidates derived from baseline.
    champion_id = graph["champion"]
    assert champion_id != baseline_id, \
        f"baseline should have been demoted, but champion is still {champion_id}"
    champ_node = graph["nodes"][champion_id]
    assert champ_node["parent"] == baseline_id
    assert champ_node["status"] == "completed"
    assert champion_id in champ_node["champion_at"][-1:0:-1] or 1 in champ_node["champion_at"]

    # All non-champion candidates should be shelved (still in the frontier).
    losing_ids = [
        nid for nid in graph["nodes"]
        if nid not in (baseline_id, champion_id)
    ]
    assert len(losing_ids) >= 1
    for lid in losing_ids:
        assert graph["nodes"][lid]["status"] == "shelved", graph["nodes"][lid]

    # The demoted baseline AND every shelved candidate live in the frontier.
    front = graph["frontier"]
    assert champion_id in front
    assert baseline_id in front
    for lid in losing_ids:
        assert lid in front

    # Both axes the engine emitted MUST be in exhausted_neighborhoods around
    # the baseline (the round-1 derive_parent). The engine-path control
    # rerun has no axis and therefore should NOT appear.
    exhausted_axes = {
        (row["around"], row["axis"]): set(row["tried"])
        for row in graph["exhausted_neighborhoods"]
    }
    assert (baseline_id, "turbo_deepep_num_cu") in exhausted_axes
    assert (baseline_id, "turbo_deepep_use_comm_stream") in exhausted_axes
    assert (baseline_id, "_control") not in exhausted_axes

    # rounds_since_explore should be 1 (one non-explore round just completed).
    assert graph["metadata"]["rounds_since_explore"] == 1
    assert graph["metadata"]["rounds_since_promotion"] == 0  # we just promoted


def test_run_session_shelves_marginal_gain_without_promoting(tmp_path: Path) -> None:
    """If every candidate gains <2%, champion stays as baseline and the
    completed children get shelved into the frontier. rounds_since_promotion
    must increment."""

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({
        "modules": {"pre_trainer": {"overrides": {
            "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1, "micro_batch_size": 1,
            "global_batch_size": 8, "seq_length": 128, "train_iters": 4,
        }}}
    }))
    cluster_yaml = tmp_path / "cluster.yaml"
    cluster_yaml.write_text(yaml.safe_dump({
        "schema_version": "1.0", "cluster_id": "fake", "mode": "single",
        "single": {"max_local_gpus": 8},
    }))

    # Scenario:
    #   BASELINE       → 100 ms
    #   candidate 1    → 99 ms (gain +1.0% < ε_promote=2%, NOT promoted)
    #   candidate 2    → 99 ms (gain +1.0%, NOT promoted)
    #   control rerun  → 100 ms
    iter_time_by_call = iter([100.0, 100.0, 99.0, 99.0, 100.0])

    def fake_run_one(*, cfg, launch_plan, plan_path, run_id, overrides, train_iters,
                    timeout_s, log_dir, env_overrides=None, analyze_trace=True):
        iter_time = next(iter_time_by_call)
        snap = _fake_snapshot(iter_time)
        m = tune_single.summarize_snapshot(snap)
        m["status"] = "completed"
        return {
            "id": run_id, "run_id": run_id,
            "overrides": {**overrides, "train_iters": train_iters},
            "env_overrides": env_overrides or {},
            "submit": {"status": "completed"},
            "snapshot": snap, "measurement": m,
            "trace_analysis": _fake_trace_compute_bound(),
        }

    fake_load_cluster_obj = SimpleNamespace(
        __dict__={"mode": "single", "single": {"max_local_gpus": 8}, "cluster_id": "fake"},
        mode="single",
    )

    with mock.patch.object(tune_single, "preflight_check", return_value=(_FAKE_CFG, _FAKE_LAUNCH_PLAN)), \
         mock.patch.object(tune_single, "load_cluster_config", return_value=fake_load_cluster_obj), \
         mock.patch.object(tune_single, "_latest_cluster_profile", return_value=None), \
         mock.patch.object(tune_single, "_run_one", side_effect=fake_run_one), \
         mock.patch.object(tune_single, "diagnose", side_effect=_fake_diagnose_factory()), \
         mock.patch.object(tune_single.report, "build",
                           return_value={"artifacts": [{"ref": str(tmp_path / "rep.yaml")}]}):
        result = tune_single.run_session(
            cluster_config=str(cluster_yaml),
            plan_path=str(plan_yaml),
            session_id="e2e_marginal",
            rounds=1,
            candidates_per_round=3,
            smoke_iters=4,
            train_iters=4,
            timeout_s=60,
            root=str(tmp_path / "state"),
            log_dir=str(tmp_path / "state" / "runs"),
        )

    graph = pg.load(Path(result["plan_graph_ref"]))
    # Champion should still be the baseline (the marginal gains failed to clear ε_promote=2%).
    baseline_id = next(nid for nid, n in graph["nodes"].items() if n["parent"] is None)
    assert graph["champion"] == baseline_id

    # All candidates should be shelved (kept for future backtrack).
    candidate_ids = [nid for nid, n in graph["nodes"].items() if n["parent"] == baseline_id]
    assert len(candidate_ids) >= 2
    for cid in candidate_ids:
        assert graph["nodes"][cid]["status"] == "shelved", \
            f"expected shelved, got {graph['nodes'][cid]}"
        assert cid in graph["frontier"]

    # rounds_since_promotion bumped (no promote happened).
    assert graph["metadata"]["rounds_since_promotion"] == 1
    # rounds_since_explore also bumped (non-explore round).
    assert graph["metadata"]["rounds_since_explore"] == 1


def test_run_session_marks_dead_for_oom_candidate(tmp_path: Path) -> None:
    """OOM/failed candidates land as `dead` nodes in the PlanGraph, with a
    measurable contribution to `dead_rate_in_subtree[baseline]`."""

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({
        "modules": {"pre_trainer": {"overrides": {
            "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": 1, "micro_batch_size": 1,
            "global_batch_size": 8, "seq_length": 128, "train_iters": 4,
        }}}
    }))
    cluster_yaml = tmp_path / "cluster.yaml"
    cluster_yaml.write_text(yaml.safe_dump({
        "schema_version": "1.0", "cluster_id": "fake", "mode": "single",
        "single": {"max_local_gpus": 8},
    }))

    # SMOKE + BASELINE complete normally; round-1 candidate 1 OOMs.
    # Candidate slots: SMOKE(1), BASELINE(2), C1=OOM(3), C2(4), C3=control(5).
    call_idx = {"n": 0}

    def fake_run_one(*, cfg, launch_plan, plan_path, run_id, overrides, train_iters,
                    timeout_s, log_dir, env_overrides=None, analyze_trace=True):
        call_idx["n"] += 1
        if call_idx["n"] in (1, 2):
            # SMOKE + BASELINE
            snap = _fake_snapshot(100.0)
        elif call_idx["n"] == 3:
            # Candidate 1: OOM (no scoreable iter timing)
            snap = {
                "run_id": run_id, "status": "oom",
                "symptoms": {"oom_detected": True, "hang_suspected": False,
                             "nccl_error": False, "cuda_error": False,
                             "python_error": False, "loss_nan_or_inf": False},
                "metrics": {"loss_finite": False, "history": {}, "latest": {}},
                "progress": {"iters": 0},
            }
        else:
            # Candidate 2 / control rerun: complete normally.
            snap = _fake_snapshot(98.0)

        m = tune_single.summarize_snapshot(snap)
        m["status"] = snap["status"]
        return {
            "id": run_id, "run_id": run_id,
            "overrides": {**overrides, "train_iters": train_iters},
            "env_overrides": env_overrides or {},
            "submit": {"status": "completed" if snap["status"] == "completed" else "failed"},
            "snapshot": snap, "measurement": m,
            "trace_analysis": _fake_trace_compute_bound() if snap["status"] == "completed" else None,
        }

    fake_load_cluster_obj = SimpleNamespace(
        __dict__={"mode": "single", "single": {"max_local_gpus": 8}, "cluster_id": "fake"},
        mode="single",
    )

    with mock.patch.object(tune_single, "preflight_check", return_value=(_FAKE_CFG, _FAKE_LAUNCH_PLAN)), \
         mock.patch.object(tune_single, "load_cluster_config", return_value=fake_load_cluster_obj), \
         mock.patch.object(tune_single, "_latest_cluster_profile", return_value=None), \
         mock.patch.object(tune_single, "_run_one", side_effect=fake_run_one), \
         mock.patch.object(tune_single, "diagnose", side_effect=_fake_diagnose_factory()), \
         mock.patch.object(tune_single.report, "build",
                           return_value={"artifacts": [{"ref": str(tmp_path / "rep.yaml")}]}):
        result = tune_single.run_session(
            cluster_config=str(cluster_yaml),
            plan_path=str(plan_yaml),
            session_id="e2e_oom",
            rounds=1,
            candidates_per_round=3,
            smoke_iters=4,
            train_iters=4,
            timeout_s=60,
            root=str(tmp_path / "state"),
            log_dir=str(tmp_path / "state" / "runs"),
        )

    graph = pg.load(Path(result["plan_graph_ref"]))
    baseline_id = next(nid for nid, n in graph["nodes"].items() if n["parent"] is None)
    dead_children = [
        nid for nid, n in graph["nodes"].items()
        if n.get("parent") == baseline_id and n["status"] == "dead"
    ]
    assert len(dead_children) == 1, \
        f"expected exactly 1 dead child, got {dead_children}"
    # 1 dead out of 3 candidates → dead_rate ≈ 0.333.
    rate = graph["metadata"]["dead_rate_in_subtree"][baseline_id]
    assert rate == pytest.approx(1 / 3), f"expected dead rate 1/3, got {rate}"


# ---------------------------------------------------------------------------
# E2E for the escape mechanisms (settle.md §5):
#   - Periodic Exploration Round at K=3
#   - Backtrack rescue when subtree dead-rate > 0.5
# ---------------------------------------------------------------------------


def _basic_inputs(tmp_path: Path):
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.write_text(yaml.safe_dump({"modules": {"pre_trainer": {"overrides": {
        "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1, "micro_batch_size": 1,
        "global_batch_size": 8, "seq_length": 128, "train_iters": 4,
    }}}}))
    cluster_yaml = tmp_path / "cluster.yaml"
    cluster_yaml.write_text(yaml.safe_dump({
        "schema_version": "1.0", "cluster_id": "fake", "mode": "single",
        "single": {"max_local_gpus": 8},
    }))
    fake_cluster = SimpleNamespace(
        __dict__={"mode": "single", "single": {"max_local_gpus": 8}, "cluster_id": "fake"},
        mode="single",
    )
    return plan_yaml, cluster_yaml, fake_cluster


def test_run_session_triggers_periodic_exploration_after_K_rounds(tmp_path: Path) -> None:
    """With rounds=3 and explore_period_K=3 (the default), the third round
    must be an explore round (derives from a shelved node) AND must not
    stop the loop even if it's regressive (settle.md §6 last ¶).

    Setup:
      - Every round, both candidates are marginally slower than baseline
        (gain <2%) so no promotion ever happens. The graph accumulates
        shelved siblings under baseline.
      - rounds_since_explore grows: 0 → 1 → 2 → 3.
      - On round 3, run_session sees rounds_since_explore ≥ K=3 and
        switches derive_parent from `baseline` to the highest-tps shelved
        node. We verify by checking the PlanGraph: the round-3 children
        must have `parent != baseline`.
    """
    plan_yaml, cluster_yaml, fake_cluster = _basic_inputs(tmp_path)

    # SMOKE + BASELINE + 4 rounds × ≤2 candidates each (rotate_values
    # collapses the engine to a single fresh weakly_local axis per round,
    # plus an optional control rerun). Use a generous buffer so the mock
    # never raises StopIteration. All candidates report 99.0 ms (1.0%
    # gain over baseline 100.0 — below ε_promote=2%, so no round promotes
    # and the loop must rely on explore-round + stop-deferral to keep
    # going).
    iter_time_by_call = itertools.chain([100.0, 100.0], itertools.repeat(99.0))

    def fake_run_one(*, cfg, launch_plan, plan_path, run_id, overrides, train_iters,
                    timeout_s, log_dir, env_overrides=None, analyze_trace=True):
        iter_time = next(iter_time_by_call)
        snap = _fake_snapshot(iter_time)
        m = tune_single.summarize_snapshot(snap); m["status"] = "completed"
        return {
            "id": run_id, "run_id": run_id,
            "overrides": {**overrides, "train_iters": train_iters},
            "env_overrides": env_overrides or {},
            "submit": {"status": "completed"}, "snapshot": snap, "measurement": m,
            "trace_analysis": _fake_trace_compute_bound(),
        }

    with mock.patch.object(tune_single, "preflight_check",
                          return_value=(_FAKE_CFG, _FAKE_LAUNCH_PLAN)), \
         mock.patch.object(tune_single, "load_cluster_config", return_value=fake_cluster), \
         mock.patch.object(tune_single, "_latest_cluster_profile", return_value=None), \
         mock.patch.object(tune_single, "_run_one", side_effect=fake_run_one), \
         mock.patch.object(tune_single, "diagnose",
                           side_effect=_fake_diagnose_factory(rotate_values=True)), \
         mock.patch.object(tune_single.report, "build",
                           return_value={"artifacts": [{"ref": str(tmp_path / "rep.yaml")}]}):
        result = tune_single.run_session(
            cluster_config=str(cluster_yaml), plan_path=str(plan_yaml),
            session_id="e2e_explore",
            rounds=4, candidates_per_round=3,
            smoke_iters=4, train_iters=4, timeout_s=60,
            root=str(tmp_path / "state"),
            log_dir=str(tmp_path / "state" / "runs"),
        )

    graph = pg.load(Path(result["plan_graph_ref"]))
    baseline_id = next(nid for nid, n in graph["nodes"].items() if n["parent"] is None)

    rounds_executed = {n["round_id"] for n in graph["nodes"].values()
                       if n["round_id"] not in (0, None)}
    assert rounds_executed == {1, 2, 3, 4}, \
        f"expected rounds {{1,2,3,4}}; got {rounds_executed}"

    # The R2 and R3 settle stages must have explicitly deferred the stop.
    settle_stages = [s for s in result["stage_history"]
                     if s["stage"] == "OPTIMIZE_LOOP.SETTLE"]
    deferred = [s for s in settle_stages
                if "stop deferred for explore round" in s["headline"]]
    assert deferred, f"expected at least one stop-deferred settle; got {settle_stages}"

    # R4 must be flagged as the explore round.
    r4_replan = [s for s in result["stage_history"]
                 if s["stage"] == "OPTIMIZE_LOOP.REPLAN" and s["round"] == 4]
    assert r4_replan and "(explore)" in r4_replan[-1]["headline"], \
        f"round 4 was not the explore round; replan stage headline: {r4_replan}"

    # Round 4 must be an explore round (rounds_since_explore=3 ≥ K=3): every
    # round-4 child must derive from a NON-baseline node (a shelved sibling
    # picked by ``pg.should_explore_round`` in run_session).
    round_4_children = [n for n in graph["nodes"].values() if n["round_id"] == 4]
    assert round_4_children, "round 4 produced no nodes"
    for n in round_4_children:
        assert n["parent"] != baseline_id, \
            f"round-4 node should derive from a shelved sibling, not baseline: {n}"

    # rounds_since_explore was reset to 0 by the explore round.
    assert graph["metadata"]["rounds_since_explore"] == 0


def test_run_session_backtrack_rescues_champion_when_subtree_dies(tmp_path: Path) -> None:
    """If the champion's subtree dies (>50% dead in 1 round), Settle must
    fire Backtrack and the loop must rebase the champion onto the
    highest-tps shelved survivor.

    Setup:
      - Round 1 spawns 3 candidates: 2 OOM (dead) + 1 healthy completer
        that beats baseline (gets promoted to champion).
      - Round 2 spawns 3 more candidates under that new champion, but
        we make them ALL OOM. Now `subtree_dead_rate[champion] = 1.0`,
        which exceeds the 0.5 threshold. Settle emits Backtrack with the
        highest-tps shelved node as the new target (the original baseline,
        if it's still in the frontier as shelved).
      - We verify: by the end of round 2, the PlanGraph champion is NOT
        the round-1 champion, and `champion_history` reflects the rebase.
    """
    plan_yaml, cluster_yaml, fake_cluster = _basic_inputs(tmp_path)

    # Counter-driven mock for both diagnose and _run_one, keyed off the
    # number of _run_one calls so far (SMOKE=1, BASELINE=2, R1c1=3,
    # R1c2=4, R1c3=5, R2c1=6, R2c2=7, R2c3=8...). We emit a single fresh
    # weakly_local axis per round so override-dedup doesn't collapse the
    # pool, and we choose iter_times so R1 has exactly one healthy
    # candidate at 85 ms (promote) and R2 has all OOM candidates.
    call_idx = {"n": 0}
    num_cu_seq = [80, 160, 320, 640]
    diag_idx = {"i": 0}

    def fake_run_one(*, cfg, launch_plan, plan_path, run_id, overrides, train_iters,
                    timeout_s, log_dir, env_overrides=None, analyze_trace=True):
        call_idx["n"] += 1
        n = call_idx["n"]
        # Calls 1–2: SMOKE + BASELINE @ 100 ms.
        # Call 3: R1 candidate — healthy at 85 ms (+17% gain → promotion).
        # Call 4+: all subsequent (R2…) candidates OOM.
        if n in (1, 2):
            snap = _fake_snapshot(100.0)
        elif n == 3:
            snap = _fake_snapshot(85.0)
        else:
            snap = {
                "run_id": run_id, "status": "oom",
                "symptoms": {"oom_detected": True, "hang_suspected": False,
                             "nccl_error": False, "cuda_error": False,
                             "python_error": False, "loss_nan_or_inf": False},
                "metrics": {"loss_finite": False, "history": {}, "latest": {}},
                "progress": {"iters": 0},
            }
        m = tune_single.summarize_snapshot(snap); m["status"] = snap["status"]
        return {
            "id": run_id, "run_id": run_id,
            "overrides": {**overrides, "train_iters": train_iters},
            "env_overrides": env_overrides or {},
            "submit": {"status": "completed" if snap["status"] == "completed" else "failed"},
            "snapshot": snap, "measurement": m,
            "trace_analysis": _fake_trace_compute_bound() if snap["status"] == "completed" else None,
        }

    def fake_diagnose(snapshot, **kwargs):
        i = diag_idx["i"] % len(num_cu_seq)
        diag_idx["i"] += 1
        num_cu = num_cu_seq[i]
        return {
            "status": "success",
            "bottleneck": "COMPUTE",
            "candidate_axes": [],
            "measurement": tune_single.summarize_snapshot(snapshot),
            "failure": None,
            "source": "engine",
            "meta": {
                "engine_report": {
                    "schema_version": "1.0",
                    "bottleneck": "COMPUTE_BOUND",
                    "confidence": 0.85,
                    "candidate_axes": [{
                        "axis": "turbo_deepep_num_cu",
                        "type": "weakly_local",
                        "candidates": [num_cu],
                        "expected_gain_band_pct": [10, 20],
                        "rationale": f"try num_cu={num_cu}",
                    }],
                    "env_suspect": [],
                    "meta": {"rule_id": "TEST"},
                },
            },
        }

    with mock.patch.object(tune_single, "preflight_check",
                          return_value=(_FAKE_CFG, _FAKE_LAUNCH_PLAN)), \
         mock.patch.object(tune_single, "load_cluster_config", return_value=fake_cluster), \
         mock.patch.object(tune_single, "_latest_cluster_profile", return_value=None), \
         mock.patch.object(tune_single, "_run_one", side_effect=fake_run_one), \
         mock.patch.object(tune_single, "diagnose", side_effect=fake_diagnose), \
         mock.patch.object(tune_single.report, "build",
                           return_value={"artifacts": [{"ref": str(tmp_path / "rep.yaml")}]}):
        result = tune_single.run_session(
            cluster_config=str(cluster_yaml), plan_path=str(plan_yaml),
            session_id="e2e_backtrack",
            rounds=2, candidates_per_round=3,
            smoke_iters=4, train_iters=4, timeout_s=60,
            root=str(tmp_path / "state"),
            log_dir=str(tmp_path / "state" / "runs"),
        )

    graph = pg.load(Path(result["plan_graph_ref"]))
    baseline_id = next(nid for nid, n in graph["nodes"].items() if n["parent"] is None)

    # Round 1 produced a healthy candidate that took the champion crown.
    r1_winner_candidates = [
        nid for nid, n in graph["nodes"].items()
        if n.get("parent") == baseline_id and 1 in n.get("champion_at", [])
    ]
    assert r1_winner_candidates, \
        f"round 1 should have promoted a champion; champion_history={graph['champion_history']}"
    r1_winner = r1_winner_candidates[0]

    # Round 2 must have seeded its candidate under r1_winner and that
    # candidate must be dead → subtree_dead_rate[r1_winner] > 0.5.
    r1_winner_rate = graph["metadata"]["dead_rate_in_subtree"].get(r1_winner, 0.0)
    assert r1_winner_rate > 0.5, \
        f"expected high subtree dead-rate on r1_winner; got {r1_winner_rate}"

    # Backtrack must have rescued the champion off of r1_winner.
    assert graph["champion"] != r1_winner, \
        f"backtrack should have rebased the champion off of {r1_winner}; got {graph['champion']}"

    # champion_history must show the rebase at round 2 (Backtrack's promote).
    round_2_promotions = [e for e in graph["champion_history"] if e["round"] == 2]
    assert round_2_promotions, \
        f"expected a champion_history entry at round 2 from Backtrack; got {graph['champion_history']}"

    # The Settle reason for round 2 must mention the backtrack rebase.
    r2_settle = [s for s in result["stage_history"]
                 if s["stage"] == "OPTIMIZE_LOOP.SETTLE" and s["round"] == 2]
    assert r2_settle and "backtrack→" in r2_settle[-1]["headline"], \
        f"round 2 settle should report backtrack→<target>; got {r2_settle}"
