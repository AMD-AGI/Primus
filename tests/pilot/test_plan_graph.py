"""Unit tests for ``pilot.tools.plan_graph`` (PlanGraph engine).

Covers:
  - construction + root invariants
  - node lifecycle (running → completed → shelved, running → dead)
  - promote() champion bookkeeping
  - frontier_excluding_dead() sorting
  - exhausted_neighborhoods with weakly_local numeric radius
  - subtree_dead_rate recomputation
  - escape-mechanism helpers (should_backtrack / pick_backtrack_target /
    should_explore_round / novelty_bonus_for / stability_bonus_for)
  - persistence round-trip
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pilot.tools import plan_graph as pg


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_new_creates_root_champion() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=12000.0, root_bottleneck="COMM_BOUND")
    assert g["schema_version"] == "1.0"
    assert g["session_id"] == "s1"
    assert g["champion"] == "baseline"
    root = g["nodes"]["baseline"]
    assert root["status"] == "completed"
    assert root["parent"] is None
    assert root["tps"] == 12000.0
    assert root["champion_at"] == [0]
    assert root["derived_axis"] is None
    assert g["frontier"] == ["baseline"]
    assert g["champion_history"][0]["id"] == "baseline"
    assert g["metadata"]["rounds_since_promotion"] == 0
    assert g["metadata"]["rounds_since_explore"] == 0


def test_add_node_requires_known_parent() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    with pytest.raises(pg.PlanGraphError):
        pg.add_node(
            g, plan_id="x", parent="nope", round_id=1,
            derived_axis={"axis": "mbs", "value": 2, "type": "structural"},
        )


def test_add_node_then_record_completed_lands_in_frontier() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.add_node(
        g, plan_id="r1c1", parent="baseline", round_id=1,
        derived_axis={"axis": "mbs", "value": 2, "type": "structural"},
        reason="probe mbs+1",
    )
    assert g["nodes"]["r1c1"]["status"] == "running"
    assert "r1c1" not in g["frontier"]  # running ≠ derivable

    g = pg.record_result(g, plan_id="r1c1", status="completed", tps=110.0, bottleneck="COMPUTE_BOUND")
    # Completed children are NOT auto-added to frontier; they appear once
    # they're shelved or promoted (they aren't shelved yet so they aren't
    # in `frontier` either — only champion + shelved are).
    assert g["nodes"]["r1c1"]["status"] == "completed"
    assert g["nodes"]["r1c1"]["tps"] == 110.0
    assert g["frontier"] == ["baseline"]


def test_record_dead_excluded_from_frontier_forever() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="r1c1", parent="baseline", round_id=1,
                   derived_axis={"axis": "mbs", "value": 2, "type": "structural"})
    g = pg.record_result(g, plan_id="r1c1", status="dead", reason="OOM at step 47")
    assert g["nodes"]["r1c1"]["status"] == "dead"
    assert "r1c1" not in g["frontier"]
    # Dead is one-way: no API path back to a non-dead status.


# ---------------------------------------------------------------------------
# Promote / shelve
# ---------------------------------------------------------------------------


def test_promote_demotes_old_champion_and_resets_counter() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.add_node(g, plan_id="r1c1", parent="baseline", round_id=1,
                   derived_axis={"axis": "mbs", "value": 2, "type": "structural"})
    g = pg.record_result(g, plan_id="r1c1", status="completed", tps=120.0)
    g = pg.bump_promotion_counter(g)
    assert g["metadata"]["rounds_since_promotion"] == 1

    g = pg.promote(g, plan_id="r1c1", round_id=1)
    assert g["champion"] == "r1c1"
    assert g["nodes"]["baseline"]["status"] == "shelved"
    assert g["nodes"]["r1c1"]["status"] == "completed"
    assert g["nodes"]["r1c1"]["champion_at"] == [1]
    assert g["metadata"]["rounds_since_promotion"] == 0
    # Frontier now contains the new champion + the shelved baseline,
    # sorted by tps desc.
    assert g["frontier"] == ["r1c1", "baseline"]
    assert g["champion_history"][-1] == {"round": 1, "id": "r1c1", "at": g["champion_history"][-1]["at"]}


def test_shelve_requires_completed() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="r1c1", parent="baseline", round_id=1, derived_axis=None)
    with pytest.raises(pg.PlanGraphError):
        pg.shelve(g, plan_id="r1c1")  # still running
    g = pg.record_result(g, plan_id="r1c1", status="completed", tps=110.0)
    g = pg.shelve(g, plan_id="r1c1", reason="marginal gain")
    assert g["nodes"]["r1c1"]["status"] == "shelved"
    assert "r1c1" in g["frontier"]


def test_promote_to_shelved_node_revives_it() -> None:
    """Backtrack picks a shelved node and promotes it; the engine must
    transition it back to 'completed' as part of the promotion."""
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.add_node(g, plan_id="r1c1", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="r1c1", status="completed", tps=105.0)
    g = pg.shelve(g, plan_id="r1c1")
    assert g["nodes"]["r1c1"]["status"] == "shelved"

    g = pg.promote(g, plan_id="r1c1", round_id=3)
    assert g["champion"] == "r1c1"
    assert g["nodes"]["r1c1"]["status"] == "completed"
    assert g["nodes"]["r1c1"]["champion_at"] == [3]


# ---------------------------------------------------------------------------
# Frontier ordering
# ---------------------------------------------------------------------------


def test_frontier_is_tps_sorted_desc() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    for cid, tps in [("a", 90.0), ("b", 130.0), ("c", 70.0)]:
        g = pg.add_node(g, plan_id=cid, parent="baseline", round_id=1, derived_axis=None)
        g = pg.record_result(g, plan_id=cid, status="completed", tps=tps)
        g = pg.shelve(g, plan_id=cid)
    # champion (baseline=100) + 3 shelved → sorted desc by tps
    assert pg.frontier_excluding_dead(g) == ["b", "baseline", "a", "c"]


# ---------------------------------------------------------------------------
# exhausted_neighborhoods radius
# ---------------------------------------------------------------------------


def test_exhausted_exact_match_for_structural() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.mark_exhausted(g, around="baseline", axis="micro_batch_size", value=14,
                         axis_type="structural")
    assert pg.is_exhausted(g, around="baseline", axis="micro_batch_size", value=14,
                          axis_type="structural")
    assert not pg.is_exhausted(g, around="baseline", axis="micro_batch_size", value=15,
                              axis_type="structural")


def test_exhausted_radius_for_weakly_local_numeric() -> None:
    """Numeric weakly_local axes have a ±25% radius (per plan_graph.md §5)."""
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.mark_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=80,
                         axis_type="weakly_local")
    # 80 ± 25% = [60, 100]
    assert pg.is_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=80,
                          axis_type="weakly_local")
    assert pg.is_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=64,
                          axis_type="weakly_local")
    assert pg.is_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=96,
                          axis_type="weakly_local")
    assert not pg.is_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=120,
                              axis_type="weakly_local")
    assert not pg.is_exhausted(g, around="baseline", axis="turbo_deepep_num_cu", value=40,
                              axis_type="weakly_local")


def test_exhausted_exact_match_for_boolean_weakly_local() -> None:
    """Booleans/enums always use exact match, even when type=weakly_local."""
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.mark_exhausted(g, around="baseline", axis="turbo_deepep_use_comm_stream",
                         value=True, axis_type="weakly_local")
    assert pg.is_exhausted(g, around="baseline", axis="turbo_deepep_use_comm_stream",
                          value=True, axis_type="weakly_local")
    assert not pg.is_exhausted(g, around="baseline", axis="turbo_deepep_use_comm_stream",
                              value=False, axis_type="weakly_local")


def test_mark_exhausted_merges_rows() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.mark_exhausted(g, around="baseline", axis="mbs", value=14, axis_type="structural")
    g = pg.mark_exhausted(g, around="baseline", axis="mbs", value=15, axis_type="structural")
    g = pg.mark_exhausted(g, around="baseline", axis="mbs", value=14, axis_type="structural")
    rows = [r for r in g["exhausted_neighborhoods"] if r["around"] == "baseline" and r["axis"] == "mbs"]
    assert len(rows) == 1
    assert rows[0]["tried"] == [14, 15]


# ---------------------------------------------------------------------------
# subtree_dead_rate
# ---------------------------------------------------------------------------


def test_subtree_dead_rate_counts_descendants() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1, derived_axis=None)
    g = pg.add_node(g, plan_id="b", parent="baseline", round_id=1, derived_axis=None)
    g = pg.add_node(g, plan_id="c", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="a", status="completed", tps=100.0)
    g = pg.record_result(g, plan_id="b", status="dead", reason="OOM")
    g = pg.record_result(g, plan_id="c", status="dead", reason="HANG")
    assert pg.subtree_dead_rate(g, "baseline") == pytest.approx(2 / 3)

    # `a` is alive, no descendants → 0.0
    assert pg.subtree_dead_rate(g, "a") == 0.0
    # `b` is dead with no descendants → 0.0 (counts only DESCENDANTS, not self)
    assert pg.subtree_dead_rate(g, "b") == 0.0


# ---------------------------------------------------------------------------
# Backtrack helpers
# ---------------------------------------------------------------------------


def test_should_backtrack_fires_above_threshold() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    # Two siblings under baseline, both dead → 100% dead rate.
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1, derived_axis=None)
    g = pg.add_node(g, plan_id="b", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="a", status="dead", reason="OOM")
    g = pg.record_result(g, plan_id="b", status="dead", reason="HANG")
    # baseline's subtree dead rate is now 1.0
    assert g["metadata"]["dead_rate_in_subtree"]["baseline"] == pytest.approx(1.0)
    assert pg.should_backtrack(g)


def test_should_backtrack_quiet_when_subtree_healthy() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="a", status="completed", tps=100.0)
    assert g["metadata"]["dead_rate_in_subtree"]["baseline"] == 0.0
    assert not pg.should_backtrack(g)


def test_pick_backtrack_target_returns_highest_tps_shelved() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.add_node(g, plan_id="x", parent="baseline", round_id=1, derived_axis=None)
    g = pg.add_node(g, plan_id="y", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="x", status="completed", tps=95.0)
    g = pg.record_result(g, plan_id="y", status="completed", tps=110.0)
    g = pg.shelve(g, plan_id="x")
    g = pg.shelve(g, plan_id="y")
    # baseline is champion, so pick_backtrack_target picks the highest-tps shelved one
    assert pg.pick_backtrack_target(g) == "y"


def test_pick_backtrack_target_returns_none_when_no_shelved() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    assert pg.pick_backtrack_target(g) is None


# ---------------------------------------------------------------------------
# Explore-round + counters
# ---------------------------------------------------------------------------


def test_should_explore_round_after_K() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    assert not pg.should_explore_round(g)
    g = pg.bump_explore_counter(g)
    g = pg.bump_explore_counter(g)
    assert not pg.should_explore_round(g)  # K=3 default; rounds_since_explore=2
    g = pg.bump_explore_counter(g)
    assert pg.should_explore_round(g)
    g = pg.reset_explore_counter(g)
    assert not pg.should_explore_round(g)


def test_should_explore_round_honors_custom_K() -> None:
    g = pg.new(session_id="s1", root_id="baseline", explore_period_K=5)
    for _ in range(4):
        g = pg.bump_explore_counter(g)
    assert not pg.should_explore_round(g)
    g = pg.bump_explore_counter(g)
    assert pg.should_explore_round(g)


# ---------------------------------------------------------------------------
# Diversification + Stability bonuses
# ---------------------------------------------------------------------------


def test_novelty_bonus_applies_to_unseen_axis() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1,
                   derived_axis={"axis": "mbs", "value": 2, "type": "structural"})
    # baseline has a sibling that already covers `mbs`; a new mbs candidate gets no bonus.
    assert pg.novelty_bonus_for(g, parent="baseline", axis="mbs") == 1.0
    # But a fresh axis under baseline does get the bonus.
    assert pg.novelty_bonus_for(g, parent="baseline", axis="tp") == pg.DEFAULT_NOVELTY_BONUS


def test_novelty_bonus_handles_composite_axes() -> None:
    g = pg.new(session_id="s1", root_id="baseline")
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1,
                   derived_axis={
                       "axis": None,
                       "composite": [
                           {"axis": "mbs", "value": 2, "type": "structural"},
                           {"axis": "tp",  "value": 2, "type": "structural"},
                       ],
                   })
    # Both composite axes count as "seen" siblings.
    assert pg.novelty_bonus_for(g, parent="baseline", axis="mbs") == 1.0
    assert pg.novelty_bonus_for(g, parent="baseline", axis="tp") == 1.0
    assert pg.novelty_bonus_for(g, parent="baseline", axis="ep") == pg.DEFAULT_NOVELTY_BONUS


def test_stability_bonus_requires_two_championship_rounds() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    # baseline.champion_at = [0] → one round so far, no bonus.
    assert pg.stability_bonus_for(g, parent="baseline") == 1.0

    # Promote and demote a few rounds so baseline reclaims championship.
    g = pg.add_node(g, plan_id="x", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="x", status="completed", tps=105.0)
    g = pg.promote(g, plan_id="x", round_id=1)         # baseline shelved
    g = pg.promote(g, plan_id="baseline", round_id=2)  # baseline champion again
    # baseline.champion_at = [0, 2] → two rounds, bonus applies
    assert pg.stability_bonus_for(g, parent="baseline") == pg.DEFAULT_STABILITY_BONUS


# ---------------------------------------------------------------------------
# Promotion / explore counter maintenance
# ---------------------------------------------------------------------------


def test_promotion_counter_resets_on_promote() -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.bump_promotion_counter(g)
    g = pg.bump_promotion_counter(g)
    assert g["metadata"]["rounds_since_promotion"] == 2
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=3, derived_axis=None)
    g = pg.record_result(g, plan_id="a", status="completed", tps=110.0)
    g = pg.promote(g, plan_id="a", round_id=3)
    assert g["metadata"]["rounds_since_promotion"] == 0


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persist_and_load_round_trip(tmp_path: Path) -> None:
    g = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    g = pg.add_node(g, plan_id="r1c1", parent="baseline", round_id=1,
                   derived_axis={"axis": "mbs", "value": 2, "type": "structural"})
    g = pg.record_result(g, plan_id="r1c1", status="completed", tps=110.0)
    g = pg.mark_exhausted(g, around="baseline", axis="mbs", value=2, axis_type="structural")
    path = pg.persist(g, tmp_path / "plan_graph.yaml")
    assert Path(path).exists()
    g2 = pg.load(path)
    assert g2["champion"] == "baseline"
    assert g2["nodes"]["r1c1"]["tps"] == 110.0
    assert g2["exhausted_neighborhoods"][0]["tried"] == [2]
    assert g2["schema_version"] == "1.0"


# ---------------------------------------------------------------------------
# Purity: no operation mutates its input
# ---------------------------------------------------------------------------


def test_settle_with_plan_graph_uses_rounds_since_promotion_for_stop() -> None:
    """When `plan_graph` is supplied, Settle's stagnation check must use the
    graph's `rounds_since_promotion` counter (settle.md §6.2) rather than
    the legacy flat-history heuristic."""
    from pilot.tools import tune_single

    g = pg.new(session_id="s", root_id="baseline", root_tps=100.0)
    g["metadata"]["rounds_since_promotion"] = 1  # one stagnant round already

    history = [
        {"id": "baseline", "measurement": {"status": "completed",
                                            "median_iter_time_ms": 100.0,
                                            "loss_finite": True}},
        # round 1: marginal gain, ≥ ε_stop but < ε_promote, no promotion
        {"id": "r1", "measurement": {"status": "completed",
                                      "median_iter_time_ms": 99.5,
                                      "loss_finite": True},
         "gain_vs_champion": 0.005},
    ]
    result = tune_single.settle(
        history, champion_id="baseline", plan_graph=g, round_id=1,
    )
    # rounds_since_promotion is 1 going in; if we don't promote, the
    # post-round value is 2 → ≥ stagnation_rounds=2 → stop.
    assert result["promoted"] is False
    assert result["stop"] is True
    assert result["reason"].startswith("stagnation")


def test_settle_explore_round_does_not_stop_even_when_stagnant() -> None:
    """Per settle.md §6 last paragraph, an explore-round result is exempt
    from stagnation."""
    from pilot.tools import tune_single

    g = pg.new(session_id="s", root_id="baseline", root_tps=100.0)
    g["metadata"]["rounds_since_promotion"] = 5  # well past stagnation

    history = [
        {"id": "baseline", "measurement": {"status": "completed",
                                            "median_iter_time_ms": 100.0,
                                            "loss_finite": True}},
        {"id": "explore_cand", "measurement": {"status": "completed",
                                                "median_iter_time_ms": 101.0,
                                                "loss_finite": True}},
    ]
    result = tune_single.settle(
        history, champion_id="baseline", plan_graph=g,
        round_id=6, is_explore_round=True,
    )
    assert result["stop"] is False
    assert result["is_explore_round"] is True


def test_settle_emits_backtrack_signal_when_dead_rate_high() -> None:
    """Settle must emit a backtrack recommendation (without auto-applying)
    when the champion's subtree dead-rate exceeds the threshold."""
    from pilot.tools import tune_single

    g = pg.new(session_id="s", root_id="baseline", root_tps=100.0)
    # Two siblings: both dead → subtree dead-rate = 1.0 > 0.5
    g = pg.add_node(g, plan_id="a", parent="baseline", round_id=1, derived_axis=None)
    g = pg.add_node(g, plan_id="b", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="a", status="dead", reason="OOM")
    g = pg.record_result(g, plan_id="b", status="dead", reason="HANG")
    # Plus one shelved survivor available for rescue.
    g = pg.add_node(g, plan_id="c", parent="baseline", round_id=1, derived_axis=None)
    g = pg.record_result(g, plan_id="c", status="completed", tps=105.0)
    g = pg.shelve(g, plan_id="c")

    history = [
        {"id": "baseline", "measurement": {"status": "completed",
                                            "median_iter_time_ms": 100.0,
                                            "loss_finite": True}},
    ]
    result = tune_single.settle(
        history, champion_id="baseline", plan_graph=g, round_id=1,
    )
    assert result["backtrack"]["fired"] is True
    assert result["backtrack"]["new_champion"] == "c"
    assert "subtree dead-rate" in result["backtrack"]["reason"]
    # The signal alone should NOT stop the loop (the engine acts on it instead).
    assert result["stop"] is False


def test_settle_legacy_path_without_plan_graph_still_works() -> None:
    """Callers that don't pass `plan_graph` should still get the legacy
    history-based stagnation check (backward compat)."""
    from pilot.tools import tune_single

    history = [
        {"id": "baseline", "measurement": {"status": "completed",
                                            "median_iter_time_ms": 100.0,
                                            "loss_finite": True}},
        {"id": "c1", "measurement": {"status": "completed",
                                      "median_iter_time_ms": 99.9,
                                      "loss_finite": True},
         "gain_vs_champion": 0.001},
        {"id": "c2", "measurement": {"status": "completed",
                                      "median_iter_time_ms": 99.8,
                                      "loss_finite": True},
         "gain_vs_champion": 0.001},
    ]
    result = tune_single.settle(history, champion_id="baseline")
    # Legacy semantics: last 2 entries both have gain < 0.005 → stop=True.
    assert result["stop"] is True
    assert result["backtrack"]["fired"] is False
    assert result["is_explore_round"] is False


def test_operations_are_pure() -> None:
    g0 = pg.new(session_id="s1", root_id="baseline", root_tps=100.0)
    snapshot = dict(g0)
    g1 = pg.add_node(g0, plan_id="a", parent="baseline", round_id=1, derived_axis=None)
    g2 = pg.record_result(g1, plan_id="a", status="completed", tps=120.0)
    g3 = pg.promote(g2, plan_id="a", round_id=1)
    g4 = pg.mark_exhausted(g3, around="baseline", axis="mbs", value=14, axis_type="structural")
    # Original unchanged.
    assert g0["champion"] == "baseline"
    assert g0["nodes"] == {"baseline": snapshot["nodes"]["baseline"]}
    assert "a" not in g0["nodes"]
    assert g0["exhausted_neighborhoods"] == []
    # Final has all the modifications.
    assert g4["champion"] == "a"
    assert g4["exhausted_neighborhoods"][0]["tried"] == [14]
