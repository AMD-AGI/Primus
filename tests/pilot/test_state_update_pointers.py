"""Tests for `pilot.tools.state.update_pointers`.

This is the per-round bookkeeping primitive added after the gap report
flagged that every Orchestrator round of session 20260513T024603Z hand-rolled
the same `read yaml → mutate dict in python → write back` pattern (see
`IMPL_VS_DESIGN.md §4`).
"""

from __future__ import annotations

import yaml

from pilot.tools import state


def _seed_state(root, session, **extra) -> str:
    sd = root / session
    sd.mkdir(parents=True, exist_ok=True)
    base = {
        "schema_version": "1.0",
        "session_id": session,
        "current_stage": "BASELINE",
        "round_id": 0,
        "stage_history": [
            {"stage": "PREFLIGHT", "status": "success", "headline": "1/1 nodes"},
        ],
    }
    base.update(extra)
    (sd / "tuning_state.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    return str(sd / "tuning_state.yaml")


def _read(path) -> dict:
    return yaml.safe_load(open(path).read())


def test_update_pointers_merges_fields_and_writes_checkpoint(tmp_path) -> None:
    root = tmp_path / "state"
    _seed_state(root, "sess1")
    out = state.update_pointers(
        "sess1",
        fields={
            "current_stage": "OPTIMIZE_LOOP.SETTLE",
            "round_id": 4,
            "champion_id": "r4_c2_fp8_delayed",
            "budget_used": {"rounds": 4, "gpu_h": 0.42},
        },
        root=str(root),
    )
    assert "checkpoints/r4" in out
    doc = _read(root / "sess1" / "tuning_state.yaml")
    assert doc["current_stage"] == "OPTIMIZE_LOOP.SETTLE"
    assert doc["round_id"] == 4
    assert doc["champion_id"] == "r4_c2_fp8_delayed"
    assert doc["budget_used"]["gpu_h"] == 0.42
    # Existing fields preserved.
    assert doc["stage_history"][0]["stage"] == "PREFLIGHT"


def test_update_pointers_appends_stage_history_entry(tmp_path) -> None:
    root = tmp_path / "state"
    _seed_state(root, "sess2")
    state.update_pointers(
        "sess2",
        append_history={
            "stage": "OPTIMIZE_LOOP.DIAGNOSE",
            "status": "success",
            "headline": "COMPUTE_BOUND, conf=0.85",
        },
        root=str(root),
    )
    doc = _read(root / "sess2" / "tuning_state.yaml")
    assert len(doc["stage_history"]) == 2
    assert doc["stage_history"][-1]["headline"] == "COMPUTE_BOUND, conf=0.85"


def test_update_pointers_combined_call(tmp_path) -> None:
    """Both arguments together: a single tool call that an Orchestrator
    round would actually make at stage exit."""
    root = tmp_path / "state"
    _seed_state(root, "sess3")
    state.update_pointers(
        "sess3",
        fields={"current_stage": "OPTIMIZE_LOOP.EXECUTE", "round_id": 1},
        append_history={"stage": "OPTIMIZE_LOOP.EXECUTE", "status": "success", "headline": "3 trials ok"},
        root=str(root),
    )
    doc = _read(root / "sess3" / "tuning_state.yaml")
    assert doc["round_id"] == 1
    assert doc["stage_history"][-1]["stage"] == "OPTIMIZE_LOOP.EXECUTE"


def test_update_pointers_raises_on_missing_session(tmp_path) -> None:
    import pytest

    with pytest.raises(state._StateError):
        state.update_pointers("ghost", fields={"x": 1}, root=str(tmp_path / "state"))
