"""Tests for pilot.tools.learn (between-session learn loop).

Covers:
  * catalog_gap detection on a synthetic stage_history (SETTLE-promoted).
  * blocklist filter (rc=0 / champion=... must not become axes).
  * mutex fingerprint detection from a fabricated log_tail.
  * calibration drift verdict (in_band / above_band / regressed).
  * anti-pattern detection (consistent regression).
  * draft suffix uniqueness (no file collision in a tight emit loop).
  * knowledge.write extra_content passthrough.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot.tools import knowledge as knowledge_mod
from pilot.tools import learn as learn_mod

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    """A minimal session directory with tuning_state + one exec_results."""
    sd = tmp_path / "session_test"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text(
        "schema_version: '1.0'\n"
        "session_id: session_test\n"
        "current_stage: DONE\n"
        "round_id: 3\n"
        "champion_id: r1_c0\n"
        "stage_history:\n"
        "- stage: SETTLE\n"
        "  status: promoted\n"
        "  headline: 'R1 promote: champion=r1_c0 (turbo_deepep_num_cu=80) +2.13% TFLOPS'\n"
        "- stage: SETTLE\n"
        "  status: promoted\n"
        "  headline: 'R4 promoted: c2 mystery_axis=delayed +24.85% TFLOPS PROMOTED'\n"
        "- stage: EXECUTE\n"
        "  status: failed\n"
        "  headline: 'R3 c0 cuda_graph_impl mutex FAIL'\n"
    )
    return sd


# ---------------------------------------------------------------------------
# Headline parsing
# ---------------------------------------------------------------------------


def test_blocklist_filters_noise_tokens() -> None:
    """`champion=...` and `rc=...` MUST NOT be parsed as axes."""
    text = "R1 promote: champion=r1_c0 rc=0 +2.13% TFLOPS"
    out = learn_mod._extract_axis_value_gain(text, require_promote_marker=False)
    assert all(axis not in {"champion", "rc"} for axis, *_ in out)


def test_promote_marker_required_outside_settle() -> None:
    """A non-SETTLE headline without 'promote' yields no axes."""
    text = "R3 c2 mystery_axis=foo +24.85% TFLOPS"
    out = learn_mod._extract_axis_value_gain(text, require_promote_marker=True)
    assert out == []


def test_promote_marker_satisfied_emits() -> None:
    text = "R3 c2 mystery_axis=foo +24.85% TFLOPS PROMOTED"
    out = learn_mod._extract_axis_value_gain(text, require_promote_marker=True)
    assert ("mystery_axis", "foo", 24.85) in out


def test_alias_resolution_canonicalizes_omp() -> None:
    out = learn_mod._extract_axis_value_gain(
        "R7 c2 OMP=4 +3.34% TFLOPS PROMOTED", require_promote_marker=True
    )
    # OMP should be resolved to its catalog-canonical name.
    assert any(axis == "OMP_NUM_THREADS" for axis, *_ in out)


def test_min_gain_threshold_filters_noise() -> None:
    out = learn_mod._extract_axis_value_gain(
        "R3 mystery_axis=foo +0.1% TFLOPS PROMOTED", require_promote_marker=True
    )
    assert out == []


# ---------------------------------------------------------------------------
# Catalog gap finder
# ---------------------------------------------------------------------------


def test_catalog_gap_emits_for_unknown_axis(session_dir: Path) -> None:
    analysis = learn_mod.analyze(session_dir)
    axes_emitted = {g["axis"] for g in analysis.findings["catalog_gaps"]}
    # mystery_axis is not in the catalog, so it should surface as a gap.
    assert "mystery_axis" in axes_emitted
    # turbo_deepep_num_cu IS in the catalog (added by the P0 patch), so it
    # must NOT surface as a gap — that would be a false positive.
    assert "turbo_deepep_num_cu" not in axes_emitted


# ---------------------------------------------------------------------------
# Constraint gap finder (mutex fingerprint matching)
# ---------------------------------------------------------------------------


def test_mutex_fingerprint_matches_defer_emb(tmp_path: Path) -> None:
    sd = tmp_path / "ses"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text("session_id: ses\nstage_history: []\n")
    log = sd / "train.log"
    log.write_text(
        "Megatron-LM error: Cannot defer embedding wgrad compute when "
        "pipeline model parallel is not used\n"
    )
    inner = {
        "status": "failed",
        "failure": {"kind": "TRAIN_LAUNCH", "message": "job exited with status=failed"},
        "artifacts": [{"kind": "TrainLog", "ref": str(log)}],
    }
    (sd / "r3_exec_results.json").write_text(
        json.dumps([{"run_id": "t_r3_c0", "exit": 1, "stdout": json.dumps(inner), "stderr": ""}])
    )
    analysis = learn_mod.analyze(sd)
    rule_ids = [g["rule_id"] for g in analysis.findings["constraint_gaps"]]
    assert "REQ-PP-DEFER-EMB" in rule_ids


def test_mutex_fingerprint_matches_known_blocker_cg(tmp_path: Path) -> None:
    sd = tmp_path / "ses"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text("session_id: ses\nstage_history: []\n")
    log = sd / "train.log"
    log.write_text("RuntimeError: HIP error: invalid argument at backward step 4\n")
    inner = {
        "status": "failed",
        "failure": {"kind": "TRAIN_LAUNCH", "message": "job exited"},
        "artifacts": [{"kind": "TrainLog", "ref": str(log)}],
    }
    (sd / "r4_exec_results.json").write_text(
        json.dumps([{"run_id": "t_r4_c1", "exit": 1, "stdout": json.dumps(inner), "stderr": ""}])
    )
    analysis = learn_mod.analyze(sd)
    rule_ids = [g["rule_id"] for g in analysis.findings["constraint_gaps"]]
    assert "KNOWN-BLOCKER-CG-HIP" in rule_ids


def test_mutex_fingerprint_no_match_on_clean_run(tmp_path: Path) -> None:
    sd = tmp_path / "ses"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text("session_id: ses\nstage_history: []\n")
    inner = {
        "status": "completed",
        "failure": None,
        "artifacts": [],
    }
    (sd / "r5_exec_results.json").write_text(
        json.dumps([{"run_id": "t_r5_c0", "exit": 0, "stdout": json.dumps(inner), "stderr": ""}])
    )
    analysis = learn_mod.analyze(sd)
    assert analysis.findings["constraint_gaps"] == []


# ---------------------------------------------------------------------------
# Calibration drift
# ---------------------------------------------------------------------------


def test_calibration_drift_classifies_above_band(session_dir: Path) -> None:
    diagnosis = [
        {
            "candidate_axes": [
                {"axis": "mystery_axis", "expected_gain_band_pct": [1.0, 5.0]},
            ]
        }
    ]
    analysis = learn_mod.analyze(session_dir, diagnosis_reports=diagnosis)
    drifts = analysis.findings["calibration_drifts"]
    assert len(drifts) == 1
    assert drifts[0]["axis"] == "mystery_axis"
    assert drifts[0]["verdict"] == "above_band"
    assert drifts[0]["measured_gain_pct"] == pytest.approx(24.85, abs=0.01)


# ---------------------------------------------------------------------------
# Anti-pattern detection
# ---------------------------------------------------------------------------


def test_anti_pattern_requires_consistent_regression(tmp_path: Path) -> None:
    sd = tmp_path / "ses"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text(
        "session_id: ses\n"
        "run_history:\n"
        "- id: t1\n"
        "  overrides: {}\n"
        "  env_overrides: {DANGER_KNOB: 0}\n"
        "  gain_vs_champion_pct: -13.28\n"
        "- id: t2\n"
        "  overrides: {}\n"
        "  env_overrides: {DANGER_KNOB: 0}\n"
        "  gain_vs_champion_pct: -8.4\n"
        "stage_history: []\n"
    )
    analysis = learn_mod.analyze(sd)
    sigs = analysis.findings["anti_pattern_signals"]
    assert any(s["axis"] == "DANGER_KNOB" and s["value"] == 0 for s in sigs)


def test_anti_pattern_skips_mixed_signal(tmp_path: Path) -> None:
    sd = tmp_path / "ses"
    sd.mkdir()
    (sd / "tuning_state.yaml").write_text(
        "session_id: ses\n"
        "run_history:\n"
        "- id: t1\n"
        "  overrides: {}\n"
        "  env_overrides: {NEUTRAL_KNOB: 1}\n"
        "  gain_vs_champion_pct: -7.0\n"
        "- id: t2\n"
        "  overrides: {}\n"
        "  env_overrides: {NEUTRAL_KNOB: 1}\n"
        "  gain_vs_champion_pct: 2.0\n"
        "stage_history: []\n"
    )
    analysis = learn_mod.analyze(sd)
    assert analysis.findings["anti_pattern_signals"] == []


# ---------------------------------------------------------------------------
# Draft emission
# ---------------------------------------------------------------------------


def test_emit_drafts_produces_unique_files(session_dir: Path, tmp_path: Path) -> None:
    drafts_root = tmp_path / "drafts"
    drafts_root.mkdir()
    diagnosis = [
        {
            "candidate_axes": [
                {"axis": "mystery_axis", "expected_gain_band_pct": [1.0, 5.0]},
            ]
        }
    ]
    analysis = learn_mod.analyze(session_dir, diagnosis_reports=diagnosis)
    results = learn_mod.emit_drafts(analysis, drafts_root=str(drafts_root))
    assert len(results) >= 1
    paths = [r["written_path"] for r in results if r["written_path"]]
    assert len(paths) == len(set(paths)), f"draft path collision: {paths}"
    for r in results:
        assert (
            r["accepted"] is True
            or "no artifact evidence" in " ".join(r["reasons"])
            or "missing plan binding" in " ".join(r["reasons"])
        )


def test_extra_content_passthrough(tmp_path: Path) -> None:
    """knowledge.write must carry report['content'] into the draft."""
    drafts_root = tmp_path / "drafts"
    drafts_root.mkdir()
    report = {
        "session": {"plan_name": "demo", "plan_ref": "ref://demo"},
        "verdict": {"headline": "demo finding"},
        "tuning": {"champion": {}},
        "artifacts": [{"kind": "round_result", "ref": "t_demo_1"}],
        "content": {
            "rule_id": "MUTEX-DEMO",
            "axes": ["axis_a", "axis_b"],
            "rationale": "demo rationale",
        },
    }
    res = knowledge_mod.write(report, "failure_pattern", drafts_root=str(drafts_root), id_suffix="MUTEX-DEMO")
    assert res["accepted"] is True
    import yaml

    written = yaml.safe_load(open(res["written_path"]))
    assert written["content"]["rule_id"] == "MUTEX-DEMO"
    assert written["content"]["axes"] == ["axis_a", "axis_b"]
    assert "MUTEX-DEMO" in written["draft_id"]


def test_write_analysis_persists_yaml(
    session_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(learn_mod, "_PILOT_ROOT", tmp_path)
    analysis = learn_mod.analyze(session_dir)
    out_path = learn_mod.write_analysis(analysis)
    assert Path(out_path).exists()
    import yaml

    payload = yaml.safe_load(open(out_path))
    assert payload["session_id"] == "session_test"
    assert "findings" in payload
