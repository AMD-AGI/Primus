###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "tools" / "preflight_bisect" / "bisect.py"
MODULE_NAME = "preflight_bisect_script"

spec = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert spec is not None and spec.loader is not None
preflight_bisect = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = preflight_bisect
spec.loader.exec_module(preflight_bisect)


def _make_args(max_concurrent_trials: int = 2) -> argparse.Namespace:
    return argparse.Namespace(max_concurrent_trials=max_concurrent_trials)


def test_bisect_prunes_passing_sibling_subtree(monkeypatch, tmp_path):
    bad_node = "node07"
    nodes = [f"node{i:02d}" for i in range(1, 9)]

    def fake_run_trial(nodes, trial_idx, state, args, runner, out_dir):
        time.sleep(0.01 if nodes[0] >= "node05" else 0.0)
        return "fail" if bad_node in nodes else "pass"

    monkeypatch.setattr(preflight_bisect, "run_trial", fake_run_trial)

    state = preflight_bisect.BisectState(max_concurrent_trials=2)
    suspects = preflight_bisect.bisect(
        nodes,
        state,
        _make_args(max_concurrent_trials=2),
        runner=tmp_path / "runner.sh",
        out_dir=tmp_path,
    )

    assert suspects == [bad_node]

    ordered_trials = state.ordered_trials()
    trial_ids = [trial["idx"] for trial in ordered_trials]
    assert trial_ids == list(range(len(ordered_trials)))

    good_half = set(nodes[:4])
    assert any(trial["nodes"] == nodes[:4] and trial["status"] == "pass" for trial in ordered_trials)
    assert not any(set(trial["nodes"]).issubset(good_half) and len(trial["nodes"]) < 4 for trial in ordered_trials)


def test_write_summary_sorts_trials_by_idx(tmp_path):
    nodes = ["node01", "node02", "node03", "node04"]
    trials = [
        {"idx": 2, "n": 2, "status": "fail", "nodes": ["node03", "node04"]},
        {"idx": 0, "n": 4, "status": "fail", "nodes": nodes},
        {"idx": 1, "n": 2, "status": "pass", "nodes": ["node01", "node02"]},
    ]

    preflight_bisect.write_summary(tmp_path, nodes, ["node03"], trials)

    summary_lines = (tmp_path / "summary.txt").read_text(encoding="utf-8").splitlines()
    assert summary_lines[1].startswith("[000]")
    assert summary_lines[2].startswith("[001]")
    assert summary_lines[3].startswith("[002]")
    assert summary_lines[-1] == "SUSPECT_NODES: node03"


def test_main_rejects_parallel_scancel(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["bisect.py", "--nodelist", "node[01-02]", "--scancel-user-on-hang"],
    )

    with pytest.raises(SystemExit) as exc_info:
        preflight_bisect.main()

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--max-concurrent-trials=1" in captured.err
