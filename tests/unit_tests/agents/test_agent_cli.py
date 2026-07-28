###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the tuning-agent CLI (``cli.py``) arg parsing + early exits.

The full main() pipeline (load_config -> evaluator -> LLM) is out of scope; we
cover argument parsing and the missing-file guards that return before any heavy
work.
"""

from __future__ import annotations

import pytest

pytest.importorskip("primus.agents.tuning_agent.cli")

from primus.agents.tuning_agent.cli import _parse_args, main  # noqa: E402


def test_parse_args_defaults():
    ns = _parse_args(["--workload", "w.yaml", "--target-cluster", "tc.yaml"])
    assert ns.mode == "full"
    assert ns.profiling_mode == "simulate"
    assert ns.seed_budget == 12
    assert ns.dry_run is False and ns.seed_only is False


def test_parse_args_flags_and_choices():
    ns = _parse_args(
        [
            "--workload",
            "w",
            "--target-cluster",
            "tc",
            "--dry-run",
            "--seed-only",
            "--seed-budget",
            "5",
            "--mode",
            "memory-real",
            "--profiling-mode",
            "benchmark",
        ]
    )
    assert ns.dry_run is True and ns.seed_only is True
    assert ns.seed_budget == 5 and ns.mode == "memory-real" and ns.profiling_mode == "benchmark"


def test_parse_args_requires_workload_and_cluster():
    with pytest.raises(SystemExit):
        _parse_args(["--target-cluster", "tc"])
    with pytest.raises(SystemExit):
        _parse_args(["--workload", "w"])


def test_parse_args_rejects_invalid_mode():
    with pytest.raises(SystemExit):
        _parse_args(["--workload", "w", "--target-cluster", "tc", "--mode", "bogus"])


def test_main_returns_2_on_missing_workload(tmp_path):
    rc = main(["--workload", str(tmp_path / "nope.yaml"), "--target-cluster", str(tmp_path / "tc.yaml")])
    assert rc == 2


def test_main_returns_2_on_missing_target_cluster(tmp_path):
    wl = tmp_path / "wl.yaml"
    wl.write_text("modules: {}\n")
    rc = main(["--workload", str(wl), "--target-cluster", str(tmp_path / "nope.yaml")])
    assert rc == 2
