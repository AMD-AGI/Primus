###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``projection.register_subcommand`` (pure argparse surface).

Only the parser wiring is exercised: suite registration, per-suite argument
defaults, choice validation, and deprecated aliases. ``run()`` imports the
backend / projection engines (GPU) and is intentionally out of scope.
"""

from __future__ import annotations

import argparse

import pytest

from primus.cli.subcommands import projection


def _build_parser():
    parser = argparse.ArgumentParser(prog="primus")
    subparsers = parser.add_subparsers(dest="cmd")
    returned = projection.register_subcommand(subparsers)
    return parser, returned


def test_register_returns_parser_wired_to_run():
    _, returned = _build_parser()
    assert returned.get_default("func") is projection.run


def test_suite_is_required():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["projection"])


@pytest.mark.parametrize("suite", ["memory", "performance", "both"])
def test_each_suite_parses_config(suite):
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", suite, "--config", "exp.yaml"])
    assert args.suite == suite
    assert args.config == "exp.yaml"


def test_exp_alias_maps_to_config():
    # add_pretrain_parser registers --config with a --exp alias.
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "memory", "--exp", "y.yaml"])
    assert args.config == "y.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# memory suite
# ─────────────────────────────────────────────────────────────────────────────


def test_memory_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "memory", "--config", "x"])
    assert args.memory_mode == "benchmark"
    assert args.memory_safety_margin == pytest.approx(0.05)
    assert args.pipeline_schedule_algorithm == "auto"
    assert args.target_nodes is None
    assert args.benchmark_gpus is None
    assert args.save_benchmark is None
    assert args.load_benchmark is None


def test_memory_compute_baseline_alias_present():
    # Memory side exposes --compute-baseline as a deprecated load alias.
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "memory", "--config", "x", "--compute-baseline", "a.json"])
    assert args.compute_baseline == "a.json"


def test_memory_save_profiling_alias_present():
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "memory", "--config", "x", "--save-profiling", "out.json"])
    assert args.save_profiling == "out.json"


def test_memory_mode_invalid_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["projection", "memory", "--config", "x", "--memory-mode", "bogus"])


# ─────────────────────────────────────────────────────────────────────────────
# performance suite
# ─────────────────────────────────────────────────────────────────────────────


def test_performance_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "performance", "--config", "x"])
    assert args.profiling_mode == "benchmark"
    assert args.sync_free_stage == 0
    assert args.enable_zero_bubble is False
    assert args.enable_deepep is False
    assert args.gemm_backend is None


def test_performance_store_true_flags():
    parser, _ = _build_parser()
    args = parser.parse_args(
        ["projection", "performance", "--config", "x", "--enable-zero-bubble", "--enable-deepep"]
    )
    assert args.enable_zero_bubble is True
    assert args.enable_deepep is True


def test_performance_gemm_backend_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["projection", "performance", "--config", "x", "--gemm-backend", "cutlass"])


def test_performance_profiling_mode_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["projection", "performance", "--config", "x", "--profiling-mode", "bogus"])


# ─────────────────────────────────────────────────────────────────────────────
# both suite
# ─────────────────────────────────────────────────────────────────────────────


def test_both_exposes_safety_margin_but_not_memory_mode():
    parser, _ = _build_parser()
    args = parser.parse_args(["projection", "both", "--config", "x"])
    # `both` always runs a fresh bench, so --memory-mode is not registered.
    assert not hasattr(args, "memory_mode")
    assert args.memory_safety_margin == pytest.approx(0.05)
    # perf knobs are present on `both`.
    assert args.profiling_mode == "benchmark"
