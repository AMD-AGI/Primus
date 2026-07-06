###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for ``benchmark.register_subcommand`` (pure argparse surface).

Exercises only the parser wiring (suite registration, per-suite arg defaults
and choice validation). This also covers the ``primus/tools/benchmark/*_args``
helpers that ``register_subcommand`` imports to build each suite parser.
``run()`` initializes torch.distributed and runs GPU/RCCL benches, so it is
intentionally out of scope.
"""

from __future__ import annotations

import argparse

import pytest

from primus.cli.subcommands import benchmark

_SUITES = ["gemm", "attention", "gemm-dense", "gemm-deepseek", "strided-allgather", "rccl"]


def _build_parser():
    parser = argparse.ArgumentParser(prog="primus")
    subparsers = parser.add_subparsers(dest="cmd")
    returned = benchmark.register_subcommand(subparsers)
    return parser, returned


def test_register_returns_parser_wired_to_run():
    _, returned = _build_parser()
    assert returned.get_default("func") is benchmark.run


def test_suite_is_required():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark"])


@pytest.mark.parametrize("suite", _SUITES)
def test_each_suite_parses(suite):
    parser, _ = _build_parser()
    args = parser.parse_args(["benchmark", suite])
    assert args.suite == suite


def test_unknown_suite_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "bogus"])


# ─────────────────────────────────────────────────────────────────────────────
# gemm suite (tools/benchmark/gemm_bench_args)
# ─────────────────────────────────────────────────────────────────────────────


def test_gemm_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["benchmark", "gemm"])
    assert (args.M, args.N, args.K) == (4096, 4096, 4096)
    assert args.dtype == "bf16"
    assert args.trans_a is False and args.trans_b is False


def test_gemm_dtype_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "gemm", "--dtype", "int4"])


# ─────────────────────────────────────────────────────────────────────────────
# rccl suite (tools/benchmark/rccl_bench_args)
# ─────────────────────────────────────────────────────────────────────────────


def test_rccl_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["benchmark", "rccl"])
    assert args.scale == "log2"
    assert args.min_bytes == "1K" and args.max_bytes == "128M"


def test_rccl_op_accepts_multiple_values():
    parser, _ = _build_parser()
    args = parser.parse_args(["benchmark", "rccl", "--op", "all_reduce", "all_gather"])
    assert args.op == ["all_reduce", "all_gather"]


def test_rccl_op_invalid_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "rccl", "--op", "bogus_collective"])


# ─────────────────────────────────────────────────────────────────────────────
# attention suite (tools/benchmark/attention_bench_args)
# ─────────────────────────────────────────────────────────────────────────────


def test_attention_defaults():
    parser, _ = _build_parser()
    args = parser.parse_args(["benchmark", "attention"])
    assert args.backend == "flash"
    assert args.dtype == "bf16"


def test_attention_backend_choice_rejected():
    parser, _ = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["benchmark", "attention", "--backend", "bogus"])
