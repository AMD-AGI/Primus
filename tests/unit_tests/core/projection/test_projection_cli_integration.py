###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""End-to-end integration tests for the ``primus projection`` CLI.

These drive the real CLI entry point in-process (no subprocess), exercising the
performance-projection engine and the analytical simulation backends:

* ``projection memory``      — pure analytical, no extra deps (always runs).
* ``projection performance --profiling-mode simulate`` — uses the Origami GEMM
  model + SDPA simulator on CPU (no GPU). Skipped when ``origami`` is missing.

The Origami backend is the ROCm package
(``git+https://github.com/ROCm/rocm-libraries.git#subdirectory=shared/origami/python``);
it ships in the CI image. Where it is absent, the simulate cases skip cleanly.
"""

import argparse
from pathlib import Path

import pytest

PRIMUS_ROOT = Path(__file__).resolve().parents[4]
CFG_DIR = PRIMUS_ROOT / "examples" / "megatron" / "configs" / "MI300X"
DENSE_CFG = CFG_DIR / "llama3_8B-BF16-pretrain.yaml"
MOE_CFG = CFG_DIR / "deepseek_v2_lite-BF16-pretrain.yaml"


def _run_projection(argv):
    """Build the real projection CLI parser and invoke it in-process."""
    from primus.cli.subcommands import projection as proj_cmd

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    proj_cmd.register_subcommand(subparsers)
    args = parser.parse_args(argv)
    args.func(args, [])


@pytest.mark.parametrize("cfg", [DENSE_CFG, MOE_CFG], ids=["dense", "moe"])
def test_memory_projection_cli(cfg, capsys):
    """`projection memory` runs end-to-end (analytical, no GPU/origami)."""
    _run_projection(["projection", "memory", "--config", str(cfg)])
    out = capsys.readouterr().out
    assert "Memory Projection Summary" in out
    # Regression guard for the dense-config MoE None bug (would print an error line).
    assert "Error calculating metrics" not in out


@pytest.mark.parametrize("cfg", [DENSE_CFG, MOE_CFG], ids=["dense", "moe"])
def test_performance_simulate_cli(cfg, capsys):
    """`projection performance --profiling-mode simulate` runs on CPU via Origami."""
    pytest.importorskip("origami", reason="ROCm origami GEMM backend not installed")
    _run_projection(
        [
            "projection",
            "performance",
            "--config",
            str(cfg),
            "--profiling-mode",
            "simulate",
            "--gpu-arch",
            "mi300x",
            "--target-nodes",
            "2",
        ]
    )
    out = capsys.readouterr().out
    assert any(k in out for k in ("Iteration Time", "Projection Results", "Tokens/s"))
