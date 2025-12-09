###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Train Launcher (CLI entry)

This module provides:
  - add_train_parser(): register CLI arguments
  - launch_train():    entry point used by Primus CLI or direct execution

It delegates the actual orchestration to PrimusRuntime.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from primus.core.runtime.train_runtime import PrimusRuntime


def add_train_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Register train-related arguments to the given parser.

    This function is intended to be used by the top-level Primus CLI.
    """
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config file (alias: --exp)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to data directory [default: ./data]",
    )
    parser.add_argument(
        "--backend_path",
        nargs="?",
        default=None,
        help=(
            "Optional backend import path for the selected backend "
            "(e.g., Megatron or TorchTitan). If provided, it will be "
            "appended to PYTHONPATH dynamically."
        ),
    )
    parser.add_argument(
        "--export_config",
        type=str,
        help="Optional path to export the final merged config to a file.",
    )
    return parser


def launch_train(args, overrides: Optional[List[str]], module: str) -> None:
    """
    Unified training entry point (framework-agnostic).

    This function is intentionally thin and delegates orchestration to
    PrimusRuntime so it can be reused from different frontends
    (direct CLI / Slurm / container / Python API).
    """
    runtime = PrimusRuntime(args=args)
    runtime.run_train_module(module_name=module, overrides=overrides or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Primus Train Launcher")
    add_train_parser(parser)

    # parse_known_args allows arbitrary backend-specific flags to pass through
    args, unknown_args = parser.parse_known_args()

    # Use unknown_args as CLI overrides (e.g., batch_size=32, lr=0.001)
    # NOTE: module is hard-coded to "pretrain" for standalone usage.
    launch_train(args, overrides=unknown_args, module="pretrain")
