###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
from typing import List, Optional

from primus.cli.base import CommandBase


class TrainCommand(CommandBase):
    """Command for training workflows."""

    @classmethod
    def name(cls) -> str:
        return "train"

    @classmethod
    def help(cls) -> str:
        return "Launch Primus pretrain with Megatron or TorchTitan"

    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments.

        Supported suites (training workflows):
            - pretrain: Pre-training workflow (Megatron, TorchTitan, etc.)

        Future extensions:
            - posttrain: Post-training workflow (alignment, preference tuning, etc.)

        Example:
            primus train pretrain --config exp.yaml --backend-path /path/to/megatron
        """
        parser.description = "Primus training entry. Supports pretrain now; posttrain/finetune/evaluate reserved for future use."
        suite_parsers = parser.add_subparsers(dest="suite", required=True)

        # ---------- pretrain ----------
        pretrain = suite_parsers.add_parser("pretrain", help="Pre-training workflow.")
        from primus.core.launcher.parser import add_pretrain_parser

        add_pretrain_parser(pretrain)

    @classmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        if args.suite == "pretrain":
            from primus.pretrain import launch_pretrain_from_cli

            launch_pretrain_from_cli(args, unknown_args)
        else:
            raise NotImplementedError(f"Unsupported train suite: {args.suite}")
