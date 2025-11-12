###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
from typing import List, Optional

from primus.cli.base import CommandBase

logger = logging.getLogger(__name__)


class TrainCommand(CommandBase):
    """Command for training workflows.

    This command provides access to various training workflows:
    - pretrain: Pre-training large language models
    - posttrain: Post-training (alignment, preference tuning) - Coming Soon
    - finetune: Fine-tuning on downstream tasks - Coming Soon
    """

    @classmethod
    def name(cls) -> str:
        return "train"

    @classmethod
    def help(cls) -> str:
        return "Launch Primus pretrain with Megatron or TorchTitan"

    @classmethod
    def description(cls) -> str:
        return """
Training workflow management for large language models.

Supported workflows:
  pretrain   - Pre-training with Megatron-LM or TorchTitan
  posttrain  - Post-training workflows (Coming Soon)
  finetune   - Fine-tuning on downstream tasks (Coming Soon)

Examples:
  # Launch pre-training with Megatron
  primus train pretrain --config config.yaml --backend megatron

  # Launch pre-training with TorchTitan
  primus train pretrain --config config.yaml --backend torchtitan
        """

    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register command-specific arguments.

        Supported suites (training workflows):
            - pretrain: Pre-training workflow (Megatron, TorchTitan, etc.)

        Future extensions:
            - posttrain: Post-training workflow (alignment, preference tuning, etc.)
            - finetune: Fine-tuning workflow for downstream tasks

        Example:
            primus train pretrain --config exp.yaml --backend-path /path/to/megatron
        """
        parser.description = cls.description()
        suite_parsers = parser.add_subparsers(dest="suite", required=True)

        # ---------- pretrain ----------
        pretrain = suite_parsers.add_parser(
            "pretrain",
            help="Pre-training workflow.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        from primus.core.launcher.parser import add_pretrain_parser

        add_pretrain_parser(pretrain)

    @classmethod
    def validate_args(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> bool:
        """Validate training arguments.

        Args:
            args: Parsed arguments.
            unknown_args: Unknown arguments (passed to backend).

        Returns:
            True if validation passes, False otherwise.
        """
        if not hasattr(args, "suite"):
            logger.error("No training suite specified")
            return False

        valid_suites = ["pretrain"]
        if args.suite not in valid_suites:
            logger.error(f"Invalid training suite: {args.suite}")
            logger.error(f"Valid suites: {', '.join(valid_suites)}")
            return False

        return True

    @classmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the training command.

        Args:
            args: Parsed arguments.
            unknown_args: Unknown arguments to pass to training backend.
        """
        logger.info(f"Starting training suite: {args.suite}")

        if args.suite == "pretrain":
            logger.info("Launching pre-training workflow")
            from primus.pretrain import launch_pretrain_from_cli

            launch_pretrain_from_cli(args, unknown_args)
            logger.info("Pre-training workflow completed")
        else:
            # This should not happen due to validation, but kept for safety
            raise NotImplementedError(f"Training suite '{args.suite}' is not yet implemented")
