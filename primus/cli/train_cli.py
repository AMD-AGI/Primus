###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.
    Dispatch to pretrain, prepare, etc.
    """
    if args.suite == "pretrain":
        from primus.pretrain import launch_pretrain_from_cli

        launch_pretrain_from_cli(args, overrides)
    else:
        raise NotImplementedError(f"Unsupported train suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'train' subcommand to the main CLI parser.

    Supported suites (training workflows):
        - pretrain: Pre-training workflow (Megatron, TorchTitan, etc.)
        # Future extensions:
        - finetune: Fine-tuning workflow
        - evaluate: Evaluation workflow

    Example:
        primus-cli train pretrain --config exp.yaml --backend-path /path/to/megatron

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser("train", help="Launch Primus pretrain with Megatron or TorchTitan")
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- pretrain ----------
    pretrain = suite_parsers.add_parser("pretrain", help="Pre-training workflow.")
    from primus.core.launcher.parser import add_pretrain_parser

    add_pretrain_parser(pretrain)

    parser.set_defaults(func=run)

    return parser
