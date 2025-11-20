###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


from primus.core.launcher.parser import add_pretrain_parser


def run(args, overrides):
    """
    Entry point for the 'train' subcommand.
    """
    if args.suite == "pretrain":
        from primus.core.launcher.pretrain_launcher import launch_pretrain

        launch_pretrain(args, overrides)

    # elif args.suite == "sft":
    #     from primus.sft import launch_sft_from

    #     launch_sft_from(args, overrides)
    else:
        raise NotImplementedError(f"Unsupported train suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'train' subcommand to the main CLI parser.

    Supported suites (training workflows):
        - pretrain: Pre-training workflow (Megatron, TorchTitan, etc.)

    Future extensions:
        - posttrain: Post-training workflow (alignment, preference tuning, etc.)

    Example:
        primus train pretrain --config exp.yaml --backend-path /path/to/megatron

    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser(
        "train",
        help="Launch Primus pretrain with Megatron or TorchTitan",
        description="Primus training entry. Supports pretrain now; posttrain/finetune/evaluate reserved for future use.",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- pretrain ----------
    pretrain = suite_parsers.add_parser("pretrain", help="Pre-training workflow.")

    add_pretrain_parser(pretrain)

    parser.set_defaults(func=run)

    return parser
