###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################


def run(args, overrides):
    """
    Entry point for the 'projection' subcommand.
    """
    if args.suite == "memory":
        from primus.core.projection.memory_projection import launch_projection_from_cli
        launch_projection_from_cli(args, overrides)
    else:
        raise NotImplementedError(f"Unsupported projection suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'projection' subcommand to the main CLI parser.

    Example:
        primus projection memory --config exp.yaml
    Args:
        subparsers: argparse subparsers object from main.py

    Returns:
        parser: The parser for this subcommand
    """

    parser = subparsers.add_parser(
        "projection",
        help="Pre-training performance projection tool",
        description="Primus performance projection entry point.",
    )
    suite_parsers = parser.add_subparsers(dest="suite", required=True)

    # ---------- pretrain ----------
    pretrain = suite_parsers.add_parser("memory", help="Memory projection.")
    from primus.core.launcher.parser import add_pretrain_parser

    add_pretrain_parser(pretrain)

    parser.set_defaults(func=run)

    return parser
