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
    elif args.suite == "performance":
        from primus.pretrain import setup_backend_path

        setup_backend_path(framework="megatron", verbose=True)

        from primus.core.projection.performance_projection import (
            launch_projection_from_cli,
        )

        launch_projection_from_cli(args, overrides)
    else:
        raise NotImplementedError(f"Unsupported projection suite: {args.suite}")


def register_subcommand(subparsers):
    """
    Register the 'projection' subcommand to the main CLI parser.

    Example:
        # Memory projection
        primus projection memory --config exp.yaml
        
        # Performance projection (benchmarking on baseline nodes)
        primus projection performance --config exp.yaml
        
        # Performance projection with multinode scaling (set PROJECTION_NNODES)
        PROJECTION_NNODES=4 primus projection performance --config exp.yaml
        
        # Skip GPU benchmarking and provide baseline time manually
        primus projection performance --config exp.yaml --baseline-time 1234.56 --baseline-nodes 1
        PROJECTION_NNODES=4 primus projection performance --config exp.yaml --baseline-time 1234.56 --baseline-nodes 1
        
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

    # ---------- memory ----------
    memory = suite_parsers.add_parser("memory", help="Memory projection only (per-GPU memory analysis).")
    from primus.core.launcher.parser import add_pretrain_parser

    add_pretrain_parser(memory)

    # ---------- performance ----------
    performance = suite_parsers.add_parser(
        "performance", 
        help="Performance projection. Set PROJECTION_NNODES env var for multinode scaling projection."
    )
    add_pretrain_parser(performance)
    performance.add_argument(
        "--baseline-time",
        type=float,
        required=False,
        default=None,
        help="Optional: Measured baseline iteration time in milliseconds. If not provided, will automatically run GPU benchmarking. Use with --baseline-nodes to specify the number of nodes this time was measured on.",
    )
    performance.add_argument(
        "--baseline-nodes",
        type=int,
        required=False,
        default=None,
        help="Optional: Number of nodes used for the baseline time measurement. If not provided, defaults to minimum nodes required by parallelism config.",
    )
    performance.add_argument(
        "--hardware-config",
        type=str,
        required=False,
        default=None,
        help="Path to YAML file with hardware configuration (optional)",
    )

    parser.set_defaults(func=run)

    return parser
