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
        
        # Performance projection (single-node benchmarking only)
        primus projection performance --config exp.yaml
        
        # Performance projection with multinode scaling to 4 nodes
        primus projection performance --config exp.yaml --target-nodes 4
        
        # Performance projection with custom hardware config
        primus projection performance --config exp.yaml --target-nodes 8 \\
            --hardware-config hardware_mi300x.yaml
        
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
        "performance", help="Performance projection with optional multinode scaling."
    )
    add_pretrain_parser(performance)
    performance.add_argument(
        "--target-nodes",
        type=int,
        required=False,
        default=None,
        help=(
            "Target number of nodes for multinode scaling projection. "
            "If not specified, defaults to minimum nodes required by the parallelism config "
            "(TP × PP × EP × CP / GPUs_per_node). When specified, projects performance from "
            "the minimum required nodes to the target node count by scaling data parallelism."
        ),
    )
    performance.add_argument(
        "--hardware-config",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to YAML file with hardware configuration for collective communication modeling. "
            "If not provided, uses default cluster parameters.\n\n"
        ),
    )

    parser.set_defaults(func=run)

    return parser
