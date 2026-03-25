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
        profiling_mode = getattr(args, "profiling_mode", "benchmark")
        needs_backend = profiling_mode != "simulate"

        if needs_backend:
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
        "--benchmark-gpus",
        type=int,
        required=False,
        default=None,
        help=(
            "Number of GPUs to use for benchmarking. When set lower than GPUS_PER_NODE,\n"
            "enables sub-node benchmarking with analytical upscaling to the full node.\n"
            "If not specified, defaults to GPUS_PER_NODE (full node benchmarking).\n"
            "Parallelism dimensions (PP, EP, TP) are automatically reduced to fit the\n"
            "benchmark GPU count and restored analytically during projection.\n"
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
    performance.add_argument(
        "--profiling-mode",
        type=str,
        required=False,
        default="benchmark",
        choices=["benchmark", "simulate", "both"],
        help=(
            "Profiling mode for layer timing:\n"
            "  benchmark  - Run actual GPU benchmarks (default, requires GPU)\n"
            "  simulate   - Use simulation backends (origami for GEMM,\n"
            "               analytical model for SDPA). No GPU required.\n"
            "  both       - Run both benchmark and simulation, report side-by-side\n"
        ),
    )
    performance.add_argument(
        "--gemm-backend",
        type=str,
        required=False,
        default=None,
        choices=["origami"],
        help=(
            "GEMM simulation backend (only used when --profiling-mode is 'simulate' or 'both').\n"
            "  origami  - Open-source GEMM performance model (default)\n"
        ),
    )
    performance.add_argument(
        "--gpu-arch",
        type=str,
        required=False,
        default=None,
        help=(
            "Target GPU architecture for simulation (e.g. 'mi300x', 'gfx942', 'mi355x', 'gfx950').\n"
            "If not specified, auto-detected or uses PRIMUS_GPU_ARCH env var.\n"
        ),
    )
    performance.add_argument(
        "--gpu-clock-mhz",
        type=int,
        required=False,
        default=None,
        help=(
            "Override the GPU compute clock frequency in MHz for simulation.\n"
            "If not specified, uses the default from the hardware profile for the\n"
            "given --gpu-arch (e.g. 2100 MHz for MI300X/MI325X).\n"
            "Can also be set via the PRIMUS_GPU_CLOCK_MHZ env var.\n"
            "Example: --gpu-clock-mhz 1500\n"
        ),
    )
    performance.add_argument(
        "--pipeline-schedule-algorithm",
        type=str,
        required=False,
        default="auto",
        choices=["auto", "zerobubble", "zbv-formatted", "zbv-greedy", "megatron-ilp", "all"],
        help=(
            "Pipeline scheduling algorithm to use for simulation:\n"
            "  auto          - Default Primus behavior (zerobubble if enabled, else 1f1b)\n"
            "  zerobubble    - Primus zero-bubble scheduler (VPP=1 only)\n"
            "  zbv-formatted - ZBV Formatted scheduler (VPP=2 only)\n"
            "  zbv-greedy    - ZBV Greedy scheduler with half mem config (VPP=2 only)\n"
            "  megatron-ilp  - Megatron ILP-based zero-bubble scheduler\n"
            "  all           - Run all available schedulers and compare results\n"
        ),
    )
    parser.set_defaults(func=run)

    return parser
