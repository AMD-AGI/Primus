###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import os


def add_rccl_parser(parser: argparse.ArgumentParser):
    """
    Register RCCL benchmark arguments to the CLI parser.
    """

    parser.add_argument(
        "--op",
        nargs="+",
        default=["allreduce"],
        choices=["all_reduce", "broadcast", "reduce_scatter", "all_gather", "alltoall"],
        help="Collectives to run",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default=None,
        help="Explicit list of sizes, e.g. '1K,2K,4K,8K,1M'. Overrides min/max/num.",
    )
    parser.add_argument("--min-bytes", type=str, default="1K", help="Minimum message size (e.g. 1K / 1M)")
    parser.add_argument("--max-bytes", type=str, default="128M", help="Maximum message size")
    parser.add_argument("--num-sizes", type=int, default=12, help="Number of sizes if generated")
    parser.add_argument(
        "--scale",
        type=str,
        choices=["log2", "linear"],
        default="log2",
        help="Sweep scale when generating sizes",
    )

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each {op,size} combination multiple times for statistical stability.",
    )
    parser.add_argument(
        "--aggregate-repeat",
        action="store_true",
        help="Emit an additional summary row that aggregates repeat results (mean/stats).",
    )
    parser.add_argument("--check", action="store_true", help="Enable lightweight correctness checks")
    parser.add_argument(
        "--output-file",
        type=str,
        default="./rccl_report.md",
        help="Path to save results (.md/.csv/.tsv/.jsonl[.gz]). If not set or '-', print to stdout (Markdown).",
    )
    parser.add_argument("--append", action="store_true", help="Append to file instead of overwrite.")

    parser.add_argument(
        "--per-rank", action="store_true", help="Emit per-rank summary stats (one line per {op,size,rank})."
    )
    parser.add_argument(
        "--per-rank-file",
        type=str,
        default="",
        help="Output path for per-rank stats. If empty, will derive from --output-file with suffix '_rank'.",
    )

    parser.add_argument(
        "--per-iter-trace",
        action="store_true",
        help="Emit per-iteration trace (large). Use filters to limit.",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="",
        help="Output path for per-iteration trace (JSONL/CSV/TSV). If empty, derive from --output-file with suffix '_trace'.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=0,
        help="Max iters to record per {op,size}. 0 = all (careful: large).",
    )
    parser.add_argument(
        "--trace-ops",
        type=str,
        default="",
        help="Comma-separated ops to trace (e.g., 'alltoall,allreduce'). Empty = all.",
    )
    parser.add_argument(
        "--trace-sizes",
        type=str,
        default="",
        help="Comma-separated sizes to trace (e.g., '1M,16M'). Empty = all.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=os.getenv("PRIMUS_CLUSTER", "amd-aig-poolside"),
        help="Cluster/Location label to include in the report preamble.",
    )

    return parser
