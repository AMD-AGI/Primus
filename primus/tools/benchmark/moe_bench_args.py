###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_moe_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--op",
        type=str,
        nargs="+",
        default=["router", "dispatch"],
        choices=["router", "dispatch", "grouped-gemm"],
        help="MoE microbench operations.",
    )
    parser.add_argument("--tokens", type=int, default=8192, help="Token count for synthetic MoE workload.")
    parser.add_argument("--hidden-size", type=int, default=4096, help="Hidden size for synthetic tensors.")
    parser.add_argument("--num-experts", type=int, default=64, help="Total number of experts.")
    parser.add_argument("--topk", type=int, default=8, help="Router top-k experts per token.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each op multiple times for statistical stability.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Tensor dtype used for synthetic benchmark.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a fast smoke benchmark with reduced iterations and token count.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append benchmark results to output files instead of overwrite.",
    )
    parser.add_argument(
        "--per-rank",
        action="store_true",
        help="Emit per-rank summary stats for each op.",
    )
    parser.add_argument(
        "--per-rank-file",
        type=str,
        default="",
        help="Output file for per-rank summary. Empty means derive from --output-file.",
    )
    parser.add_argument(
        "--per-iter-trace",
        action="store_true",
        help="Emit per-iteration trace rows (can be large).",
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="",
        help="Output file for trace rows. Empty means derive from --output-file.",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=0,
        help="Max iters to record per repeat. 0 means all.",
    )
    parser.add_argument(
        "--trace-ops",
        type=str,
        default="",
        help="Comma-separated ops to trace, e.g. 'dispatch,router'. Empty means all.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output/benchmark/moe_bench.csv",
        help="Output report file path.",
    )
    return parser
