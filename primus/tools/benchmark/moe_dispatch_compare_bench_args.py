###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_moe_dispatch_compare_parser(parser: argparse.ArgumentParser):
    """Register MoE dispatcher comparison benchmark arguments."""
    parser.add_argument(
        "--cases",
        type=str,
        default="deepep,comet-ll",
        help="Comma-separated cases to run. Supported: deepep, comet-ll",
    )
    parser.add_argument("--tokens", type=int, default=128, help="Number of tokens per rank")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden size")
    parser.add_argument("--num-experts", type=int, default=256, help="Total experts across all ranks")
    parser.add_argument("--topk", type=int, default=8, help="Top-k experts per token")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16"],
        help="Input dtype (currently bf16 only)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for each case",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Measured iterations for each case",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    return parser
