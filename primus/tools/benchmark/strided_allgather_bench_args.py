###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_arguments(parser: argparse.ArgumentParser):
    """
    Register Strided Allgather benchmark arguments to the CLI parser.
    """
    parser.add_argument(
        "--sizes-mb",
        type=str,
        default="64,128,256",
        help="Comma-separated message sizes in MB (per-rank send size)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Rank stride for the group, default is 8. For example, "
        "if the world size is 32, and the stride is 8, then the groups will be "
        "[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], "
        "[4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]."
        "[4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31].",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run the allgather of multiple groups in parallel, default is False. "
        "If True, the allgather of multiple groups will be run in parallel for each rank.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Timed iterations per size",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per size (not timed)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Tensor dtype",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        choices=["nccl", "gloo", "mpi"],
        help="Distributed backend",
    )

    return parser
