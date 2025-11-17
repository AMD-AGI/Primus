###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_gemm_parser(parser: argparse.ArgumentParser):
    """
    Register GEMM benchmark arguments to the CLI parser.
    """
    parser.add_argument("--M", type=int, default=4096, help="GEMM M dimension (default: 4096)")
    parser.add_argument("--N", type=int, default=4096, help="GEMM N dimension (default: 4096)")
    parser.add_argument("--K", type=int, default=4096, help="GEMM K dimension (default: 4096)")
    parser.add_argument("--trans_a", action="store_true", help="Transpose A matrix")
    parser.add_argument("--trans_b", action="store_true", help="Transpose B matrix")
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Data type for GEMM computation."
    )
    parser.add_argument("--duration", type=int, default=10, help="Benchmark duration in seconds.")
    parser.add_argument(
        "--output-file",
        default="./gemm_report.md",
        help="Path to save results (.md/.csv/.tsv/.jsonl[.gz]). If not set or '-', print to stdout (Markdown).",
    )

    return parser
