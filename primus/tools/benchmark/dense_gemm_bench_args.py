###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_gemm_parser(parser: argparse.ArgumentParser):
    """
    Register Dense GEMM benchmark arguments to the CLI parser.
    """
    parser.add_argument("--model", default=None, help="Model name (e.g., Llama3.1_8B)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2"],
        default="bf16",
        help="Data type for GEMM operations. fp8_e4m3 (E4M3) for training, fp8_e5m2 (E5M2) for inference",
    )
    parser.add_argument("--mbs", type=int, default=1, help="Microbatch size")
    parser.add_argument("--output-file", default="./gemm-dense_report.md")
    parser.add_argument("--duration", type=int, default=3, help="Benchmark duration per shape (sec)")
    return parser
