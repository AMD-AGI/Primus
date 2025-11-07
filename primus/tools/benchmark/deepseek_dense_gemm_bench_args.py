###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_gemm_parser(parser: argparse.ArgumentParser):
    """
    Register DeepSeek GEMM benchmark arguments to the CLI parser.
    """
    parser.add_argument("--model", default=None, help="Model name (Deepseek_V2, Deepseek_V3, etc.)")
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=12288)
    parser.add_argument("--kv-lora-rank", type=int, default=512)
    parser.add_argument("--moe-intermediate-size", type=int, default=1536)
    parser.add_argument("--num-attention-heads", type=int, default=64)
    parser.add_argument("--num-experts-per-tok", type=int, default=6)
    parser.add_argument("--n-routed-experts", type=int, default=128)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--q-lora-rank", type=int, default=None)
    parser.add_argument("--qk-nope-head-dim", type=int, default=128)
    parser.add_argument("--qk-rope-head-dim", type=int, default=64)
    parser.add_argument("--v-head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=128256)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--duration", type=int, default=3, help="Benchmark duration per shape (sec)")
    parser.add_argument("--output-file", default="./gemm-deepseek_report.md")
    parser.add_argument("--append", action="store_true", help="Append to existing report")
    return parser

