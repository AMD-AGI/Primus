###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse


def add_attention_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Attention benchmark args (ported from benchmark/kernel/attention/benchmark_attention.py).
    """
    parser.add_argument(
        "--shapes-json-path",
        type=str,
        default="",
        help=("Path to attention shape JSON (list[dict]). " "If empty, uses an internal default shape set."),
    )
    parser.add_argument(
        "--report-csv-path",
        type=str,
        default="./attention_benchmark.csv",
        help="Output CSV path (written by rank0).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="flash",
        choices=["flash", "ck", "all"],
        help="Which backend(s) to benchmark. 'ck' requires tile_example_fmha_* binaries in PATH.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="Attention dtype (flash-attn supports bf16/fp16).",
    )
    parser.add_argument(
        "--mbs-list",
        type=str,
        default="1,2,3,4,5,6,7,8",
        help="Comma-separated microbatch sizes to benchmark.",
    )
    parser.add_argument(
        "--skip-models",
        type=str,
        default="deepseek_v2_lite,deepseek_v2/v3",
        help="Comma-separated model names to skip (exact match).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model allowlist (exact match). Use 'all' to include all models.",
    )
    return parser
