###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import itertools
from datetime import datetime
from typing import Tuple

import torch
from git import List
from tqdm import tqdm

from primus.tools.benchmark.gemm_bench import profile_gemm
from primus.tools.report import write_table_simple
from primus.tools.utils import gather_records, is_rank_0


def add_gemm_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model", default="llama3-7B")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--mbs", type=int, default=1, help="Microbatch size")
    parser.add_argument("--output_file", default="./gemm-dense_report.md")
    parser.add_argument("--duration", type=int, default=3, help="Benchmark duration per shape (sec)")
    return parser


def profile_fwd(m, n, k, dtype, duration):
    return profile_gemm(m, n, k, dtype, False, True, duration)


def profile_wgrad(m, n, k, dtype, duration):
    return profile_gemm(n, k, m, dtype, True, False, duration)


def profile_dgrad(m, n, k, dtype, duration):
    return profile_gemm(m, k, n, dtype, False, False, duration)


def build_gemm_preamble(args, shape_defs: List[Tuple[str, List[int]]]) -> str:
    lines = [
        "# Dense GEMM Benchmark Report",
        "",
        f"- Model: {args.model}",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Cluster: amd-aig-poolside",
        f"- Duration per shape: {args.duration} sec",
        "",
        "## Configuration",
        f"- mbs: {args.mbs}",
        f"- num_attention_heads: {args.num_attention_heads}",
        f"- num_key_value_heads: {args.num_key_value_heads}",
        f"- head_dim: {args.head_dim}",
        f"- hidden_size: {args.hidden_size}",
        f"- intermediate_size: {args.intermediate_size}",
        f"- vocab_size: {args.vocab_size}",
        f"- seqlen: {args.seqlen}",
        f"- dtype: {args.dtype}",
        "",
        "## GEMM Shapes (M, N, K)",
    ]

    for name, shape in shape_defs:
        m, n, k = shape
        lines.append(f"- {name}: ({m}, {n}, {k})")

    lines += [
        "",
        "## Phases",
        "- fwd: forward pass",
        "- wgrad: weight gradient",
        "- dgrad: data gradient",
        "",
    ]

    return "\n".join(lines)


def run_gemm_benchmark(args):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp34": torch.float32}
    dtype = dtype_map[args.dtype]

    shape_defs = [
        (
            "attn_qkv",
            [
                args.seqlen,
                (args.num_attention_heads + 2 * args.num_key_value_heads) * args.head_dim,
                args.hidden_size,
            ],
        ),
        ("attn_out", [args.seqlen, args.hidden_size, args.hidden_size]),
        ("mlp_up", [args.seqlen, 2 * args.intermediate_size, args.hidden_size]),
        ("mlp_down", [args.seqlen, args.hidden_size, args.intermediate_size]),
        ("vocab", [args.seqlen, args.vocab_size, args.hidden_size]),
    ]

    func_defs = [
        ("fwd", profile_fwd),
        ("wgrad", profile_wgrad),
        ("dgrad", profile_dgrad),
    ]

    record = {}
    for (phase, shape), (tag, func) in tqdm(
        itertools.product(shape_defs, func_defs), total=len(shape_defs) * len(func_defs), desc="Dense GEMM"
    ):
        m = args.mbs * shape[0]
        n = shape[1]
        k = shape[2]

        res = func(m, n, k, dtype, args.duration)
        summary = (
            f"{res['avg_time_ms']:.6f}s / "
            f"{res['tflops']:.2f}TF/s / "
            f"{res['bandwidth_gbps']:.2f}GB/s / "
            f"AI={res['arith_intensity']:.2f}"
        )
        record[f"{phase}_{tag}"] = summary

    gathered = gather_records(record)
    if is_rank_0():
        all_keys = set()
        for r in gathered:
            all_keys.update(r.keys())
        header = ["host", "world", "rank"] + sorted(
            [k for k in all_keys if k not in {"host", "rank", "world"}]
        )

        rows = [[r.get(col, "") for col in header] for r in gathered]

        preamble = build_gemm_preamble(args, shape_defs)

        append = getattr(args, "append", False)

        write_table_simple(
            header=header,
            rows=rows,
            output_file=args.output_file or f"benchmark_gemm_dense_{args.model}.md",
            append=append,
            preamble=preamble if not append else None,
        )

        print(
            f"[✔] GEMM benchmark finished. Results saved to {args.output_file or f'benchmark_gemm_dense_{args.model}.md'}"
        )


def build_gemm_dense_parser() -> argparse.ArgumentParser:
    """
    Build a standalone parser for local execution.
    """
    parser = argparse.ArgumentParser(description="GEMM benchmark")
    add_gemm_parser(parser)
    return parser


if __name__ == "__main__":
    parser = build_gemm_dense_parser()
    args = parser.parse_args()
    run_gemm_benchmark(args)
