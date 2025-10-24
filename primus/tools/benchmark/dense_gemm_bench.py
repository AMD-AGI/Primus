###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import itertools
from datetime import datetime
from typing import Any, Dict, Tuple

import torch
from git import List
from tqdm import tqdm

from primus.tools.benchmark.gemm_bench import profile_gemm
from primus.tools.report import write_table_simple
from primus.tools.utils import gather_records, is_rank_0

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Llama2_7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    "Llama2_70B": {
        "seqlen": 4096,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    "Llama3.1_8B": {
        "seqlen": 8192,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    "Llama3.1_70B": {
        "seqlen": 8192,
        "hidden_size": 8192,
        "intermediate_size": 28672,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    "Llama3.1_405B": {
        "seqlen": 8192,
        "hidden_size": 16384,
        "intermediate_size": 53248,
        "num_attention_heads": 128,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 128256,
    },
    "Mistral_8x7B": {
        "seqlen": 4096,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
    "Mistral_8x22B": {
        "seqlen": 4096,
        "hidden_size": 6144,
        "intermediate_size": 16384,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": 32000,
    },
}


def add_gemm_parser(parser: argparse.ArgumentParser):
    parser.add_argument("--model", default=None, help="Model name (e.g., Llama3.1_8B)")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=11008)
    parser.add_argument("--num-attention-heads", type=int, default=32)
    parser.add_argument("--num-key-value-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--mbs", type=int, default=1, help="Microbatch size")
    parser.add_argument("--output-file", default="./gemm-dense_report.md")
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
        f"- Model: {args.model or 'Custom'}",
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
    if args.model:
        model_lower_map = {k.lower(): k for k in MODEL_CONFIGS.keys()}
        model_key = args.model.lower()

        if model_key not in model_lower_map:
            raise ValueError(
                f"[ERROR] Unknown model '{args.model}'. Supported models: {', '.join(MODEL_CONFIGS.keys())}"
            )

        true_key = model_lower_map[model_key]
        cfg = MODEL_CONFIGS[true_key]
        args.model = true_key  # 规范化模型名
        for k, v in cfg.items():
            setattr(args, k, v)
    else:
        print("[INFO] No model specified. Using CLI-provided parameters.")

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

        print(f"[✔] DENSE GEMM benchmark finished. Results saved to {args.output_file}")


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
