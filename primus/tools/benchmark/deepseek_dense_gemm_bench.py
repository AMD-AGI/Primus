###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import itertools
from datetime import datetime
from typing import List, Tuple

from tqdm import tqdm

try:
    import torch

    from primus.tools.benchmark.gemm_bench import profile_gemm
    from primus.tools.utils import gather_records, is_rank_0

    TORCH_AVAILABLE = True
except ImportError:
    print("[WARNING] dense gemm benchmark depends on torch, which does not exist in current environment!")
    TORCH_AVAILABLE = False

from primus.tools.report import write_table_simple

MODEL_CONFIGS = {
    "Deepseek_V2_Lite": {
        "seqlen": 4096,
        "hidden_size": 2048,
        "intermediate_size": 10944,
        "kv_lora_rank": 512,
        "moe_intermediate_size": 1408,
        "num_attention_heads": 16,
        "num_experts_per_tok": 6,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "q_lora_rank": None,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 102400,
    },
    "Deepseek_V2": {
        "seqlen": 4096,
        "hidden_size": 5120,
        "intermediate_size": 12288,
        "kv_lora_rank": 512,
        "moe_intermediate_size": 1536,
        "num_attention_heads": 128,
        "num_experts_per_tok": 6,
        "n_routed_experts": 160,
        "n_shared_experts": 2,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 102400,
    },
    "Deepseek_V3": {
        "seqlen": 4096,
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "kv_lora_rank": 512,
        "moe_intermediate_size": 2048,
        "num_attention_heads": 128,
        "num_experts_per_tok": 8,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "vocab_size": 129280,
    },
}


def add_gemm_parser(parser: argparse.ArgumentParser):
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


def build_preamble(args, shapes: List[Tuple[str, List[int]]]) -> str:
    lines = [
        "# DeepSeek GEMM Benchmark Report",
        "",
        f"- Model: {args.model or 'Custom'}",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Duration per shape: {args.duration}s",
        "",
        "## Configuration",
        f"- seqlen: {args.seqlen}",
        f"- hidden_size: {args.hidden_size}",
        f"- intermediate_size: {args.intermediate_size}",
        f"- kv_lora_rank: {args.kv_lora_rank}",
        f"- moe_intermediate_size: {args.moe_intermediate_size}",
        f"- num_attention_heads: {args.num_attention_heads}",
        f"- num_experts_per_tok: {args.num_experts_per_tok}",
        f"- n_routed_experts: {args.n_routed_experts}",
        f"- n_shared_experts: {args.n_shared_experts}",
        f"- q_lora_rank: {args.q_lora_rank}",
        f"- dtype: {args.dtype}",
        "",
        "## GEMM Shapes (M, N, K)",
    ]
    for name, (m, n, k) in shapes:
        lines.append(f"- {name}: ({m}, {n}, {k})")
    lines += ["", "## Phases", "- fwd", "- wgrad", "- dgrad", ""]
    return "\n".join(lines)


if TORCH_AVAILABLE:

    def profile_fwd(m, n, k, dtype, duration):
        return profile_gemm(m, n, k, dtype, False, True, duration)

    def profile_wgrad(m, n, k, dtype, duration):
        return profile_gemm(n, k, m, dtype, True, False, duration)

    def profile_dgrad(m, n, k, dtype, duration):
        return profile_gemm(m, k, n, dtype, False, False, duration)

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

        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        dtype = dtype_map[args.dtype]

        q_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        shape_defs = []

        # q-proj
        if args.q_lora_rank is None:
            shape_defs.append(
                ("attn_q", [args.seqlen, args.num_attention_heads * q_head_dim, args.hidden_size])
            )
        else:
            shape_defs.append(("attn_q_down", [args.seqlen, args.q_lora_rank, args.hidden_size]))
            shape_defs.append(
                ("attn_q_up", [args.seqlen, args.num_attention_heads * q_head_dim, args.q_lora_rank])
            )

        # kv projections
        shape_defs += [
            ("attn_kv_down", [args.seqlen, args.kv_lora_rank + args.qk_rope_head_dim, args.hidden_size]),
            (
                "attn_kv_up",
                [
                    args.seqlen,
                    args.num_attention_heads * (args.qk_nope_head_dim + args.v_head_dim),
                    args.kv_lora_rank,
                ],
            ),
            ("attn_out", [args.seqlen, args.hidden_size, args.num_attention_heads * args.v_head_dim]),
            ("router", [args.seqlen, args.n_routed_experts, args.hidden_size]),
        ]

        # shared experts
        if args.n_shared_experts > 0:
            shape_defs.append(("shared_gateup", [args.seqlen, args.intermediate_size * 2, args.hidden_size]))
            shape_defs.append(("shared_down", [args.seqlen, args.hidden_size, args.intermediate_size]))

        # routed experts (balance)
        balance_seq = int(args.seqlen * args.num_experts_per_tok // args.n_routed_experts)
        shape_defs.append(("moe_gateup", [balance_seq, args.moe_intermediate_size * 2, args.hidden_size]))
        shape_defs.append(("moe_down", [balance_seq, args.hidden_size, args.moe_intermediate_size]))

        # vocab
        shape_defs.append(("vocab", [args.seqlen, args.vocab_size, args.hidden_size]))

        func_defs = [
            ("fwd", profile_fwd),
            ("wgrad", profile_wgrad),
            ("dgrad", profile_dgrad),
        ]

        record = {}
        for (phase, shape), (tag, func) in tqdm(
            itertools.product(shape_defs, func_defs),
            total=len(shape_defs) * len(func_defs),
            desc=f"[DeepSeek GEMM] {args.model or 'Custom'}",
        ):
            m, n, k = [args.mbs * shape[0], shape[1], shape[2]]

            res = func(m, n, k, dtype, args.duration)
            summary = (
                f"{res['tflops']:.2f}TF/s / "
                f"{res['bandwidth_gbps']:.2f}GB/s / "
                f"{res['avg_time_ms']:.6f}s / "
                f"AI={res['arith_intensity']:.2f}"
            )
            record[f"{phase}_{tag}"] = summary

        gathered = gather_records(record)
        if is_rank_0():
            all_keys = set().union(*(r.keys() for r in gathered))
            header = ["host", "world", "rank"] + sorted(
                [k for k in all_keys if k not in {"host", "rank", "world"}]
            )

            rows = [[r.get(col, "") for col in header] for r in gathered]

            preamble = build_preamble(args, shape_defs)

            append = getattr(args, "append", False)

            write_table_simple(
                header=header,
                rows=rows,
                output_file=args.output_file or f"benchmark_gemm_dense_{args.model}.md",
                append=append,
                preamble=preamble if not append else None,
            )

            print(f"[✔] DeepSeek GEMM benchmark finished. Results saved to {args.output_file}")


def build_gemm_dense_parser() -> argparse.ArgumentParser:
    """
    Build a standalone parser for local execution.
    """
    parser = argparse.ArgumentParser(description="DEEPSEEK-GEMM benchmark")
    add_gemm_parser(parser)
    return parser


if __name__ == "__main__":
    parser = build_gemm_dense_parser()
    args = parser.parse_args()
    run_gemm_benchmark(args)
