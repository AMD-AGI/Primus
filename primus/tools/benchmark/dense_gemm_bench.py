###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import itertools
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = None  # type: ignore

TORCH_AVAILABLE = torch is not None

from primus.tools.report import write_table_simple

from .dense_gemm_bench_args import add_gemm_parser

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


def build_gemm_preamble(args, shape_defs: List[Tuple[str, List[int]]]) -> str:
    lines = [
        "# Dense GEMM Benchmark Report",
        "",
        f"- Model: {args.model or 'Custom'}",
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Cluster: amd-aig-poolside",
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


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "[ERROR] primus.tools.benchmark.dense_gemm_bench requires PyTorch. "
            "Install torch or run inside the Primus ROCm container."
        )
    return torch


@lru_cache(maxsize=1)
def _load_helpers():
    from primus.tools.benchmark.gemm_bench import profile_gemm
    from primus.tools.utils import gather_records, is_rank_0

    return profile_gemm, gather_records, is_rank_0


def _profile_fwd(m, n, k, dtype, duration, is_fp8=False):
    if is_fp8:
        from primus.tools.benchmark.gemm_bench import profile_gemm_fp8

        return profile_gemm_fp8(m, n, k, dtype, False, True, duration)
    else:
        profile_gemm, _, _ = _load_helpers()
        return profile_gemm(m, n, k, dtype, False, True, duration)


def _profile_wgrad(m, n, k, dtype, duration, is_fp8=False):
    if is_fp8:
        from primus.tools.benchmark.gemm_bench import profile_gemm_fp8

        return profile_gemm_fp8(n, k, m, dtype, True, False, duration)
    else:
        profile_gemm, _, _ = _load_helpers()
        return profile_gemm(n, k, m, dtype, True, False, duration)


def _profile_dgrad(m, n, k, dtype, duration, is_fp8=False):
    if is_fp8:
        from primus.tools.benchmark.gemm_bench import profile_gemm_fp8

        return profile_gemm_fp8(m, k, n, dtype, False, False, duration)
    else:
        profile_gemm, _, _ = _load_helpers()
        return profile_gemm(m, k, n, dtype, False, False, duration)


def run_gemm_benchmark(args):
    torch_mod = _require_torch()
    _, gather_records, is_rank_0 = _load_helpers()

    if args.model:
        model_lower_map = {k.lower(): k for k in MODEL_CONFIGS.keys()}
        model_key = args.model.lower()

        if model_key not in model_lower_map:
            raise ValueError(
                f"[ERROR] Unknown model '{args.model}'. Supported models: {', '.join(MODEL_CONFIGS.keys())}"
            )

        true_key = model_lower_map[model_key]
        cfg = MODEL_CONFIGS[true_key]
        args.model = true_key
        for k, v in cfg.items():
            setattr(args, k, v)
    else:
        print("[INFO] No model specified. Using CLI-provided parameters.")

    # Check FP8 availability
    is_fp8 = args.dtype == "fp8"
    if is_fp8:
        from primus.tools.benchmark.gemm_bench import FP8_AVAILABLE

        if not FP8_AVAILABLE:
            raise RuntimeError(
                "FP8 dtype requested but torchao is not available. " "Install it with: pip install torchao"
            )
        dtype = None  # FP8 doesn't use torch dtype directly
    else:
        dtype_map = {"bf16": torch_mod.bfloat16, "fp16": torch_mod.float16, "fp32": torch_mod.float32}
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
        ("fwd", _profile_fwd),
        ("wgrad", _profile_wgrad),
        ("dgrad", _profile_dgrad),
    ]

    record: Dict[str, str] = {}
    for (phase, shape), (tag, func) in tqdm(
        itertools.product(shape_defs, func_defs),
        total=len(shape_defs) * len(func_defs),
        desc="Dense GEMM",
    ):
        m = args.mbs * shape[0]
        n = shape[1]
        k = shape[2]

        res = func(m, n, k, dtype, args.duration, is_fp8=is_fp8)
        summary = (
            f"{res['avg_time_ms']:.6f}s / "
            f"{res['tflops']:.2f}TF/s / "
            f"{res['bandwidth_gbps']:.2f}GB/s / "
            f"AI={res['arith_intensity']:.2f}"
        )
        record[f"{phase}_{tag}"] = summary

    gathered = gather_records(record)
    if not is_rank_0():
        return

    all_keys = set()
    for rec in gathered:
        all_keys.update(rec.keys())
    header = ["host", "world", "rank"] + sorted([k for k in all_keys if k not in {"host", "rank", "world"}])

    rows: List[List[Any]] = []
    for rec in gathered:
        rows.append([rec.get(col, "") for col in header])

    preamble = build_gemm_preamble(args, shape_defs)
    append = getattr(args, "append", False)
    output_file = args.output_file or f"benchmark_gemm_dense_{args.model}.md"

    write_table_simple(
        header=header,
        rows=rows,
        output_file=output_file,
        append=append,
        preamble=preamble if not append else None,
    )

    print(f"[✔] DENSE GEMM benchmark finished. Results saved to {output_file}")


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
    if not TORCH_AVAILABLE:
        parser.error("PyTorch is required to run the dense GEMM benchmark. Please install torch first.")
    run_gemm_benchmark(args)
