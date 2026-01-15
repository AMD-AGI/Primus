###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = None  # type: ignore

from primus.tools.utils import (
    gather_all_results,
    get_current_device,
    get_hostname,
    get_rank_world,
    pick_dtype,
)

from .attention_profiler import profile_one, resolve_backend

DEFAULT_SHAPES: List[Dict[str, Any]] = [
    {
        "model": "Llama2_7B",
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_head_q": 32,
        "num_head_kv": 32,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
    {
        "model": "Llama2_70B",
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_head_q": 64,
        "num_head_kv": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
    {
        "model": "Llama3_8B",
        "seqlen_q": 8192,
        "seqlen_kv": 8192,
        "num_head_q": 32,
        "num_head_kv": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
    {
        "model": "Llama3_70B",
        "seqlen_q": 8192,
        "seqlen_kv": 8192,
        "num_head_q": 64,
        "num_head_kv": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
    {
        "model": "mistral_8x7B",
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_head_q": 32,
        "num_head_kv": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
    {
        "model": "mistral_8x22B",
        "seqlen_q": 4096,
        "seqlen_kv": 4096,
        "num_head_q": 48,
        "num_head_kv": 8,
        "head_dim_qk": 128,
        "head_dim_v": 128,
        "causal": True,
    },
]


def _parse_int_list(s: str) -> List[int]:
    if not s:
        return []
    out: List[int] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _parse_str_set(s: str) -> set[str]:
    if not s:
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}


def run_attention_benchmark(args) -> None:
    """
    Attention benchmark runner (ported from benchmark/kernel/attention/benchmark_attention.py).
    - Shards shape cases across ranks: case_idx % WORLD_SIZE == RANK
    - Gathers per-rank result rows and writes a single CSV on rank0.
    """
    rank, world = get_rank_world()

    if torch is None:
        raise RuntimeError(
            "[ERROR] primus.tools.benchmark.attention_bench requires PyTorch. "
            "Install torch or run inside the Primus ROCm container."
        )

    device = get_current_device()
    dtype = pick_dtype(getattr(args, "dtype", "bf16"))
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("[ERROR] attention benchmark supports only bf16/fp16.")

    shapes_path = getattr(args, "shapes_json_path", None) or getattr(args, "shapes_json", None)
    if shapes_path:
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"[ERROR] shapes json not found: {shapes_path}")
        with open(shapes_path, "r", encoding="utf-8") as f:
            shape_dicts: List[Dict[str, Any]] = json.load(f)
    else:
        shape_dicts = list(DEFAULT_SHAPES)

    mbs_list = _parse_int_list(getattr(args, "mbs_list", "1,2,3,4,5,6,7,8"))
    if not mbs_list:
        raise ValueError("[ERROR] --mbs-list is empty.")

    skip_models = _parse_str_set(getattr(args, "skip_models", ""))
    models = getattr(args, "models", "all")
    allow_models = set()
    if isinstance(models, str) and models.strip().lower() != "all":
        allow_models = _parse_str_set(models)

    backends = resolve_backend(getattr(args, "backend", "flash"))

    # Shard cases across ranks to avoid duplicated work.
    local_rows: List[Dict[str, Any]] = []
    cases_iter = enumerate(shape_dicts)
    if tqdm is not None and rank == 0:
        cases_iter = tqdm(list(cases_iter), desc="Attention cases")

    for case_idx, sd in cases_iter:
        model = str(sd.get("model", ""))
        if allow_models and model not in allow_models:
            continue
        if model in skip_models:
            continue

        # Original script assumes seqlen_q == seqlen_kv.
        seqlen_q = int(sd["seqlen_q"])
        seqlen_kv = int(sd["seqlen_kv"])
        if seqlen_q != seqlen_kv:
            continue

        if world > 1 and (case_idx % world) != rank:
            continue

        for backend in backends:
            for mbs in mbs_list:
                res = profile_one(
                    backend=backend,
                    batch_size=int(mbs),
                    seq_len=seqlen_q,
                    num_head_q=int(sd["num_head_q"]),
                    num_head_kv=int(sd["num_head_kv"]),
                    head_dim_qk=int(sd["head_dim_qk"]),
                    head_dim_v=int(sd["head_dim_v"]),
                    causal=bool(sd.get("causal", True)),
                    dtype=dtype,
                    device=device,
                )

                local_rows.append(
                    {
                        "host": get_hostname(),
                        "rank": rank,
                        "world": world,
                        "model": model,
                        "batch_size": int(mbs),
                        "seqlen_q": seqlen_q,
                        "seqlen_kv": seqlen_kv,
                        "num_head_q": int(sd["num_head_q"]),
                        "num_head_kv": int(sd["num_head_kv"]),
                        "head_dim_qk": int(sd["head_dim_qk"]),
                        "head_dim_v": int(sd["head_dim_v"]),
                        "causal": bool(sd.get("causal", True)),
                        "fwd_tflops": float(res.fwd_tflops),
                        "fwd_time": float(res.fwd_time_s),
                        "bwd_tflops": float(res.bwd_tflops),
                        "bwd_time": float(res.bwd_time_s),
                        "backend": backend,
                        "dtype": str(getattr(args, "dtype", "bf16")),
                    }
                )

    all_rows: List[Dict[str, Any]] = gather_all_results(local_rows)
    if rank != 0:
        return

    out_csv = getattr(args, "report_csv_path", "./attention_benchmark.csv")
    fieldnames = [
        "host",
        "world",
        "rank",
        "model",
        "batch_size",
        "seqlen_q",
        "seqlen_kv",
        "num_head_q",
        "num_head_kv",
        "head_dim_qk",
        "head_dim_v",
        "causal",
        "fwd_tflops",
        "fwd_time",
        "bwd_tflops",
        "bwd_time",
        "backend",
        "dtype",
    ]

    with open(out_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"[✔] Attention benchmark finished. Results saved to {out_csv}")
