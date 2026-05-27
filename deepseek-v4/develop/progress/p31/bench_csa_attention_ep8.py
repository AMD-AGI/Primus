#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the P31 Triton CSA attention path at proxy-EP8 shape.

This avoids launching the full training stack for each kernel experiment while
still exercising the real CSA tensor shape used by the Plan-5 proxy:

    B=1, H=64, S=4096, D=512, P=S/4=1024, K_topk=512, SWA=128.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.megatron.core.transformer.v4_attention_kernels import (
    v4_csa_attention_from_pool,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--compress-ratio", type=int, default=4)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--swa-window", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--random-topk", action="store_true", help="Allow duplicate top-k indices.")
    parser.add_argument("--sort-topk", action="store_true", help="Sort top-k pool ids within each query row.")
    parser.add_argument(
        "--n-input-copies",
        type=int,
        default=4,
        help=(
            "Proxy-mode buffer rotation: allocate N independent copies of Q/K/V/pool "
            "and rotate through them per iteration, so each call reads fresh HBM "
            "addresses (defeats HBM row-buffer / L2 reuse). Set to 1 for the legacy "
            "single-buffer microbench."
        ),
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=512,
        help=(
            "Proxy-mode L2 / last-level-cache eviction buffer (MiB). Written between "
            "iterations to evict cached Q/K/V/pool tiles. Set to 0 to disable."
        ),
    )
    parser.add_argument("--profile", action="store_true", help="Emit a one-step torch.profiler trace.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("deepseek-v4/develop/progress/p31/csa_bench_traces"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name!r}")


def _stats(values: Iterable[float]) -> dict[str, float]:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _make_topk(
    args: argparse.Namespace, *, pool_size: int, device: torch.device, gen: torch.Generator
) -> torch.Tensor:
    shape = (args.batch, args.seq_len, args.topk)
    if args.random_topk:
        topk = torch.randint(0, pool_size, shape, device=device, dtype=torch.int64, generator=gen)
        return torch.sort(topk, dim=-1).values if args.sort_topk else topk

    # The real indexer uses top-k, which yields unique pool ids per query row.
    # Build that pattern once outside the timed region.
    scores = torch.rand(args.batch, args.seq_len, pool_size, device=device, generator=gen)
    topk = torch.topk(scores, k=args.topk, dim=-1).indices.to(torch.int64)
    del scores
    return torch.sort(topk, dim=-1).values if args.sort_topk else topk


def _make_inputs(args: argparse.Namespace, *, seed: int | None = None) -> dict[str, torch.Tensor | None]:
    if not torch.cuda.is_available():
        raise RuntimeError("CSA benchmark requires CUDA/HIP.")

    device = torch.device("cuda")
    dtype = _dtype(args.dtype)
    pool_size = args.seq_len // args.compress_ratio
    if args.topk > pool_size:
        raise ValueError(f"topk={args.topk} must be <= pool_size={pool_size}.")

    gen = torch.Generator(device=device).manual_seed(args.seed if seed is None else seed)
    q = torch.randn(
        args.batch, args.heads, args.seq_len, args.head_dim, device=device, dtype=dtype, generator=gen
    )
    k_local = torch.randn_like(q)
    v_local = torch.randn_like(q)
    pool = torch.randn(args.batch, pool_size, args.head_dim, device=device, dtype=dtype, generator=gen)
    dout = torch.randn_like(q)
    sink = None
    if not args.no_sink:
        sink = (
            torch.randn(args.heads, device=device, dtype=torch.float32, generator=gen) * 0.1
        ).requires_grad_(True)

    q.requires_grad_(True)
    k_local.requires_grad_(True)
    v_local.requires_grad_(True)
    pool.requires_grad_(True)

    return {
        "q": q,
        "k_local": k_local,
        "v_local": v_local,
        "pool": pool,
        "topk_idxs": _make_topk(args, pool_size=pool_size, device=device, gen=gen),
        "dout": dout,
        "sink": sink,
    }


def _make_input_pool(args: argparse.Namespace) -> list[dict[str, torch.Tensor | None]]:
    n = max(1, args.n_input_copies)
    return [_make_inputs(args, seed=args.seed + i) for i in range(n)]


class L2Flusher:
    """Evict L2 / last-level cache between bench iterations.

    Writes a contiguous int32 buffer that is larger than any plausible
    L2 + Infinity-Cache footprint on AMD MI300/MI350, so every read in
    the timed region pays HBM (matches proxy steady state, not the
    cache-warm microbench artifact).
    """

    def __init__(self, size_mb: int, device: torch.device | str = "cuda") -> None:
        n = max(0, size_mb) * 1024 * 1024 // 4
        self.buf = torch.empty(n, dtype=torch.int32, device=device) if n > 0 else None

    @property
    def enabled(self) -> bool:
        return self.buf is not None

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


def _forward(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]) -> torch.Tensor:
    return v4_csa_attention_from_pool(
        tensors["q"],
        tensors["k_local"],
        tensors["v_local"],
        tensors["pool"],
        topk_idxs=tensors["topk_idxs"],
        sink=tensors["sink"],
        swa_window=args.swa_window,
        attn_dropout=0.0,
        training=True,
        scale=1.0 / math.sqrt(args.head_dim),
    )


def _backward_from_out(
    out: torch.Tensor,
    tensors: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    leaves = [tensors["q"], tensors["k_local"], tensors["v_local"], tensors["pool"]]
    if tensors["sink"] is not None:
        leaves.append(tensors["sink"])
    return torch.autograd.grad(
        out, leaves, grad_outputs=tensors["dout"], retain_graph=False, allow_unused=True
    )


def _backward(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
    return _backward_from_out(_forward(args, tensors), tensors)


def _time_forward_ms(
    args: argparse.Namespace,
    pool: list[dict[str, torch.Tensor | None]],
    *,
    iters: int,
    flusher: L2Flusher,
) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        tensors = pool[i % len(pool)]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        result = _forward(args, tensors)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del result
    return times


def _time_backward_ms(
    args: argparse.Namespace,
    pool: list[dict[str, torch.Tensor | None]],
    *,
    iters: int,
    flusher: L2Flusher,
) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        tensors = pool[i % len(pool)]
        out = _forward(args, tensors)
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        result = _backward_from_out(out, tensors)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out, result
    return times


def _profile_once(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]) -> None:
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    handler = torch.profiler.tensorboard_trace_handler(str(args.trace_dir))
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False, on_trace_ready=handler) as prof:
        _backward(args, tensors)
        prof.step()
    print(f"[profile] trace_dir={args.trace_dir}")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=12))


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)

    pool = _make_input_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    bench_mode = {
        "n_input_copies": len(pool),
        "l2_flush_mb": args.l2_flush_mb if flusher.enabled else 0,
        "proxy_mode": flusher.enabled and len(pool) > 1,
    }

    shape = {
        "B": args.batch,
        "H": args.heads,
        "S": args.seq_len,
        "D": args.head_dim,
        "P": args.seq_len // args.compress_ratio,
        "K_topk": args.topk,
        "swa_window": args.swa_window,
        "dtype": args.dtype,
        "sink": not args.no_sink,
        "unique_topk": not args.random_topk,
        "sorted_topk": args.sort_topk,
    }
    print("[shape]", json.dumps(shape, sort_keys=True))
    print("[bench_mode]", json.dumps(bench_mode, sort_keys=True))

    for _ in range(args.warmup):
        _backward(args, pool[0])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_ms = _time_forward_ms(args, pool, iters=args.iters, flusher=flusher)
    bwd_ms = _time_backward_ms(args, pool, iters=args.iters, flusher=flusher)

    result = {
        "shape": shape,
        "bench_mode": bench_mode,
        "forward": _stats(fwd_ms),
        "backward": _stats(bwd_ms),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.profile:
        _profile_once(args, pool[0])


if __name__ == "__main__":
    main()
