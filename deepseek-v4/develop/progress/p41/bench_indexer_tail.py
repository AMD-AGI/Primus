#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P41 fused Indexer post-einsum tail kernel.

V4-Flash widths: ``[B=1, S=4096, P=1024, H=8]`` -- the CSA Indexer's
post-einsum tail is invoked at the V4-Flash 8-layer slice 3 times per
iter (layers where ``compress_ratio == 4``).

Three paths bench-compared:

* ``eager_tail`` — the eager body extracted from ``Indexer.forward``
  (``relu + mul + sum(H) + causal_mask``); the einsum is **not**
  included so this measures the same thing the Triton tail does.
* ``triton_tail`` — the P41 Triton kernel
  (`indexer_score_post_triton`).
* ``triton_full`` — the legacy P38 full-fuse kernel
  (`indexer_score_triton`), which includes the einsum.  Bench
  driver subtracts the cuBLAS einsum cost to make this comparable
  to the tail-only paths.

Mirrors the P34..P39 microbench convention (`--mode {v4, small}`,
JSON output via `--json-out`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.indexer_score import (  # noqa: E402
    IndexerScoreFn,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.indexer_score_post import (  # noqa: E402
    IndexerScorePostFn,
)

_MODES = {
    "v4": dict(B=1, S=4096, P=1024, H=8, HD=128),
    "small": dict(B=2, S=128, P=32, H=8, HD=128),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(_MODES.keys()), default="v4")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--n-input-copies", type=int, default=4)
    p.add_argument("--l2-flush-mb", type=int, default=512)
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument(
        "--include-full",
        action="store_true",
        help="Also bench the legacy P38 full-fuse kernel.",
    )
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _stats(values: Iterable[float]) -> dict:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _bytes_tail_fwd(*, B, S, P, H, HD, dtype: torch.dtype) -> int:
    """HBM read + write footprint for the tail kernel.

    Reads ``dot [B, S, H, P]`` (bandwidth-bound) + ``w [B, S, H]``.
    Writes ``scores [B, S, P]``.
    """
    bytes_per = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4}[dtype]
    return (
        B * S * H * P * bytes_per  # read dot
        + B * S * H * bytes_per  # read w
        + B * S * P * bytes_per  # write scores
    )


class L2Flusher:
    def __init__(self, mb: int):
        n = (max(0, mb) * 1024 * 1024) // 4
        self.buf = torch.empty(n, dtype=torch.int32, device="cuda") if n > 0 else None

    def flush(self):
        if self.buf is not None:
            self.buf.zero_()


def _eager_tail(dot, w, *, compress_ratio: int, out_dtype):
    relu = F.relu(dot.float())
    scores = (relu * w.float().unsqueeze(-1)).sum(dim=2)
    B, S, P = scores.shape
    t_idx = torch.arange(S, device=scores.device).unsqueeze(1)
    s_end = (torch.arange(P, device=scores.device).unsqueeze(0) + 1) * compress_ratio - 1
    allowed = s_end <= t_idx
    mask = torch.where(
        allowed,
        torch.zeros_like(scores[0]),
        torch.full_like(scores[0], float("-inf")),
    )
    return (scores + mask.unsqueeze(0)).to(out_dtype)


def _build_pool(args):
    """Build a pool of (q, k, w) and pre-compute (dot, w) for the tail bench."""
    cfg = _MODES[args.mode]
    B, S, P, H, HD = cfg["B"], cfg["S"], cfg["P"], cfg["H"], cfg["HD"]
    device = torch.device("cuda")
    full_pool = []
    tail_pool = []
    for c in range(max(1, args.n_input_copies)):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        q = torch.randn((B, S, H, HD), dtype=_dtype(args.dtype), device=device, generator=gen).requires_grad_(
            True
        )
        k = torch.randn((B, P, HD), dtype=_dtype(args.dtype), device=device, generator=gen).requires_grad_(
            True
        )
        w = (
            torch.randn((B, S, H), dtype=_dtype(args.dtype), device=device, generator=gen)
            .abs()
            .requires_grad_(True)
        )
        full_pool.append((q, k, w))
        # Eager einsum to produce dot [B, S, H, P] -- this is what the
        # P41 tail kernel consumes.  Detach so backward through the
        # tail does not back-prop into the einsum (matches the
        # production wiring where the einsum is its own ATen op).
        with torch.no_grad():
            dot = torch.einsum("bshd,bpd->bshp", q.detach(), k.detach())
        dot = dot.detach().clone().requires_grad_(True)
        w_t = w.detach().clone().requires_grad_(True)
        tail_pool.append((dot, w_t))
    return full_pool, tail_pool, cfg


def _time_tail_fwd(tail_pool, dtype, *, triton_path: bool, iters: int, flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(tail_pool)
    for i in range(iters):
        dot, w = tail_pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = IndexerScorePostFn.apply(dot, w, 4, dtype)
        else:
            out = _eager_tail(dot, w, compress_ratio=4, out_dtype=dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_tail_bwd(tail_pool, dtype, *, triton_path: bool, iters: int, flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(tail_pool)
    for i in range(iters):
        dot, w = tail_pool[i % n]
        for t in (dot, w):
            if t.grad is not None:
                t.grad = None
        if triton_path:
            out = IndexerScorePostFn.apply(dot, w, 4, dtype)
        else:
            out = _eager_tail(dot, w, compress_ratio=4, out_dtype=dtype)
        g = torch.randn_like(out)
        finite = torch.isfinite(out)
        g = torch.where(finite, g, torch.zeros_like(g))
        torch.cuda.synchronize()
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        (out * g).sum().backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _time_full_fwd(full_pool, dtype, *, iters: int, flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(full_pool)
    for i in range(iters):
        q, k, w = full_pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        out = IndexerScoreFn.apply(q, k, w, 4, dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Indexer tail bench requires CUDA / HIP.")

    cfg = _MODES[args.mode]
    dtype = _dtype(args.dtype)
    tail_bytes = _bytes_tail_fwd(**cfg, dtype=dtype)

    print(f"== indexer_score_post (P41) bench: mode={args.mode} {cfg} dtype={args.dtype} ==")
    print(
        f"   tail HBM footprint = {tail_bytes / 1e6:.1f} MB / call; "
        f"iters={args.iters} warmup={args.warmup}"
    )

    full_pool, tail_pool, cfg = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    results = {}

    for name, triton_path in [("triton_tail", True), ("eager_tail", False)]:
        _time_tail_fwd(tail_pool, dtype, triton_path=triton_path, iters=args.warmup, flusher=flusher)
        fwd = _time_tail_fwd(tail_pool, dtype, triton_path=triton_path, iters=args.iters, flusher=flusher)
        bwd = _time_tail_bwd(tail_pool, dtype, triton_path=triton_path, iters=args.iters, flusher=flusher)
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {**_stats(fwd), "gbps_median": tail_bytes / (fwd_med * 1e-3) / 1e9},
            "bwd": {**_stats(bwd), "gbps_median": 3 * tail_bytes / (bwd_med * 1e-3) / 1e9},
        }

    if args.include_full:
        _time_full_fwd(full_pool, dtype, iters=args.warmup, flusher=flusher)
        fwd_full = _time_full_fwd(full_pool, dtype, iters=args.iters, flusher=flusher)
        results["triton_full"] = {
            "fwd": {**_stats(fwd_full), "gbps_median": float("nan")},
            "bwd": None,
        }

    print()
    print(f"   {'path':<14}  {'fwd ms':>10}  {'fwd GB/s':>10}  {'bwd ms':>10}  {'bwd GB/s':>10}")
    print(f"   {'-' * 14}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name in ("triton_tail", "eager_tail") + (("triton_full",) if args.include_full else ()):
        r = results[name]
        bwd_md = r["bwd"]["median_ms"] if r["bwd"] is not None else float("nan")
        bwd_gb = r["bwd"]["gbps_median"] if r["bwd"] is not None else float("nan")
        print(
            f"   {name:<14}  "
            f"{r['fwd']['median_ms']:>10.3f}  "
            f"{r['fwd']['gbps_median']:>10.1f}  "
            f"{bwd_md:>10.3f}  "
            f"{bwd_gb:>10.1f}"
        )

    e = results["eager_tail"]
    t = results["triton_tail"]
    print()
    print(f"   tail FWD speedup vs eager: {e['fwd']['median_ms'] / t['fwd']['median_ms']:.2f}x")
    if t["bwd"] is not None and e["bwd"] is not None:
        print(f"   tail BWD speedup vs eager: {e['bwd']['median_ms'] / t['bwd']['median_ms']:.2f}x")

    if args.json_out is not None:
        payload = {
            "shape": {"mode": args.mode, **cfg, "dtype": args.dtype},
            "iters": args.iters,
            "warmup": args.warmup,
            "n_input_copies": args.n_input_copies,
            "l2_flush_mb": args.l2_flush_mb,
            "tail_bytes_fwd": tail_bytes,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n   wrote {args.json_out}")


if __name__ == "__main__":
    main()
