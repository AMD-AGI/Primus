#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P38 fused Indexer score kernel.

V4-Flash widths: ``[B=1, S=4096, P=1024, H=8, Hd=128]`` -- the CSA
Indexer is invoked at the V4-Flash 8-layer slice 3 times per iter
(layers where ``compress_ratio == 4``).

Mirrors the P34..P37 microbench convention.
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


_MODES = {
    "v4": dict(B=1, S=4096, P=1024, H=8, HD=128),
    "small": dict(B=2, S=128, P=32, H=8, HD=128),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(_MODES.keys()), default="v4")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--n-input-copies", type=int, default=4)
    p.add_argument("--l2-flush-mb", type=int, default=512)
    p.add_argument("--json-out", type=Path, default=None)
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


def _gbps(bytes_total: int, time_ms: float) -> float:
    return bytes_total / (time_ms * 1e-3) / 1e9


def _flops(*, B, S, P, H, HD) -> int:
    # Per (b, s, p, h): dot of HD -> 2*HD FLOPs; +1 relu; *w_i: 1 mul; sum_h: 1 add per head.
    # Approx: 2 * B * S * P * H * HD MAC ops.
    return 2 * B * S * P * H * HD


def _build_pool(args):
    cfg = _MODES[args.mode]
    B, S, P, H, HD = cfg["B"], cfg["S"], cfg["P"], cfg["H"], cfg["HD"]
    device = torch.device("cuda")
    pool = []
    for c in range(max(1, args.n_input_copies)):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        q = torch.randn((B, S, H, HD), dtype=_dtype(args.dtype), device=device, generator=gen).requires_grad_(True)
        k = torch.randn((B, P, HD), dtype=_dtype(args.dtype), device=device, generator=gen).requires_grad_(True)
        w = torch.randn((B, S, H), dtype=_dtype(args.dtype), device=device, generator=gen).abs().requires_grad_(True)
        pool.append((q, k, w))
    return pool, cfg


class L2Flusher:
    def __init__(self, mb: int):
        n = (max(0, mb) * 1024 * 1024) // 4
        self.buf = torch.empty(n, dtype=torch.int32, device="cuda") if n > 0 else None

    def flush(self):
        if self.buf is not None:
            self.buf.zero_()


def _eager_score(q, k, w, *, compress_ratio: int, out_dtype):
    relu = F.relu(torch.einsum("bshd,bpd->bshp", q.float(), k.float()))
    s = (relu * w.float().unsqueeze(-1)).sum(dim=2)
    B, S, P = s.shape
    t_idx = torch.arange(S, device=s.device).unsqueeze(1)
    s_end = (torch.arange(P, device=s.device).unsqueeze(0) + 1) * compress_ratio - 1
    allowed = s_end <= t_idx
    mask = torch.where(allowed, torch.zeros_like(s[0]), torch.full_like(s[0], float("-inf")))
    return (s + mask.unsqueeze(0)).to(out_dtype)


def _time_fwd(pool, cfg, dtype, *, triton_path: bool, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        q, k, w = pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = IndexerScoreFn.apply(q, k, w, 4, dtype)
        else:
            out = _eager_score(q, k, w, compress_ratio=4, out_dtype=dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(pool, cfg, dtype, *, triton_path: bool, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        q, k, w = pool[i % n]
        for t in (q, k, w):
            if t.grad is not None:
                t.grad = None
        if triton_path:
            out = IndexerScoreFn.apply(q, k, w, 4, dtype)
        else:
            out = _eager_score(q, k, w, compress_ratio=4, out_dtype=dtype)
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


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Indexer bench requires CUDA / HIP.")

    cfg = _MODES[args.mode]
    flops = _flops(**cfg)

    print(f"== indexer_score bench: mode={args.mode} {cfg} dtype={args.dtype} ==")
    print(f"   compute = {flops / 1e9:.1f} GFLOP / call; iters={args.iters} warmup={args.warmup}")

    pool, cfg = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    results = {}
    for name, triton_path in [("triton", True), ("eager", False)]:
        _time_fwd(pool, cfg, _dtype(args.dtype), triton_path=triton_path, iters=args.warmup, flusher=flusher)
        fwd = _time_fwd(pool, cfg, _dtype(args.dtype), triton_path=triton_path, iters=args.iters, flusher=flusher)
        bwd = _time_bwd(pool, cfg, _dtype(args.dtype), triton_path=triton_path, iters=args.iters, flusher=flusher)
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {**_stats(fwd), "tflops_median": flops / (fwd_med * 1e-3) / 1e12},
            "bwd": {**_stats(bwd), "tflops_median": 3 * flops / (bwd_med * 1e-3) / 1e12},
        }

    print()
    print(f"   {'path':<8}  {'fwd ms':>10}  {'fwd TF':>10}  {'bwd ms':>10}  {'bwd TF':>10}")
    print(f"   {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name in ("triton", "eager"):
        r = results[name]
        print(
            f"   {name:<8}  "
            f"{r['fwd']['median_ms']:>10.3f}  "
            f"{r['fwd']['tflops_median']:>10.1f}  "
            f"{r['bwd']['median_ms']:>10.3f}  "
            f"{r['bwd']['tflops_median']:>10.1f}"
        )

    e = results["eager"]
    t = results["triton"]
    print()
    print(f"   Triton FWD speedup vs eager: {e['fwd']['median_ms'] / t['fwd']['median_ms']:.2f}x")
    print(f"   Triton BWD speedup vs eager: {e['bwd']['median_ms'] / t['bwd']['median_ms']:.2f}x")

    if args.json_out is not None:
        payload = {
            "shape": {"mode": args.mode, **cfg, "dtype": args.dtype},
            "iters": args.iters,
            "warmup": args.warmup,
            "n_input_copies": args.n_input_copies,
            "l2_flush_mb": args.l2_flush_mb,
            "flops_fwd": flops,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n   wrote {args.json_out}")


if __name__ == "__main__":
    main()
