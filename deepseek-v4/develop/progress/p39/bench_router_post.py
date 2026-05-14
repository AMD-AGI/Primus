#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P39 fused V4 router post-logits kernel.

Compares the Triton FWD/BWD kernel against the eager body in
``v4_topk_router._compute_route`` (or
``v4_hash_router.DeepseekV4HashRouter.forward``) -- post-topk path.

V4-Flash widths: ``N = batch * seq_len = 4096``, ``E = 256``,
``K = 6`` (kernel constraint requires power-of-2 K so we use ``K = 8``
when ``K = 6`` is unsupported; both V4 routers use K=6 in production).
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

from primus.backends.megatron.core.transformer.moe._triton.v4_router_post import (  # noqa: E402
    V4RouterPostFn,
)

_MODES = {
    "v4": dict(N=4096, E=256, K=8),  # K=8 because K must be pow-of-2; V4 uses K=6 in prod
    "small": dict(N=128, E=32, K=4),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(_MODES.keys()), default="v4")
    p.add_argument("--score-fn", choices=("softmax", "sigmoid", "sqrtsoftplus"), default="sqrtsoftplus")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--n-input-copies", type=int, default=4)
    p.add_argument("--l2-flush-mb", type=int, default=512)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def _stats(values: Iterable[float]) -> dict:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _bandwidth_gbps(bytes_total: int, time_ms: float) -> float:
    return bytes_total / (time_ms * 1e-3) / 1e9


def _build_pool(args):
    cfg = _MODES[args.mode]
    N, E, K = cfg["N"], cfg["E"], cfg["K"]
    device = torch.device("cuda")
    pool = []
    for c in range(max(1, args.n_input_copies)):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        logits = torch.randn((N, E), dtype=torch.float32, device=device, generator=gen).requires_grad_(True)
        idx = torch.stack([torch.randperm(E, generator=gen, device=device)[:K] for _ in range(N)], dim=0).to(
            torch.int64
        )
        pool.append((logits, idx))
    return pool, cfg


class L2Flusher:
    def __init__(self, mb: int):
        n = (max(0, mb) * 1024 * 1024) // 4
        self.buf = torch.empty(n, dtype=torch.int32, device="cuda") if n > 0 else None

    def flush(self):
        if self.buf is not None:
            self.buf.zero_()


def _eager_post(logits, indices, *, score_function: str, topk_scaling_factor: float):
    if score_function == "softmax":
        sc = torch.softmax(logits, dim=-1)
    elif score_function == "sigmoid":
        sc = torch.sigmoid(logits)
    else:
        sc = torch.sqrt(F.softplus(logits))
    weights = sc.gather(1, indices)
    if score_function != "softmax":
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom
    if topk_scaling_factor != 1.0:
        weights = weights * float(topk_scaling_factor)
    N, E = logits.shape
    probs = torch.zeros(N, E, dtype=weights.dtype, device=logits.device)
    probs.scatter_(1, indices, weights)
    rmap = torch.zeros(N, E, dtype=torch.bool, device=logits.device)
    rmap.scatter_(1, indices, True)
    return probs, rmap


def _time_fwd(pool, cfg, *, triton_path: bool, score_fn: str, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        logits, idx = pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = V4RouterPostFn.apply(logits, idx, score_fn, 2.5, torch.float32)
        else:
            out = _eager_post(logits, idx, score_function=score_fn, topk_scaling_factor=2.5)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(pool, cfg, *, triton_path: bool, score_fn: str, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        logits, idx = pool[i % n]
        if logits.grad is not None:
            logits.grad = None
        if triton_path:
            probs, _ = V4RouterPostFn.apply(logits, idx, score_fn, 2.5, torch.float32)
        else:
            probs, _ = _eager_post(logits, idx, score_function=score_fn, topk_scaling_factor=2.5)
        g = torch.randn_like(probs)
        torch.cuda.synchronize()
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        (probs * g).sum().backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("router_post bench requires CUDA / HIP.")

    cfg = _MODES[args.mode]
    N, E, K = cfg["N"], cfg["E"], cfg["K"]
    # FWD traffic ~ logits (NxE fp32) + indices (NxK i64) + probs (NxE fp32) + rmap (NxE bool)
    bytes_fwd = 4 * N * E + 8 * N * K + 4 * N * E + 1 * N * E
    bytes_bwd = 4 * N * E + 8 * N * K + 4 * N * E + 4 * N * E  # dprobs + saved + dlogits

    print(f"== v4_router_post bench: mode={args.mode} {cfg} score_fn={args.score_fn} ==")
    print(
        f"   FWD traffic ~ {bytes_fwd / 1024 / 1024:.2f} MiB; "
        f"BWD traffic ~ {bytes_bwd / 1024 / 1024:.2f} MiB; "
        f"iters={args.iters} warmup={args.warmup}"
    )

    pool, cfg = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    results = {}
    for name, triton_path in [("triton", True), ("eager", False)]:
        _time_fwd(
            pool, cfg, triton_path=triton_path, score_fn=args.score_fn, iters=args.warmup, flusher=flusher
        )
        fwd = _time_fwd(
            pool, cfg, triton_path=triton_path, score_fn=args.score_fn, iters=args.iters, flusher=flusher
        )
        bwd = _time_bwd(
            pool, cfg, triton_path=triton_path, score_fn=args.score_fn, iters=args.iters, flusher=flusher
        )
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {**_stats(fwd), "bw_gbps_median": _bandwidth_gbps(bytes_fwd, fwd_med)},
            "bwd": {**_stats(bwd), "bw_gbps_median": _bandwidth_gbps(bytes_bwd, bwd_med)},
        }

    print()
    print(f"   {'path':<8}  {'fwd ms':>10}  {'fwd GB/s':>10}  {'bwd ms':>10}  {'bwd GB/s':>10}")
    print(f"   {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name in ("triton", "eager"):
        r = results[name]
        print(
            f"   {name:<8}  "
            f"{r['fwd']['median_ms']:>10.3f}  "
            f"{r['fwd']['bw_gbps_median']:>10.1f}  "
            f"{r['bwd']['median_ms']:>10.3f}  "
            f"{r['bwd']['bw_gbps_median']:>10.1f}"
        )

    e = results["eager"]
    t = results["triton"]
    print()
    print(f"   Triton FWD speedup vs eager: {e['fwd']['median_ms'] / t['fwd']['median_ms']:.2f}x")
    print(f"   Triton BWD speedup vs eager: {e['bwd']['median_ms'] / t['bwd']['median_ms']:.2f}x")

    if args.json_out is not None:
        payload = {
            "shape": {"mode": args.mode, **cfg, "score_fn": args.score_fn},
            "iters": args.iters,
            "warmup": args.warmup,
            "n_input_copies": args.n_input_copies,
            "l2_flush_mb": args.l2_flush_mb,
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n   wrote {args.json_out}")


if __name__ == "__main__":
    main()
