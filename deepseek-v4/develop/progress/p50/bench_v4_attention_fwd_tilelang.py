#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-8 P50 tilelang dense FWD kernel vs the
plan-4 P25 Triton FWD kernel at V4-Flash production widths.

The harness wires through both code paths via the existing
`v4_attention` functional wrapper (the same call site V4
attention uses in production); switching is via the
`PRIMUS_V4_TILELANG_ATTN` env knob inside the wrapper.
"""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_MODES = {
    # V4-Flash production widths (cr=0 dense layer)
    "v4_flash": dict(B=1, HQ=64, HK=1, Sq=4096, Sk=4096, D=512),
    # Small smoke shape (fast bench loop in CI)
    "smoke": dict(B=2, HQ=4, HK=1, Sq=128, Sk=128, D=64),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(_MODES.keys()), default="smoke")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--swa-window", type=int, default=128)
    p.add_argument("--has-sink", action="store_true", default=True)
    p.add_argument("--no-sink", dest="has_sink", action="store_false")
    p.add_argument("--seed", type=int, default=20260515)
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


def _flops_fwd(*, B: int, HQ: int, HK: int, Sq: int, Sk: int, D: int, swa_window: int) -> int:
    # Approx visible-pair count for SWA-causal:
    #   visible_pairs = sum_m min(m + 1, swa_window) for m in [0, Sq)
    # FWD FLOPs = 2 * B * HQ * visible_pairs * D  (Q @ K^T)
    #           + 2 * B * HQ * visible_pairs * D  (P @ V)
    if swa_window <= 0:
        visible_pairs = Sq * (Sq + 1) // 2
    else:
        visible_pairs = sum(min(m + 1, swa_window) for m in range(Sq))
    return 4 * B * HQ * visible_pairs * D


def _build_pool(args):
    cfg = _MODES[args.mode]
    B, HQ, HK, Sq, Sk, D = cfg["B"], cfg["HQ"], cfg["HK"], cfg["Sq"], cfg["Sk"], cfg["D"]
    dt = _dtype(args.dtype)
    pool = []
    for c in range(max(1, args.n_input_copies)):
        gen = torch.Generator(device="cuda").manual_seed(args.seed + c)
        q = torch.randn((B, HQ, Sq, D), dtype=dt, device="cuda", generator=gen)
        k = torch.randn((B, HK, Sk, D), dtype=dt, device="cuda", generator=gen)
        v = torch.randn((B, HK, Sk, D), dtype=dt, device="cuda", generator=gen)
        sink = torch.randn((HQ,), dtype=dt, device="cuda", generator=gen) if args.has_sink else None
        pool.append((q, k, v, sink))
    return pool, cfg


class L2Flusher:
    def __init__(self, mb: int):
        n = (max(0, mb) * 1024 * 1024) // 4
        self.buf = torch.empty(n, dtype=torch.int32, device="cuda") if n > 0 else None

    def flush(self):
        if self.buf is not None:
            self.buf.zero_()


def _time_path(*, pool, path: str, swa_window: int, iters: int, flusher: L2Flusher):
    """Time `iters` invocations of `v4_attention` with the given path
    (the env knob is set per-loop on the dispatcher side).
    """
    import os

    if path == "tilelang":
        os.environ["PRIMUS_V4_TILELANG_ATTN"] = "1"
    else:
        os.environ["PRIMUS_V4_TILELANG_ATTN"] = "0"

    # Import here so the env is in effect before the dispatcher caches.
    from primus.backends.megatron.core.transformer.v4_attention_kernels.v4_attention import (
        v4_attention,
    )

    times: List[float] = []
    n = len(pool)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    scale = 1.0 / (pool[0][0].shape[-1] ** 0.5)
    for i in range(iters):
        q, k, v, sink = pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        out = v4_attention(
            q,
            k,
            v,
            sink=sink,
            swa_window=int(swa_window),
            additive_mask=None,
            attn_dropout=0.0,
            training=False,
            scale=scale,
        )
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bench requires CUDA / HIP.")
    cfg = _MODES[args.mode]
    flops = _flops_fwd(**cfg, swa_window=args.swa_window)

    print(f"== v4_attention FWD bench: mode={args.mode} {cfg} dtype={args.dtype} swa={args.swa_window} ==")
    print(f"   compute = {flops / 1e9:.1f} GFLOP / call; iters={args.iters} warmup={args.warmup}")

    pool, _ = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    results = {}
    for path in ("triton", "tilelang"):
        # Warmup (includes tilelang JIT compile cost on first iter).
        _time_path(pool=pool, path=path, swa_window=args.swa_window, iters=args.warmup, flusher=flusher)
        times = _time_path(
            pool=pool, path=path, swa_window=args.swa_window, iters=args.iters, flusher=flusher
        )
        med = median(times)
        results[path] = {**_stats(times), "tflops_median": flops / (med * 1e-3) / 1e12}

    print()
    print(f"   {'path':<10}  {'fwd ms':>10}  {'fwd TF':>10}")
    print(f"   {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for path in ("triton", "tilelang"):
        r = results[path]
        print(f"   {path:<10}  {r['median_ms']:>10.3f}  {r['tflops_median']:>10.1f}")
    e = results["triton"]
    t = results["tilelang"]
    print()
    print(f"   tilelang speedup vs triton: {e['median_ms'] / t['median_ms']:.2f}x")

    if args.json_out is not None:
        payload = {
            "shape": {
                "mode": args.mode,
                **cfg,
                "dtype": args.dtype,
                "swa_window": args.swa_window,
                "has_sink": args.has_sink,
            },
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
