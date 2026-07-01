#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P37 fused HC compute_weights tail kernel.

V4-Flash widths (B=1, S=4096, K=4): the eager body issues ~8
elementwise GPU launches per call -- 3 slices + 3 fused-multiply-adds
+ 2 sigmoids + 1 softmax + 2 eps adds (the eager path actually maps
to ~7-9 ``aten`` ops since slice / view are free).  At 16 calls per
iter the P32 trace attributes ~3-5 ms / iter to this chain.

This bench measures the Triton path's FWD + BWD vs the eager body at
two shapes:

* ``--mode v4`` -- V4-Flash production: ``[B=1, S=4096, K=4]``;
* ``--mode small`` -- coverage: ``[B=2, S=64, K=4]``.

Mirrors the P34 / P35 / P36 microbench conventions (``--n-input-copies``,
``--l2-flush-mb``) so the numbers track proxy-steady-state behaviour.
"""

from __future__ import annotations

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

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.hc_glue import (  # noqa: E402
    HCComputeTailFn,
)

# (B, S, K) per preset.
_MODES = {
    "v4": (1, 4096, 4),
    "small": (2, 64, 4),
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=tuple(_MODES.keys()), default="v4")
    p.add_argument("--out-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--seed", type=int, default=20260514)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--n-input-copies", type=int, default=4)
    p.add_argument("--l2-flush-mb", type=int, default=512)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _stats(values: Iterable[float]) -> dict[str, float]:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _bandwidth_gbps(*, bytes_total: int, time_ms: float) -> float:
    return bytes_total / (time_ms * 1e-3) / 1e9


def _build_pool(args):
    B, S, K = _MODES[args.mode]
    device = torch.device("cuda")
    out_dim = (2 + K) * K
    pool = []
    for c in range(max(1, args.n_input_copies)):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        logits = torch.randn(
            (B, S, out_dim), dtype=torch.float32, device=device, generator=gen
        ).requires_grad_(True)
        scale = torch.ones(3, dtype=torch.float32, device=device).requires_grad_(True)
        base = (
            0.01 * torch.randn(out_dim, dtype=torch.float32, device=device, generator=gen)
        ).requires_grad_(True)
        pool.append((logits, scale, base))
    return pool, K


class L2Flusher:
    def __init__(self, size_mb: int):
        n = (max(0, size_mb) * 1024 * 1024) // 4
        self.buf = torch.empty(n, dtype=torch.int32, device="cuda") if n > 0 else None

    def flush(self):
        if self.buf is not None:
            self.buf.zero_()


def _eager_tail(logits, scale, base, *, K: int, out_dtype: torch.dtype, eps: float = 1e-6):
    pre_logit = logits[..., :K] * scale[0] + base[:K]
    post_logit = logits[..., K : 2 * K] * scale[1] + base[K : 2 * K]
    comb_logit = logits[..., 2 * K :].view(*logits.shape[:-1], K, K) * scale[2] + base[2 * K :].view(K, K)
    pre = torch.sigmoid(pre_logit) + eps
    post = 2.0 * torch.sigmoid(post_logit)
    comb = torch.softmax(comb_logit, dim=-1) + eps
    return pre.to(out_dtype), post.to(out_dtype), comb.to(out_dtype)


def _time_fwd(pool, K, out_dtype, *, triton_path: bool, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        logits, scale, base = pool[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = HCComputeTailFn.apply(logits, scale, base, K, 1e-6, out_dtype)
        else:
            out = _eager_tail(logits, scale, base, K=K, out_dtype=out_dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(pool, K, out_dtype, *, triton_path: bool, iters: int, flusher: L2Flusher):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(pool)
    for i in range(iters):
        logits, scale, base = pool[i % n]
        for t in (logits, scale, base):
            if t.grad is not None:
                t.grad = None
        if triton_path:
            pre, post, comb = HCComputeTailFn.apply(logits, scale, base, K, 1e-6, out_dtype)
        else:
            pre, post, comb = _eager_tail(logits, scale, base, K=K, out_dtype=out_dtype)
        g_pre = torch.randn_like(pre)
        g_post = torch.randn_like(post)
        g_comb = torch.randn_like(comb)
        torch.cuda.synchronize()
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        ((pre * g_pre).sum() + (post * g_post).sum() + (comb * g_comb).sum()).backward()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("hc_glue bench requires CUDA / HIP.")

    B, S, K = _MODES[args.mode]
    out_dtype = _dtype(args.out_dtype)

    bpe_in = 4  # logits is fp32
    bpe_out = torch.tensor([], dtype=out_dtype).element_size()
    out_dim = (2 + K) * K
    bytes_fwd = bpe_in * B * S * out_dim + bpe_out * B * S * (2 * K + K * K) + 4 * B * S * (2 * K + K * K)
    bytes_bwd = bpe_out * B * S * (2 * K + K * K) + 4 * B * S * (2 * K + K * K) * 2 + bpe_in * B * S * out_dim

    print(f"== hc_glue bench: mode={args.mode} B={B} S={S} K={K} out_dtype={args.out_dtype} ==")
    print(
        f"   FWD traffic ~ {bytes_fwd / 1024 / 1024:.2f} MiB; "
        f"BWD traffic ~ {bytes_bwd / 1024 / 1024:.2f} MiB; "
        f"iters={args.iters}, warmup={args.warmup}, "
        f"n_input_copies={args.n_input_copies}, l2_flush_mb={args.l2_flush_mb}"
    )

    pool, K = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    results = {}
    for name, triton_path in [("triton", True), ("eager", False)]:
        _time_fwd(pool, K, out_dtype, triton_path=triton_path, iters=args.warmup, flusher=flusher)
        fwd = _time_fwd(pool, K, out_dtype, triton_path=triton_path, iters=args.iters, flusher=flusher)
        bwd = _time_bwd(pool, K, out_dtype, triton_path=triton_path, iters=args.iters, flusher=flusher)
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {**_stats(fwd), "bw_gbps_median": _bandwidth_gbps(bytes_total=bytes_fwd, time_ms=fwd_med)},
            "bwd": {**_stats(bwd), "bw_gbps_median": _bandwidth_gbps(bytes_total=bytes_bwd, time_ms=bwd_med)},
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
            "shape": {"mode": args.mode, "B": B, "S": S, "K": K, "out_dtype": args.out_dtype},
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
