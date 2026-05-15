#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-7 P45 — multi-tensor BF16 add microbench.

Compares three paths on a synthetic V4-Flash-ish parameter list:

* ``foreach``: ``torch._foreach_add_(out, b, alpha=scale)``.
* ``triton_per_tensor``: per-tensor Triton kernel launched in a loop.
* ``triton_packed``: single-kernel Triton multi-tensor add.

The eager ``foreach`` path is the closest PyTorch reference for
the V4-Flash trace's ``vec_elem<add_bf16>`` bucket (743 launches /
iter × 0.23 ms = 170.99 ms).  Production replacement requires
coordinating with the Apex / TE optimizer call site (plan-8 scope).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.megatron.core.extensions._triton.multi_tensor_add import (  # noqa: E402
    multi_tensor_add_triton_packed,
    multi_tensor_add_triton_per_tensor,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-tensors",
        type=int,
        default=128,
        help="Number of tensors in the multi-tensor batch.",
    )
    p.add_argument(
        "--shape-mix",
        choices=("v4flash", "uniform_small", "uniform_large"),
        default="v4flash",
        help="Shape distribution of the synthetic parameter list.",
    )
    p.add_argument("--seed", type=int, default=20260515)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--block-size", type=int, default=8192)
    p.add_argument("--json-out", type=Path, default=None)
    return p.parse_args()


def _shapes(name: str, n: int, gen: torch.Generator) -> List[Tuple[int, ...]]:
    if name == "v4flash":
        # Mixture: 50% small (4096), 30% mid (1M), 20% large (16M).
        # Roughly matches the V4-Flash 8-layer parameter list scale.
        cpu_gen = torch.Generator().manual_seed(int(gen.initial_seed()) + 1)
        out: List[Tuple[int, ...]] = []
        for _i in range(n):
            r = float(torch.rand(1, generator=cpu_gen).item())
            if r < 0.5:
                out.append((4096,))
            elif r < 0.8:
                out.append((1024, 1024))
            else:
                out.append((4096, 4096))
        return out
    if name == "uniform_small":
        return [(4096,)] * n
    if name == "uniform_large":
        return [(4096, 4096)] * n
    raise ValueError(name)


def _stats(values: Iterable[float]) -> dict:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _build_pool(args, dtype: torch.dtype = torch.bfloat16):
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    shapes = _shapes(args.shape_mix, args.n_tensors, gen)
    out = [torch.randn(s, dtype=dtype, device="cuda", generator=gen) for s in shapes]
    a = [torch.randn(s, dtype=dtype, device="cuda", generator=gen) for s in shapes]
    b = [torch.randn(s, dtype=dtype, device="cuda", generator=gen) for s in shapes]
    return out, a, b, shapes


def _time_path(
    path: str,
    out_list,
    a_list,
    b_list,
    *,
    scale: float,
    block_size: int,
    iters: int,
):
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        # Reset out_list to a_list so the math is reproducible per
        # iteration (out = a + scale * b each time).
        torch.cuda.synchronize()
        start.record()
        if path == "foreach":
            torch._foreach_copy_(out_list, a_list)
            torch._foreach_add_(out_list, b_list, alpha=scale)
        elif path == "triton_per_tensor":
            multi_tensor_add_triton_per_tensor(out_list, a_list, b_list, scale=scale, block_size=block_size)
        elif path == "triton_packed":
            multi_tensor_add_triton_packed(out_list, a_list, b_list, scale=scale, block_size=block_size)
        else:
            raise ValueError(path)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _total_bytes(shapes, *, dtype_bytes: int = 2) -> int:
    # 3 tensors per shape (out, a, b) * dtype_bytes.
    return sum(int(torch.tensor(list(s)).prod()) for s in shapes) * 3 * dtype_bytes


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Bench requires CUDA / HIP.")

    out_list, a_list, b_list, shapes = _build_pool(args)
    total_bytes = _total_bytes(shapes)
    scale = 0.5

    print(
        f"== multi_tensor_add bench: n_tensors={args.n_tensors} " f"shape_mix={args.shape_mix} dtype=bf16 =="
    )
    print(f"   total bytes = {total_bytes / 1e6:.1f} MB / call; " f"iters={args.iters} warmup={args.warmup}")

    results = {}
    for path in ("foreach", "triton_per_tensor", "triton_packed"):
        _time_path(path, out_list, a_list, b_list, scale=scale, block_size=args.block_size, iters=args.warmup)
        times = _time_path(
            path, out_list, a_list, b_list, scale=scale, block_size=args.block_size, iters=args.iters
        )
        med = median(times)
        results[path] = {**_stats(times), "gbps_median": total_bytes / (med * 1e-3) / 1e9}

    print()
    print(f"   {'path':<22}  {'median ms':>10}  {'GB/s':>10}")
    print(f"   {'-' * 22}  {'-' * 10}  {'-' * 10}")
    for path in ("foreach", "triton_per_tensor", "triton_packed"):
        r = results[path]
        print(f"   {path:<22}  {r['median_ms']:>10.3f}  {r['gbps_median']:>10.1f}")

    eager = results["foreach"]
    triton_packed = results["triton_packed"]
    triton_per = results["triton_per_tensor"]
    print()
    print(f"   triton_packed speedup vs foreach: " f"{eager['median_ms'] / triton_packed['median_ms']:.2f}x")
    print(f"   triton_per_tensor speedup vs foreach: " f"{eager['median_ms'] / triton_per['median_ms']:.2f}x")

    if args.json_out is not None:
        payload = {
            "config": {
                "n_tensors": args.n_tensors,
                "shape_mix": args.shape_mix,
                "iters": args.iters,
                "warmup": args.warmup,
                "block_size": args.block_size,
                "total_bytes": total_bytes,
            },
            "results": results,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"\n   wrote {args.json_out}")


if __name__ == "__main__":
    main()
