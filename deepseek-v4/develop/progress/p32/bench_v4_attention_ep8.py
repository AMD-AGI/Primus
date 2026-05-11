#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the V4 Triton dense / HCA attention path at proxy-EP8 shape.

Mirrors ``progress/p31/bench_csa_attention_ep8.py`` so plan-5 P32 kernel
experiments do not require launching full EP8 training to read out FWD /
BWD wall time.

Modes:

* ``--mode dense`` — ``compress_ratio == 0`` shape: MQA-expanded
  ``[B, H, S, D]`` K / V, SWA causal mask (``swa_window=128``), no
  additive mask. Matches the V4-Flash dense layers.
* ``--mode hca`` — ``compress_ratio == 128`` HCA split-mask shape:
  K / V are ``[B, H, S + P, D]`` (local SWA prefix + compressed pool
  suffix), ``additive_mask`` is the pool-only ``[S, P]`` causal mask,
  ``hca_local_seqlen=S``, ``swa_window=128``. Matches the V4-Flash
  HCA layers.

The reported BWD time excludes the forward pass (BWD-only timing window
created from a pre-computed forward output, identical to the CSA bench
fix shipped in P31).
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

from primus.backends.megatron.core.transformer.v4_attention_kernels import v4_attention


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dense", "hca"), default="dense")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--compress-ratio", type=int, default=128, help="HCA pool size = S/CR.")
    parser.add_argument("--swa-window", type=int, default=128)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--no-sink", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Emit a one-step torch.profiler trace.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("deepseek-v4/develop/progress/p32/attention_bench_traces"),
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


def _make_pool_mask(B: int, S: int, P: int, device: torch.device) -> torch.Tensor:
    """Build a ``[S, P]`` pool-visibility mask matching production HCA.

    The mask is causal at the *pool block* granularity: query row ``q``
    can see pool slot ``p`` iff ``(p + 1) * CR <= q`` (the compressed
    pool slot covers ``CR`` raw key positions ending at ``(p + 1) * CR``).
    Invisible entries are set to a finite ``-1e30`` sentinel.
    """
    cr = S // P
    q_idx = torch.arange(S, device=device).unsqueeze(-1)
    p_end = (torch.arange(P, device=device) + 1) * cr
    visible = p_end.unsqueeze(0) <= q_idx
    mask = torch.where(visible, 0.0, -1.0e30).to(torch.float32)
    return mask  # broadcast across B, H


def _make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor | None]:
    if not torch.cuda.is_available():
        raise RuntimeError("V4 attention benchmark requires CUDA / HIP.")
    device = torch.device("cuda")
    dtype = _dtype(args.dtype)

    B, H, S, D = args.batch, args.heads, args.seq_len, args.head_dim
    if args.mode == "hca":
        if S % args.compress_ratio != 0:
            raise ValueError(
                f"HCA mode expects S ({S}) to be divisible by compress_ratio ({args.compress_ratio})."
            )
        P = S // args.compress_ratio
        Sk = S + P
        hca_local_seqlen = S
        additive_mask = _make_pool_mask(B, S, P, device)
    else:
        P = 0
        Sk = S
        hca_local_seqlen = 0
        additive_mask = None

    gen = torch.Generator(device=device).manual_seed(args.seed)
    q = torch.randn(B, H, S, D, device=device, dtype=dtype, generator=gen)
    k = torch.randn(B, H, Sk, D, device=device, dtype=dtype, generator=gen)
    v = torch.randn(B, H, Sk, D, device=device, dtype=dtype, generator=gen)
    dout = torch.randn_like(q)
    sink = None
    if not args.no_sink:
        sink = (torch.randn(H, device=device, dtype=torch.float32, generator=gen) * 0.1).requires_grad_(True)

    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    return {
        "q": q,
        "k": k,
        "v": v,
        "additive_mask": additive_mask,
        "dout": dout,
        "sink": sink,
        "hca_local_seqlen": hca_local_seqlen,
        "P": P,
        "Sk": Sk,
    }


def _forward(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]) -> torch.Tensor:
    swa_window = (
        args.swa_window if (tensors["additive_mask"] is None or tensors["hca_local_seqlen"] > 0) else 0
    )
    return v4_attention(
        tensors["q"],
        tensors["k"],
        tensors["v"],
        sink=tensors["sink"],
        swa_window=swa_window,
        additive_mask=tensors["additive_mask"],
        attn_dropout=0.0,
        training=True,
        scale=1.0 / math.sqrt(args.head_dim),
        hca_local_seqlen=tensors["hca_local_seqlen"],
    )


def _backward_from_out(
    out: torch.Tensor,
    tensors: dict[str, torch.Tensor | None],
) -> tuple[torch.Tensor, ...]:
    leaves = [tensors["q"], tensors["k"], tensors["v"]]
    if tensors["sink"] is not None:
        leaves.append(tensors["sink"])
    return torch.autograd.grad(
        out, leaves, grad_outputs=tensors["dout"], retain_graph=False, allow_unused=True
    )


def _backward(args: argparse.Namespace, tensors: dict[str, torch.Tensor | None]) -> tuple[torch.Tensor, ...]:
    return _backward_from_out(_forward(args, tensors), tensors)


def _time_ms(fn, *, iters: int) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        torch.cuda.synchronize()
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del result
    return times


def _time_backward_ms(
    args: argparse.Namespace, tensors: dict[str, torch.Tensor | None], *, iters: int
) -> list[float]:
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        out = _forward(args, tensors)
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
    tensors = _make_inputs(args)

    shape = {
        "mode": args.mode,
        "B": args.batch,
        "H": args.heads,
        "S": args.seq_len,
        "D": args.head_dim,
        "P": tensors["P"],
        "Sk": tensors["Sk"],
        "hca_local_seqlen": tensors["hca_local_seqlen"],
        "swa_window": args.swa_window,
        "dtype": args.dtype,
        "sink": not args.no_sink,
    }
    print("[shape]", json.dumps(shape, sort_keys=True))

    for _ in range(args.warmup):
        _backward(args, tensors)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    fwd_ms = _time_ms(lambda: _forward(args, tensors), iters=args.iters)
    bwd_ms = _time_backward_ms(args, tensors, iters=args.iters)

    result = {
        "shape": shape,
        "forward": _stats(fwd_ms),
        "backward": _stats(bwd_ms),
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.profile:
        _profile_once(args, tensors)


if __name__ == "__main__":
    main()
