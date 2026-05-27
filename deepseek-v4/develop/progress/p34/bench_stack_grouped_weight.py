#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P34 `stack_grouped_weight` Triton kernel vs eager.

The plan-5 P32 final EP=8 proxy trace attributes ``hipMemcpyWithStream``
**289.6 ms / 32 calls** to the eager
``torch.stack(weights, dim=0).transpose(1, 2).contiguous()`` chain inside
:meth:`PrimusTurboGroupedMLP._stack_grouped_linear_weight`.  At V4-Flash
EP=8 widths the chain runs twice per layer (``linear_fc1`` + ``linear_fc2``,
~9 ms / call at ~57 GB/s effective HBM bandwidth — far below the MI355X
HBM peak of ~3.2 TB/s).  This bench measures the Triton path's wall time
and effective bandwidth at the same shapes as a standalone microbench so
P34 kernel experiments do not require launching full EP8 training.

Modes:

* ``--mode fc1`` — ``linear_fc1`` shape at V4-Flash EP=8:
  ``E=32, K=2*ffn=4096, N=hidden=4096``.
* ``--mode fc2`` — ``linear_fc2`` shape at V4-Flash EP=8:
  ``E=32, K=hidden=4096, N=ffn=2048``.

Reports ``<ms> ms | <GB/s effective bandwidth>`` per call for the FWD,
BWD, and the eager reference.  Effective bandwidth is computed as
``2 * elements * bytes_per_element / time_s`` (one read pass over inputs
+ one write pass over outputs; both passes are the full data size since
the operation is a pure layout transform).

Same proxy-mode flags as the plan-5 P32 attention bench
(``--n-input-copies``, ``--l2-flush-mb``) so the measured numbers track
proxy-steady-state behaviour, not L2-cache-warm artefacts.
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

from primus.backends.megatron.core.extensions._triton.stack_grouped_weight import (  # noqa: E402
    StackGroupedWeightFn,
    eager_stack_grouped_weight,
)

# ---------------------------------------------------------------------------
# CLI + helpers
# ---------------------------------------------------------------------------


_MODES = {
    # mode -> (K, N) where weight{i}.shape == [K, N], output is [E, N, K]
    "fc1": (4096, 4096),  # K=2*ffn, N=hidden  (V4-Flash MoE w/ SwiGLU)
    "fc2": (4096, 2048),  # K=hidden, N=ffn
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(_MODES.keys()), default="fc1")
    parser.add_argument("--num-experts", "-E", type=int, default=32)
    parser.add_argument("--K", type=int, default=None, help="Override mode K (=N_out).")
    parser.add_argument("--N", type=int, default=None, help="Override mode N (=N_in).")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--n-input-copies",
        type=int,
        default=4,
        help=(
            "Proxy-mode buffer rotation: allocate N independent copies of the per-expert "
            "weights and rotate through them per iteration, so each call reads fresh HBM "
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
            "iterations to evict cached weights. Set to 0 to disable."
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--no-eager",
        action="store_true",
        help="Skip the eager reference timing (useful when iterating on the Triton path).",
    )
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _bytes_per_element(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _stats(values: Iterable[float]) -> dict[str, float]:
    seq = list(values)
    return {
        "mean_ms": mean(seq),
        "median_ms": median(seq),
        "min_ms": min(seq),
        "max_ms": max(seq),
    }


def _bandwidth_gbps(*, elements: int, bytes_per_element: int, time_ms: float) -> float:
    # Total bytes moved = 1 read pass over inputs + 1 write pass over outputs.
    total_bytes = 2 * elements * bytes_per_element
    return total_bytes / (time_ms * 1e-3) / 1e9


def _make_pool(args: argparse.Namespace) -> List[List[torch.Tensor]]:
    """Allocate ``n_input_copies`` independent sets of ``E`` weights."""
    K = args.K if args.K is not None else _MODES[args.mode][0]
    N = args.N if args.N is not None else _MODES[args.mode][1]
    E = args.num_experts
    dtype = _dtype(args.dtype)
    device = torch.device("cuda")
    n = max(1, args.n_input_copies)

    pool: List[List[torch.Tensor]] = []
    for c in range(n):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        pool.append(
            [
                torch.randn(K, N, dtype=dtype, device=device, generator=gen).requires_grad_(True)
                for _ in range(E)
            ]
        )
    return pool


class L2Flusher:
    """Evict L2 / last-level cache between bench iterations (matches the
    plan-5 P32 attention bench helper).
    """

    def __init__(self, size_mb: int, device: torch.device | str = "cuda") -> None:
        n_bytes = max(0, size_mb) * 1024 * 1024
        n_int32 = n_bytes // 4
        self.buf = torch.empty(n_int32, dtype=torch.int32, device=device) if n_int32 > 0 else None

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


# ---------------------------------------------------------------------------
# Per-mode timing
# ---------------------------------------------------------------------------


def _time_fwd(
    *,
    pool: List[List[torch.Tensor]],
    triton_path: bool,
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        weights = pool[i % len(pool)]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = StackGroupedWeightFn.apply(*weights)
        else:
            out = eager_stack_grouped_weight(weights)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(
    *,
    pool: List[List[torch.Tensor]],
    triton_path: bool,
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    """Time only the BWD pass (FWD computed outside the timing window)."""
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(iters):
        weights = pool[i % len(pool)]
        for w in weights:
            if w.grad is not None:
                w.grad = None
        if triton_path:
            out = StackGroupedWeightFn.apply(*weights)
        else:
            out = eager_stack_grouped_weight(weights)
        dout = torch.randn_like(out)
        torch.cuda.synchronize()
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        out.backward(dout)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("stack_grouped_weight bench requires CUDA / HIP.")

    K = args.K if args.K is not None else _MODES[args.mode][0]
    N = args.N if args.N is not None else _MODES[args.mode][1]
    E = args.num_experts
    dtype = _dtype(args.dtype)
    bpe = _bytes_per_element(dtype)
    elements_total = E * K * N

    print(f"== stack_grouped_weight bench: mode={args.mode} " f"E={E} K={K} N={N} dtype={args.dtype} ==")
    print(
        f"   per-call data size = {elements_total * bpe / 1024 / 1024:.1f} MiB; "
        f"iters={args.iters}, warmup={args.warmup}, "
        f"n_input_copies={args.n_input_copies}, l2_flush_mb={args.l2_flush_mb}"
    )

    pool = _make_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    paths = [("triton", True)]
    if not args.no_eager:
        paths.append(("eager", False))

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, triton_flag in paths:
        # Warmup
        _time_fwd(pool=pool, triton_path=triton_flag, iters=args.warmup, flusher=flusher)
        # Timed
        fwd = _time_fwd(pool=pool, triton_path=triton_flag, iters=args.iters, flusher=flusher)
        bwd = _time_bwd(pool=pool, triton_path=triton_flag, iters=args.iters, flusher=flusher)
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {
                **_stats(fwd),
                "bw_gbps_median": _bandwidth_gbps(
                    elements=elements_total, bytes_per_element=bpe, time_ms=fwd_med
                ),
            },
            "bwd": {
                **_stats(bwd),
                "bw_gbps_median": _bandwidth_gbps(
                    elements=elements_total, bytes_per_element=bpe, time_ms=bwd_med
                ),
            },
        }

    # Print table
    print()
    print(f"   {'path':<8}  {'fwd ms':>10}  {'fwd GB/s':>10}  {'bwd ms':>10}  {'bwd GB/s':>10}")
    print(f"   {'-' * 8}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name in [n for n, _ in paths]:
        r = results[name]
        print(
            f"   {name:<8}  "
            f"{r['fwd']['median_ms']:>10.3f}  "
            f"{r['fwd']['bw_gbps_median']:>10.1f}  "
            f"{r['bwd']['median_ms']:>10.3f}  "
            f"{r['bwd']['bw_gbps_median']:>10.1f}"
        )

    if "triton" in results and "eager" in results:
        eager = results["eager"]
        triton_r = results["triton"]
        print()
        print(
            f"   Triton FWD speedup vs eager: "
            f"{eager['fwd']['median_ms'] / triton_r['fwd']['median_ms']:.2f}x"
        )
        print(
            f"   Triton BWD speedup vs eager: "
            f"{eager['bwd']['median_ms'] / triton_r['bwd']['median_ms']:.2f}x"
        )

    if args.json_out is not None:
        payload = {
            "shape": {"mode": args.mode, "E": E, "K": K, "N": N, "dtype": args.dtype},
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
