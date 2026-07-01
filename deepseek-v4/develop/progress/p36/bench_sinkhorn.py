#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P36 fused Sinkhorn-Knopp Triton kernel.

The plan-5 P32 final EP=8 proxy trace attributes ~62 ms / 16 calls (sum
of ``Torch-Compiled Region`` + ``CompiledFunctionBackward`` for the
cached ``_build_compiled_sinkhorn`` artefact) to the FWD+BWD of
``sinkhorn_normalize`` inside :class:`HyperConnection.compute_weights`.
At 8 layers x 2 ``HyperConnection`` calls per layer the per-call cost
is roughly ``62 / 16 = 3.9 ms``.

This bench measures the Triton path's FWD and BWD latency against:

* the eager body (``primus...hyper_connection.sinkhorn_normalize`` with
  ``use_compiled=False``), and
* the plan-5 P29 compiled body (``use_compiled=True``).

Three pre-set shapes:

* ``--mode k4`` -- V4-Flash production:  ``[B=1, S=4096, K=4, K=4]``;
* ``--mode k4_small`` -- coverage shape:  ``[B=2, S=64,   K=4, K=4]``;
* ``--mode k8`` -- forward-compat:        ``[B=1, S=4096, K=8, K=8]``.

Reports ``<ms>`` and effective HBM bandwidth (``GB/s``) per call for
FWD, BWD, and the two reference paths.  ``--n-input-copies`` and
``--l2-flush-mb`` mirror the P34/P35 microbench conventions so the
numbers track proxy-steady-state behaviour and not L2-warm artefacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.megatron.core.transformer.hyper_connection import (  # noqa: E402
    _get_compiled_sinkhorn,
    sinkhorn_normalize,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.sinkhorn import (  # noqa: E402
    SinkhornNormalizeFn,
    eager_sinkhorn_normalize,
)

# ---------------------------------------------------------------------------
# CLI + helpers
# ---------------------------------------------------------------------------


# (B, S, K) for each preset shape; the last-two-dims are (K, K).
_MODES = {
    "k4": (1, 4096, 4),
    "k4_small": (2, 64, 4),
    "k8": (1, 4096, 8),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(_MODES.keys()), default="k4")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--n-iters", type=int, default=20, help="Sinkhorn iterations.")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--n-input-copies",
        type=int,
        default=4,
        help=(
            "Proxy-mode buffer rotation: allocate N independent copies of x and "
            "rotate through them per iteration so each call reads fresh HBM "
            "addresses (defeats HBM row-buffer / L2 reuse). Set to 1 for the "
            "legacy single-buffer microbench."
        ),
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=512,
        help="L2 / last-level-cache eviction buffer (MiB). Set to 0 to disable.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-compiled", action="store_true")
    return parser.parse_args()


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


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


def _build_pool(args: argparse.Namespace) -> List[torch.Tensor]:
    """Allocate ``n_input_copies`` independent x tensors (non-negative)."""
    B, S, K = _MODES[args.mode]
    dtype = _dtype(args.dtype)
    device = torch.device("cuda")
    n = max(1, args.n_input_copies)

    xs: List[torch.Tensor] = []
    for c in range(n):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        x = torch.rand((B, S, K, K), dtype=dtype, device=device, generator=gen) + 1e-3
        x.requires_grad_(True)
        xs.append(x)
    return xs


class L2Flusher:
    def __init__(self, size_mb: int, device: torch.device | str = "cuda") -> None:
        n_bytes = max(0, size_mb) * 1024 * 1024
        n_int32 = n_bytes // 4
        self.buf = torch.empty(n_int32, dtype=torch.int32, device=device) if n_int32 > 0 else None

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


# ---------------------------------------------------------------------------
# Path factories
# ---------------------------------------------------------------------------


def _make_runner(*, name: str, n_iters: int, dtype: torch.dtype) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``fn(x) -> y`` for the requested path."""
    if name == "triton":
        return lambda x: SinkhornNormalizeFn.apply(x, n_iters, 1e-6)
    if name == "eager":
        return lambda x: eager_sinkhorn_normalize(x, n_iters=n_iters, eps=1e-6)
    if name == "compiled":
        compiled_fn = _get_compiled_sinkhorn(n_iters, 1e-6, dtype)
        return compiled_fn
    raise ValueError(f"unknown path {name!r}")


def _time_fwd(
    *,
    xs: List[torch.Tensor],
    runner: Callable[[torch.Tensor], torch.Tensor],
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(xs)
    for i in range(iters):
        x = xs[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        out = runner(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(
    *,
    xs: List[torch.Tensor],
    runner: Callable[[torch.Tensor], torch.Tensor],
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(xs)
    for i in range(iters):
        x = xs[i % n]
        if x.grad is not None:
            x.grad = None
        out = runner(x)
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
        raise RuntimeError("sinkhorn bench requires CUDA / HIP.")

    B, S, K = _MODES[args.mode]
    dtype = _dtype(args.dtype)
    bpe = _bytes(dtype)
    n_elem = B * S * K * K
    # FWD HBM traffic: read x (in dtype) + write y (in dtype).  Internal
    # fp32 compute lives in registers (the state buffer is also written
    # but is small relative to x at K=4; we count it explicitly).
    n_states = 2 * args.n_iters
    state_bytes = B * S * n_states * K * K * 4  # fp32 state buffer
    bytes_fwd = bpe * n_elem * 2 + state_bytes
    # BWD HBM traffic: read state buffer + read dy + write dx.
    bytes_bwd = bpe * n_elem * 2 + state_bytes

    print(
        f"== sinkhorn bench: mode={args.mode} B={B} S={S} K={K} "
        f"n_iters={args.n_iters} dtype={args.dtype} =="
    )
    print(
        f"   per-call x = {n_elem * bpe / 1024:.1f} KiB; "
        f"state buf = {state_bytes / 1024 / 1024:.2f} MiB; "
        f"FWD traffic = {bytes_fwd / 1024 / 1024:.2f} MiB; "
        f"BWD traffic = {bytes_bwd / 1024 / 1024:.2f} MiB; "
        f"iters={args.iters}, warmup={args.warmup}, "
        f"n_input_copies={args.n_input_copies}, l2_flush_mb={args.l2_flush_mb}"
    )

    xs = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    paths = ["triton", "eager"]
    if not args.no_compiled:
        paths.append("compiled")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name in paths:
        runner = _make_runner(name=name, n_iters=args.n_iters, dtype=dtype)
        _time_fwd(xs=xs, runner=runner, iters=args.warmup, flusher=flusher)
        fwd = _time_fwd(xs=xs, runner=runner, iters=args.iters, flusher=flusher)
        bwd = _time_bwd(xs=xs, runner=runner, iters=args.iters, flusher=flusher)
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        results[name] = {
            "fwd": {
                **_stats(fwd),
                "bw_gbps_median": _bandwidth_gbps(bytes_total=bytes_fwd, time_ms=fwd_med),
            },
            "bwd": {
                **_stats(bwd),
                "bw_gbps_median": _bandwidth_gbps(bytes_total=bytes_bwd, time_ms=bwd_med),
            },
        }

    print()
    print(f"   {'path':<10}  {'fwd ms':>10}  {'fwd GB/s':>10}  {'bwd ms':>10}  {'bwd GB/s':>10}")
    print(f"   {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}  {'-' * 10}")
    for name in paths:
        r = results[name]
        print(
            f"   {name:<10}  "
            f"{r['fwd']['median_ms']:>10.3f}  "
            f"{r['fwd']['bw_gbps_median']:>10.1f}  "
            f"{r['bwd']['median_ms']:>10.3f}  "
            f"{r['bwd']['bw_gbps_median']:>10.1f}"
        )

    if "triton" in results:
        triton_r = results["triton"]
        for ref in ("eager", "compiled"):
            if ref not in results:
                continue
            ref_r = results[ref]
            print()
            print(
                f"   Triton FWD speedup vs {ref}: "
                f"{ref_r['fwd']['median_ms'] / triton_r['fwd']['median_ms']:.2f}x"
            )
            print(
                f"   Triton BWD speedup vs {ref}: "
                f"{ref_r['bwd']['median_ms'] / triton_r['bwd']['median_ms']:.2f}x"
            )

    if args.json_out is not None:
        payload = {
            "shape": {
                "mode": args.mode,
                "B": B,
                "S": S,
                "K": K,
                "dtype": args.dtype,
                "n_iters": args.n_iters,
            },
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
