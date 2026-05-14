#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Microbenchmark the plan-6 P35 fused interleaved partial RoPE Triton kernel.

The plan-5 P32 final EP=8 proxy trace attributes ``CatArrayBatchedCopy_contig``
~ **10.0 ms / 24 calls** + a non-trivial share of
``elementwise_kernel_manual_unroll<128, 8>`` to the 9-op
``apply_interleaved_partial_rope`` chain (slice / reshape / 4 broadcast
muls / stack / reshape / cat).  At 16 invocations per iter (q + k per
``DualRoPE`` call × 8 layers) the per-call cost is **~3-5 ms**.

This bench measures the Triton path's per-call latency and effective
HBM bandwidth against the eager body at the V4-Flash EP=8 shapes:

* ``--mode q`` -- Q tensor: ``[B=1, S=4096, H=64, head_dim=512, rd=64]``;
* ``--mode k`` -- K tensor: ``[B=1, S=4096, H=1,  head_dim=64,  rd=64]``.

Reports ``<ms> ms | <GB/s effective bandwidth>`` per call for FWD, BWD,
and the eager reference.  Effective bandwidth is computed as
``(x_bytes + cos_bytes + sin_bytes + out_bytes) / time_s`` to keep
parity with the plan-6 P34 microbench convention.

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
from typing import Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.rope_interleaved_partial import (  # noqa: E402
    RoPEInterleavedPartialFn,
    eager_apply_interleaved_partial_rope,
)

# ---------------------------------------------------------------------------
# CLI + helpers
# ---------------------------------------------------------------------------


# (B, S, H, head_dim, rotary_dim) for each V4-Flash EP=8 mode.
_MODES = {
    "q": (1, 4096, 64, 512, 64),
    "k": (1, 4096, 1, 64, 64),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(_MODES.keys()), default="q")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260514)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--n-input-copies",
        type=int,
        default=4,
        help=(
            "Proxy-mode buffer rotation: allocate N independent copies of x / cos / sin "
            "and rotate through them per iteration so each call reads fresh HBM addresses "
            "(defeats HBM row-buffer / L2 reuse). Set to 1 for the legacy single-buffer "
            "microbench."
        ),
    )
    parser.add_argument(
        "--l2-flush-mb",
        type=int,
        default=512,
        help="L2 / last-level-cache eviction buffer (MiB). Set to 0 to disable.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--no-eager", action="store_true")
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


def _build_pool(
    args: argparse.Namespace,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Allocate ``n_input_copies`` independent x / cos / sin tuples."""
    B, S, H, head_dim, rd = _MODES[args.mode]
    dtype = _dtype(args.dtype)
    device = torch.device("cuda")
    rd_half = rd // 2
    n = max(1, args.n_input_copies)

    xs: List[torch.Tensor] = []
    coss: List[torch.Tensor] = []
    sins: List[torch.Tensor] = []
    for c in range(n):
        gen = torch.Generator(device=device).manual_seed(args.seed + c)
        x = torch.randn((B, S, H, head_dim), dtype=dtype, device=device, generator=gen).requires_grad_(True)
        freqs = torch.randn((B, S, rd_half), dtype=torch.float32, device=device, generator=gen)
        cos = freqs.cos().to(dtype)
        sin = freqs.sin().to(dtype)
        xs.append(x)
        coss.append(cos)
        sins.append(sin)
    return xs, coss, sins


class L2Flusher:
    def __init__(self, size_mb: int, device: torch.device | str = "cuda") -> None:
        n_bytes = max(0, size_mb) * 1024 * 1024
        n_int32 = n_bytes // 4
        self.buf = torch.empty(n_int32, dtype=torch.int32, device=device) if n_int32 > 0 else None

    def flush(self) -> None:
        if self.buf is not None:
            self.buf.zero_()


def _time_fwd(
    *,
    xs: List[torch.Tensor],
    coss: List[torch.Tensor],
    sins: List[torch.Tensor],
    rotary_dim: int,
    triton_path: bool,
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(xs)
    for i in range(iters):
        x = xs[i % n]
        cos = coss[i % n]
        sin = sins[i % n]
        flusher.flush()
        torch.cuda.synchronize()
        start.record()
        if triton_path:
            out = RoPEInterleavedPartialFn.apply(x, cos, sin, rotary_dim)
        else:
            out = eager_apply_interleaved_partial_rope(x, cos, sin, rotary_dim=rotary_dim)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        del out
    return times


def _time_bwd(
    *,
    xs: List[torch.Tensor],
    coss: List[torch.Tensor],
    sins: List[torch.Tensor],
    rotary_dim: int,
    triton_path: bool,
    iters: int,
    flusher: L2Flusher,
) -> List[float]:
    times: List[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    n = len(xs)
    for i in range(iters):
        x = xs[i % n]
        cos = coss[i % n]
        sin = sins[i % n]
        if x.grad is not None:
            x.grad = None
        if triton_path:
            out = RoPEInterleavedPartialFn.apply(x, cos, sin, rotary_dim)
        else:
            out = eager_apply_interleaved_partial_rope(x, cos, sin, rotary_dim=rotary_dim)
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
        raise RuntimeError("rope_triton bench requires CUDA / HIP.")

    B, S, H, head_dim, rd = _MODES[args.mode]
    dtype = _dtype(args.dtype)
    bpe = _bytes(dtype)
    n_elem_x = B * S * H * head_dim
    n_elem_cs = B * S * (rd // 2)
    bytes_per_call = bpe * (
        n_elem_x + n_elem_cs + n_elem_cs + n_elem_x  # read x  # read cos  # read sin  # write out
    )

    print(
        f"== rope_triton bench: mode={args.mode} B={B} S={S} H={H} "
        f"head_dim={head_dim} rd={rd} dtype={args.dtype} =="
    )
    print(
        f"   per-call x tensor = {n_elem_x * bpe / 1024 / 1024:.1f} MiB; "
        f"total traffic = {bytes_per_call / 1024 / 1024:.1f} MiB; "
        f"iters={args.iters}, warmup={args.warmup}, "
        f"n_input_copies={args.n_input_copies}, l2_flush_mb={args.l2_flush_mb}"
    )

    xs, coss, sins = _build_pool(args)
    flusher = L2Flusher(args.l2_flush_mb)

    paths = [("triton", True)]
    if not args.no_eager:
        paths.append(("eager", False))

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, triton_flag in paths:
        _time_fwd(
            xs=xs,
            coss=coss,
            sins=sins,
            rotary_dim=rd,
            triton_path=triton_flag,
            iters=args.warmup,
            flusher=flusher,
        )
        fwd = _time_fwd(
            xs=xs,
            coss=coss,
            sins=sins,
            rotary_dim=rd,
            triton_path=triton_flag,
            iters=args.iters,
            flusher=flusher,
        )
        bwd = _time_bwd(
            xs=xs,
            coss=coss,
            sins=sins,
            rotary_dim=rd,
            triton_path=triton_flag,
            iters=args.iters,
            flusher=flusher,
        )
        fwd_med = median(fwd)
        bwd_med = median(bwd)
        # BWD traffic = dout read + dx write + cos / sin read = same shape sum as FWD.
        bwd_bytes = bytes_per_call
        results[name] = {
            "fwd": {
                **_stats(fwd),
                "bw_gbps_median": _bandwidth_gbps(bytes_total=bytes_per_call, time_ms=fwd_med),
            },
            "bwd": {
                **_stats(bwd),
                "bw_gbps_median": _bandwidth_gbps(bytes_total=bwd_bytes, time_ms=bwd_med),
            },
        }

    # Table
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
            "shape": {
                "mode": args.mode,
                "B": B,
                "S": S,
                "H": H,
                "head_dim": head_dim,
                "rotary_dim": rd,
                "dtype": args.dtype,
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
