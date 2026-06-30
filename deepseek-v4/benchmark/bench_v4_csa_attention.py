###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 CSA attention kernel benchmark: Triton vs FlyDSL.

Compares the **forward** and **backward** CSA (``compress_ratio == 4``)
attention kernels for the two production model sizes at the real
attention shapes:

* V4-Flash:  H=64,  head_dim=512, index_topk=512,  swa_window=128
* V4-Pro:    H=128, head_dim=512, index_topk=1024, swa_window=128

with ``seq_len = 4096`` and ``micro_batch_size = 1`` (``B = 1``). KV is
the V4 MQA single latent broadcast across heads; ``K_topk =
min(index_topk, S // compress_ratio)``.

Two backends are timed via their raw kernel launchers (no autograd
wrapper, so the comparison is kernel-to-kernel):

* Triton:  ``_launch_v4_csa_attention_fwd`` / ``_launch_v4_csa_attention_bwd``
* FlyDSL:  ``_launch_v4_attention_fwd_csa`` /
           ``flydsl_v4_csa_attention_bwd`` (with both FlyDSL bwd knobs on
           so a single FlyDSL full-kernel launch produces all 5 grads)

bf16, sink enabled (production). The backward uses each backend's own
forward ``(out, lse)`` so the lse convention stays self-consistent.

Run inside the dev container (gfx950 / MI355X):

    python deepseek-v4/benchmark/bench_v4_csa_attention.py
    python deepseek-v4/benchmark/bench_v4_csa_attention.py --variant pro --iters 50
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Tuple

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl.kernels.v4_attention_fwd_flydsl_csa import (
    _launch_v4_attention_fwd_csa,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl.kernels.v4_csa_attention_bwd_flydsl_mqa import (
    flydsl_v4_csa_attention_bwd,
)

# Import local primus Triton launchers first so they are cached before the
# FlyDSL wrappers (which insert /workspace/Primus onto sys.path) load.
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_csa_attention_bwd import (
    _launch_v4_csa_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_csa_attention_fwd import (
    _launch_v4_csa_attention_fwd,
)

# (variant, H, index_topk)  — head_dim / swa_window / cr shared below.
_VARIANTS = {
    "flash": dict(H=64, index_topk=512),
    "pro": dict(H=128, index_topk=1024),
}
_HEAD_DIM = 512
_SWA_WINDOW = 128
_COMPRESS_RATIO = 4


def _build_inputs(*, B: int, H: int, S: int, D: int, K_topk: int, seed: int = 0):
    """Production-like bf16 CSA inputs with MQA (head-identical) K/V."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    # MQA single latent: FlyDSL forward consumes the [B, 1, S, D] tensor
    # (mqa_kv stride trick); Triton consumes the [B, H, S, D] broadcast
    # (its production input, post head-expand). Heads are identical.
    k_mqa = torch.randn(B, 1, S, D, generator=g, device=device, dtype=dtype)
    v_mqa = torch.randn(B, 1, S, D, generator=g, device=device, dtype=dtype)
    k_full = k_mqa.expand(B, H, S, D).contiguous()
    v_full = v_mqa.expand(B, H, S, D).contiguous()

    gathered = torch.randn(B, S, K_topk, D, generator=g, device=device, dtype=dtype)
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.25
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )
    gathered = gathered * valid.unsqueeze(-1).to(dtype)
    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
    dout = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    return q, k_full, v_full, k_mqa, v_mqa, gathered, sparse_mask, sink, dout


def _time_ms(fn, *, warmup: int, iters: int) -> Tuple[float, float]:
    """Return (median_ms, mean_ms) over ``iters`` timed launches."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    median = times[len(times) // 2]
    mean = sum(times) / len(times)
    return median, mean


def _safe_time(fn, *, warmup: int, iters: int):
    """Time ``fn``; on failure return (None, short_error_string)."""
    try:
        med, _ = _time_ms(fn, warmup=warmup, iters=iters)
        return med, None
    except Exception as exc:  # noqa: BLE001 - benchmark must survive one cell failing
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        return None, f"{type(exc).__name__}: {msg}"


def _bench_variant(variant: str, *, B: int, S: int, warmup: int, iters: int):
    cfg = _VARIANTS[variant]
    H = cfg["H"]
    D = _HEAD_DIM
    P = max(S // _COMPRESS_RATIO, 1)
    K_topk = min(cfg["index_topk"], P)
    scale = 1.0 / math.sqrt(D)

    q, k_full, v_full, k_mqa, v_mqa, gathered, sparse_mask, sink, dout = _build_inputs(
        B=B, H=H, S=S, D=D, K_topk=K_topk
    )

    print(
        f"\n=== V4-{variant.upper()} CSA | B={B} H={H} S={S} D={D} "
        f"K_topk={K_topk} swa={_SWA_WINDOW} sink=on bf16 ===",
        flush=True,
    )

    # --- forward launchers ---
    def fwd_triton():
        return _launch_v4_csa_attention_fwd(
            q,
            k_full,
            v_full,
            gathered,
            sparse_mask,
            sink=sink,
            swa_window=_SWA_WINDOW,
            scale=scale,
        )

    def fwd_flydsl():
        return _launch_v4_attention_fwd_csa(
            q,
            k_mqa,
            v_mqa,
            gathered,
            sink=sink,
            swa_window=_SWA_WINDOW,
            sparse_mask=sparse_mask,
            scale=scale,
        )

    # Force the FlyDSL bwd to actually run its own kernel (both knobs on).
    os.environ["V4_FLYDSL_CSA_BWD_FLY_DQ"] = "1"
    os.environ["V4_FLYDSL_CSA_BWD_FLY_DKV"] = "1"
    os.environ.setdefault("V4_FLYDSL_BWD_VERBOSE", "0")

    # Pre-compute each backend's own fwd (out, lse) for its bwd input. Guarded
    # so a backend that cannot run this shape (e.g. FlyDSL int32 element-count
    # overflow at K_topk=1024) does not abort the whole sweep.
    try:
        out_t, lse_t = fwd_triton()
    except Exception:  # noqa: BLE001
        out_t = lse_t = None
    try:
        out_f, lse_f = fwd_flydsl()
    except Exception:  # noqa: BLE001
        out_f = lse_f = None

    def bwd_triton():
        return _launch_v4_csa_attention_bwd(
            q,
            k_full,
            v_full,
            gathered,
            sparse_mask,
            out_t,
            dout,
            lse_t,
            sink=sink,
            swa_window=_SWA_WINDOW,
            scale=scale,
        )

    def bwd_flydsl():
        return flydsl_v4_csa_attention_bwd(
            q,
            k_full,
            v_full,
            gathered,
            sparse_mask,
            out_f,
            dout,
            lse_f,
            sink=sink,
            swa_window=_SWA_WINDOW,
            scale=scale,
        )

    rows = []
    for op, tri_fn, fly_fn in (
        ("fwd", fwd_triton, fwd_flydsl),
        ("bwd", bwd_triton, bwd_flydsl),
    ):
        tri_med, tri_err = _safe_time(tri_fn, warmup=warmup, iters=iters)
        fly_med, fly_err = _safe_time(fly_fn, warmup=warmup, iters=iters)
        rows.append((op, tri_med, tri_err, fly_med, fly_err))

    print(f"  {'op':4s} {'Triton(ms)':>12s} {'FlyDSL(ms)':>12s} {'speedup(T/F)':>14s}", flush=True)
    for op, tri_med, tri_err, fly_med, fly_err in rows:
        tri_s = f"{tri_med:12.3f}" if tri_med is not None else f"{'FAIL':>12s}"
        fly_s = f"{fly_med:12.3f}" if fly_med is not None else f"{'FAIL':>12s}"
        if tri_med is not None and fly_med is not None and fly_med > 0:
            sp = f"{tri_med / fly_med:13.2f}x"
        else:
            sp = f"{'-':>14s}"
        print(f"  {op:4s} {tri_s} {fly_s} {sp}", flush=True)
        if tri_err:
            print(f"       Triton {op} error: {tri_err}", flush=True)
        if fly_err:
            print(f"       FlyDSL {op} error: {fly_err}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=["flash", "pro", "both"],
        default="both",
        help="model size to benchmark (default: both)",
    )
    parser.add_argument("--seq", type=int, default=4096, help="sequence length (default 4096)")
    parser.add_argument("--mbs", type=int, default=1, help="micro batch size / B (default 1)")
    parser.add_argument("--warmup", type=int, default=10, help="warmup launches (default 10)")
    parser.add_argument("--iters", type=int, default=30, help="timed launches (default 30)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / HIP device required for this benchmark")

    torch.backends.cuda.matmul.allow_tf32 = True
    variants = ["flash", "pro"] if args.variant == "both" else [args.variant]

    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"seq={args.seq} mbs={args.mbs} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    for v in variants:
        _bench_variant(v, B=args.mbs, S=args.seq, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
