###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 CSA attention kernel benchmark: Triton vs FlyDSL vs Gluon.

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

* Triton:  ``_launch_v4_csa_attention_pool_fwd`` / ``_pool_bwd`` — the PRODUCTION
           CSA path (``v4_csa_attention_from_pool``: split FWD + segreduce BWD,
           in-kernel gather from the compressed pool). NOT the legacy ``gathered``
           reference API, which is ~30-260x slower (see attention_perf.md).
* FlyDSL:  ``_launch_v4_attention_fwd_csa`` /
           ``flydsl_v4_csa_attention_bwd`` (with both FlyDSL bwd knobs on
           so a single FlyDSL full-kernel launch produces all 5 grads)
* Gluon:   ``sparse_mla_fwd_v4_gluon`` / ``sparse_mla_bwd_v4_gluon`` (gfx950).
           NOTE: the Gluon backend uses the **sparse-MLA latent** I/O contract
           (q/kv ``[T, H, D+rope]``, a single ``topk_indices`` list = SWA
           window + sparse top-k pre-merged), NOT the CSA gathered/sparse_mask
           layout. It is the same logical CSA workload but a different
           representation, so the T/G speedup is indicative, not a same-kernel
           comparison. Gluon timings are omitted (FAIL row) where unavailable.

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
#
# Triton is benchmarked on its PRODUCTION CSA path: the pool + in-kernel-topk
# launchers (``v4_csa_attention_from_pool`` -> split FWD + segreduce BWD), which
# is what ``DeepseekV4Attention._csa_forward`` runs when
# ``use_v4_triton_csa_attention=True``. The older ``gathered`` API
# (``_launch_v4_csa_attention_fwd``) is only the eager-fallback / P26 reference
# and is ~30-260x slower, so it is NOT used here. See
# ``deepseek-v4/develop/perf/attention_perf.md`` (cr=4 FWD 1.43ms / BWD 5.11ms).
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_csa_attention_bwd import (
    _launch_v4_csa_attention_pool_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_csa_attention_fwd import (
    _launch_v4_csa_attention_pool_fwd,
)

# Gluon DSA (sparse-MLA latent backend, gfx950). Optional: guarded so the
# benchmark still runs on archs / triton builds where Gluon is unavailable.
try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_dsa import (
        sparse_mla_bwd_v4_gluon,
        sparse_mla_fwd_v4_gluon,
    )

    _GLUON_AVAIL = True
except Exception:  # noqa: BLE001
    _GLUON_AVAIL = False

# (variant, H, index_topk)  — head_dim / swa_window / cr shared below.
_VARIANTS = {
    "flash": dict(H=64, index_topk=512),
    "pro": dict(H=128, index_topk=1024),
}
_HEAD_DIM = 512
_SWA_WINDOW = 128
_COMPRESS_RATIO = 4
_ROPE_DIM = 64  # sparse-MLA d_qk = kv_lora_rank (=_HEAD_DIM) + rope


def _build_gluon_inputs(*, H: int, S: int, D: int, topk: int, seed: int = 0):
    """Sparse-MLA latent inputs for the Gluon backend.

    Different representation than the CSA gathered form: q/kv carry the
    [.,.,D+rope] latent (K includes rope, V = first D channels), and a single
    ``topk_indices`` list selects the keys (SWA window + sparse, pre-merged).
    Tokens flatten the (B=1) batch, so total_tokens = S.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    device = "cuda"
    dtype = torch.bfloat16
    d_qk = D + _ROPE_DIM
    q = torch.randn(S, H, d_qk, generator=g, device=device, dtype=dtype)
    kv = torch.randn(S, 1, d_qk, generator=g, device=device, dtype=dtype)
    topk_idx = torch.randint(0, S, (S, topk), generator=g, device=device, dtype=torch.int32)
    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
    do = torch.randn(S, H, D, generator=g, device=device, dtype=dtype)
    return q, kv, topk_idx, sink, do


def _build_inputs(*, B: int, H: int, S: int, D: int, K_topk: int, P: int, seed: int = 0):
    """Production-like bf16 CSA inputs with MQA (head-identical) K/V.

    Produces BOTH representations so each backend runs its native path:
      * Triton (production): ``pool`` [B, P, D] + ``topk_idxs`` [B, Sq, K] int32
        (in-kernel gather, split FWD + segreduce BWD);
      * FlyDSL: pre-gathered ``gathered`` [B, Sq, K, D] + additive ``sparse_mask``.
    The same valid/invalid pattern (~25% dropped) is shared so the two
    representations describe the same sparse workload.
    """
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

    # Compressed pool + per-query top-K indices (Triton production path).
    pool = torch.randn(B, P, D, generator=g, device=device, dtype=dtype)
    valid = torch.rand(B, S, K_topk, generator=g, device=device) > 0.25
    topk_idxs = torch.randint(0, P, (B, S, K_topk), generator=g, device=device, dtype=torch.int32)
    topk_idxs = torch.where(valid, topk_idxs, torch.full_like(topk_idxs, -1))

    # Pre-gathered equivalent (FlyDSL path): gather the pool by topk + mask.
    safe = topk_idxs.clamp(min=0).long()
    gathered = torch.gather(
        pool.unsqueeze(1).expand(B, S, P, D), dim=2, index=safe.unsqueeze(-1).expand(B, S, K_topk, D)
    )
    gathered = gathered * valid.unsqueeze(-1).to(dtype)
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dtype, device=device),
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    sink = torch.randn(H, generator=g, device=device, dtype=torch.float32) * 0.1
    dout = torch.randn(B, H, S, D, generator=g, device=device, dtype=dtype)
    return q, k_full, v_full, k_mqa, v_mqa, pool, topk_idxs, gathered, sparse_mask, sink, dout


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


def _tflops(flops: float, med_ms) -> float:
    """Effective TFLOP/s from total FLOPs and per-launch wall time (ms)."""
    if med_ms is None or med_ms <= 0:
        return float("nan")
    return flops / (med_ms * 1e-3) / 1e12


def _cell(med, flops) -> str:
    """Format one ``ms | tflops`` table cell (or FAIL)."""
    if med is None:
        return f"{'FAIL':>18s}"
    return f"{med:9.2f} | {_tflops(flops, med):6.1f}"


def _bench_variant(variant: str, *, B: int, S: int, warmup: int, iters: int):
    cfg = _VARIANTS[variant]
    H = cfg["H"]
    D = _HEAD_DIM
    P = max(S // _COMPRESS_RATIO, 1)
    K_topk = min(cfg["index_topk"], P)
    scale = 1.0 / math.sqrt(D)

    q, k_full, v_full, k_mqa, v_mqa, pool, topk_idxs, gathered, sparse_mask, sink, dout = _build_inputs(
        B=B, H=H, S=S, D=D, K_topk=K_topk, P=P
    )

    print(
        f"\n=== V4-{variant.upper()} CSA | B={B} H={H} S={S} D={D} "
        f"K_topk={K_topk} P={P} swa={_SWA_WINDOW} sink=on bf16 ===",
        flush=True,
    )

    # --- forward launchers ---
    # Triton: production pool path (split FWD, in-kernel gather from pool).
    def fwd_triton():
        return _launch_v4_csa_attention_pool_fwd(
            q,
            k_full,
            v_full,
            pool,
            topk_idxs,
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
        return _launch_v4_csa_attention_pool_bwd(
            q,
            k_full,
            v_full,
            pool,
            topk_idxs,
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

    # --- Gluon (sparse-MLA latent): different I/O contract, same logical CSA
    # workload. TOPK = SWA window + sparse top-k pre-merged into one index list
    # (Flash 128+512=640, Pro 128+1024=1152). q/kv carry the [.,.,D+rope] latent.
    glu_fwd = glu_bwd = None
    if _GLUON_AVAIL:
        gtopk = _SWA_WINDOW + K_topk
        gq, gkv, gtopk_idx, gsink, gdo = _build_gluon_inputs(H=H, S=S, D=D, topk=gtopk)
        gscale = 1.0 / math.sqrt(D + _ROPE_DIM)

        def fwd_gluon():
            return sparse_mla_fwd_v4_gluon(gq, gkv, gtopk_idx, attn_sink=gsink, kv_lora_rank=D, scale=gscale)

        try:
            out_g, lse_g = fwd_gluon()
        except Exception:  # noqa: BLE001
            out_g = lse_g = None

        def bwd_gluon():
            return sparse_mla_bwd_v4_gluon(
                gq, gkv, out_g, gdo, gtopk_idx, lse_g, attn_sink=gsink, kv_lora_rank=D, scale=gscale
            )

        glu_fwd, glu_bwd = fwd_gluon, bwd_gluon

    # Effective FLOPs aligned with aiter's bench_sparse_mla_v4_fwd.py:
    #   FWD = 2 * T * H * TOPK * (D_QK + D_V)   (QK over D_QK + PV over D_V)
    #   BWD = 2.5 * FWD                          (5 matmuls vs 2; matches the
    #                                             attention_perf.md 10:4 ratio)
    # computed PER BACKEND with its real dims, since the representations differ:
    #   * Gluon (sparse-MLA): QK is a D_QK=576 dot (512 lora + 64 *extra* rope),
    #     PV is D_V=512  -> per-pair FLOP 2*(576+512).
    #   * Triton / FlyDSL (CSA): K=V=head_dim=512 (rope is partial, in-place
    #     within the 512), so QK and PV are both 512 -> per-pair 2*(512+512).
    # TOPK = SWA window + sparse top-k (Flash 128+512=640, Pro 128+1024=1152),
    # matching aiter's ``topk_for_layer`` and the actual key count attended.
    T = B * S
    topk_eff = _SWA_WINDOW + K_topk

    def _mk_flops(d_qk: int, d_v: int):
        fwd = 2.0 * T * H * topk_eff * (d_qk + d_v)
        return {"fwd": fwd, "bwd": 2.5 * fwd}

    flops_csa = _mk_flops(D, D)  # Triton / FlyDSL (gathered, K=V=512)
    flops_glu = _mk_flops(D + _ROPE_DIM, D)  # Gluon (sparse-MLA, QK over 576)
    print(
        f"  TOPK_eff={topk_eff} (swa {_SWA_WINDOW} + sparse {K_topk})  "
        f"FWD GFLOP: CSA={flops_csa['fwd'] / 1e9:.1f} Gluon={flops_glu['fwd'] / 1e9:.1f}  "
        f"(cells: ms | TFLOP/s; per-backend FLOPs)",
        flush=True,
    )

    rows = []
    for op, tri_fn, fly_fn, glu_fn in (
        ("fwd", fwd_triton, fwd_flydsl, glu_fwd),
        ("bwd", bwd_triton, bwd_flydsl, glu_bwd),
    ):
        tri_med, tri_err = _safe_time(tri_fn, warmup=warmup, iters=iters)
        fly_med, fly_err = _safe_time(fly_fn, warmup=warmup, iters=iters)
        if glu_fn is not None:
            glu_med, glu_err = _safe_time(glu_fn, warmup=warmup, iters=iters)
        else:
            glu_med, glu_err = None, ("gluon backend unavailable" if not _GLUON_AVAIL else None)
        rows.append((op, tri_med, tri_err, fly_med, fly_err, glu_med, glu_err))

    print(
        f"  {'op':4s} {'Triton (ms|TF)':>18s} {'FlyDSL (ms|TF)':>18s} "
        f"{'Gluon (ms|TF)':>18s} {'T/F':>7s} {'T/G':>7s}",
        flush=True,
    )
    for op, tri_med, tri_err, fly_med, fly_err, glu_med, glu_err in rows:
        tf = f"{tri_med / fly_med:6.2f}x" if (tri_med and fly_med) else f"{'-':>7s}"
        tg = f"{tri_med / glu_med:6.2f}x" if (tri_med and glu_med) else f"{'-':>7s}"
        print(
            f"  {op:4s} {_cell(tri_med, flops_csa[op])} {_cell(fly_med, flops_csa[op])} "
            f"{_cell(glu_med, flops_glu[op])} {tf} {tg}",
            flush=True,
        )
        if tri_err:
            print(f"       Triton {op} error: {tri_err}", flush=True)
        if fly_err:
            print(f"       FlyDSL {op} error: {fly_err}", flush=True)
        if glu_err:
            print(f"       Gluon  {op} error: {glu_err}", flush=True)
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

    # Production-optimal Triton CSA config (matches attention_perf.md P57 final
    # defaults): split FWD (monolithic OFF), split + segreduce BWD.
    os.environ.setdefault("PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC", "0")
    os.environ.setdefault("PRIMUS_V4_ATTN_BWD_USE_SPLIT", "1")
    os.environ.setdefault("PRIMUS_V4_CSA_BWD_SEGREDUCE", "1")

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
