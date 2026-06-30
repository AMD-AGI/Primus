###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention kernel benchmark: Triton vs FlyDSL vs Gluon.

Benchmarks the **forward** and **backward** V4 attention kernels for the two
production model sizes across all three layer kinds:

* ``cr=0``   — dense / sliding-window (SWA-only) attention
* ``cr=4``   — CSA (local SWA + sparse top-k from the compressed pool)
* ``cr=128`` — HCA (local SWA + full compressed pool, joint softmax)

at the real attention shapes (``seq_len=4096``, ``mbs=1``, bf16, sink on,
``swa_window=128``):

* V4-Flash:  H=64,  head_dim=512, index_topk=512
* V4-Pro:    H=128, head_dim=512, index_topk=1024

Backends (each on its native / production path; ``ms | TFLOP/s`` per cell):

* Triton (ALL crs): the PRODUCTION launchers used by ``DeepseekV4Attention``:
    - cr=0   : ``_launch_v4_attention_fwd`` / ``_bwd`` (dense, SWA-pruned)
    - cr=4   : ``_launch_v4_csa_attention_pool_fwd`` / ``_pool_bwd`` (split FWD +
               segreduce BWD, in-kernel gather; NOT the legacy ``gathered`` API
               which is ~30-260x slower, see attention_perf.md)
    - cr=128 : ``_launch_v4_attention_fwd`` / ``_bwd`` with a pool-only joint
               mask + ``hca_local_seqlen=S``
* FlyDSL (cr=4 ONLY): ``_launch_v4_attention_fwd_csa`` /
    ``flydsl_v4_csa_attention_bwd`` (both bwd knobs on -> one full-kernel launch).
* Gluon (ALL crs): ``sparse_mla_fwd_v4_gluon`` / ``sparse_mla_bwd_v4_gluon``
    (gfx950). Sparse-MLA latent I/O (q/kv ``[T, H, D+rope]``, a single
    ``topk_indices`` list); the layer kind is just a different TOPK
    (cr=0: swa 128; cr=4: 128+sparse; cr=128: 128+pool). Different
    representation than Triton/FlyDSL, so T/G is indicative, not same-kernel.

Effective TFLOP/s uses aiter's formula ``2*T*H*TOPK*(D_QK+D_V)`` (BWD = 2.5x
FWD), computed per-backend with real dims (Gluon QK over D_QK=576; Triton/FlyDSL
over 512). ``TOPK = swa_window + sparse/pool`` per layer kind.

Run inside the dev container (gfx950 / MI355X):

    python deepseek-v4/benchmark/bench_v4_csa_attention.py
    python deepseek-v4/benchmark/bench_v4_csa_attention.py --variant pro --cr 4
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
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_bwd import (
    _launch_v4_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_fwd import (
    _launch_v4_attention_fwd,
)
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

_VARIANTS = {
    "flash": dict(H=64, index_topk=512),
    "pro": dict(H=128, index_topk=1024),
}
_HEAD_DIM = 512
_SWA_WINDOW = 128
_ROPE_DIM = 64  # sparse-MLA d_qk = kv_lora_rank (=_HEAD_DIM) + rope
_CR_NAME = {0: "SWA/dense", 4: "CSA", 128: "HCA"}


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------


def _common_inputs(B: int, H: int, S: int, D: int, seed: int = 0):
    """q + MQA single-latent K/V (full + [.,1,.,.]) + sink + dout, all bf16."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    q = torch.randn(B, H, S, D, generator=g, device=dev, dtype=dt)
    k_mqa = torch.randn(B, 1, S, D, generator=g, device=dev, dtype=dt)
    v_mqa = torch.randn(B, 1, S, D, generator=g, device=dev, dtype=dt)
    k_full = k_mqa.expand(B, H, S, D).contiguous()
    v_full = v_mqa.expand(B, H, S, D).contiguous()
    sink = torch.randn(H, generator=g, device=dev, dtype=torch.float32) * 0.1
    dout = torch.randn(B, H, S, D, generator=g, device=dev, dtype=dt)
    return q, k_mqa, v_mqa, k_full, v_full, sink, dout


def _csa_sparse(B: int, H: int, S: int, D: int, K: int, P: int, g: torch.Generator):
    """CSA sparse branch: compressed pool + per-query top-K indices, plus the
    pre-gathered equivalent (FlyDSL) sharing the same valid/invalid pattern."""
    dev, dt = "cuda", torch.bfloat16
    pool = torch.randn(B, P, D, generator=g, device=dev, dtype=dt)
    valid = torch.rand(B, S, K, generator=g, device=dev) > 0.25
    topk_idxs = torch.randint(0, P, (B, S, K), generator=g, device=dev, dtype=torch.int32)
    topk_idxs = torch.where(valid, topk_idxs, torch.full_like(topk_idxs, -1))
    safe = topk_idxs.clamp(min=0).long()
    gathered = torch.gather(
        pool.unsqueeze(1).expand(B, S, P, D), dim=2, index=safe.unsqueeze(-1).expand(B, S, K, D)
    )
    gathered = gathered * valid.unsqueeze(-1).to(dt)
    sparse_mask = torch.where(
        valid,
        torch.zeros((), dtype=dt, device=dev),
        torch.tensor(float("-inf"), dtype=dt, device=dev),
    )
    return pool, topk_idxs, gathered, sparse_mask


def _hca_mask(S: int, P: int, ratio: int, device, dtype):
    """HCA pool-only additive causal mask [S, P]: pool slot s visible to query t
    iff (s+1)*ratio - 1 <= t (matches DeepseekV4Attention._hca_extra_mask)."""
    t = torch.arange(S, device=device).unsqueeze(1)
    s_end = (torch.arange(P, device=device).unsqueeze(0) + 1) * ratio - 1
    return torch.where(s_end <= t, 0.0, float("-inf")).to(dtype)


def _build_gluon_v4form(*, cr: int, H: int, S: int, D: int, K: int, P: int, W: int, seed: int = 0):
    """Gluon inputs in the **V4 form** (matches the training adapter):

    The 512 V4 latent (RoPE baked in-place) is the gluon "lora" with a zero rope
    pad (kernel needs D_ROPE>0); kv = [local latent ++ pool] and topk = [SWA
    window ++ (cr=4: sparse pool top-k | cr=128: full causal pool | cr=0: none)].
    ``topk`` is padded to a multiple of 64 so the gluon bwd dKV tiling (TILE_K=64)
    is valid (notably HCA: 128+32=160 -> 192). scale = 1/sqrt(D) (V4, over 512).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, dt = "cuda", torch.bfloat16
    latent = torch.randn(S, D, generator=g, device=dev, dtype=dt)
    q512 = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)
    z_q = torch.zeros(S, H, _ROPE_DIM, device=dev, dtype=dt)
    q_g = torch.cat([q512, z_q], dim=-1).contiguous()  # [S, H, D+rope_pad]
    sink = torch.randn(H, generator=g, device=dev, dtype=torch.float32) * 0.1
    do = torch.randn(S, H, D, generator=g, device=dev, dtype=dt)

    ti = torch.arange(S, device=dev).view(S, 1)
    win = ti - W + 1 + torch.arange(W, device=dev).view(1, W)  # [S, W] local token idx
    win = torch.where(win >= 0, win, torch.full_like(win, -1))

    if cr == 0:
        kv512 = latent.unsqueeze(1)  # [S, 1, D]
        topk = win
    else:
        pool = torch.randn(P, D, generator=g, device=dev, dtype=dt)
        kv512 = torch.cat([latent, pool], dim=0).unsqueeze(1)  # [S+P, 1, D]
        if cr == 4:
            sp = torch.randint(0, P, (S, K), generator=g, device=dev)
            pool_topk = S + sp
        else:  # cr == 128: HCA full causal pool
            ps = torch.arange(P, device=dev).view(1, P)
            vis = ((ps + 1) * cr - 1) <= ti  # [S, P]
            pool_topk = torch.where(vis, S + ps, torch.full_like(ps.expand(S, P), -1))
        topk = torch.cat([win, pool_topk], dim=1)

    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat([topk, torch.full((S, pad), -1, device=dev, dtype=topk.dtype)], dim=1)
    topk_g = topk.to(torch.int32).contiguous()

    z_kv = torch.zeros(kv512.shape[0], 1, _ROPE_DIM, device=dev, dtype=dt)
    kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()
    return q_g, kv_g, topk_g, sink, do


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


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
    return times[len(times) // 2], sum(times) / len(times)


def _safe_time(fn, *, warmup: int, iters: int):
    """Time ``fn``; on failure return (None, short_error_string)."""
    if fn is None:
        return None, None
    try:
        med, _ = _time_ms(fn, warmup=warmup, iters=iters)
        return med, None
    except Exception as exc:  # noqa: BLE001 - benchmark must survive one cell failing
        msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
        return None, f"{type(exc).__name__}: {msg}"


def _call_fwd(fn):
    """Call a fwd launcher once for the bwd's (out, lse); (None, None) on failure."""
    if fn is None:
        return None, None
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return None, None


def _tflops(flops: float, med_ms) -> float:
    if med_ms is None or med_ms <= 0:
        return float("nan")
    return flops / (med_ms * 1e-3) / 1e12


def _cell(med, flops) -> str:
    if med is None:
        return f"{'FAIL':>18s}"
    return f"{med:9.2f} | {_tflops(flops, med):6.1f}"


# ---------------------------------------------------------------------------
# Per-(variant, cr) benchmark
# ---------------------------------------------------------------------------


def _bench_cr(variant: str, cr: int, *, B: int, S: int, warmup: int, iters: int):
    cfg = _VARIANTS[variant]
    H = cfg["H"]
    D = _HEAD_DIM
    scale = 1.0 / math.sqrt(D)
    g = torch.Generator(device="cuda").manual_seed(11)
    q, k_mqa, v_mqa, k_full, v_full, sink, dout = _common_inputs(B, H, S, D)

    fwd_triton = bwd_triton = fwd_flydsl = bwd_flydsl = None

    if cr == 0:
        topk_eff = _SWA_WINDOW
        glu_K, glu_P = 0, 0
        extra = ""

        def fwd_triton():
            return _launch_v4_attention_fwd(
                q,
                k_full,
                v_full,
                sink=sink,
                swa_window=_SWA_WINDOW,
                additive_mask=None,
                scale=scale,
                hca_local_seqlen=0,
            )

        out_t, lse_t = _call_fwd(fwd_triton)

        def bwd_triton():
            return _launch_v4_attention_bwd(
                q,
                k_full,
                v_full,
                out_t,
                dout,
                lse_t,
                sink=sink,
                swa_window=_SWA_WINDOW,
                additive_mask=None,
                scale=scale,
                hca_local_seqlen=0,
            )

    elif cr == 4:
        P = max(S // 4, 1)
        K = min(cfg["index_topk"], P)
        topk_eff = _SWA_WINDOW + K
        glu_K, glu_P = K, P
        extra = f" K_topk={K} P={P}"
        pool, topk_idxs, gathered, sparse_mask = _csa_sparse(B, H, S, D, K, P, g)

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

        out_t, lse_t = _call_fwd(fwd_triton)

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

        # FlyDSL: CSA only. Both bwd knobs on -> single full-kernel launch.
        os.environ["V4_FLYDSL_CSA_BWD_FLY_DQ"] = "1"
        os.environ["V4_FLYDSL_CSA_BWD_FLY_DKV"] = "1"
        os.environ.setdefault("V4_FLYDSL_BWD_VERBOSE", "0")

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

        out_f, lse_f = _call_fwd(fwd_flydsl)

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

    elif cr == 128:
        P = max(S // cr, 1)
        topk_eff = _SWA_WINDOW + P
        glu_K, glu_P = 0, P
        extra = f" P={P}"
        pool_bh = torch.randn(B, H, P, D, generator=g, device="cuda", dtype=torch.bfloat16)
        k_hca = torch.cat([k_full, pool_bh], dim=2).contiguous()
        v_hca = torch.cat([v_full, pool_bh], dim=2).contiguous()
        hca_mask = _hca_mask(S, P, cr, "cuda", torch.bfloat16)

        def fwd_triton():
            return _launch_v4_attention_fwd(
                q,
                k_hca,
                v_hca,
                sink=sink,
                swa_window=_SWA_WINDOW,
                additive_mask=hca_mask,
                scale=scale,
                hca_local_seqlen=S,
            )

        out_t, lse_t = _call_fwd(fwd_triton)

        def bwd_triton():
            return _launch_v4_attention_bwd(
                q,
                k_hca,
                v_hca,
                out_t,
                dout,
                lse_t,
                sink=sink,
                swa_window=_SWA_WINDOW,
                additive_mask=hca_mask,
                scale=scale,
                hca_local_seqlen=S,
            )

    else:
        raise ValueError(f"unsupported cr={cr}")

    # Gluon: V4-form (the production training invocation) for every cr — zero
    # rope pad + [local ++ pool] buffer + [SWA window ++ pool] topk.
    glu_fwd = glu_bwd = None
    if _GLUON_AVAIL:
        gq, gkv, gtopk_idx, gsink, gdo = _build_gluon_v4form(
            cr=cr, H=H, S=S, D=D, K=glu_K, P=glu_P, W=_SWA_WINDOW
        )
        gscale = 1.0 / math.sqrt(D)  # V4 scale (score over 512)

        def fwd_gluon():
            return sparse_mla_fwd_v4_gluon(gq, gkv, gtopk_idx, attn_sink=gsink, kv_lora_rank=D, scale=gscale)

        out_g, lse_g = _call_fwd(fwd_gluon)

        def bwd_gluon():
            return sparse_mla_bwd_v4_gluon(
                gq, gkv, out_g, gdo, gtopk_idx, lse_g, attn_sink=gsink, kv_lora_rank=D, scale=gscale
            )

        glu_fwd, glu_bwd = fwd_gluon, bwd_gluon

    # Effective FLOPs over the USEFUL work (score+value both over head_dim=512,
    # TOPK = real key count); BWD = 2.5x FWD. Same formula for all backends so
    # TFLOP/s is comparable — the gluon zero-rope-pad overhead shows up in ms,
    # not in counted FLOPs.
    T = B * S

    def _mk_flops(d_qk: int, d_v: int):
        fwd = 2.0 * T * H * topk_eff * (d_qk + d_v)
        return {"fwd": fwd, "bwd": 2.5 * fwd}

    flops_csa = _mk_flops(D, D)  # Triton / FlyDSL / Gluon — useful work over 512
    flops_glu = flops_csa

    print(
        f"\n=== V4-{variant.upper()} cr={cr} ({_CR_NAME[cr]}) | B={B} H={H} S={S} D={D}"
        f"{extra} TOPK_eff={topk_eff} swa={_SWA_WINDOW} sink=on bf16 ===\n"
        f"  FWD GFLOP={flops_csa['fwd'] / 1e9:.1f} (useful, over head_dim={D})  "
        f"(cells: ms | TFLOP/s; Gluon=V4-form w/ zero rope pad)",
        flush=True,
    )

    rows = []
    for op, tri_fn, fly_fn, glu_fn in (
        ("fwd", fwd_triton, fwd_flydsl, glu_fwd),
        ("bwd", bwd_triton, bwd_flydsl, glu_bwd),
    ):
        tri_med, tri_err = _safe_time(tri_fn, warmup=warmup, iters=iters)
        fly_med, fly_err = _safe_time(fly_fn, warmup=warmup, iters=iters)
        glu_med, glu_err = _safe_time(glu_fn, warmup=warmup, iters=iters)
        if cr != 4:
            fly_err = "FlyDSL: cr=4 only"
        rows.append((op, tri_med, tri_err, fly_med, fly_err, glu_med, glu_err))

    print(
        f"  {'op':4s} {'Triton (ms|TF)':>18s} {'FlyDSL (ms|TF)':>18s} "
        f"{'Gluon (ms|TF)':>18s} {'T/F':>7s} {'T/G':>7s}",
        flush=True,
    )
    for op, tri_med, tri_err, fly_med, fly_err, glu_med, glu_err in rows:
        tf = f"{tri_med / fly_med:6.2f}x" if (tri_med and fly_med) else f"{'-':>7s}"
        tg = f"{tri_med / glu_med:6.2f}x" if (tri_med and glu_med) else f"{'-':>7s}"
        fly_cell = _cell(fly_med, flops_csa[op]) if cr == 4 else f"{'n/a':>18s}"
        print(
            f"  {op:4s} {_cell(tri_med, flops_csa[op])} {fly_cell} "
            f"{_cell(glu_med, flops_glu[op])} {tf} {tg}",
            flush=True,
        )
        if tri_err:
            print(f"       Triton {op} error: {tri_err}", flush=True)
        if fly_err and cr == 4:
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
    parser.add_argument(
        "--cr",
        choices=["0", "4", "128", "all"],
        default="all",
        help="compress ratio / layer kind to benchmark (default: all)",
    )
    parser.add_argument("--seq", type=int, default=4096, help="sequence length (default 4096)")
    parser.add_argument("--mbs", type=int, default=1, help="micro batch size / B (default 1)")
    parser.add_argument("--warmup", type=int, default=10, help="warmup launches (default 10)")
    parser.add_argument("--iters", type=int, default=30, help="timed launches (default 30)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA / HIP device required for this benchmark")

    # Production-optimal Triton config (matches attention_perf.md P57 defaults):
    # split CSA FWD (monolithic OFF), split + segreduce BWD.
    os.environ.setdefault("PRIMUS_V4_CSA_FWD_FORCE_MONOLITHIC", "0")
    os.environ.setdefault("PRIMUS_V4_ATTN_BWD_USE_SPLIT", "1")
    os.environ.setdefault("PRIMUS_V4_CSA_BWD_SEGREDUCE", "1")

    torch.backends.cuda.matmul.allow_tf32 = True
    variants = ["flash", "pro"] if args.variant == "both" else [args.variant]
    crs = [0, 4, 128] if args.cr == "all" else [int(args.cr)]

    print(
        f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"seq={args.seq} mbs={args.mbs} warmup={args.warmup} iters={args.iters}",
        flush=True,
    )
    for v in variants:
        for cr in crs:
            _bench_cr(v, cr, B=args.mbs, S=args.seq, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
