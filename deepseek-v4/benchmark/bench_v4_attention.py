###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""DeepSeek-V4 attention kernel benchmark — all backends in one table.

Unified benchmark of the **forward** and **backward** V4 attention kernels for
every backend, so there is a single place to compare them (one row per backend,
``ms | TFLOP/s`` cells; TFLOP/s over the useful work so all rows are comparable):

* ``triton``    — PRODUCTION Triton (separate K/V; pool/SWA/HCA launchers)
* ``gluon``     — fused single-latent (K==V) sparse-MLA, hand-tuned gfx950
* ``triton_v2`` — fused single-latent sparse-MLA in plain Triton (tl.dot / MFMA)
* ``flydsl_v1`` — fused single-latent sparse-MLA in native FlyDSL MFMA (fwd + bwd)

The legacy ``_flydsl_v0_deprecated`` gathered-CSA backend (scalarized GEMV) is
NOT benchmarked: it has known correctness issues and depends on the
/workspace/FlyDSL-amd source tree.

The fused ``gluon`` / ``triton_v2`` / ``flydsl_v1`` backends share ONE kernel-pair
API and are timed on IDENTICAL V4-form inputs (zero rope pad + [local ++ pool]
kv + [SWA window ++ pool] topk). Each backend is guarded; unavailable ones are
simply skipped.

Benchmarks for the two production model sizes across all three layer kinds:

* ``cr=0``   — dense / sliding-window (SWA-only) attention
* ``cr=4``   — CSA (local SWA + sparse top-k from the compressed pool)
* ``cr=128`` — HCA (local SWA + full compressed pool, joint softmax)

at the real attention shapes (``seq_len=4096``, ``mbs=1``, bf16, sink on,
``swa_window=128``):

* V4-Flash:  H=64,  head_dim=512, index_topk=512
* V4-Pro:    H=128, head_dim=512, index_topk=1024

Backend detail:

* ``triton`` (ALL crs): PRODUCTION launchers used by ``DeepseekV4Attention`` —
    cr=0/128 ``_launch_v4_attention_fwd``/``_bwd`` (dense/HCA); cr=4
    ``_launch_v4_csa_attention_pool_fwd``/``_pool_bwd`` (split FWD + segreduce
    BWD in-kernel gather; NOT the legacy gathered API, ~30-260x slower).
* ``gluon`` / ``triton_v2`` / ``flydsl_v1`` (ALL crs): fused single-latent
    sparse-MLA (``sparse_mla_{fwd,bwd}_v4_*``); the layer kind is just a
    different TOPK (cr=0: swa 128; cr=4: 128+sparse; cr=128: 128+pool).

Effective TFLOP/s uses ``2*T*H*TOPK*(D_V+D_V)`` (useful work over head_dim=512),
BWD = 2.5x FWD, the SAME formula for every backend so rows are directly
comparable (the fused backends' zero-rope-pad overhead shows up in ms).

Run inside the dev container (gfx950 / MI355X):

    python deepseek-v4/benchmark/bench_v4_attention.py
    python deepseek-v4/benchmark/bench_v4_attention.py --variant pro --cr 4
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Tuple

import torch

# NOTE: the legacy `_flydsl_v0_deprecated` CSA backend (scalarized GEMV) is
# intentionally NOT imported/benchmarked here — it has known correctness issues
# and depends on the /workspace/FlyDSL-amd source tree. The native FlyDSL MFMA
# backend is `_flydsl_v1` (benchmarked below as `flydsl_v1`).
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention_bwd import (
    _launch_v4_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention_fwd import (
    _launch_v4_attention_fwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_csa_attention_bwd import (
    _launch_v4_csa_attention_pool_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_csa_attention_fwd import (
    _launch_v4_csa_attention_pool_fwd,
)

# Fused single-latent (K==V) sparse-MLA backends. All share ONE kernel-pair
# API (fwd(q, kv, topk, attn_sink, kv_lora_rank, scale) -> (o, lse); bwd ->
# (dq, dkv, d_sink)) so they are timed on identical V4-form inputs. Each is
# guarded so the benchmark still runs where a backend is unavailable.
_SPARSE_MLA_BACKENDS = {}  # name -> (fwd_fn, bwd_fn)
try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_dsa import (
        sparse_mla_bwd_v4_gluon,
        sparse_mla_fwd_v4_gluon,
    )

    _SPARSE_MLA_BACKENDS["gluon"] = (sparse_mla_fwd_v4_gluon, sparse_mla_bwd_v4_gluon)
except Exception:  # noqa: BLE001
    pass
try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
        sparse_mla_bwd_v4_triton,
        sparse_mla_fwd_v4_triton,
    )

    _SPARSE_MLA_BACKENDS["triton_v2"] = (sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton)
except Exception:  # noqa: BLE001
    pass
try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._flydsl_v1 import (
        sparse_mla_bwd_v4_flydsl,
        sparse_mla_fwd_v4_flydsl,
    )

    _SPARSE_MLA_BACKENDS["flydsl_v1"] = (sparse_mla_fwd_v4_flydsl, sparse_mla_bwd_v4_flydsl)
except Exception:  # noqa: BLE001
    pass
# gluon_v2: our aiter-gluon-inspired backend (guarded; skipped until present).
try:
    from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_v2 import (
        sparse_mla_bwd_v4_gluon_v2,
        sparse_mla_fwd_v4_gluon_v2,
    )

    _SPARSE_MLA_BACKENDS["gluon_v2"] = (sparse_mla_fwd_v4_gluon_v2, sparse_mla_bwd_v4_gluon_v2)
except Exception:  # noqa: BLE001
    pass
# aiter PR#3833 gluon sparse-MLA prefill (fwd-only) — OPTIONAL reference for
# comparison. Lives under agent/workspace; set PRIMUS_V4_AITER_DIR to override.
# Guarded so the benchmark is unchanged when the extracted adapter is absent.
try:
    _aiter_dir = os.environ.get(
        "PRIMUS_V4_AITER_DIR",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "agent",
            "workspace",
            "aiter_dsv4_prefill_20260706",
        ),
    )
    _aiter_dir = os.path.abspath(_aiter_dir)
    if _aiter_dir not in sys.path:
        sys.path.insert(0, _aiter_dir)
    from aiter_v4_adapter import sparse_mla_fwd_v4_aiter

    _SPARSE_MLA_BACKENDS["aiter_gluon"] = (sparse_mla_fwd_v4_aiter, None)
except Exception:  # noqa: BLE001
    pass

_GLUON_AVAIL = "gluon" in _SPARSE_MLA_BACKENDS

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

    fwd_triton = bwd_triton = None

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

    # Fused single-latent (K==V) sparse-MLA backends (gluon / triton_v2 /
    # flydsl_v2): all on the SAME V4-form inputs (zero rope pad + [local ++ pool]
    # kv + [SWA window ++ pool] topk). Time each by swapping the kernel pair.
    sparse_fb = {}  # name -> (fwd_closure, bwd_closure)
    if _SPARSE_MLA_BACKENDS:
        gq, gkv, gtopk_idx, gsink, gdo = _build_gluon_v4form(
            cr=cr, H=H, S=S, D=D, K=glu_K, P=glu_P, W=_SWA_WINDOW
        )
        gscale = 1.0 / math.sqrt(D)  # V4 scale (score over 512)
        for _name, (_fwd_k, _bwd_k) in _SPARSE_MLA_BACKENDS.items():
            fwd_c = (
                lambda fk: (lambda: fk(gq, gkv, gtopk_idx, attn_sink=gsink, kv_lora_rank=D, scale=gscale))
            )(_fwd_k)
            _out, _lse = _call_fwd(fwd_c)
            if _out is None:  # fwd unavailable (e.g. flydsl_v2 kernel WIP) -> skip bwd
                bwd_c = None
            else:
                bwd_c = (
                    lambda bk, o, l: (
                        lambda: bk(
                            gq, gkv, o, gdo, gtopk_idx, l, attn_sink=gsink, kv_lora_rank=D, scale=gscale
                        )
                    )
                )(_bwd_k, _out, _lse)
            sparse_fb[_name] = (fwd_c, bwd_c)

    # Effective FLOPs over the USEFUL work (score+value both over head_dim=512,
    # TOPK = real key count); BWD = 2.5x FWD. Same formula for ALL backends so
    # TFLOP/s is comparable (the sparse-MLA zero-rope-pad overhead shows up in
    # ms, not in counted FLOPs).
    T = B * S
    fwd_flop = 2.0 * T * H * topk_eff * (D + D)
    flops = {"fwd": fwd_flop, "bwd": 2.5 * fwd_flop}

    # Backend table: native production Triton (separate K/V), then the fused
    # sparse-MLA backends (legacy `_flydsl_v0` gathered CSA is excluded).
    backends = [("triton", fwd_triton, bwd_triton)]
    for _name in ("gluon", "triton_v2", "gluon_v2", "flydsl_v1", "aiter_gluon"):
        if _name in sparse_fb:
            backends.append((_name, sparse_fb[_name][0], sparse_fb[_name][1]))

    print(
        f"\n=== V4-{variant.upper()} cr={cr} ({_CR_NAME[cr]}) | B={B} H={H} S={S} D={D}"
        f"{extra} TOPK_eff={topk_eff} swa={_SWA_WINDOW} sink=on bf16 ===\n"
        f"  FWD GFLOP={fwd_flop / 1e9:.1f} (useful, over head_dim={D})  "
        f"(cells: ms | TFLOP/s; *_v2/gluon = V4-form fused single-latent)",
        flush=True,
    )
    print(
        f"  {'backend':12s} {'fwd (ms|TF)':>18s} {'bwd (ms|TF)':>18s}",
        flush=True,
    )
    rows = []
    for name, fwd_fn, bwd_fn in backends:
        fwd_med, fwd_err = _safe_time(fwd_fn, warmup=warmup, iters=iters)
        bwd_med, bwd_err = _safe_time(bwd_fn, warmup=warmup, iters=iters)
        print(f"  {name:12s} {_cell(fwd_med, flops['fwd'])} {_cell(bwd_med, flops['bwd'])}", flush=True)
        for op, err in (("fwd", fwd_err), ("bwd", bwd_err)):
            if err:
                print(f"       {name} {op} error: {err}", flush=True)
        rows.append((name, fwd_med, bwd_med))
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
