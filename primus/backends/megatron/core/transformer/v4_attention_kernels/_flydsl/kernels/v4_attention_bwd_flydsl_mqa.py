"""V4 SWA attention backward FlyDSL launcher (Phase B STEP 1b, cr=0, SWA-only).

API contract (matches the Triton ``_launch_v4_attention_bwd`` signature; the
bwd_modes harness calls this function):

    flydsl_v4_attention_bwd(
        q, k, v, out, dout, lse,
        *,
        sink, swa_window, additive_mask, scale, hca_local_seqlen,
    ) -> (dq, dk, dv, dsink)

STEP 1b SCOPE
-------------
Current state of each compute stage:
  * preprocess (D scalar) -> FlyDSL kernel (``v4_swa_bwd_preprocess_kernel``)
  * dq                    -> FlyDSL kernel (``v4_swa_bwd_dq_kernel``)
                              forked from kernels/sla_bwd_dq.py
  * dk / dv               -> Triton kernel (``_v4_attention_bwd_dkv_kernel``)
  * dsink                 -> FlyDSL kernel writes it (via atomic_fadd in dq)

Env knobs:
  V4_FLYDSL_BWD_FLY_PREPROCESS  default 1 (1=FlyDSL, 0=Triton)
  V4_FLYDSL_BWD_FLY_DQ          default 1 (1=FlyDSL, 0=Triton)
  V4_FLYDSL_BWD_VERBOSE         default 0 (1=print provenance line)
"""
from __future__ import annotations

import math
import os
import sys
import threading
from typing import Optional, Tuple

import torch

_FLYDSL_SRC = "/workspace/FlyDSL-amd"
if _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

os.environ.setdefault("FLYDSL_WAVES_PER_EU", "2")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

if "/workspace/Primus" not in sys.path:
    sys.path.insert(0, "/workspace/Primus")
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_bwd import (  # noqa: E402
    _v4_attention_bwd_preprocess_kernel,
    _v4_attention_bwd_dq_kernel,
    _v4_attention_bwd_dkv_kernel,
    _v4_attention_bwd_dkv_pool_kernel,
)
import triton  # noqa: E402


# Built lazily on first call; module-level kernel build is expensive.
_PREPROCESS_KERNEL_CACHE = {}
_PREPROCESS_KERNEL_LOCK = threading.Lock()

_DQ_KERNEL_CACHE = {}
_DQ_KERNEL_LOCK = threading.Lock()
_DKV_KERNEL_CACHE = {}
_DKV_KERNEL_LOCK = threading.Lock()
_DQ_POOL_KERNEL_CACHE = {}
_DQ_POOL_KERNEL_LOCK = threading.Lock()
_DKV_POOL_KERNEL_CACHE = {}
_DKV_POOL_KERNEL_LOCK = threading.Lock()


def _get_fly_preprocess(head_dim: int, dtype_str: str, block_rows: int):
    key = (head_dim, dtype_str, block_rows)
    with _PREPROCESS_KERNEL_LOCK:
        if key in _PREPROCESS_KERNEL_CACHE:
            return _PREPROCESS_KERNEL_CACHE[key]
        from v4_sla_bwd_kernel import build_v4_swa_bwd_preprocess_module
        launch = build_v4_swa_bwd_preprocess_module(
            head_dim=head_dim, dtype_str=dtype_str, block_rows=block_rows,
        )
        _PREPROCESS_KERNEL_CACHE[key] = launch
        return launch


def _get_fly_dq(num_heads: int, head_dim: int, swa_window: int,
                dtype_str: str, mqa_kv: bool, has_sink: bool):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv, has_sink)
    with _DQ_KERNEL_LOCK:
        if key in _DQ_KERNEL_CACHE:
            return _DQ_KERNEL_CACHE[key]
        from v4_sla_bwd_kernel import build_v4_swa_bwd_dq_module
        launch = build_v4_swa_bwd_dq_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
            has_sink=has_sink,
        )
        _DQ_KERNEL_CACHE[key] = launch
        return launch




def _get_fly_dq_pool(num_heads: int, head_dim: int, pool_size: int,
                     hca_local_seqlen: int, dtype_str: str, mqa_kv: bool):
    key = (num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv)
    with _DQ_POOL_KERNEL_LOCK:
        if key in _DQ_POOL_KERNEL_CACHE:
            return _DQ_POOL_KERNEL_CACHE[key]
        from v4_hca_bwd_dq_pool_kernel import build_v4_hca_bwd_dq_pool_module
        launch = build_v4_hca_bwd_dq_pool_module(
            num_heads=num_heads,
            head_dim=head_dim,
            pool_size=pool_size,
            hca_local_seqlen=hca_local_seqlen,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DQ_POOL_KERNEL_CACHE[key] = launch
        return launch


def _get_fly_dkv_pool(num_heads: int, head_dim: int, pool_size: int,
                      hca_local_seqlen: int, dtype_str: str, mqa_kv: bool):
    key = (num_heads, head_dim, pool_size, hca_local_seqlen, dtype_str, mqa_kv)
    with _DKV_POOL_KERNEL_LOCK:
        if key in _DKV_POOL_KERNEL_CACHE:
            return _DKV_POOL_KERNEL_CACHE[key]
        from v4_hca_bwd_dkv_pool_kernel import build_v4_hca_bwd_dkv_pool_module
        launch = build_v4_hca_bwd_dkv_pool_module(
            num_heads=num_heads,
            head_dim=head_dim,
            pool_size=pool_size,
            hca_local_seqlen=hca_local_seqlen,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DKV_POOL_KERNEL_CACHE[key] = launch
        return launch


def _get_fly_dkv(num_heads: int, head_dim: int, swa_window: int,
                 dtype_str: str, mqa_kv: bool):
    key = (num_heads, head_dim, swa_window, dtype_str, mqa_kv)
    with _DKV_KERNEL_LOCK:
        if key in _DKV_KERNEL_CACHE:
            return _DKV_KERNEL_CACHE[key]
        from v4_sla_bwd_dkv_kernel import build_v4_swa_bwd_dkv_module
        launch = build_v4_swa_bwd_dkv_module(
            num_heads=num_heads,
            head_dim=head_dim,
            swa_window=swa_window,
            dtype_str=dtype_str,
            mqa_kv=mqa_kv,
        )
        _DKV_KERNEL_CACHE[key] = launch
        return launch


def _run_preprocess(
    out: torch.Tensor,
    dout: torch.Tensor,
    *,
    use_flydsl: bool,
    block_m: int,
    block_dmodel: int,
) -> torch.Tensor:
    """Compute D[b,h,m] = sum_d (out[b,h,m,d] * dout[b,h,m,d]) in fp32."""
    B, HQ, Sq, D = out.shape
    d_buf = torch.empty((B, HQ, Sq), device=out.device, dtype=torch.float32)
    if use_flydsl:
        out_f = out.contiguous().view(-1, D)
        dout_f = dout.contiguous().view(-1, D)
        delta_f = d_buf.view(-1)
        n_rows = out_f.shape[0]
        dtype_str = "bf16" if out.dtype == torch.bfloat16 else "f16"
        max_block_threads = 256
        threads_per_row = D // 8
        for br in (8, 4, 2, 1):
            if n_rows % br == 0 and br * threads_per_row <= max_block_threads:
                block_rows = br
                break
        else:
            block_rows = 1
        launch = _get_fly_preprocess(D, dtype_str, block_rows)
        launch(out_f, dout_f, delta_f, n_rows)
        return d_buf
    pre_grid = (triton.cdiv(Sq, block_m), B * HQ)
    _v4_attention_bwd_preprocess_kernel[pre_grid](
        out, dout, d_buf,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        d_buf.stride(0), d_buf.stride(1), d_buf.stride(2),
        Sq,
        HEAD=HQ,
        BLOCK_M=block_m,
        BLOCK_DMODEL=block_dmodel,
        num_warps=4, num_stages=1,
    )
    return d_buf


def _run_dq_triton(
    q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_arg, mask_arg,
    *, scale, swa_window_constexpr, has_sink, has_add_mask,
    hca_local_seqlen, use_causal,
    block_m, block_n, block_dmodel,
    stride_ms, stride_mn,
    Sq, Sk, HQ, HK, B,
    exact_tiles_m, exact_tiles_n,
):
    dq_grid = (triton.cdiv(Sq, block_m), B * HQ)
    _v4_attention_bwd_dq_kernel[dq_grid](
        q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_arg, mask_arg,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        d_buf.stride(0), d_buf.stride(1), d_buf.stride(2),
        dq_fp32.stride(0), dq_fp32.stride(1), dq_fp32.stride(2), dq_fp32.stride(3),
        stride_ms, stride_mn,
        Sq, Sk, float(scale),
        HEAD_Q=HQ, HEAD_K=HK,
        SWA_WINDOW=swa_window_constexpr,
        HAS_SINK=has_sink, HAS_ADD_MASK=has_add_mask,
        HCA_LOCAL_SEQLEN=hca_local_seqlen,
        USE_CAUSAL=use_causal,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_DMODEL=block_dmodel,
        EXACT_TILES_M=exact_tiles_m, EXACT_TILES_N=exact_tiles_n,
        num_warps=int(os.getenv("PRIMUS_V4_ATTN_BWD_DQ_NUM_WARPS", "2")),
        num_stages=int(os.getenv("PRIMUS_V4_ATTN_BWD_DQ_NUM_STAGES", "1")),
    )


def _run_dq_flydsl(
    q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_arg,
    *, scale, swa_window, has_sink,
    Sq, Sk, HQ, HK, B, D,
):
    """V4 SWA bwd dQ via FlyDSL. Writes dq_fp32 (fp32) and adds into
    dsink_fp32 (fp32, atomic) when has_sink. Sink_arg is a fp32 dummy
    buffer when has_sink=False.

    Expects BHLD contiguous: q [B,HQ,Sq,D], k/v [B,HK,Sk,D], dout/lse/d_buf
    same as Triton path. For MQA we ENFORCE HK==1 and pass k/v as-is
    (the FlyDSL kernel uses stride_kh=0 via the mqa_kv codepath, which
    expects K/V to be flat [B, Sk, D]-contiguous regardless of stride).
    """
    mqa_kv = (HK == 1)
    # Sanity: V4 STEP 1 only supports MQA (HK==1). Keep this gate; if HK!=1
    # we fall back to the Triton path upstream.
    assert mqa_kv, f"FlyDSL dq STEP 1b only supports MQA (HK==1); got HK={HK}"
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"

    launch = _get_fly_dq(
        num_heads=HQ, head_dim=D, swa_window=int(swa_window),
        dtype_str=dtype_str, mqa_kv=True, has_sink=has_sink,
    )

    # The FlyDSL kernel reads K/V via flat pointer ops, so they must be
    # contiguous and rank-4 doesn't strictly matter. We pass the rank-4
    # tensors directly; the BHLD path computes ((b*1 + 0)*Sk + n)*D + col
    # for KV when mqa_kv=True. That matches the stride pattern of a
    # [B, 1, Sk, D]-contiguous tensor.
    # NOTE: K and V come in as [B, 1, Sk, D] from the harness MQA leaves
    # (see bwd_modes _build_swa_inputs).
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"
    assert v.is_contiguous(), "v must be contiguous"
    assert dout.is_contiguous(), "dout must be contiguous"
    assert lse.is_contiguous(), "lse must be contiguous"
    assert d_buf.is_contiguous(), "d_buf must be contiguous"
    assert dq_fp32.is_contiguous(), "dq_fp32 must be contiguous"

    launch(
        q,                 # Q [B, HQ, Sq, D]
        k,                 # K [B, 1, Sk, D] (MQA broadcast view)
        v,                 # V [B, 1, Sk, D]
        dout,              # DOS [B, HQ, Sq, D]
        lse,               # LSE [B, HQ, Sq] fp32
        d_buf,             # DELTAS [B, HQ, Sq] fp32
        dq_fp32,           # DQ [B, HQ, Sq, D] fp32 (OUTPUT)
        dsink_fp32,        # DSINK [HQ] fp32 (OUTPUT, atomic)
        sink_arg,          # SINK [HQ] fp32 (INPUT, dummy if !has_sink)
        int(B),
        int(Sq),
        int(Sk),
    )



def _run_dkv_flydsl(
    q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32,
    *, scale, swa_window, has_sink,
    Sq, Sk, HQ, HK, B, D,
):
    """V4 SWA bwd dKdV via FlyDSL. Writes dk_fp32 / dv_fp32 (fp32 buffers).
    MQA only (HK==1). One program per (b, n_block), head-loop accumulator,
    no atomics."""
    mqa_kv = (HK == 1)
    assert mqa_kv, f"FlyDSL dkv STEP 1c only supports MQA (HK==1); got HK={HK}"
    print(
        f"[provenance] FlyDSL dkv launcher invoked, B={B} H={HQ} Sk={Sk} D={D}",
        flush=True,
    )
    dtype_str = "bf16" if q.dtype == __import__("torch").bfloat16 else "f16"
    launch = _get_fly_dkv(
        num_heads=HQ, head_dim=D, swa_window=int(swa_window),
        dtype_str=dtype_str, mqa_kv=True,
    )
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"
    assert v.is_contiguous(), "v must be contiguous"
    assert dout.is_contiguous(), "dout must be contiguous"
    assert lse.is_contiguous(), "lse must be contiguous"
    assert d_buf.is_contiguous(), "d_buf must be contiguous"
    assert dk_fp32.is_contiguous(), "dk_fp32 must be contiguous"
    assert dv_fp32.is_contiguous(), "dv_fp32 must be contiguous"
    launch(
        q,           # Q [B, HQ, Sq, D]
        k,           # K [B, 1, Sk, D]
        v,           # V [B, 1, Sk, D]
        dout,        # DOS [B, HQ, Sq, D]
        lse,         # LSE [B, HQ, Sq] fp32 (RAW domain)
        d_buf,       # DELTAS [B, HQ, Sq] fp32
        dk_fp32,     # DK [B, 1, Sk, D] fp32 (OUTPUT)
        dv_fp32,     # DV [B, 1, Sk, D] fp32 (OUTPUT)
        int(B),
        int(Sq),
        int(Sk),
    )


def _run_dq_pool_flydsl(
    q, k, v, dout, lse, d_buf, dq_fp32, add_mask,
    *, pool_size, hca_local_seqlen,
    Sq, Sk, HQ, HK, B, D,
):
    """V4 HCA bwd dq POOL stream via FlyDSL. ACCUMULATES into dq_fp32.

    Called AFTER ``_run_dq_flydsl`` has written the LOCAL stream dq into
    dq_fp32. Race-free since each program owns a unique
    (b, qhid, m_block) slice and the launch is sequenced after the local
    dq launch.
    """
    mqa_kv = (HK == 1)
    assert mqa_kv, f"FlyDSL dq_pool only supports MQA (HK==1); got HK={HK}"
    assert pool_size <= 64, (
        f"pool_size must fit in one BLOCK_N=64 block; got pool_size={pool_size}"
    )
    assert hca_local_seqlen % 64 == 0, (
        f"hca_local_seqlen must be multiple of BLOCK_N=64; got {hca_local_seqlen}"
    )
    assert (hca_local_seqlen + pool_size) == Sk, (
        f"expected hca_local_seqlen+pool_size == Sk; got {hca_local_seqlen}+{pool_size}!={Sk}"
    )
    assert add_mask is not None and add_mask.shape == (Sq, pool_size), (
        f"add_mask shape mismatch: expected ({Sq},{pool_size}); got {tuple(add_mask.shape) if add_mask is not None else None}"
    )
    assert add_mask.dtype == q.dtype, (
        f"add_mask dtype must match q.dtype; got {add_mask.dtype} vs {q.dtype}"
    )
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    launch = _get_fly_dq_pool(
        num_heads=HQ, head_dim=D,
        pool_size=int(pool_size),
        hca_local_seqlen=int(hca_local_seqlen),
        dtype_str=dtype_str, mqa_kv=True,
    )
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"
    assert v.is_contiguous(), "v must be contiguous"
    assert dout.is_contiguous(), "dout must be contiguous"
    assert lse.is_contiguous(), "lse must be contiguous"
    assert d_buf.is_contiguous(), "d_buf must be contiguous"
    assert dq_fp32.is_contiguous(), "dq_fp32 must be contiguous"
    assert add_mask.is_contiguous(), "add_mask must be contiguous"
    launch(
        q,
        k,
        v,
        dout,
        lse,
        d_buf,
        dq_fp32,
        add_mask,
        int(B),
        int(Sq),
        int(Sk),
    )


def _run_dkv_pool_flydsl(
    q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, add_mask,
    *, pool_size, hca_local_seqlen,
    Sq, Sk, HQ, HK, B, D,
):
    """V4 HCA bwd dKdV POOL stream via FlyDSL. WRITES into the POOL slice of
    dk_fp32/dv_fp32 (which are zero-initialized by the wrapper and have the
    LOCAL slice already populated by ``_run_dkv_flydsl``). Race-free:
    LOCAL writes [..., :hca_local_seqlen, :] and POOL writes
    [..., hca_local_seqlen:hca_local_seqlen+pool_size, :] -- disjoint.
    """
    mqa_kv = (HK == 1)
    assert mqa_kv, f"FlyDSL dkv_pool only supports MQA (HK==1); got HK={HK}"
    assert pool_size <= 32, (
        f"pool_size must fit in one BLOCK_N=32 block; got pool_size={pool_size}"
    )
    assert hca_local_seqlen % 32 == 0, (
        f"hca_local_seqlen must be multiple of BLOCK_N=32; got {hca_local_seqlen}"
    )
    assert (hca_local_seqlen + pool_size) == Sk, (
        f"expected hca_local_seqlen+pool_size == Sk; got "
        f"{hca_local_seqlen}+{pool_size}!={Sk}"
    )
    assert add_mask is not None and add_mask.shape == (Sq, pool_size), (
        f"add_mask shape mismatch: expected ({Sq},{pool_size}); got "
        f"{tuple(add_mask.shape) if add_mask is not None else None}"
    )
    assert add_mask.dtype == q.dtype, (
        f"add_mask dtype must match q.dtype; got {add_mask.dtype} vs {q.dtype}"
    )
    dtype_str = "bf16" if q.dtype == torch.bfloat16 else "f16"
    launch = _get_fly_dkv_pool(
        num_heads=HQ, head_dim=D,
        pool_size=int(pool_size),
        hca_local_seqlen=int(hca_local_seqlen),
        dtype_str=dtype_str, mqa_kv=True,
    )
    assert q.is_contiguous(), "q must be contiguous"
    assert k.is_contiguous(), "k must be contiguous"
    assert v.is_contiguous(), "v must be contiguous"
    assert dout.is_contiguous(), "dout must be contiguous"
    assert lse.is_contiguous(), "lse must be contiguous"
    assert d_buf.is_contiguous(), "d_buf must be contiguous"
    assert dk_fp32.is_contiguous(), "dk_fp32 must be contiguous"
    assert dv_fp32.is_contiguous(), "dv_fp32 must be contiguous"
    assert add_mask.is_contiguous(), "add_mask must be contiguous"
    launch(
        q,
        k,
        v,
        dout,
        lse,
        d_buf,
        dk_fp32,
        dv_fp32,
        add_mask,
        int(B),
        int(Sq),
        int(Sk),
    )


def _run_dkv_triton(
    q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, mask_arg,
    *, scale, swa_window_constexpr, has_add_mask,
    hca_local_seqlen, use_causal,
    block_m, block_n, block_dmodel,
    stride_ms, stride_mn,
    Sq, Sk, HQ, HK, B,
    exact_tiles_m,
):
    dkv_block_n = block_n
    dkv_n_blocks = triton.cdiv(Sk, dkv_block_n)
    num_head_groups = 1
    if HQ > HK:
        # MQA/HQ>=128 (V4-Pro): HG=2 measured 1.47 vs 2.08 ms (1.41x) on gfx950 —
        # mirrors the prod _launch_v4_attention_bwd default. Overridable via env.
        _hg_default = "2" if (HQ >= 64 and HK == 1) else "1"
        target = int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_HEAD_GROUPS", _hg_default))
        while target > 1 and HQ % target != 0:
            target //= 2
        num_head_groups = max(1, target)
    dkv_grid = (dkv_n_blocks, B * HK, num_head_groups)
    _v4_attention_bwd_dkv_kernel[dkv_grid](
        q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, mask_arg,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        d_buf.stride(0), d_buf.stride(1), d_buf.stride(2),
        dk_fp32.stride(0), dk_fp32.stride(1), dk_fp32.stride(2), dk_fp32.stride(3),
        dv_fp32.stride(0), dv_fp32.stride(1), dv_fp32.stride(2), dv_fp32.stride(3),
        stride_ms, stride_mn,
        Sq, Sk, float(scale),
        HEAD_Q=HQ, HEAD_K=HK,
        SWA_WINDOW=swa_window_constexpr,
        HAS_ADD_MASK=has_add_mask,
        HCA_LOCAL_SEQLEN=hca_local_seqlen,
        USE_CAUSAL=use_causal,
        BLOCK_M=block_m, BLOCK_N=dkv_block_n, BLOCK_DMODEL=block_dmodel,
        NUM_HEAD_GROUPS=num_head_groups,
        EXACT_TILES_M=exact_tiles_m,
        EXACT_TILES_N=(Sk % dkv_block_n) == 0,
        num_warps=int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_NUM_WARPS", "2")),
        num_stages=int(os.getenv("PRIMUS_V4_ATTN_BWD_DKV_NUM_STAGES", "1")),
    )





# ---------------------------------------------------------------------------
# HCA pool-only dq accumulator (Triton, inline).
#
# This is a TEMPORARY Triton fallback for the POOL stream of HCA dq while
# the FlyDSL pool kernel is being authored. It is structurally equivalent
# to the pool branch of ``_v4_attention_bwd_dq_kernel`` (Triton ref) but
# written as a standalone kernel so we can call it as a pure accumulator
# (``DQ += pool_contrib``, no local loop). One program per (m_block, b*qhid).
# ---------------------------------------------------------------------------


import triton.language as tl  # noqa: E402


@triton.jit
def _hca_pool_dq_accumulator(
    Q, K, V, DOUT, LSE, D, DQ, ADD_MASK,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_dob, stride_doh, stride_dom, stride_dod,
    stride_lb, stride_lh, stride_lm,
    stride_db, stride_dh, stride_dm,
    stride_dqb, stride_dqh, stride_dqm, stride_dqd,
    stride_ms, stride_mn,
    seqlen_q, seqlen_k, pool_size, sm_scale,
    HEAD_Q: tl.constexpr,
    HEAD_K: tl.constexpr,
    HCA_LOCAL_SEQLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    bid = pid_bh // HEAD_Q
    qhid = pid_bh % HEAD_Q
    if HEAD_K == HEAD_Q:
        khid = qhid
    else:
        khid = 0

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    pool_n = tl.arange(0, BLOCK_N)
    offs_n = HCA_LOCAL_SEQLEN + pool_n
    NEG_INF: tl.constexpr = -1.0e30
    pool_n_mask = pool_n < pool_size

    q_ptrs = Q + bid * stride_qb + qhid * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    dout_ptrs = DOUT + bid * stride_dob + qhid * stride_doh + offs_m[:, None] * stride_dom + offs_d[None, :] * stride_dod
    lse_ptrs = LSE + bid * stride_lb + qhid * stride_lh + offs_m * stride_lm
    dvec_ptrs = D + bid * stride_db + qhid * stride_dh + offs_m * stride_dm

    q_load_mask = offs_m[:, None] < seqlen_q
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    dout = tl.load(dout_ptrs, mask=q_load_mask, other=0.0)
    lse = tl.load(lse_ptrs, mask=offs_m < seqlen_q, other=0.0)
    dvec = tl.load(dvec_ptrs, mask=offs_m < seqlen_q, other=0.0)

    k_ptrs = K + bid * stride_kb + khid * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + bid * stride_vb + khid * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    kv_load_mask = pool_n_mask[:, None]
    k = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
    v = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

    qk = tl.dot(q, tl.trans(k)) * sm_scale
    mask_ptrs = ADD_MASK + offs_m[:, None] * stride_ms + pool_n[None, :] * stride_mn
    mask_load_mask = (offs_m[:, None] < seqlen_q) & pool_n_mask[None, :]
    add_bias = tl.load(mask_ptrs, mask=mask_load_mask, other=0.0).to(tl.float32)
    qk = qk + add_bias
    qk = tl.where(pool_n_mask[None, :], qk, NEG_INF)
    qk = tl.where(offs_m[:, None] < seqlen_q, qk, NEG_INF)

    p = tl.exp(qk - lse[:, None])
    dp = tl.dot(dout, tl.trans(v))
    ds = p * (dp - dvec[:, None])
    dq_contrib = tl.dot(ds.to(k.dtype), k) * sm_scale

    dq_ptrs = DQ + bid * stride_dqb + qhid * stride_dqh + offs_m[:, None] * stride_dqm + offs_d[None, :] * stride_dqd
    dq_prev = tl.load(dq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
    tl.store(dq_ptrs, dq_prev + dq_contrib, mask=offs_m[:, None] < seqlen_q)


def flydsl_v4_attention_bwd(
    q: torch.Tensor,         # [B, HQ, Sq, D]
    k: torch.Tensor,         # [B, HK, Sk, D]  (HK == 1 for MQA, HK == HQ for MHA)
    v: torch.Tensor,         # [B, HK, Sk, D]
    out: torch.Tensor,       # [B, HQ, Sq, D]
    dout: torch.Tensor,      # [B, HQ, Sq, D]
    lse: torch.Tensor,       # [B, HQ, Sq] fp32
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    scale: float,
    hca_local_seqlen: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """V4 SWA backward launcher (STEP 1b scope: SWA-only, no HCA)."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("rank-4 q/k/v required")
    if dout.shape != out.shape or out.shape != q.shape:
        raise ValueError(
            f"shape mismatch: q={tuple(q.shape)} out={tuple(out.shape)} dout={tuple(dout.shape)}"
        )
    B, HQ, Sq, D = q.shape
    HK = k.shape[1]
    Sk = k.shape[2]
    if k.shape != (B, HK, Sk, D) or v.shape != k.shape:
        raise ValueError(f"k/v shape mismatch: k={tuple(k.shape)} v={tuple(v.shape)}")
    if q.dtype != torch.bfloat16:
        raise NotImplementedError(f"bf16 only; got q.dtype={q.dtype}")
    if D % 16 != 0:
        raise NotImplementedError(f"head_dim must be multiple of 16; got D={D}")
    # STEP 2 HCA: accept SWA-only (additive_mask=None, hca_local_seqlen=0)
    # AND HCA (additive_mask is [Sq, P], hca_local_seqlen=Sq, Sk=Sq+P).
    is_hca = (additive_mask is not None) and (int(hca_local_seqlen) > 0)
    if (additive_mask is not None) != (int(hca_local_seqlen) > 0):
        raise NotImplementedError(
            "additive_mask and hca_local_seqlen must both be set (HCA) or both unset (SWA)"
        )
    if is_hca:
        if int(hca_local_seqlen) != Sq:
            raise NotImplementedError(
                f"HCA requires hca_local_seqlen == Sq; got {int(hca_local_seqlen)} vs Sq={Sq}"
            )
        if int(swa_window) <= 0:
            raise NotImplementedError("HCA requires swa_window > 0 for the local stream")
        if Sk <= Sq:
            raise NotImplementedError(f"HCA requires Sk>Sq; got Sk={Sk} Sq={Sq}")
        pool_size = Sk - Sq
        if additive_mask.shape != (Sq, pool_size):
            raise NotImplementedError(
                f"HCA additive_mask must be [Sq={Sq}, P={pool_size}]; got {tuple(additive_mask.shape)}"
            )
        if B != 1:
            raise NotImplementedError(
                f"HCA path currently only supports B=1 (got B={B}); B>1 would mis-stride K/V."
            )
    else:
        if int(swa_window) <= 0:
            raise NotImplementedError("SWA requires swa_window > 0")
        if Sq != Sk:
            raise NotImplementedError("SWA requires Sq == Sk")

    has_sink = sink is not None
    has_add_mask = False
    use_causal = False
    swa_window_constexpr = int(swa_window)

    BLOCK_M = int(os.getenv("PRIMUS_V4_ATTN_BWD_BLOCK_M", "32"))
    BLOCK_N = int(os.getenv("PRIMUS_V4_ATTN_BWD_BLOCK_N", "16"))
    BLOCK_DMODEL = D
    exact_tiles_m = (Sq % BLOCK_M) == 0
    exact_tiles_n = (Sk % BLOCK_N) == 0

    dq_fp32 = torch.zeros((B, HQ, Sq, D), device=q.device, dtype=torch.float32)
    dk_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    dv_fp32 = torch.zeros((B, HK, Sk, D), device=q.device, dtype=torch.float32)
    if has_sink:
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        sink_arg = sink.to(torch.float32) if sink.dtype != torch.float32 else sink
    else:
        # Dummy buffers; FlyDSL kernel still takes them but doesn't touch
        # them when has_sink=False at build time. For Triton path we mirror
        # the original behavior of passing q-shape sentinels.
        dsink_fp32 = torch.zeros((HQ,), device=q.device, dtype=torch.float32)
        sink_arg = torch.zeros((HQ,), device=q.device, dtype=torch.float32)

    use_fly_preprocess = (os.getenv("V4_FLYDSL_BWD_FLY_PREPROCESS", "1") == "1")
    d_buf = _run_preprocess(
        out, dout, use_flydsl=use_fly_preprocess,
        block_m=BLOCK_M, block_dmodel=BLOCK_DMODEL,
    )

    mask_arg = q
    stride_ms = 0
    stride_mn = 0

    use_fly_dq = (os.getenv("V4_FLYDSL_BWD_FLY_DQ", "1") == "1")
    use_fly_dkv = (os.getenv("V4_FLYDSL_BWD_FLY_DKV", "1") == "1")
    verbose = (os.getenv("V4_FLYDSL_BWD_VERBOSE", "0") == "1")

    if verbose:
        pp_tag = "flydsl" if use_fly_preprocess else "triton"
        dq_tag = "flydsl" if use_fly_dq else "triton"
        dkv_tag = "flydsl" if use_fly_dkv else "triton"
        print(
            f"[v4_bwd] preproc={pp_tag} dq={dq_tag} dkv={dkv_tag} "
            f"has_sink={has_sink} Sq={Sq} Sk={Sk} HQ={HQ} HK={HK} D={D} swa={swa_window}",
            flush=True,
        )

    if is_hca:
        # ---- HCA mode (split-mask) ----
        # 1. LOCAL stream (n < HCA_LOCAL_SEQLEN): existing FlyDSL dq + dkv,
        #    with effective seq_len_k = HCA_LOCAL_SEQLEN. Kernel uses
        #    seq_len_k for both n-loop bound and batch base; B=1 makes the
        #    base zero so the reduced seq_len_k cleanly bounds the n-loop
        #    without mis-striding K/V. (B=1-only; gated above.) The LOCAL
        #    pass uses the JOINT lse (saved from fwd with both streams) and
        #    JOINT delta (preprocess used full out, dout).
        # 2. POOL stream (n in [HCA_LOCAL_SEQLEN, Sk)): Triton pool kernels
        #    for this round; FlyDSL pool kernels are next-round work.
        Sk_local = int(hca_local_seqlen)  # == Sq
        pool_size = Sk - Sk_local
        sink_dq_arg = sink_arg  # both fp32 [HQ]
        use_fly_dq_pool = (os.getenv("V4_FLYDSL_BWD_FLY_DQ_POOL", "0") == "1")
        use_fly_dkv_pool = (os.getenv("V4_FLYDSL_BWD_FLY_DKV_POOL", "0") == "1")
        # Provenance: tag dq_pool / dkv_pool only based on knob (silent
        # fallback would violate the strict-gate rule).
        dq_pool_tag = "FlyDSL" if use_fly_dq_pool else "Triton"
        dkv_pool_tag = "FlyDSL" if use_fly_dkv_pool else "Triton"
        print(
            f"[v4_bwd_hca] provenance: dq_local=FlyDSL dkv_local=FlyDSL "
            f"dsink=FlyDSL dq_pool={dq_pool_tag} dkv_pool={dkv_pool_tag} "
            f"B={B} HQ={HQ} HK={HK} Sq={Sq} Sk={Sk} pool={pool_size} D={D} "
            f"swa={swa_window} hca_local_seqlen={Sk_local}",
            flush=True,
        )
        # ---- LOCAL FlyDSL dq (computes dq_local + dsink) ----
        _run_dq_flydsl(
            q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_dq_arg,
            scale=scale, swa_window=swa_window_constexpr, has_sink=has_sink,
            Sq=Sq, Sk=Sk_local, HQ=HQ, HK=HK, B=B, D=D,
        )
        # ---- LOCAL FlyDSL dkv ----
        _run_dkv_flydsl(
            q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32,
            scale=scale, swa_window=swa_window_constexpr, has_sink=has_sink,
            Sq=Sq, Sk=Sk_local, HQ=HQ, HK=HK, B=B, D=D,
        )
        # ---- POOL dq accumulator: FlyDSL (knob ON) or Triton (default) ----
        if use_fly_dq_pool:
            _run_dq_pool_flydsl(
                q, k, v, dout, lse, d_buf, dq_fp32, additive_mask,
                pool_size=int(pool_size), hca_local_seqlen=Sk_local,
                Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B, D=D,
            )
        else:
            pool_block_n_dq = max(16, triton.next_power_of_2(pool_size))
            _hca_pool_dq_accumulator[(triton.cdiv(Sq, BLOCK_M), B * HQ)](
                q, k, v, dout, lse, d_buf, dq_fp32, additive_mask,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                d_buf.stride(0), d_buf.stride(1), d_buf.stride(2),
                dq_fp32.stride(0), dq_fp32.stride(1), dq_fp32.stride(2), dq_fp32.stride(3),
                additive_mask.stride(0), additive_mask.stride(1),
                Sq, Sk, pool_size, float(scale),
                HEAD_Q=HQ, HEAD_K=HK,
                HCA_LOCAL_SEQLEN=Sk_local,
                BLOCK_M=BLOCK_M, BLOCK_N=pool_block_n_dq,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=2, num_stages=1,
            )
        # ---- POOL dkv: FlyDSL (knob ON) or Triton (default) ----
        if use_fly_dkv_pool:
            _run_dkv_pool_flydsl(
                q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, additive_mask,
                pool_size=int(pool_size), hca_local_seqlen=Sk_local,
                Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B, D=D,
            )
        else:
            pool_block_n_dkv = max(16, triton.next_power_of_2(pool_size))
            pool_grid_m = triton.cdiv(Sq, BLOCK_M)
            _v4_attention_bwd_dkv_pool_kernel[(pool_grid_m, B)](
                q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, additive_mask,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
                lse.stride(0), lse.stride(1), lse.stride(2),
                d_buf.stride(0), d_buf.stride(1), d_buf.stride(2),
                dk_fp32.stride(0), dk_fp32.stride(1), dk_fp32.stride(2), dk_fp32.stride(3),
                dv_fp32.stride(0), dv_fp32.stride(1), dv_fp32.stride(2), dv_fp32.stride(3),
                additive_mask.stride(0), additive_mask.stride(1),
                Sq, Sk, pool_size, float(scale),
                HEAD_Q=HQ, HEAD_K=HK,
                HCA_LOCAL_SEQLEN=Sk_local,
                BLOCK_M=BLOCK_M, BLOCK_N=pool_block_n_dkv,
                BLOCK_DMODEL=BLOCK_DMODEL,
                num_warps=2, num_stages=1,
            )
    elif use_fly_dq:
        # SWA-only path (STEP 1b unchanged).
        sink_dq_arg = sink_arg  # both are fp32 [HQ]
        _run_dq_flydsl(
            q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_dq_arg,
            scale=scale, swa_window=swa_window_constexpr, has_sink=has_sink,
            Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B, D=D,
        )
    else:
        _run_dq_triton(
            q, k, v, dout, lse, d_buf, dq_fp32, dsink_fp32, sink_arg, mask_arg,
            scale=scale, swa_window_constexpr=swa_window_constexpr,
            has_sink=has_sink, has_add_mask=has_add_mask,
            hca_local_seqlen=0, use_causal=use_causal,
            block_m=BLOCK_M, block_n=BLOCK_N, block_dmodel=BLOCK_DMODEL,
            stride_ms=stride_ms, stride_mn=stride_mn,
            Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B,
            exact_tiles_m=exact_tiles_m, exact_tiles_n=exact_tiles_n,
        )

    if not is_hca:
        if use_fly_dkv:
            _run_dkv_flydsl(
                q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32,
                scale=scale, swa_window=swa_window_constexpr, has_sink=has_sink,
                Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B, D=D,
            )
        else:
            _run_dkv_triton(
                q, k, v, dout, lse, d_buf, dk_fp32, dv_fp32, mask_arg,
                scale=scale, swa_window_constexpr=swa_window_constexpr,
                has_add_mask=has_add_mask, hca_local_seqlen=0, use_causal=use_causal,
                block_m=BLOCK_M, block_n=BLOCK_N, block_dmodel=BLOCK_DMODEL,
                stride_ms=stride_ms, stride_mn=stride_mn,
                Sq=Sq, Sk=Sk, HQ=HQ, HK=HK, B=B,
                exact_tiles_m=exact_tiles_m,
            )

    dq_out = dq_fp32.to(q.dtype)
    dk_out = dk_fp32.to(k.dtype)
    dv_out = dv_fp32.to(v.dtype)
    dsink_out = dsink_fp32.to(sink.dtype) if has_sink else None
    return dq_out, dk_out, dv_out, dsink_out


__all__ = ["flydsl_v4_attention_bwd"]
