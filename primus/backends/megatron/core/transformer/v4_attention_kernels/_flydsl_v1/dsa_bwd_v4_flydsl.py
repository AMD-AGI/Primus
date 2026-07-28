###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""FlyDSL-v1 sparse-MLA backward — native FlyDSL dQ + shared dKV gather.

``sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk, lse, ...) -> (dq, dkv, d_sink)``.

The **dQ** kernel is native FlyDSL MFMA (``dsa_bwd_dq_flydsl_kernel``): it
recomputes S = Q·Kᵀ, P = exp(scale·S − lse), dP = dO·Kᵀ, dS = P·(dP − Δ)·scale,
accumulates dQ = dS·K, and writes the per-token dS/P buffers. The dKV path
(intermediate GEMM + CSR inverted-topk scatter-reduce) reuses the shared,
proven Triton kernels — the dKV gather is a variable-length scatter-reduction
with no MFMA content, so there is nothing to gain from a FlyDSL rewrite there.

This runs the whole top-k as a single chunk (R_CHUNK = TOPK), so dQ needs no
cross-chunk read-modify-write and the dKV intermediate is built in one pass.

Depends only on the installed ``flydsl`` pip package (no /workspace source).
"""

from __future__ import annotations

import os
import sys
import threading

import torch
import triton

from .._gluon_dsa._dsa_bwd_gather import _build_inverted_topk_slice
from .._triton_v2.dsa_bwd_kernels import (
    _bwd_compute_dkv_intermediate,
    _bwd_dkv_gather_acc,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dsa_bwd_dq_flydsl_kernel import build_dsa_bwd_dq_module  # noqa: E402

_DQ_CACHE = {}
_DQ_LOCK = threading.Lock()
_BLOCK_N = int(os.environ.get("PRIMUS_DSA_FLYDSL_BWD_BLOCK_N", "64"))
_BLOCK_H = int(os.environ.get("PRIMUS_DSA_FLYDSL_BWD_BLOCK_H", "64"))


def _get_dq_kernel(num_heads, kv_lora_rank, d_qk, topk, single_latent, scale):
    block_h = min(_BLOCK_H, num_heads)
    while num_heads % block_h != 0:
        block_h -= 32
    key = (num_heads, kv_lora_rank, d_qk, topk, block_h, single_latent, round(float(scale), 8))
    with _DQ_LOCK:
        launch = _DQ_CACHE.get(key)
        if launch is None:
            launch = build_dsa_bwd_dq_module(
                num_heads=num_heads,
                kv_lora_rank=kv_lora_rank,
                d_qk=d_qk,
                topk=topk,
                dtype_str="bf16",
                sm_scale=float(scale),
                block_n=_BLOCK_N,
                block_h=block_h,
                single_latent=single_latent,
            )
            _DQ_CACHE[key] = launch
        return launch


def sparse_mla_bwd_v4_flydsl(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward: native FlyDSL dQ + Triton dKV gather."""
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    D = int(kv_lora_rank)
    rope_rank = d_qk - D
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]
    assert q.dtype == torch.bfloat16

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # pad topk to a BLOCK_N multiple (-1 = invalid); usually already padded upstream
    topk = topk_indices.shape[1]
    if topk % _BLOCK_N != 0:
        pad = ((topk + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N - topk
        topk_p = torch.cat(
            [topk_indices, torch.full((total_tokens, pad), -1, dtype=torch.int32, device=q.device)], dim=1
        ).contiguous()
    else:
        topk_p = topk_indices
    TOPK = topk_p.shape[1]

    # Delta = rowsum(O * dO)  (o is [T,H,D])
    delta = (o[:, :, :D].float() * do.float()).sum(-1).contiguous()
    lse_c = lse.contiguous()

    # ---- native FlyDSL dQ (whole top-k) + dS/P buffers ----
    dq = torch.zeros(total_tokens, num_heads, d_qk, dtype=q.dtype, device=q.device)  # rope cols stay 0
    chunk_dS = torch.empty(total_tokens, num_heads, TOPK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, TOPK, dtype=torch.bfloat16, device=q.device)
    single_latent = num_heads <= 64
    dq_launch = _get_dq_kernel(int(num_heads), D, int(d_qk), int(TOPK), single_latent, scale)
    dq_launch(
        q.reshape(-1),
        kv.reshape(-1),
        do.reshape(-1),
        topk_p.reshape(-1),
        lse_c.reshape(-1),
        delta.reshape(-1),
        dq.reshape(-1),
        chunk_dS.reshape(-1),
        chunk_P.reshape(-1),
        int(total_tokens),
    )

    # ---- dKV: intermediate GEMM + CSR inverted-topk scatter-reduce (Triton) ----
    # BH_DKV=32/TK_DKV=64 whole-top-k (R_CHUNK=TOPK) in one dKV-intermediate pass.
    # This config's LDS stays < 160 KB in both the standalone and training Triton
    # contexts (the wider 64/128 config overflows under the training runtime).
    HAS_ROPE = False
    BH_DKV, TK_DKV = 32, 64
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    interm = torch.empty(total_tokens, TOPK, d_qk, dtype=torch.bfloat16, device=q.device)
    _bwd_compute_dkv_intermediate[(total_tokens,)](
        q,
        do,
        chunk_dS,
        chunk_P,
        interm,
        q.stride(0),
        q.stride(1),
        do.stride(0),
        do.stride(1),
        chunk_dS.stride(0),
        chunk_dS.stride(1),
        interm.stride(0),
        interm.stride(1),
        num_heads,
        R_CHUNK=TOPK,
        TILE_K=TK_DKV,
        BLOCK_H=BH_DKV,
        NUM_HG=num_hg_dkv,
        D_V=D,
        D_ROPE=rope_rank,
        HAS_ROPE=HAS_ROPE,
        num_warps=4,
    )

    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    inv_ptr, inv_data = _build_inverted_topk_slice(topk_p, 0, TOPK, num_kv=num_kv)
    _bwd_dkv_gather_acc[(num_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv_acc,
        interm.stride(1),
        dkv_acc.stride(0),
        D_V=D,
        D_ROPE=rope_rank,
        HAS_ROPE=HAS_ROPE,
        num_warps=4,
    )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink


__all__ = ["sparse_mla_bwd_v4_flydsl"]
