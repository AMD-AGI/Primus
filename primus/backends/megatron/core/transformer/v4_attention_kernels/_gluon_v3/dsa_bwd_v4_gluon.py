###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gluon DeepSeek-V4 sparse-MLA backward for the "gluon_v2" backend.

Companion to the gluon_v2 forward (:func:`sparse_mla_fwd_v4_gluon_v2`). Wires the Gluon
dQ + Gluon dKV-intermediate compute kernels with a Triton Delta preprocess and the
backend-neutral CSR inverted-topk gather + torch d_sink reduction (non-atomic
chunked-gather scheme).

The dQ / dKV-intermediate Gluon kernels apply the forward campaign's accepted techniques
to the backward: rope-skip (the V4 zero-rope-pad makes the rope gradients provably zero)
+ MFMA K=32 for the D_V=512-reduction matmuls, plus a single-chunk dQ read-modify-write
for high head counts. Beats the plain-Triton backward ~1.12x geomean over the 6
flash/pro x cr{0,4,128} shapes (eager-UT 9/9).
"""

import torch
import triton

from .._gluon_dsa._dsa_bwd_gather import _build_inverted_topk_slice, _bwd_dkv_gather_acc
from .._gluon_dsa._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess
from .dsa_bwd_dkv_interm_gluon import _sparse_mla_bwd_dkv_interm_gl_kernel
from .dsa_bwd_dq_gluon import _sparse_mla_bwd_dq_gl_kernel


def sparse_mla_bwd_v4_gluon_v2(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward (Gluon dQ/dKV). Returns ``(dq, dkv, d_sink)``."""
    assert q.is_contiguous() and kv.is_contiguous() and o.is_contiguous()
    assert do.is_contiguous() and topk_indices.is_contiguous() and lse.is_contiguous()

    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    num_kv = kv.shape[0]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # ---- preprocess: Delta = rowsum(O*dO) (Triton, unchanged) ----
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    BLOCK_H_PRE = triton.next_power_of_2(min(64, num_heads))
    _sparse_mla_bwd_preprocess[(total_tokens, triton.cdiv(num_heads, BLOCK_H_PRE))](
        O_ptr=o,
        dO_ptr=do,
        Delta_ptr=delta,
        stride_o_t=o.stride(0),
        stride_o_h=o.stride(1),
        num_heads=num_heads,
        D_V=kv_lora_rank,
        BLOCK_H=BLOCK_H_PRE,
    )

    # ---- config ----
    # R2: dQ is read-modify-written across chunks, so more chunks = more redundant dq
    # reload passes + repeated CSR builds. H=64 CSA has TOPK=640, so the old 256 cap
    # split it into 3 chunks; a 320 cap tests a two-chunk schedule without changing
    # high-head whole-topk behavior.
    if num_heads >= 128:
        R_CHUNK = min(topk, 1536)
    elif num_heads >= 64:
        R_CHUNK = min(topk, 320)
    else:
        R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 16
    BH_DKV, TK_DKV = 32, 64
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # ---- pad topk to R_CHUNK multiple ----
    topk_padded_len = ((topk + R_CHUNK - 1) // R_CHUNK) * R_CHUNK
    if topk_padded_len != topk:
        pad = torch.full((total_tokens, topk_padded_len - topk), -1, dtype=torch.int32, device=q.device)
        topk_padded = torch.cat([topk_indices, pad], dim=1).contiguous()
    else:
        topk_padded = topk_indices

    all_csr = [
        _build_inverted_topk_slice(topk_padded[:, rs : rs + R_CHUNK], rs, R_CHUNK, num_kv=num_kv)
        for rs in range(0, topk, R_CHUNK)
    ]

    for chunk_idx, r_start in enumerate(range(0, topk, R_CHUNK)):
        is_first = r_start == 0

        _sparse_mla_bwd_dq_gl_kernel[(total_tokens, num_hg_dq)](
            q,
            kv,
            do,
            topk_padded,
            lse,
            delta,
            dq,
            chunk_dS,
            chunk_P,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            do.stride(0),
            do.stride(1),
            dq.stride(0),
            dq.stride(1),
            topk_padded.stride(0),
            chunk_dS.stride(0),
            chunk_dS.stride(1),
            scale,
            num_heads,
            r_start,
            R_CHUNK=R_CHUNK,
            BLOCK_H=BH_DQ,
            TILE_K=TK_DQ,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            HAS_ROPE=False,  # V4 zero-rope-pad: dQ_rope is provably zero + discarded by the adapter
            IS_FIRST_CHUNK=is_first,
            num_warps=4,
            waves_per_eu=1,
        )

        _sparse_mla_bwd_dkv_interm_gl_kernel[(total_tokens,)](
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
            R_CHUNK=R_CHUNK,
            TILE_K=TK_DKV,
            BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            HAS_ROPE=False,  # V4 zero-rope-pad: dKV_rope provably zero + discarded downstream
            num_warps=4,
        )

        inv_ptr, inv_data = all_csr[chunk_idx]
        _bwd_dkv_gather_acc[(num_kv,)](
            interm,
            inv_ptr,
            inv_data,
            dkv_acc,
            interm.stride(1),
            dkv_acc.stride(0),
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            num_warps=4,
        )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv_out = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv_out, d_sink
