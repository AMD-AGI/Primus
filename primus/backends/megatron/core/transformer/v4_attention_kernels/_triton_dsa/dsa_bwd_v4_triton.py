###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plain-Triton DeepSeek-V4 sparse-MLA backward (the "triton v2" backend).

Companion to :func:`sparse_mla_fwd_v4_triton`; API mirrors
:func:`sparse_mla_bwd_v4_gluon` -> ``(dq, dkv, d_sink)``.

Uses the **non-atomic chunked-gather** scheme (the same one the gluon backward
shares for its dKV gather): per rank-chunk it runs a pure-Triton dQ kernel and a
dKV-intermediate kernel (both ``tl.dot`` / MFMA, plain stores — no atomics),
then reduces the intermediate into ``dkv`` via a CSR inverted-topk gather. This
is the fully-Triton analogue of the gluon dQ/dKV-interm kernels, so its dKV is
not bottlenecked by global atomics. ``d_sink`` is the closed-form torch
reduction ``-sum_t exp(sink - lse) * delta``.
"""

import torch
import triton

# These bwd-gather kernels are plain @triton.jit (not gluon) and backend-neutral;
# reuse them rather than duplicate ~600 lines.
from .._gluon_dsa._dsa_bwd_gather import (
    _build_inverted_topk_slice,
    _bwd_chunk_dkv_interm,
    _bwd_chunk_dq,
    _bwd_dkv_gather_acc,
)
from .._gluon_dsa._dsa_bwd_preprocess import _sparse_mla_bwd_preprocess


def sparse_mla_bwd_v4_triton(q, kv, o, do, topk_indices, lse, attn_sink=None, kv_lora_rank=512, scale=None):
    """DeepSeek-V4 sparse-MLA backward (plain Triton / MFMA, non-atomic).

    Returns ``(dq, dkv, d_sink)`` with ``dkv`` shaped like ``kv`` and ``d_sink``
    ``[num_heads]`` fp32 (or ``None`` when ``attn_sink`` is None).
    """
    assert q.is_contiguous() and o.is_contiguous() and do.is_contiguous()
    assert topk_indices.is_contiguous() and lse.is_contiguous()
    total_tokens, num_heads, d_qk = q.shape
    rope_rank = d_qk - kv_lora_rank
    topk = topk_indices.shape[1]
    if scale is None:
        scale = 1.0 / (d_qk**0.5)
    if kv.dim() == 2:
        kv = kv.unsqueeze(1)
    assert kv.is_contiguous()
    num_kv = kv.shape[0]

    has_sink = attn_sink is not None
    if has_sink:
        assert attn_sink.dtype == torch.float32 and attn_sink.shape == (num_heads,)

    # ---- preprocess: Delta = rowsum(O*dO) ----
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

    # ---- config (mirror the gluon bwd chunking) ----
    R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 16
    BH_DKV, TK_DKV = 32, 64
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    dq = torch.empty_like(q)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

    # dKV-interm consumes q/do transposed to [T, D, H].
    q_t = q.transpose(1, 2).contiguous()
    do_t = do[..., :kv_lora_rank].transpose(1, 2).contiguous()

    # pad topk to an R_CHUNK multiple (-1 = invalid).
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

        _bwd_chunk_dq[(total_tokens, num_hg_dq)](
            q,
            kv,
            do,
            topk_padded,
            lse,
            delta,
            dq,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            do.stride(0),
            do.stride(1),
            dq.stride(0),
            dq.stride(1),
            topk_padded.stride(0),
            scale,
            num_heads,
            r_start,
            R_CHUNK=R_CHUNK,
            BLOCK_H=BH_DQ,
            TILE_K=TK_DQ,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            IS_FIRST_CHUNK=is_first,
            num_warps=4,
            waves_per_eu=1,
        )

        _bwd_chunk_dkv_interm[(total_tokens,)](
            q_t,
            do_t,
            topk_padded,
            lse,
            delta,
            kv,
            interm,
            q_t.stride(0),
            do_t.stride(0),
            topk_padded.stride(0),
            kv.stride(0),
            interm.stride(0),
            interm.stride(1),
            scale,
            num_heads,
            r_start,
            R_CHUNK=R_CHUNK,
            TILE_K=TK_DKV,
            BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
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

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink
