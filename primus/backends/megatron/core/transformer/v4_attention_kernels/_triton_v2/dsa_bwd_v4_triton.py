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

# CSR-inverted-topk builder + Delta preprocess are backend-neutral torch/host
# helpers; reuse them. The compute kernels are owned locally (dsa_bwd_kernels)
# so this backend can be tuned independently of the gluon path.
from .._gluon_dsa._dsa_bwd_gather import _build_inverted_topk_slice
from .dsa_bwd_kernels import (
    _bwd_chunk_dq_store_ds,
    _bwd_compute_dkv_intermediate,
    _bwd_dkv_gather_acc,
)


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

    # Delta = rowsum(O*dO) is fused into the first dQ chunk (no preprocess kernel).
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)

    # ---- config (mirror the gluon bwd chunking) ----
    # R_CHUNK (rank-chunk width): dQ is read-modify-written across chunks, so more
    # chunks = more redundant dq reload passes + repeated launches/CSR builds. For
    # high head counts (H>=128) the dq RMW volume is large, so a single chunk over
    # the whole topk (bounded for memory) is a big win (-22% on pro cr4). For low
    # head counts (H<=64) the smaller dq RMW is outweighed by the larger per-chunk
    # buffers/occupancy, and 256 stays best — so keep the cap there.
    if num_heads >= 128:
        R_CHUNK = min(topk, 1536)
    else:
        R_CHUNK = min(256, topk)
    BH_DQ, TK_DQ = 64, 16
    # dKV tiling: the default (BH_DKV=32, TK_DKV=64) is best for high head counts
    # (H>=128) and for chunk widths that are not 128-aligned. For low head counts
    # (H<=64) with a 128-aligned chunk, a wider TILE_K=128 over a single head-group
    # (BH_DKV=64, NUM_HG=1) reduces redundant Q/dO re-loads and issues fuller MMAs,
    # which is faster there; it regresses for H>=128 (register pressure) and for
    # non-128-aligned chunks (partial tiles), so it is guarded on both.
    if num_heads <= 64 and R_CHUNK % 128 == 0:
        BH_DKV, TK_DKV = 64, 128
    else:
        BH_DKV, TK_DKV = 32, 64
    num_hg_dq = triton.cdiv(num_heads, BH_DQ)
    num_hg_dkv = triton.cdiv(num_heads, BH_DKV)

    # In the V4 single-latent form the D_ROPE block of q/kv is a zero pad (RoPE is
    # baked in-place over the 512 latent) and its gradient is discarded by the
    # adapter (dq[..., :D_V], dkv[..., :D_V]). So all rope MMAs / K_rope loads /
    # interm-rope traffic compute a provably-zero result that is thrown away.
    # Skip them: bit-identical non-rope outputs, zero rope outputs (== gluon).
    HAS_ROPE = False

    dq = torch.empty_like(q)
    chunk_dS = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    chunk_P = torch.empty(total_tokens, num_heads, R_CHUNK, dtype=torch.bfloat16, device=q.device)
    dkv_acc = torch.zeros(num_kv, d_qk, dtype=torch.float32, device=q.device)
    interm = torch.empty(total_tokens, R_CHUNK, d_qk, dtype=torch.bfloat16, device=q.device)

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

        _bwd_chunk_dq_store_ds[(total_tokens, num_hg_dq)](
            q,
            kv,
            do,
            topk_padded,
            lse,
            delta,
            o,
            dq,
            chunk_dS,
            chunk_P,
            q.stride(0),
            q.stride(1),
            kv.stride(0),
            do.stride(0),
            do.stride(1),
            o.stride(0),
            o.stride(1),
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
            HAS_ROPE=HAS_ROPE,
            IS_FIRST_CHUNK=is_first,
            num_warps=4,
            waves_per_eu=1,
        )

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
            R_CHUNK=R_CHUNK,
            TILE_K=TK_DKV,
            BLOCK_H=BH_DKV,
            NUM_HG=num_hg_dkv,
            D_V=kv_lora_rank,
            D_ROPE=rope_rank,
            HAS_ROPE=HAS_ROPE,
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
            HAS_ROPE=HAS_ROPE,
            num_warps=4,
        )

    d_sink = None
    if has_sink:
        d_sink = -(torch.exp(attn_sink.unsqueeze(0) - lse) * delta).sum(0)

    dkv = dkv_acc.to(kv.dtype).unsqueeze(1)
    return dq, dkv, d_sink
