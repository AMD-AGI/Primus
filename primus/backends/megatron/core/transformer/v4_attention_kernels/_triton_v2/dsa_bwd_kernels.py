###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Owned plain-Triton backward compute kernels for the "triton v2" sparse-MLA
backend. Forked from the shared ``_gluon_dsa/_dsa_bwd_gather.py`` plain-Triton
kernels so this backend can be tuned independently of the gluon path.

Three kernels implement the non-atomic chunked-gather backward:
  * ``_bwd_chunk_dq_store_ds`` — dQ accumulation for one rank chunk, ALSO stores
    per-tile dS and P to [T, H, R_CHUNK] buffers for reuse by the dKV kernel.
  * ``_bwd_compute_dkv_intermediate`` — dKV intermediate for one chunk, REUSES
    the stored dS/P (no S/P/dS recompute). Consumes q/do transposed to [T, D, H].
  * ``_bwd_dkv_gather_acc`` — CSR inverted-topk reduce of the intermediate into
    the fp32 dKV accumulator.
"""

import triton
import triton.language as tl


@triton.jit
def _bwd_chunk_dq_store_ds(
    Q_ptr,  # [T, H, D] bf16
    KV_ptr,  # [T, 1, D] bf16
    dO_ptr,  # [T, H, D_V] bf16
    TopK_ptr,  # [T, TOPK] int32
    LSE_ptr,  # [T, H] fp32
    Delta_ptr,  # [T, H] fp32 (computed here on the first chunk, read after)
    O_ptr,  # [T, H, D_V] bf16 (fwd output, for Delta = rowsum(O*dO))
    dQ_ptr,  # [T, H, D] bf16 — read-modify-write across chunks
    dS_ptr,  # [T, H, R_CHUNK] bf16 — output chunk dS
    P_ptr,  # [T, H, R_CHUNK] bf16 — output chunk P
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_o_t: tl.int64,
    stride_o_h: tl.int64,
    stride_dq_t: tl.int64,
    stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    scale: tl.float32,
    num_heads: tl.int32,
    R_START: tl.int32,
    R_CHUNK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_K: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    IS_FIRST_CHUNK: tl.constexpr,
):
    """dQ accumulation for rank chunk [R_START, R_START+R_CHUNK), plus stores
    chunk dS and P to [T, H, R_CHUNK] buffers for _bwd_compute_dkv_intermediate.
    Grid: (total_tokens, num_hg).

    Delta = rowsum(O*dO) is fused here (computed + stored on the first chunk,
    reloaded on later chunks) instead of a separate preprocess kernel — dO is
    already resident, so this only adds the O load and drops a whole kernel + a
    duplicate dO read.

    HAS_ROPE=False (V4 zero-pad): skips all rope MMAs/loads and writes the dQ rope
    columns as zero (they are discarded downstream but kept well-defined)."""
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(
        Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :], mask=mask_h[:, None], other=0.0
    )
    if HAS_ROPE:
        Q_rope = tl.load(
            Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
            mask=mask_h[:, None],
            other=0.0,
        )
    do_base = token_idx * stride_do_t
    dO_val = tl.load(
        dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :], mask=mask_h[:, None], other=0.0
    )
    lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    if IS_FIRST_CHUNK:
        # Delta = rowsum(O * dO): fold the preprocess in (dO already loaded).
        O_val = tl.load(
            O_ptr + token_idx * stride_o_t + offs_h[:, None] * stride_o_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        )
        delta = tl.sum(O_val.to(tl.float32) * dO_val.to(tl.float32), axis=1)
        tl.store(Delta_ptr + token_idx * num_heads + offs_h, delta, mask=mask_h)
    else:
        delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dq_base = token_idx * stride_dq_t
    if IS_FIRST_CHUNK:
        dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    else:
        dQ_lora = tl.load(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
            mask=mask_h[:, None],
            other=0.0,
        ).to(tl.float32)
    if HAS_ROPE:
        if IS_FIRST_CHUNK:
            dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
        else:
            dQ_rope = tl.load(
                dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
                mask=mask_h[:, None],
                other=0.0,
            ).to(tl.float32)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t + R_START
    offs_tile = tl.arange(0, TILE_K)
    ds_base = token_idx * stride_ds_t + hg_idx * BLOCK_H * stride_ds_h

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs, mask=valid, other=-1)
        valid = valid & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(
            KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None], mask=valid[None, :], other=0.0
        )

        S = tl.dot(Q_lora, K_lora_T)
        if HAS_ROPE:
            K_rope_T = tl.load(
                KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                mask=valid[None, :],
                other=0.0,
            )
            S += tl.dot(Q_rope, K_rope_T)
        S = tl.where(valid[None, :] & mask_h[:, None], S * scale, float("-inf"))
        P = tl.exp(S - lse[:, None])
        P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)
        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        dS_bf = dS.to(tl.bfloat16)
        dQ_lora += tl.dot(dS_bf, tl.trans(K_lora_T)).to(tl.float32)
        if HAS_ROPE:
            dQ_rope += tl.dot(dS_bf, tl.trans(K_rope_T)).to(tl.float32)

        local_h = tl.arange(0, BLOCK_H)
        tl.store(
            dS_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            dS_bf,
            mask=mask_h[:, None] & valid[None, :],
        )
        tl.store(
            P_ptr + ds_base + local_h[:, None] * stride_ds_h + tile_offs[None, :],
            P.to(tl.bfloat16),
            mask=mask_h[:, None] & valid[None, :],
        )

    tl.store(
        dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
        dQ_lora.to(Q_lora.dtype),
        mask=mask_h[:, None],
    )
    if HAS_ROPE:
        tl.store(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
            dQ_rope.to(Q_lora.dtype),
            mask=mask_h[:, None],
        )
    else:
        tl.store(
            dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
            tl.zeros([BLOCK_H, D_ROPE], dtype=Q_lora.dtype),
            mask=mask_h[:, None],
        )


@triton.jit
def _bwd_compute_dkv_intermediate(
    Q_ptr,  # [T, H, D_QK] bf16  (UNtransposed; loaded transposed via strided index)
    dO_ptr,  # [T, H, D_V]  bf16 (UNtransposed)
    dS_ptr,  # [T, H, R_CHUNK] bf16
    P_ptr,  # [T, H, R_CHUNK] bf16
    Interm_ptr,  # [T, R_CHUNK, D_QK] bf16 — output, one writer per (q, rank)
    stride_q_t: tl.int64,
    stride_q_h: tl.int64,
    stride_do_t: tl.int64,
    stride_do_h: tl.int64,
    stride_ds_t: tl.int64,
    stride_ds_h: tl.int64,
    stride_interm_t: tl.int64,  # R_CHUNK * D_QK
    stride_interm_k: tl.int64,  # D_QK
    num_heads: tl.int32,
    R_CHUNK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """dKV intermediate for one chunk, REUSING stored dS/P (no recompute).
    Loads Q/dO UNtransposed with a [D, BLOCK_H] strided index pattern (contiguous
    along D per head), removing the external q.transpose(1,2).contiguous() /
    do.transpose(...).contiguous() copies. Grid: (total_tokens,).

    HAS_ROPE=False (V4 zero-pad): skips Q_rope load, the dKV_rope MMA and the
    interm rope store (interm rope columns are never read by the gather)."""
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (R_CHUNK + TILE_K - 1) // TILE_K
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    interm_base_t = token_idx * stride_interm_t
    q_base = token_idx * stride_q_t
    do_base = token_idx * stride_do_t
    ds_base = token_idx * stride_ds_t

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        valid = tile_offs < R_CHUNK

        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        if HAS_ROPE:
            dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            # [D_V, BLOCK_H] loaded transposed: rows=offs_v (stride 1, contiguous),
            # cols=offs_h (stride stride_q_h). No HBM transpose copy needed.
            Q_lora_T = tl.load(
                Q_ptr + q_base + offs_h[None, :] * stride_q_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )
            dO_T = tl.load(
                dO_ptr + do_base + offs_h[None, :] * stride_do_h + offs_v[:, None],
                mask=mask_h[None, :],
                other=0.0,
            )

            dS_val = tl.load(
                dS_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )
            P_val = tl.load(
                P_ptr + ds_base + offs_h[:, None] * stride_ds_h + tile_offs[None, :],
                mask=mask_h[:, None] & valid[None, :],
                other=0.0,
            )

            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)
            dKV_lora += tl.dot(dO_T, P_val.to(dO_T.dtype)).to(tl.float32)
            if HAS_ROPE:
                Q_rope_T = tl.load(
                    Q_ptr + q_base + offs_h[None, :] * stride_q_h + (D_V + offs_r[:, None]),
                    mask=mask_h[None, :],
                    other=0.0,
                )
                dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        interm_lora_ptrs = Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + offs_v[:, None]
        tl.store(interm_lora_ptrs, dKV_lora.to(tl.bfloat16), mask=valid[None, :])

        if HAS_ROPE:
            interm_rope_ptrs = (
                Interm_ptr + interm_base_t + tile_offs[None, :] * stride_interm_k + D_V + offs_r[:, None]
            )
            tl.store(interm_rope_ptrs, dKV_rope.to(tl.bfloat16), mask=valid[None, :])


@triton.jit
def _bwd_dkv_gather_acc(
    Interm_ptr,  # [T, R_CHUNK, D] bf16 — chunk intermediate
    InvPtr_ptr,  # [T+1] int32 — CSR row pointers
    InvData_ptr,  # [T*R_CHUNK] int32 — encoded as q*R_CHUNK + local_r
    dKV_acc_ptr,  # [T, D] fp32 — accumulator (read-modify-write across chunks)
    stride_interm_r: tl.int64,  # D
    stride_acc_t: tl.int64,  # D
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
):
    """Gather one chunk's bf16 intermediate into the fp32 dKV accumulator.
    Grid: (num_kv,) — one CTA per KV token, no atomics. Tiled BLOCK_K rows/iter.

    HAS_ROPE=False (V4 zero-pad): interm rope columns are never written, so skip
    reading/accumulating them; dKV_acc rope stays at its zero-init value."""
    k = tl.program_id(0)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)
    offs_k = tl.arange(0, BLOCK_K)

    start = tl.load(InvPtr_ptr + k)
    end = tl.load(InvPtr_ptr + k + 1)

    acc_base = k.to(tl.int64) * stride_acc_t
    dkv_acc_lora = tl.load(dKV_acc_ptr + acc_base + offs_v).to(tl.float32)
    if HAS_ROPE:
        dkv_acc_rope = tl.load(dKV_acc_ptr + acc_base + D_V + offs_r).to(tl.float32)

    n_entries = end - start
    for ti in range(0, tl.cdiv(n_entries, BLOCK_K)):
        e_local = ti * BLOCK_K + offs_k
        valid = e_local < n_entries
        entry = tl.load(InvData_ptr + start + e_local, mask=valid, other=0).to(tl.int64)
        rp = entry[:, None] * stride_interm_r
        rows_v = tl.load(Interm_ptr + rp + offs_v[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        dkv_acc_lora += tl.sum(rows_v, axis=0)
        if HAS_ROPE:
            rows_r = tl.load(Interm_ptr + rp + D_V + offs_r[None, :], mask=valid[:, None], other=0.0).to(
                tl.float32
            )
            dkv_acc_rope += tl.sum(rows_r, axis=0)

    tl.store(dKV_acc_ptr + acc_base + offs_v, dkv_acc_lora)
    if HAS_ROPE:
        tl.store(dKV_acc_ptr + acc_base + D_V + offs_r, dkv_acc_rope)
