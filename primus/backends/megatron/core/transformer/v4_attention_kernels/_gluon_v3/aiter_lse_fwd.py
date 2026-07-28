from __future__ import annotations

import torch
import triton
import triton.language as tl

from .aiter_mla_gluon import mla_gluon


@triton.jit
def _count_valid_topk_kernel(topk_ptr, counts_ptr, stride_t, topk: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(topk_ptr + row * stride_t + offs, mask=offs < topk, other=-1)
    valid = (offs < topk) & (vals >= 0)
    count = tl.sum(valid.to(tl.int32), axis=0)
    tl.store(counts_ptr + row, count)


@triton.jit
def _pack_valid_topk_kernel(
    topk_ptr,
    indptr_ptr,
    flat_ptr,
    stride_t,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    vals = tl.load(topk_ptr + row * stride_t + offs, mask=offs < topk, other=-1)
    valid = (offs < topk) & (vals >= 0)
    pos = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    start = tl.load(indptr_ptr + row)
    tl.store(flat_ptr + start + pos, vals.to(tl.int32), mask=valid)


def _v4_topk_to_ragged_gpu(topk_indices: torch.Tensor):
    total_tokens, topk = topk_indices.shape
    block = triton.next_power_of_2(topk)
    counts = torch.empty(total_tokens, dtype=torch.int32, device=topk_indices.device)
    _count_valid_topk_kernel[(total_tokens,)](
        topk_indices,
        counts,
        topk_indices.stride(0),
        topk,
        BLOCK=block,
    )
    indptr = torch.empty(total_tokens + 1, dtype=torch.int32, device=topk_indices.device)
    indptr[0] = 0
    torch.cumsum(counts, dim=0, out=indptr[1:])

    # Allocate the max possible nnz to avoid a CPU sync on indptr[-1]. The Gluon
    # kernel uses indptr for bounds, so unused tail elements are never read.
    flat = torch.empty(total_tokens * topk, dtype=torch.int32, device=topk_indices.device)
    _pack_valid_topk_kernel[(total_tokens,)](
        topk_indices,
        indptr,
        flat,
        topk_indices.stride(0),
        topk,
        BLOCK=block,
    )
    return flat, indptr


@triton.jit
def _pack_v4_csa_h64_kernel(
    topk_ptr,
    indptr_ptr,
    flat_ptr,
    stride_t,
    total_tokens: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    pool_k: tl.constexpr = TOPK - WINDOW
    local_count = tl.minimum(row + 1, WINDOW)

    if row < WINDOW:
        local_prefix = row * (row + 1) // 2
    else:
        local_prefix = WINDOW * (WINDOW + 1) // 2 + (row - WINDOW) * WINDOW
    start = row * pool_k + local_prefix
    tl.store(indptr_ptr + row, start)

    offs = tl.arange(0, BLOCK)
    local_mask = offs < local_count
    pool_offs = offs - local_count
    pool_mask = (offs >= local_count) & (pool_offs < pool_k)
    src = tl.where(local_mask, WINDOW - local_count + offs, WINDOW + pool_offs)
    vals = tl.load(topk_ptr + row * stride_t + src, mask=local_mask | pool_mask, other=-1)
    tl.store(flat_ptr + start + offs, vals.to(tl.int32), mask=local_mask | pool_mask)

    if row == total_tokens - 1:
        tl.store(indptr_ptr + total_tokens, start + local_count + pool_k)


def _v4_csa_h64_topk_to_ragged(topk_indices: torch.Tensor):
    total_tokens, topk = topk_indices.shape
    block = triton.next_power_of_2(topk)
    indptr = torch.empty(total_tokens + 1, dtype=torch.int32, device=topk_indices.device)
    flat = torch.empty(total_tokens * topk, dtype=torch.int32, device=topk_indices.device)
    _pack_v4_csa_h64_kernel[(total_tokens,)](
        topk_indices,
        indptr,
        flat,
        topk_indices.stride(0),
        total_tokens,
        topk,
        WINDOW=128,
        BLOCK=block,
    )
    return flat, indptr


def sparse_mla_fwd_v4_aiter_lse(q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None):
    """Aiter Gluon sparse-MLA fwd with LSE, adapted to the V4 dense-topk API."""
    _, _, d_qk = q.shape
    if scale is None:
        scale = 1.0 / (d_qk**0.5)

    q_nope = q[..., :kv_lora_rank].contiguous()
    kv2 = kv[:, 0, :] if kv.dim() == 3 else kv
    kv_c = kv2[:, :kv_lora_rank].contiguous()
    flat, indptr = _v4_topk_to_ragged_gpu(topk_indices)
    out = torch.empty(q.shape[0], q.shape[1], kv_lora_rank, dtype=q.dtype, device=q.device)

    return mla_gluon(
        q_nope,
        None,
        kv_c,
        out,
        page_table=flat,
        seq_info=indptr,
        sm_scale=float(scale),
        has_pe=False,
        min_kv_seq_len=float("inf"),
        attn_sink=attn_sink,
        return_lse=True,
    )


def sparse_mla_fwd_v4_aiter_lse_csa_formula(
    q, kv, topk_indices, attn_sink=None, kv_lora_rank=512, scale=None
):
    """CSA specialization using the V4 [SWA128 + pool topk] dense-topk layout."""
    _, _, d_qk = q.shape
    if scale is None:
        scale = 1.0 / (d_qk**0.5)

    q_nope = q[..., :kv_lora_rank].contiguous()
    kv2 = kv[:, 0, :] if kv.dim() == 3 else kv
    kv_c = kv2[:, :kv_lora_rank].contiguous()
    flat, indptr = _v4_csa_h64_topk_to_ragged(topk_indices)
    out = torch.empty(q.shape[0], q.shape[1], kv_lora_rank, dtype=q.dtype, device=q.device)

    return mla_gluon(
        q_nope,
        None,
        kv_c,
        out,
        page_table=flat,
        seq_info=indptr,
        sm_scale=float(scale),
        has_pe=False,
        min_kv_seq_len=float("inf"),
        attn_sink=attn_sink,
        return_lse=True,
    )
