###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Row top-k of the V4 indexer scores in one Triton kernel (``PRIMUS_V4_INDEXER_TOPK_TRITON=1``).

Replaces ``scores.topk(K, dim=-1)`` (sorted) + ``Indexer._finalize_topk``'s ``isneginf`` / ``full_like`` / ``where``
(11 launches per CSA layer: torch's multi-block radix select, gather and a radix KV sort) with one launch per row:
each program selects the top K of its row's 32-bit keys ``(order-preserving value bits << 16) | (0xFFFF - column)``
with ``tl.topk`` (bitonic top-K; the full ``tl.sort`` on a Triton without it, or for K not a power of two), sorted
descending, and stores the K values / columns, ``-1`` where the value is ``-inf``. A row whose columns >= K are all
``-inf`` (a causal row with at most K live pool entries) only sorts its first K columns: every ``-inf`` entry maps to
(``-inf``, ``-1``) whatever its column, so the result is the same.

Same result as torch's path, bit for bit: the value bits are mapped with torch's ``TopKTypeConfig<BFloat16>::convert``
(negative -> ~x, positive -> x ^ 0x8000, NaN -> 0xFFFF; so -0 < +0 exactly as torch orders them), and among equal
values the lower column comes first, which is what torch's gather (index order) followed by its stable descending
radix sort produces. The keys are int32 with bit 31 flipped, so the signed order is the unsigned one. Small rows
(< 1024 columns) use a different torch algorithm with other tie order, so they are not routed here.
"""

import os

import torch
import triton
import triton.language as tl

_HAS_TL_TOPK = hasattr(tl, "topk")


@triton.jit
def _topk_rows(
    S_PTR,
    V_PTR,
    I_PTR,
    row,
    P,
    stride_s,
    stride_v,
    stride_i,
    K: tl.constexpr,
    BP: tl.constexpr,
    USE_TOPK: tl.constexpr,
):
    """Top-K of the first P (<= BP) columns of one row, sorted descending, ties -> lower column first."""
    offs = tl.arange(0, BP)
    pm = offs < P
    v = tl.load(S_PTR + row * stride_s + offs, mask=pm, other=0.0)
    b = v.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    key16 = tl.where((b & 0x8000) != 0, (~b) & 0xFFFF, b | 0x8000)
    key16 = tl.where(v != v, 0xFFFF, key16)
    key = ((key16 ^ 0x8000) << 16) | (0xFFFF - offs)
    key = tl.where(pm, key, -2147483648)
    if USE_TOPK:
        srt = tl.topk(key, K)
        ko = tl.arange(0, K)
    else:
        srt = tl.sort(key, descending=True)
        ko = offs
    col = (0xFFFF - (srt & 0xFFFF)).to(tl.int64)
    k16 = ((srt >> 16) & 0xFFFF) ^ 0x8000
    vb = tl.where((k16 & 0x8000) != 0, k16 ^ 0x8000, (~k16) & 0xFFFF)
    val = vb.to(tl.int16).to(tl.bfloat16, bitcast=True)
    neginf = vb == 0xFF80  # bf16 -inf
    km = ko < K
    tl.store(V_PTR + row * stride_v + ko, val, mask=km)
    tl.store(I_PTR + row * stride_i + ko, tl.where(neginf, -1, col), mask=km)


@triton.jit
def _indexer_topk_kernel(
    S_PTR,
    V_PTR,
    I_PTR,
    P,
    stride_s,
    stride_v,
    stride_i,
    K: tl.constexpr,
    BP: tl.constexpr,
    USE_TOPK: tl.constexpr,
    HALF: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    if HALF:
        offs = tl.arange(0, BP)
        v = tl.load(S_PTR + row * stride_s + offs, mask=(offs < P) & (offs >= K), other=float("-inf"))
        n_hi = tl.sum((v != float("-inf")).to(tl.int32), axis=0)
        if n_hi == 0:  # all columns >= K are -inf: the sorted first K columns are the top K
            _topk_rows(S_PTR, V_PTR, I_PTR, row, K, stride_s, stride_v, stride_i, K, K, False)
        else:
            _topk_rows(S_PTR, V_PTR, I_PTR, row, P, stride_s, stride_v, stride_i, K, BP, USE_TOPK)
    else:
        _topk_rows(S_PTR, V_PTR, I_PTR, row, P, stride_s, stride_v, stride_i, K, BP, USE_TOPK)


def indexer_topk_supported(scores: torch.Tensor, k: int) -> bool:
    if os.environ.get("PRIMUS_V4_INDEXER_TOPK_TRITON", "0") != "1":
        return False
    P = scores.shape[-1]
    return (
        scores.is_cuda
        and not (torch.is_grad_enabled() and scores.requires_grad)  # topk_scores carry no autograd graph
        and scores.dtype == torch.bfloat16
        and scores.stride(-1) == 1
        and scores.dim() >= 2
        and 1024 <= P <= 4096
        and 0 < k <= P
        and scores.data_ptr() != 0
    )


def indexer_topk(scores: torch.Tensor, k: int):
    """-> (topk_idxs int64 [..., k] with -1 where the score is -inf, topk_scores [..., k]) for scores [..., P]."""
    lead = scores.shape[:-1]
    P = scores.shape[-1]
    s2 = scores.reshape(-1, P)
    R = s2.shape[0]
    vals = torch.empty((R, k), dtype=scores.dtype, device=scores.device)
    idxs = torch.empty((R, k), dtype=torch.int64, device=scores.device)
    pow2k = (k & (k - 1)) == 0
    if R:
        _indexer_topk_kernel[(R,)](
            s2,
            vals,
            idxs,
            P,
            s2.stride(0),
            vals.stride(0),
            idxs.stride(0),
            K=k,
            BP=triton.next_power_of_2(P),
            USE_TOPK=_HAS_TL_TOPK and pow2k,
            HALF=pow2k,
            num_warps=2,
        )
    return idxs.view(*lead, k), vals.view(*lead, k)
