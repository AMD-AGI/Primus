###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GLM-5.3 kpool lightning indexer: pooled keys, scores and token selection.

Semantics (SGLang ``dsa_indexer_kpool.py`` / Miles ``kpool_indexer.py``), per
sequence of length ``S`` with ``kpool = 4``:

* every full group of ``kpool`` consecutive tokens forms one *pool*; its key is
  a per-channel softmax over the ``kpool`` slots of ``gate + ape[slot]``
  weighting the slot keys (``P = S // kpool`` pools);
* query ``l`` scores pool ``p`` as ``sum_h w[l, h] * relu(q[l, h] . k_pool[p])``
  and only sees the ``(l + 1) // kpool`` pools that are already complete;
* if ``l + 1 <= index_topk`` the query attends to every token ``0 .. l``;
  otherwise to the ``index_topk // kpool`` best pools (expanded to their
  ``kpool`` tokens) plus the ``(l + 1) % kpool`` tail tokens of its own
  incomplete pool.

Selections are returned as flat token indices into the ``[B * S]`` token axis,
``-1``-padded to a multiple of 64 -- the sparse-MLA kernel contract.

Nothing here is differentiable: the indexer is trained (if at all) by a
separate loss, and the attention path treats the selection as a constant.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = [
    "build_pooled_keys",
    "kpool_scores",
    "kpool_scores_torch",
    "kpool_select",
    "causal_all_indices",
]

INDEX_PAD = 64


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def build_pooled_keys(
    index_k: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor, kpool: int
) -> torch.Tensor:
    """``[B, S, D]`` keys -> ``[B, S // kpool, D]`` pooled keys (fp32 math, keys' dtype out)."""
    B, S, D = index_k.shape
    P = S // kpool
    if P == 0:
        return index_k.new_zeros(B, 0, D)
    k = index_k[:, : P * kpool].float().view(B, P, kpool, D)
    g = gate[:, : P * kpool].float().view(B, P, kpool, D) + ape.float().view(1, 1, kpool, D)
    w = torch.softmax(g, dim=2)
    return (w * k).sum(dim=2).to(index_k.dtype)


@triton.jit
def _kpool_score_kernel(
    Q_ptr,  # [B*S, H, D] bf16
    K_ptr,  # [B, P, D] bf16
    W_ptr,  # [B*S, H] fp32
    O_ptr,  # [B*S, P_OUT] fp32
    S,
    P,
    P_OUT,
    stride_q_t,
    stride_q_h,
    stride_k_b,
    stride_k_p,
    stride_w_t,
    stride_o_t,
    KPOOL: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    pb = tl.program_id(1)
    b = t // S
    l = t % S
    limit = (l + 1) // KPOOL
    offs_p = pb * BLOCK_P + tl.arange(0, BLOCK_P)
    in_out = offs_p < P_OUT
    if pb * BLOCK_P >= limit:
        tl.store(O_ptr + t * stride_o_t + offs_p, tl.full([BLOCK_P], float("-inf"), tl.float32), mask=in_out)
        return
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    q = tl.load(Q_ptr + t * stride_q_t + offs_h[:, None] * stride_q_h + offs_d[None, :])
    valid_p = offs_p < limit
    k = tl.load(
        K_ptr + b * stride_k_b + offs_p[:, None].to(tl.int64) * stride_k_p + offs_d[None, :],
        mask=valid_p[:, None] & (offs_p[:, None] < P),
        other=0.0,
    )
    s = tl.dot(q, tl.trans(k))  # [H, BLOCK_P] fp32
    w = tl.load(W_ptr + t * stride_w_t + offs_h)
    score = tl.sum(tl.maximum(s, 0.0) * w[:, None], axis=0)
    score = tl.where(valid_p, score, float("-inf"))
    tl.store(O_ptr + t * stride_o_t + offs_p, score, mask=in_out)


def kpool_scores(q: torch.Tensor, pooled_k: torch.Tensor, weights: torch.Tensor, kpool: int) -> torch.Tensor:
    """Pool scores ``[B * S, P]`` (fp32, ``-inf`` for pools not yet complete).

    Args:
        q: ``[B, S, H, D]`` indexer queries.
        pooled_k: ``[B, P, D]`` pooled keys.
        weights: ``[B, S, H]`` fp32 per-head weights (scales already folded in).
    """
    B, S, H, D = q.shape
    P = pooled_k.shape[1]
    out = torch.empty(B * S, max(P, 1), dtype=torch.float32, device=q.device)
    if P == 0:
        out.fill_(float("-inf"))
        return out
    q2 = q.reshape(B * S, H, D).to(torch.bfloat16).contiguous()
    k2 = pooled_k.to(torch.bfloat16).contiguous()
    w2 = weights.reshape(B * S, H).float().contiguous()
    BLOCK_P = 64
    grid = (B * S, triton.cdiv(P, BLOCK_P))
    _kpool_score_kernel[grid](
        q2,
        k2,
        w2,
        out,
        S,
        P,
        P,
        q2.stride(0),
        q2.stride(1),
        k2.stride(0),
        k2.stride(1),
        w2.stride(0),
        out.stride(0),
        KPOOL=kpool,
        H=H,
        D=D,
        BLOCK_P=BLOCK_P,
        num_warps=4,
    )
    return out


def kpool_scores_torch(
    q: torch.Tensor, pooled_k: torch.Tensor, weights: torch.Tensor, kpool: int
) -> torch.Tensor:
    """Reference for :func:`kpool_scores` (fp32 math, chunked over queries)."""
    B, S, H, D = q.shape
    P = pooled_k.shape[1]
    out = torch.full((B, S, max(P, 1)), float("-inf"), dtype=torch.float32, device=q.device)
    if P == 0:
        return out.view(B * S, -1)
    pos = torch.arange(S, device=q.device)
    limit = (pos + 1) // kpool
    visible = torch.arange(P, device=q.device)[None, :] < limit[:, None]  # [S, P]
    kf = pooled_k.float()
    for b in range(B):
        for s0 in range(0, S, 1024):
            s1 = min(S, s0 + 1024)
            sc = torch.einsum("shd,pd->shp", q[b, s0:s1].float(), kf[b]).clamp_min(0)
            sc = (sc * weights[b, s0:s1].float().unsqueeze(-1)).sum(1)
            out[b, s0:s1] = torch.where(visible[s0:s1], sc, float("-inf"))
    return out.view(B * S, -1)


def causal_all_indices(B: int, S: int, device: torch.device) -> torch.Tensor:
    """Every query attends to every earlier token: ``[B * S, round_up(S, 64)]``."""
    width = _round_up(S, INDEX_PAD)
    cols = torch.arange(width, device=device, dtype=torch.int32)
    pos = torch.arange(S, device=device, dtype=torch.int32)
    base = torch.arange(B, device=device, dtype=torch.int32) * S
    idx = torch.where(cols[None, :] <= pos[:, None], cols[None, :], -1)  # [S, width]
    idx = torch.where(idx >= 0, idx[None] + base[:, None, None], -1)
    return idx.reshape(B * S, width).contiguous()


def kpool_select(scores: torch.Tensor, B: int, S: int, index_topk: int, kpool: int) -> torch.Tensor:
    """Token selection from pool scores ``[B * S, P]`` -> ``[B * S, width]`` int32.

    Rows with ``l + 1 <= index_topk`` select every token ``0 .. l``; the rest select
    the top ``index_topk // kpool`` pools (expanded) plus their tail tokens.
    """
    device = scores.device
    if S <= index_topk:
        return causal_all_indices(B, S, device)

    group_topk = index_topk // kpool
    width = _round_up(index_topk + kpool - 1, INDEX_PAD)
    P = scores.shape[1]
    pos = torch.arange(S, device=device, dtype=torch.int32).repeat(B)  # [B*S]
    base = (torch.arange(B, device=device, dtype=torch.int32) * S).repeat_interleave(S)  # [B*S]

    out = torch.full((B * S, width), -1, dtype=torch.int32, device=device)

    # Short rows: every token 0 .. l.
    cols = torch.arange(width, device=device, dtype=torch.int32)
    short = (pos + 1) <= index_topk
    dense = torch.where(cols[None, :] <= pos[:, None], cols[None, :] + base[:, None], -1)
    out = torch.where(short[:, None], dense, out)

    # Long rows: top pools, expanded, plus the tail of the incomplete pool.
    long_rows = torch.nonzero(~short, as_tuple=False).squeeze(1)
    if long_rows.numel() > 0:
        k = min(group_topk, P)
        sc = scores.index_select(0, long_rows)
        top_val, top_pool = torch.topk(sc, k, dim=-1)
        slots = torch.arange(kpool, device=device, dtype=torch.int32)
        lb = base.index_select(0, long_rows)
        lp = pos.index_select(0, long_rows)
        tok = (top_pool.to(torch.int32) * kpool)[:, :, None] + slots[None, None, :]  # [R, k, kpool]
        tok = torch.where(torch.isfinite(top_val)[:, :, None], tok + lb[:, None, None], -1).reshape(
            -1, k * kpool
        )
        tail_start = ((lp + 1) // kpool) * kpool
        tail_n = (lp + 1) % kpool
        tail_slots = torch.arange(kpool - 1, device=device, dtype=torch.int32)
        tail = torch.where(
            tail_slots[None, :] < tail_n[:, None], tail_start[:, None] + tail_slots[None, :] + lb[:, None], -1
        )
        sel = torch.full((long_rows.numel(), width), -1, dtype=torch.int32, device=device)
        sel[:, : k * kpool] = tok
        sel[:, index_topk : index_topk + kpool - 1] = tail
        out[long_rows] = sel
    return out.contiguous()
