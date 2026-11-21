###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused glue for the V4 sparse-MLA adapter (one kernel each instead of chains of small torch ops).

* :func:`csa_topk` -- the flat ``[B*S, round64(W+K)]`` int32 top-k map of a CSA layer (sliding window
  ++ indexer pool picks, ``-1`` padded), identical to ``_pad_topk_64(_build_csa_topk(...))`` without
  packing (``seq_starts is None``).
* :func:`rope_pad_kv` -- ``[B*(Skv+P), 1, D + pad]`` from the latent ``[B, Skv, D]`` (any strides, unit
  last stride) and an optional pool ``[B, P, D]``, pad columns zero; replaces cat + copy + zero.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _csa_topk_kernel(
    Pool, Out, S, Skv, P, K, W, TOPK, stride_pb, stride_ps, cp_dwindow, cp_gs, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    b = row // S
    t = row % S
    base = b * (Skv + P)
    s = tl.arange(0, BLOCK)
    gpos = t + cp_gs - W + 1 + s
    win = tl.where(gpos >= 0, base + gpos - cp_gs + cp_dwindow, -1)
    k = s - W
    pm = (s >= W) & (k < K)
    p = tl.load(Pool + b.to(tl.int64) * stride_pb + t.to(tl.int64) * stride_ps + k, mask=pm, other=-1)
    pool = tl.where(p >= 0, base + Skv + p, -1)
    val = tl.where(s < W, win, tl.where(pm, pool, -1))
    tl.store(Out + row.to(tl.int64) * TOPK + s, val.to(tl.int32), mask=s < TOPK)


def csa_topk(topk_idxs, S, Skv, P, W, cp_dwindow=0, cp_global_start=0):
    B, S_, K = topk_idxs.shape
    assert S_ == S and topk_idxs.stride(2) == 1
    topk = (W + K + 63) // 64 * 64
    out = torch.empty(B * S, topk, dtype=torch.int32, device=topk_idxs.device)
    if B * S:
        _csa_topk_kernel[(B * S,)](
            topk_idxs,
            out,
            S,
            Skv,
            P,
            K,
            W,
            topk,
            topk_idxs.stride(0),
            topk_idxs.stride(1),
            int(cp_dwindow),
            int(cp_global_start),
            BLOCK=triton.next_power_of_2(topk),
            num_warps=4,
        )
    return out


@triton.jit
def _rope_pad_kv_kernel(
    Lat, Pool, Out, Skv, P, n_rows, sl_b, sl_s, sp_b, sp_s, D: tl.constexpr, PAD: tl.constexpr, ROWS: tl.constexpr
):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rm = r < n_rows
    b = r // (Skv + P)
    i = r % (Skv + P)
    d = tl.arange(0, D)
    is_lat = i < Skv
    pl = Lat + b.to(tl.int64) * sl_b + i.to(tl.int64) * sl_s
    pp = Pool + b.to(tl.int64) * sp_b + (i - Skv).to(tl.int64) * sp_s
    va = tl.load(pl[:, None] + d[None, :], mask=(rm & is_lat)[:, None], other=0.0)
    vb = tl.load(pp[:, None] + d[None, :], mask=(rm & (~is_lat))[:, None], other=0.0)
    v = tl.where(is_lat[:, None], va, vb)
    o = Out + r.to(tl.int64)[:, None] * (D + PAD)
    tl.store(o + d[None, :], v, mask=rm[:, None])
    z = tl.arange(0, PAD)
    tl.store(o + D + z[None, :], tl.zeros([ROWS, PAD], dtype=v.dtype), mask=rm[:, None])


def rope_pad_kv(latent, pool=None, pad=64):
    """``[B, Skv, D]`` latent (+ ``[B, P, D]`` pool) -> ``[B*(Skv+P), 1, D+pad]``, pad zeroed."""
    B, Skv, D = latent.shape
    P = 0 if pool is None else pool.shape[1]
    assert latent.stride(2) == 1 and (pool is None or pool.stride(2) == 1)
    n = B * (Skv + P)
    out = latent.new_empty(n, 1, D + pad)
    pl = latent if pool is None else pool
    rows = 8
    if n:
        _rope_pad_kv_kernel[(triton.cdiv(n, rows),)](
            latent,
            pl,
            out,
            Skv,
            P,
            n,
            latent.stride(0),
            latent.stride(1),
            pl.stride(0),
            pl.stride(1),
            D=D,
            PAD=pad,
            ROWS=rows,
            num_warps=4,
        )
    return out
