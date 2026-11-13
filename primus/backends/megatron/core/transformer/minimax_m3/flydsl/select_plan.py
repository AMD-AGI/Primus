###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The indexer's block selection and the attention backward's :class:`BlockPlan`
in five launches instead of ~30 per MSA layer.

Everything here is integer bookkeeping or a copy/select of fp32 values, so it is
bit-identical to the eager path it replaces by construction:

* :func:`boost_forced` -- ``block_scores.scatter(-1, forced, +inf)``
  (``MinimaxM3Indexer._boost_always_visible``), one kernel instead of copy + scatter.
* the top-k itself stays ATen ``topk`` (its tie order decides the table).
* :func:`select_post` -- ``block_indices.masked_fill(topk_scores == -inf, -1)``, the
  ``_FusedSelect`` bookkeeping (``keys``, ``dead``), the int32 table the MSA
  kernels read, and the counting pass of the plan's counting sort (one launch
  instead of eq/copy/masked_fill/clamp/gather x2/isposinf/lt/or/lt/or/to(int32)
  and the FlyDSL count pass).
* :func:`plan_place` -- the exclusive cumsum of the counts, the placing pass,
  the CSR ``ptr`` and :func:`msa_token_bwd.plan_chunks` (``chunk_ptr`` and the
  chunk rows), all in one launch: every program recomputes the (tiny) prefix sums
  it needs from the counts, so nothing waits on a global scan.

The outputs match :func:`msa_token_bwd.build_block_plan` element for element on
every live entry (``entries`` past ``ptr[-1]`` are unused scratch in both).

``PRIMUS_FUSED_SELECT_PLAN=0`` falls back to the eager/FlyDSL path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_SELECT_PLAN", "1") != "0"

_SUB = 128  # msa_token_bwd._PLACE_SUB
_SEARCH_STEPS = 24  # msa_token_bwd._SEARCH_STEPS


@triton.jit
def _boost_kernel(raw_ptr, forced_ptr, out_ptr, R, S, NB: tl.constexpr, NBP: tl.constexpr,
                  F: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, NBP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < NB)
    off = rows[:, None] * NB + cols[None, :]
    v = tl.load(raw_ptr + off, mask=m, other=0.0)
    s = rows % S
    for f in tl.static_range(F):
        idx = tl.load(forced_ptr + s * F + f, mask=rmask, other=-1)
        v = tl.where(cols[None, :] == idx[:, None], float("inf"), v)
    tl.store(out_ptr + off, v, mask=m)


@triton.jit
def _post_kernel(
    tsc_ptr, tidx_ptr, argmax_ptr, boosted_ptr,
    bi_ptr, keys_ptr, keysc_ptr, dead_ptr, tbl_ptr, runs_ptr,
    S, NC, TC,
    NB: tl.constexpr, NBP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, TT: tl.constexpr,
):
    # one program per (b * Hkv + g, token chunk of TC tokens), TT tokens at a time
    bg = tl.program_id(0)
    c = tl.program_id(1)
    cols = tl.arange(0, KP)
    nb = tl.arange(0, NBP)
    cnt = tl.zeros([NBP], dtype=tl.int32)
    for t0 in range(0, TC, TT):
        tt = t0 + tl.arange(0, TT)
        t = c * TC + tt
        tmask = (t < S) & (tt < TC)
        m = tmask[:, None] & (cols[None, :] < K)
        rows = bg * S + t
        off = rows[:, None] * K + cols[None, :]
        sc = tl.load(tsc_ptr + off, mask=m, other=0.0)
        idx = tl.load(tidx_ptr + off, mask=m, other=0)
        bi = tl.where(sc == float("-inf"), -1, idx)
        slots = tl.where(bi < 0, 0, bi)
        keys = tl.load(argmax_ptr + rows[:, None] * NB + slots, mask=m, other=0)
        bsv = tl.load(boosted_ptr + rows[:, None] * NB + slots, mask=m, other=0.0)
        forced = bsv == float("inf")
        dead = (bi < 0) | forced | (keys < 0)
        tl.store(bi_ptr + off, bi, mask=m)
        tl.store(keys_ptr + off, keys, mask=m)
        tl.store(keysc_ptr + off, tl.where(keys < 0, 0, keys), mask=m)
        tl.store(dead_ptr + off, dead.to(tl.int8), mask=m)
        b32 = bi.to(tl.int32)
        tl.store(tbl_ptr + off, b32, mask=m)
        # counting pass: live entries of this chunk per block
        blk = tl.where(m, b32, -1)
        hit = (blk[:, :, None] == nb[None, None, :]).to(tl.int32)  # [TT, KP, NBP]
        cnt += tl.sum(tl.sum(hit, axis=1), axis=0)
    tl.store(runs_ptr + (bg * NB + nb) * NC + c, cnt, mask=nb < NB)


@triton.jit
def _place_kernel(
    tbl_ptr, runs_ptr, ent_ptr, ptr_ptr, chptr_ptr, chunks_ptr,
    S, NC, TC, NROWS, NCH, SIZE,
    NB: tl.constexpr, NBP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr, TT: tl.constexpr,
    RP: tl.constexpr, NCP: tl.constexpr, CPP: tl.constexpr,
):
    bg = tl.program_id(0)
    c = tl.program_id(1)
    pid = bg * NC + c
    # ---- counts [NROWS, NC] -> row totals and exclusive row prefix ----
    r = tl.arange(0, RP)
    cc = tl.arange(0, NCP)
    rm = r < NROWS
    runs = tl.load(runs_ptr + r[:, None] * NC + cc[None, :], mask=rm[:, None] & (cc[None, :] < NC), other=0)
    tot = tl.sum(runs, axis=1)  # [RP]
    rincl = tl.cumsum(tot, axis=0)
    rexcl = rincl - tot
    # ---- this chunk's start in each of its NB rows: row prefix + earlier chunks of the row ----
    nb = tl.arange(0, NBP)
    my_rows = bg * NB + nb  # [NBP]
    sel = (my_rows[:, None] == r[None, :]) & (nb[:, None] < NB)  # [NBP, RP]
    base = tl.sum(tl.where(sel, rexcl[None, :], 0), axis=1)  # [NBP]
    my_runs = tl.load(runs_ptr + my_rows[:, None] * NC + cc[None, :],
                      mask=(nb[:, None] < NB) & (cc[None, :] < c), other=0)
    base = base + tl.sum(my_runs, axis=1)
    # ---- place this chunk's entries in token order, TT tokens at a time ----
    cols = tl.arange(0, KP)
    for t0 in range(0, TC, TT):
        tt = t0 + tl.arange(0, TT)
        t = c * TC + tt
        m = ((t < S) & (tt < TC))[:, None] & (cols[None, :] < K)
        e = (bg * S + t)[:, None] * K + cols[None, :]  # [TT, KP]
        blk = tl.load(tbl_ptr + e, mask=m, other=-1)
        hit = (blk[:, :, None] == nb[None, None, :]).to(tl.int32)  # [TT, KP, NBP]
        # rank among the chunk's earlier entries of the same block, entries ordered (token, slot);
        # a token's slots name distinct blocks, so its other slots never match
        per_tok = tl.sum(hit, axis=1)  # [TT, NBP]
        tok_incl = tl.cumsum(per_tok, axis=0)
        tok_excl = tok_incl - per_tok
        pos = tl.sum(hit * (tok_excl[:, None, :] + base[None, None, :]), axis=2)  # [TT, KP]
        tl.store(ent_ptr + pos, e.to(tl.int32), mask=m & (blk >= 0))
        base += tl.sum(per_tok, axis=0)
    # ---- CSR ptr, chunk_ptr (program 0) and the chunk rows (all programs) ----
    per_row = tl.maximum((tot + (SIZE - 1)) // SIZE, 1)
    per_row = tl.where(rm, per_row, 0)
    cpi = tl.cumsum(per_row, axis=0)
    cpe = cpi - per_row  # chunk_ptr[r] for r < NROWS
    total_live = tl.sum(per_row, axis=0)
    ptr_last = tl.sum(tot, axis=0)
    if pid == 0:
        tl.store(ptr_ptr + r, rexcl, mask=rm)
        tl.store(ptr_ptr + NROWS, ptr_last)
        tl.store(chptr_ptr + r, cpe, mask=rm)
        tl.store(chptr_ptr + NROWS, total_live)
    ch = pid * CPP + tl.arange(0, CPP)
    # last row whose first chunk is <= ch (chunk_ptr strictly increases: every row has a chunk)
    le = (cpe[None, :] <= ch[:, None]) & rm[None, :]  # [CPP, RP]
    row = tl.sum(le.to(tl.int32), axis=1) - 1
    row = tl.minimum(tl.maximum(row, 0), NROWS - 1)
    pick = r[None, :] == row[:, None]
    row_ptr = tl.sum(tl.where(pick, rexcl[None, :], 0), axis=1)
    row_cp = tl.sum(tl.where(pick, cpe[None, :], 0), axis=1)
    row_end = tl.sum(tl.where(pick, rexcl[None, :] + tot[None, :], 0), axis=1)
    live = ch < total_live
    begin = row_ptr + (ch - row_cp) * SIZE
    end = tl.minimum(begin + SIZE, row_end)
    cm = ch < NCH
    tl.store(chunks_ptr + ch * 3 + 0, tl.where(live, row, 0), mask=cm)
    tl.store(chunks_ptr + ch * 3 + 1, tl.where(live, begin, 0), mask=cm)
    tl.store(chunks_ptr + ch * 3 + 2, tl.where(live, end, 0), mask=cm)


def _np2(x: int) -> int:
    return 1 << max(0, (int(x) - 1).bit_length())


def supported(B: int, H: int, S: int, n_blocks: int, topk: int) -> bool:
    """Shapes the fused path handles; anything else takes the original path."""
    tc = _SUB * -(-n_blocks // 256)
    return ENABLED and n_blocks <= 64 and topk <= 32 and tc == _SUB and tc % 16 == 0 and B * H * n_blocks <= 1024


def boost_forced(raw: torch.Tensor, forced: torch.Tensor) -> torch.Tensor:
    """``raw.scatter(-1, forced[None, None].expand(...), +inf)`` for ``raw`` ``[B, H, S, NB]`` fp32
    and ``forced`` ``[S, F]`` int64."""
    B, H, S, NB = raw.shape
    out = torch.empty_like(raw)
    forced = forced.contiguous()
    F = forced.shape[1]
    R = B * H * S
    BR = 64
    _boost_kernel[(triton.cdiv(R, BR),)](raw, forced, out, R, S, NB=NB, NBP=_np2(NB), F=F, BR=BR,
                                         num_warps=4)
    return out


def select_post_and_plan(topk_scores, topk_idx, argmax, boosted, block_size, want_plan):
    """Returns ``(block_indices int64, keys int32, keys_c int32, dead bool, table int32, plan)``.

    ``plan`` is the :class:`msa_token_bwd.BlockPlan` of ``block_indices`` when ``want_plan``.
    """
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import BlockPlan

    B, H, S, K = topk_idx.shape
    NB = argmax.shape[-1]
    dev = topk_idx.device
    BG = B * H
    tc = _SUB
    nc = -(-S // tc)
    n_rows = BG * NB
    bi = torch.empty((B, H, S, K), dtype=torch.int64, device=dev)
    keys = torch.empty((B, H, S, K), dtype=torch.int32, device=dev)
    keys_c = torch.empty_like(keys)
    dead = torch.empty((B, H, S, K), dtype=torch.bool, device=dev)
    table = torch.empty((B, H, S, K), dtype=torch.int32, device=dev)
    runs = torch.empty(n_rows * nc, dtype=torch.int32, device=dev)
    NBP, KP, TT = _np2(NB), _np2(K), 16
    _post_kernel[(BG, nc)](
        topk_scores.contiguous(), topk_idx.contiguous(), argmax.contiguous(), boosted.contiguous(),
        bi, keys, keys_c, dead.view(torch.int8), table, runs,
        S, nc, tc, NB=NB, NBP=NBP, K=K, KP=KP, TT=TT, num_warps=4,
    )
    plan = None
    if want_plan:
        max_tokens = BG * S * K
        size = max(64, -(-max_tokens // 2048))  # plan_chunks(target_chunks=2048, min_chunk=64)
        n_chunks = n_rows + -(-max_tokens // size)
        entries = torch.empty(max_tokens, dtype=torch.int32, device=dev)
        ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
        chunk_ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
        chunks = torch.empty((n_chunks, 3), dtype=torch.int32, device=dev)
        n_prog = BG * nc
        cpp = _np2(-(-n_chunks // n_prog))
        _place_kernel[(BG, nc)](
            table, runs, entries, ptr, chunk_ptr, chunks,
            S, nc, tc, n_rows, n_chunks, size,
            NB=NB, NBP=NBP, K=K, KP=KP, TT=TT, RP=_np2(n_rows), NCP=_np2(nc), CPP=cpp, num_warps=8,
        )
        plan = BlockPlan(ptr, entries, chunks, chunk_ptr, n_chunks, -(-S // block_size))
    return bi, keys, keys_c, dead, table, plan


@triton.jit
def _bwd_g_kernel(grad_ptr, bi_ptr, dead_ptr, g_ptr, R, NB: tl.constexpr, K: tl.constexpr,
                  KP: tl.constexpr, BR: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    m = (rows < R)[:, None] & (cols[None, :] < K)
    off = rows[:, None] * K + cols[None, :]
    bi = tl.load(bi_ptr + off, mask=m, other=0)
    slots = tl.where(bi < 0, 0, bi)
    g = tl.load(grad_ptr + rows[:, None] * NB + slots, mask=m, other=0.0)
    d = tl.load(dead_ptr + off, mask=m, other=1)
    g = tl.where(d != 0, 0.0, g.to(tl.float32))
    tl.store(g_ptr + off, g, mask=m)


def backward_g(grad_scores: torch.Tensor, block_indices: torch.Tensor, dead: torch.Tensor) -> torch.Tensor:
    """``grad_scores.gather(-1, block_indices.clamp_min(0)).masked_fill_(dead, 0.0).float()``."""
    B, H, S, K = block_indices.shape
    NB = grad_scores.shape[-1]
    g = torch.empty((B, H, S, K), dtype=torch.float32, device=grad_scores.device)
    R = B * H * S
    BR = 64
    _bwd_g_kernel[(triton.cdiv(R, BR),)](grad_scores.contiguous(), block_indices, dead.view(torch.int8), g,
                                         R, NB=NB, K=K, KP=_np2(K), BR=BR, num_warps=4)
    return g
