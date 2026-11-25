###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The indexer's block selection and the attention backward's :class:`BlockPlan`
in a few launches per MSA layer instead of a chain of small eager ops.

Everything here is integer bookkeeping or a copy/select of fp32 values, so it is
bit-identical to the eager path it replaces by construction:

* :func:`boost_forced` -- ``block_scores.scatter(-1, forced, +inf)``
  (``MinimaxM3Indexer._boost_always_visible``), one kernel instead of copy + scatter.
* the top-k itself stays ATen ``topk`` (its tie order decides the table).
* :func:`select_post_and_plan`, two launches:

  - ``block_indices.masked_fill(topk_scores == -inf, -1)``, the ``_FusedSelect``
    bookkeeping (``keys``, ``dead``), the int32 table the MSA kernels read, and the
    counting pass of the plan's counting sort (instead of
    eq/copy/masked_fill/clamp/gather x2/isposinf/lt/or/lt/or/to(int32) and the FlyDSL
    count pass);
  - the exclusive cumsum of the counts, the placing pass, the CSR ``ptr`` and
    :func:`msa_token_bwd.plan_chunks` (``chunk_ptr`` and the chunk rows): every
    program recomputes the (tiny) prefix sums it needs from the counts, so nothing
    waits on a global scan.

* :func:`backward_g` -- the gather and mask of the selection's backward.

The outputs match :func:`msa_token_bwd.build_block_plan` element for element on
every live entry (``entries`` past ``ptr[-1]`` are unused scratch in both).
Shapes outside :func:`supported` keep the eager/FlyDSL path, unless
``msa_select_plan_wide`` is on and :func:`wide_supported` holds: then the
bookkeeping is one launch (no counting), and the plan is the two FlyDSL passes of
``build_inverted_table`` with two Triton launches for the eager ops around them.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    _PLACE_MAX_BLOCKS,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    _PLACE_SUB as _SUB,
)


@triton.jit
def _boost_kernel(
    raw_ptr,
    forced_ptr,
    out_ptr,
    R,
    S,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    F: tl.constexpr,
    BR: tl.constexpr,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, NBP)
    rmask = rows < R
    m = rmask[:, None] & (cols[None, :] < NB)
    off = rows[:, None] * NB + cols[None, :]
    v = tl.load(raw_ptr + off, mask=m, other=0.0)
    s = rows % S
    if BS > 0:
        # msa_select_skip_fill: the scoring kernel left the blocks after the query's own
        # unwritten; they are -inf whatever the buffer holds
        v = tl.where(cols[None, :] > (s // BS)[:, None], float("-inf"), v)
    for f in tl.static_range(F):
        idx = tl.load(forced_ptr + s * F + f, mask=rmask, other=-1)
        v = tl.where(cols[None, :] == idx[:, None], float("inf"), v)
    tl.store(out_ptr + off, v, mask=m)


@triton.jit
def _post_kernel(
    tsc_ptr,
    tidx_ptr,
    argmax_ptr,
    boosted_ptr,
    bi_ptr,
    keys_ptr,
    keysc_ptr,
    dead_ptr,
    tbl_ptr,
    runs_ptr,
    S,
    NC,
    TC,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TT: tl.constexpr,
    COUNT: tl.constexpr,
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
        if COUNT:
            # counting pass: live entries of this chunk per block
            blk = tl.where(m, b32, -1)
            hit = (blk[:, :, None] == nb[None, None, :]).to(tl.int32)  # [TT, KP, NBP]
            cnt += tl.sum(tl.sum(hit, axis=1), axis=0)
    if COUNT:
        tl.store(runs_ptr + (bg * NB + nb) * NC + c, cnt, mask=nb < NB)


@triton.jit
def _place_kernel(
    tbl_ptr,
    runs_ptr,
    ent_ptr,
    ptr_ptr,
    chptr_ptr,
    chunks_ptr,
    S,
    NC,
    TC,
    NROWS,
    NCH,
    SIZE,
    NB: tl.constexpr,
    NBP: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    TT: tl.constexpr,
    RP: tl.constexpr,
    NCP: tl.constexpr,
    CPP: tl.constexpr,
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
    my_runs = tl.load(
        runs_ptr + my_rows[:, None] * NC + cc[None, :], mask=(nb[:, None] < NB) & (cc[None, :] < c), other=0
    )
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


@triton.jit
def _scan_kernel(runs_ptr, tot_ptr, NC, NCP: tl.constexpr):
    """One program per plan row: the row's chunk counts become their exclusive prefix (in place)
    and the row total goes to ``tot``."""
    r = tl.program_id(0)
    cc = tl.arange(0, NCP)
    m = cc < NC
    runs = tl.load(runs_ptr + r * NC + cc, mask=m, other=0)
    incl = tl.cumsum(runs, axis=0)
    tl.store(runs_ptr + r * NC + cc, incl - runs, mask=m)
    tl.store(tot_ptr + r, tl.sum(runs, axis=0))


@triton.jit
def _rows_kernel(
    off_ptr,
    tot_ptr,
    ptr_ptr,
    chptr_ptr,
    chunks_ptr,
    NC,
    NROWS,
    NCH,
    SIZE,
    RP: tl.constexpr,
    NCP: tl.constexpr,
    CS: tl.constexpr,
):
    """After :func:`_scan_kernel`: program ``p`` turns row ``p``'s in-row chunk offsets into
    offsets into the entry array (in place: ``torch.cumsum(runs) - runs`` of the eager path),
    and writes ``CS`` rows of the chunk table; program 0 also writes ``ptr`` and ``chunk_ptr``
    (:func:`msa_token_bwd.plan_chunks`). Everything comes from the row totals."""
    p = tl.program_id(0)
    r = tl.arange(0, RP)
    rm = r < NROWS
    tot = tl.load(tot_ptr + r, mask=rm, other=0)
    rincl = tl.cumsum(tot, axis=0)
    rexcl = rincl - tot
    # ---- row p: entries of the rows before it + the row's earlier chunks ----
    before = tl.sum(tl.where(r < p, tot, 0), axis=0)
    cc = tl.arange(0, NCP)
    m = (cc < NC) & ((cc * 0 + p) < NROWS)
    inrow = tl.load(off_ptr + p * NC + cc, mask=m, other=0)
    tl.store(off_ptr + p * NC + cc, inrow + before, mask=m)
    # ---- CSR ptr, chunk_ptr (program 0) and CS chunk rows ----
    per_row = tl.maximum((tot + (SIZE - 1)) // SIZE, 1)
    per_row = tl.where(rm, per_row, 0)
    cpi = tl.cumsum(per_row, axis=0)
    cpe = cpi - per_row  # chunk_ptr[r] for r < NROWS
    total_live = tl.sum(per_row, axis=0)
    ptr_last = tl.sum(tot, axis=0)
    if p == 0:
        tl.store(ptr_ptr + r, rexcl, mask=rm)
        tl.store(ptr_ptr + NROWS, ptr_last)
        tl.store(chptr_ptr + r, cpe, mask=rm)
        tl.store(chptr_ptr + NROWS, total_live)
    ch = p * CS + tl.arange(0, CS)
    # last row whose first chunk is <= ch (chunk_ptr strictly increases: every row has a chunk)
    le = (cpe[None, :] <= ch[:, None]) & rm[None, :]  # [CS, RP]
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


# msa_select_plan_wide: elements of the [chunk rows, plan rows] compare a _rows_kernel program holds
_WIDE_HOT = 8192


def _np2(x: int) -> int:
    return 1 << max(0, (int(x) - 1).bit_length())


def supported(B: int, H: int, S: int, n_blocks: int, topk: int) -> bool:
    """Shapes the fused path handles; anything else takes the original path. ``S`` is bounded
    through ``n_blocks`` (``S <= n_blocks * block_size``); the kernels take any ``S`` below that."""
    return 0 < n_blocks <= 64 and topk <= 32 and B * H * n_blocks <= 1024


def _wide_chunking(S: int, n_blocks: int):
    """Token chunk size and chunks per row of :func:`msa_token_bwd.build_inverted_table`."""
    tc = _SUB * -(-n_blocks // 256)
    return tc, -(-S // tc)


def wide_supported(B: int, H: int, S: int, n_blocks: int, topk: int) -> bool:
    """Shapes ``msa_select_plan_wide`` handles (used past :func:`supported`): those of the FlyDSL
    counting sort of :func:`msa_token_bwd.build_inverted_table` (not its sort fallback) with at
    most 4096 rows of blocks (the row totals fit a program) and 1024 token chunks per row: 16k
    to 128k tokens at B = 1, 32k up to B = 4, 8k up to B = 16."""
    return (
        0 < n_blocks <= _PLACE_MAX_BLOCKS
        and 0 < topk <= 64
        and B * H * n_blocks <= 4096
        and _wide_chunking(S, n_blocks)[1] <= 1024
    )


def boost_forced(raw: torch.Tensor, forced: torch.Tensor, causal_block: int = 0) -> torch.Tensor:
    """``raw.scatter(-1, forced[None, None].expand(...), +inf)`` for ``raw`` ``[B, H, S, NB]`` fp32
    and ``forced`` ``[S, F]`` int64.

    ``causal_block`` (``msa_select_skip_fill``; the block size, 0 = off): ``raw`` is taken as -inf
    on the blocks after each query's own (``block > s // causal_block``), whatever it holds there
    -- the scoring kernel's caller then need not fill them."""
    B, H, S, NB = raw.shape
    out = torch.empty_like(raw)
    forced = forced.contiguous()
    F = forced.shape[1]
    R = B * H * S
    NBP = _np2(NB)
    BR = min(64, max(1, 4096 // NBP))  # 64 rows per program up to 64 blocks
    _boost_kernel[(triton.cdiv(R, BR),)](
        raw, forced, out, R, S, NB=NB, NBP=NBP, F=F, BR=BR, BS=int(causal_block), num_warps=4
    )
    return out


def _wide_plan(table: torch.Tensor, NB: int, block_size: int, bounded_chunks: bool = False):
    """:func:`msa_token_bwd.build_block_plan` of the int32 ``table`` ``[B, H, S, K]`` in four
    launches: its two FlyDSL passes, with :func:`_scan_kernel` and :func:`_rows_kernel` for the
    eager ops between and after them. The chunk size is ``plan_chunks``' own
    (:func:`msa_token_bwd.plan_chunk_size_of`: the reference rule, or with ``bounded_chunks`` the
    ``msa_bounded_plan_chunks`` rule and its 2 GiB fallback, which these row counts can reach)."""
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        _PLACE_CACHE,
        BlockPlan,
        build_inverted_pass,
        plan_chunk_size_of,
    )

    B, H, S, K = table.shape
    dev = table.device
    BG = B * H
    n_rows = BG * NB
    tc, nc = _wide_chunking(S, NB)
    nb_cap = max(64, _np2(NB))
    stream = torch.cuda.current_stream()
    max_tokens = BG * S * K
    runs = torch.empty(n_rows * nc, dtype=torch.int32, device=dev)
    entries = torch.empty(max_tokens, dtype=torch.int32, device=dev)
    args = (table, runs, entries, int(S), int(BG), int(NB), int(nc), int(tc), stream)
    fns = _PLACE_CACHE.get((K, nb_cap))
    if fns is None:
        fns = tuple(build_inverted_pass(K, nb_cap, place).compile(*args) for place in (False, True))
        _PLACE_CACHE[(K, nb_cap)] = fns
    fns[0](*args)  # runs <- counts
    size = plan_chunk_size_of(n_rows, max_tokens, bounded=bool(bounded_chunks))  # as plan_chunks
    n_chunks = n_rows + -(-max_tokens // size)
    tot = torch.empty(n_rows, dtype=torch.int32, device=dev)
    ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
    chunk_ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
    chunks = torch.empty((n_chunks, 3), dtype=torch.int32, device=dev)
    NCP, RP = _np2(nc), _np2(n_rows)
    _scan_kernel[(n_rows,)](runs, tot, nc, NCP=NCP, num_warps=1 if NCP <= 256 else 4)
    CS = max(1, _WIDE_HOT // RP)
    _rows_kernel[(max(n_rows, -(-n_chunks // CS)),)](
        runs, tot, ptr, chunk_ptr, chunks, nc, n_rows, n_chunks, size, RP=RP, NCP=NCP, CS=CS, num_warps=4
    )
    fns[1](*args)  # runs now holds each (row, chunk)'s offset into entries
    return BlockPlan(ptr, entries, chunks, chunk_ptr, n_chunks, -(-S // block_size))


def select_post_and_plan(
    topk_scores, topk_idx, argmax, boosted, block_size, want_plan, bounded_chunks=False, wide=False
):
    """Returns ``(block_indices int64, keys int32, keys_c int32, dead bool, table int32, plan)``.

    ``plan`` is the :class:`msa_token_bwd.BlockPlan` of ``block_indices`` when ``want_plan``;
    ``bounded_chunks`` (``msa_bounded_plan_chunks``) picks its chunk size as in ``build_block_plan``.

    ``wide`` (``msa_select_plan_wide``, shapes of :func:`wide_supported`): the same outputs
    without a ``[tokens, topk, blocks]`` one-hot or a program that reads the whole run table.
    The bookkeeping kernel does not count; the plan's counting sort is the two FlyDSL passes of
    :func:`msa_token_bwd.build_inverted_table` (LDS counters), and the eager ops between and
    after them (cumsum, cat, ``plan_chunks``) are :func:`_scan_kernel` and :func:`_rows_kernel`.
    """
    from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
        BlockPlan,
        plan_chunk_size,
    )

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
    NBP, KP, TT = _np2(NB), _np2(K), 16
    if wide:
        NBP = 2  # no one-hot over the blocks: the counting part is compiled out
        runs = table  # placeholder, not written (COUNT off)
    else:
        runs = torch.empty(n_rows * nc, dtype=torch.int32, device=dev)
    _post_kernel[(BG, nc)](
        topk_scores.contiguous(),
        topk_idx.contiguous(),
        argmax.contiguous(),
        boosted.contiguous(),
        bi,
        keys,
        keys_c,
        dead.view(torch.int8),
        table,
        runs,
        S,
        nc,
        tc,
        NB=NB,
        NBP=NBP,
        K=K,
        KP=KP,
        TT=TT,
        COUNT=not wide,
        num_warps=4,
    )
    if wide:
        plan = None
        if want_plan:
            plan = _wide_plan(table, NB, block_size, bounded_chunks)
        return bi, keys, keys_c, dead, table, plan
    plan = None
    if want_plan:
        max_tokens = BG * S * K
        size = plan_chunk_size(max_tokens, bounded=bool(bounded_chunks))  # as plan_chunks
        n_chunks = n_rows + -(-max_tokens // size)
        entries = torch.empty(max_tokens, dtype=torch.int32, device=dev)
        ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
        chunk_ptr = torch.empty(n_rows + 1, dtype=torch.int32, device=dev)
        chunks = torch.empty((n_chunks, 3), dtype=torch.int32, device=dev)
        n_prog = BG * nc
        cpp = _np2(-(-n_chunks // n_prog))
        _place_kernel[(BG, nc)](
            table,
            runs,
            entries,
            ptr,
            chunk_ptr,
            chunks,
            S,
            nc,
            tc,
            n_rows,
            n_chunks,
            size,
            NB=NB,
            NBP=NBP,
            K=K,
            KP=KP,
            TT=TT,
            RP=_np2(n_rows),
            NCP=_np2(nc),
            CPP=cpp,
            num_warps=8,
        )
        plan = BlockPlan(ptr, entries, chunks, chunk_ptr, n_chunks, -(-S // block_size))
    return bi, keys, keys_c, dead, table, plan


@triton.jit
def _bwd_g_kernel(
    grad_ptr,
    bi_ptr,
    dead_ptr,
    g_ptr,
    R,
    NB: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    BR: tl.constexpr,
):
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
    _bwd_g_kernel[(triton.cdiv(R, BR),)](
        grad_scores.contiguous(),
        block_indices,
        dead.view(torch.int8),
        g,
        R,
        NB=NB,
        K=K,
        KP=_np2(K),
        BR=BR,
        num_warps=4,
    )
    return g
