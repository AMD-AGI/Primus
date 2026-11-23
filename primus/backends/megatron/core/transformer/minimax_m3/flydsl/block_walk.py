###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``msa_aligned_block_walk``: the order in which the gfx1250 MSA forward and dQ kernels walk a
token's selected blocks, chosen so that adjacent tokens meet their common blocks at the same step.

Those kernels run ``GROUP`` (forward) / 2 (dQ) adjacent tokens of a KV group per workgroup, in
lockstep, and stage a K/V tile once for every token that reads it *at the same step*
(``msa_shared_tile_staging``). The indexer's table lists a token's blocks by score, so two
neighbours that select the same blocks usually meet them at different steps and each stages its
own copy; the kernels are bound by that staging (L2 -> LDS).

The attention over a token's selected blocks does not depend on the order they are visited in,
only its rounding does. So this module permutes each token's table row -- never its contents:

* tokens are taken in groups of ``GROUP`` consecutive tokens (the forward's workgroup; the dQ
  kernel's pairs are halves of it);
* the first token of a group keeps its row;
* each later token puts a block that an earlier token of the group also selected at that token's
  position (the earliest such token; if two of its blocks want one position, the lower slot wins),
  and fills the remaining positions, in ascending order, with its remaining entries in slot order,
  valid blocks first, so ``-1`` padding stays behind the valid entries.

``aligned_block_walk`` returns the permuted table (``walk``, what the forward and dQ kernels read)
and ``slots`` (the original slot of each position), which maps the forward's per-position
``slot_lse`` / slot mass back to the indexer's slot order. Everything else (the BlockPlan, dK/dV,
the indexer loss and its backward) keeps the original table.

Relaxed numerics: the kernels then add a token's blocks in another order, so ``o``, ``lse``, the
slot mass and ``dq`` differ from the score-order walk in the last bits (the same kernels, the same
set of terms). Deterministic: the permutation is a pure integer function of the table. The Triton
kernel is compared with the PyTorch reference below on the first table of every shape
(``bitwise_check``; it also checks that every row of ``walk`` is a permutation of the table's row);
on a mismatch the run keeps the score-order walk. No barrier, wait or loop depends on the table
here or in the kernels: their trip counts stay functions of the grid position.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

# tokens aligned with each other: the forward kernel's workgroup (msa_token_fwd_w32.FWD_TOKENS_PER_WG)
GROUP = 4


@triton.jit
def _match(t, a, cols):
    """Position in row ``a`` of each entry of ``t`` (``[BR, KP]`` each), or -1: ``a``'s valid
    entries are distinct, so at most one position matches."""
    hit = (a[:, None, :] == t[:, :, None]) & (t[:, :, None] >= 0)
    return tl.sum(tl.where(hit, cols[None, None, :] + 1, 0), axis=2) - 1


@triton.jit
def _place(t, want, cols, cm):
    """Positions ``[BR, KP]`` of the slots of ``t`` given their wanted positions (-1: none)."""
    # two slots wanting one position: the lower slot keeps it
    same = (want[:, :, None] == want[:, None, :]) & (want[:, :, None] >= 0)
    dup = tl.sum((same & (cols[None, None, :] < cols[None, :, None])).to(tl.int32), axis=2) > 0
    want = tl.where(dup, -1, want)
    taken = tl.sum((want[:, :, None] == cols[None, None, :]).to(tl.int32), axis=1) > 0
    free = (~taken) & cm[None, :]
    rank_f = tl.sum((free[:, None, :] & (cols[None, None, :] < cols[None, :, None])).to(tl.int32), axis=2)
    un = (want < 0) & cm[None, :]
    valid = t >= 0
    # unplaced slots in walk order: valid entries first, then by slot
    before = un[:, None, :] & (
        (valid[:, None, :] & (~valid[:, :, None]))
        | ((valid[:, None, :] == valid[:, :, None]) & (cols[None, None, :] < cols[None, :, None]))
    )
    rank_u = tl.sum(before.to(tl.int32), axis=2)
    fill = tl.sum(
        tl.where(free[:, None, :] & (rank_f[:, None, :] == rank_u[:, :, None]), cols[None, None, :], 0),
        axis=2,
    )
    return tl.where(un, fill, want)


@triton.jit
def _row_at(t, pos, cols):
    """``t`` laid out by position: ``out[pos[s]] = t[s]``."""
    return tl.sum(tl.where(pos[:, :, None] == cols[None, None, :], t[:, :, None], 0), axis=1)


@triton.jit
def _align_kernel(tbl_ptr, walk_ptr, slots_ptr, NG, K: tl.constexpr, KP: tl.constexpr, BR: tl.constexpr):
    # one program: BR groups of 4 consecutive table rows
    grp = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, KP)
    cm = cols < K
    m = (grp < NG)[:, None] & cm[None, :]
    base = grp[:, None] * (4 * K)
    slot = cols[None, :] + tl.zeros([BR, KP], tl.int32)

    w0 = tl.load(tbl_ptr + base + cols[None, :], mask=m, other=-1)
    tl.store(walk_ptr + base + cols[None, :], w0, mask=m)
    tl.store(slots_ptr + base + cols[None, :], slot.to(tl.int64), mask=m)

    t1 = tl.load(tbl_ptr + base + K + cols[None, :], mask=m, other=-1)
    p1 = _place(t1, _match(t1, w0, cols), cols, cm)
    tl.store(walk_ptr + base + K + p1, t1, mask=m)
    tl.store(slots_ptr + base + K + p1, slot.to(tl.int64), mask=m)
    w1 = _row_at(t1, p1, cols)

    t2 = tl.load(tbl_ptr + base + 2 * K + cols[None, :], mask=m, other=-1)
    a0 = _match(t2, w0, cols)
    p2 = _place(t2, tl.where(a0 >= 0, a0, _match(t2, w1, cols)), cols, cm)
    tl.store(walk_ptr + base + 2 * K + p2, t2, mask=m)
    tl.store(slots_ptr + base + 2 * K + p2, slot.to(tl.int64), mask=m)
    w2 = _row_at(t2, p2, cols)

    t3 = tl.load(tbl_ptr + base + 3 * K + cols[None, :], mask=m, other=-1)
    a0 = _match(t3, w0, cols)
    a1 = tl.where(a0 >= 0, a0, _match(t3, w1, cols))
    p3 = _place(t3, tl.where(a1 >= 0, a1, _match(t3, w2, cols)), cols, cm)
    tl.store(walk_ptr + base + 3 * K + p3, t3, mask=m)
    tl.store(slots_ptr + base + 3 * K + p3, slot.to(tl.int64), mask=m)


def _reference(table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The alignment of the module docstring in plain PyTorch (the first-call check's reference
    and the unit tests'). ``table``: ``[B, H, S, K]`` int32, ``S`` a multiple of ``GROUP``."""
    B, H, S, K = table.shape
    t = table.reshape(-1, GROUP, K).to(torch.int64)
    N = t.shape[0]
    dev = t.device
    cols = torch.arange(K, device=dev)
    walk = torch.empty_like(t)
    slots = torch.empty_like(t)
    walk[:, 0] = t[:, 0]
    slots[:, 0] = cols
    lower = cols.view(1, 1, K) < cols.view(1, K, 1)  # [., s, s']: s' < s
    for i in range(1, GROUP):
        ti = t[:, i]
        want = torch.full((N, K), -1, dtype=torch.int64, device=dev)
        for j in range(i):
            hit = (walk[:, j].unsqueeze(1) == ti.unsqueeze(2)) & (ti.unsqueeze(2) >= 0)
            pos = (hit * (cols + 1)).sum(-1) - 1
            want = torch.where(want >= 0, want, pos)
        same = (want.unsqueeze(2) == want.unsqueeze(1)) & (want.unsqueeze(2) >= 0)
        want = torch.where((same & lower).any(-1), -1, want)
        taken = (want.unsqueeze(2) == cols.view(1, 1, K)).any(1)
        free = ~taken
        rank_f = (free.unsqueeze(1) & lower).sum(-1)
        un = want < 0
        valid = ti >= 0
        before = un.unsqueeze(1) & (
            (valid.unsqueeze(1) & ~valid.unsqueeze(2)) | ((valid.unsqueeze(1) == valid.unsqueeze(2)) & lower)
        )
        rank_u = before.sum(-1)
        fill = ((free.unsqueeze(1) & (rank_f.unsqueeze(1) == rank_u.unsqueeze(2))) * cols).sum(-1)
        pos = torch.where(un, fill, want)
        walk[:, i] = torch.empty_like(ti).scatter_(1, pos, ti)
        slots[:, i] = torch.empty_like(ti).scatter_(1, pos, cols.expand(N, K))
    return walk.reshape(B, H, S, K).to(torch.int32), slots.reshape(B, H, S, K)


def _kernel(table: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, H, S, K = table.shape
    walk = torch.empty_like(table)
    slots = torch.empty(table.shape, dtype=torch.int64, device=table.device)
    n_groups = B * H * S // GROUP
    BR = 16
    _align_kernel[(triton.cdiv(n_groups, BR),)](
        table, walk, slots, n_groups, K=K, KP=triton.next_power_of_2(K), BR=BR, num_warps=4
    )
    return walk, slots


def eligible(table: torch.Tensor, block_size: int) -> bool:
    """Shapes the alignment serves: an int32 ``[B, H, S, K]`` table on the GPU whose ``S`` and
    block size the kernels' workgroups divide (otherwise they run one token per workgroup and
    share nothing), and rows short enough for the kernel's ``K x K`` compares."""
    return (
        table.is_cuda
        and table.dtype == torch.int32
        and table.dim() == 4
        and 1 < table.shape[-1] <= 32
        and table.shape[2] >= GROUP
        and table.shape[2] % GROUP == 0
        and block_size % GROUP == 0
    )


def _is_permutation(table, walk, slots) -> bool:
    return bool(torch.equal(table.gather(-1, slots), walk)) and bool(
        torch.equal(
            slots.sort(-1).values, torch.arange(table.shape[-1], device=table.device).expand_as(slots)
        )
    )


def aligned_block_walk(table: torch.Tensor, block_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """``(walk, slots)`` for an int32 ``[B, H, S, K]`` block table: ``walk`` is ``table`` with
    every row permuted as the module docstring describes and ``slots`` (int64) the permutation,
    ``walk == table.gather(-1, slots)``. None when the shape is not :func:`eligible` or the
    kernel's first result for this shape was not the reference's (the caller keeps ``table``)."""
    if not eligible(table, block_size):
        return None
    table = table.contiguous()

    def run():
        walk, slots = _kernel(table)
        # a row of walk must hold exactly the table row's entries, whatever the alignment did
        ok = torch.tensor(_is_permutation(table, walk, slots), device=table.device)
        return walk, slots, ok

    def reference():
        walk, slots = _reference(table)
        return walk, slots, torch.tensor(True, device=table.device)

    if not bitwise_check.passed("msa_aligned_block_walk", run, reference, key=tuple(table.shape)):
        return None
    return _kernel(table)
