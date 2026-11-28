###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Logical -> physical expert ids under a replica plan.

Physical experts: rank ``d`` hosts ``L + B`` of them, its ``L`` own experts (``d * (L + B) + e % L``) followed by its
``B`` slots (``d * (L + B) + L + c``). A dispatcher built for ``R * (L + B)`` experts then sends every row to the rank
that computes it without any other change.

Routes of expert ``e`` are numbered in one global order (source rank, then this rank's ``(token, k)`` order) and the
``n``-th route goes to the first rank ``d`` with ``alloc[:d + 1].sum() > n`` (destinations in rank order, as MoonEP).
Every rank derives the same split from the same plan and counts.
"""

from typing import Optional

import torch

from .planner import ReplicaPlan


def slot_lookup(plan: ReplicaPlan) -> torch.Tensor:
    """``[R, E]``: slot index of expert ``e`` on rank ``d``, or -1."""
    R, B = plan.slot_expert.shape
    E = plan.alloc.shape[0]
    table = [[-1] * E for _ in range(R)]
    for d, row in enumerate(plan.slot_expert.tolist()):
        for c, e in enumerate(row):
            if e >= 0:
                table[d][e] = c
    return torch.tensor(table, dtype=torch.int64).view(R, E)


def remap_topk(
    topk: torch.Tensor,
    counts: torch.Tensor,
    rank: int,
    plan: ReplicaPlan,
    lut: Optional[torch.Tensor] = None,
    check: bool = False,
) -> torch.Tensor:
    """Physical expert ids for this rank's routes.

    ``topk``: ``[S, K]`` logical experts of this rank's tokens (no -1); ``counts``: ``[R, E]`` routed rows per source
    rank and expert (the all-gathered ``bincount`` of every rank's ``topk``). Works on CPU or GPU tensors; ``plan`` and
    ``lut`` are moved to ``topk``'s device. Nothing is read back to the host unless ``check`` (verify that every
    replica route has a slot).
    """
    R, E = counts.shape
    L = E // R
    B = plan.num_slots
    if plan.num_replicas() == 0:
        flat = topk.to(torch.int64)
        return (flat // L) * (L + B) + flat % L
    if lut is None:
        lut = slot_lookup(plan)
    return remap_alloc(topk, counts, rank, plan.alloc, lut, B, check)


def remap_alloc(
    topk: torch.Tensor,
    counts: torch.Tensor,
    rank: int,
    alloc: torch.Tensor,
    lut: torch.Tensor,
    num_slots: int,
    check: bool = False,
) -> torch.Tensor:
    """``remap_topk`` from an allocation ``[E, R]`` and its slot lookup ``[R, E]`` (``slot_lookup``), for an
    allocation computed on the device (no ``ReplicaPlan`` on the host)."""
    dev = topk.device
    R, E = counts.shape
    L = E // R
    B = num_slots
    lut = lut.to(dev)
    counts = counts.to(dev, torch.int64)
    cum = alloc.to(dev, torch.int64).cumsum(dim=1)  # [E, R], inclusive over destinations

    flat = topk.reshape(-1).to(torch.int64)
    local = torch.bincount(flat, minlength=E)
    order = torch.argsort(flat, stable=True)
    starts = torch.cumsum(local, 0) - local
    pos = torch.empty_like(flat)
    pos[order] = torch.arange(flat.numel(), device=dev) - starts[flat[order]]
    ordinal = counts[:rank].sum(dim=0)[flat] + pos

    dest = (cum[flat] <= ordinal.unsqueeze(1)).sum(dim=1)
    owner = flat // L
    own_id = dest * (L + B) + flat % L
    slot = lut[dest, flat]
    rep_id = dest * (L + B) + L + slot
    phys = torch.where(dest == owner, own_id, rep_id)
    if check and bool((dest != owner).any()) and bool((slot[dest != owner] < 0).any()):
        raise RuntimeError("a route was sent to a rank that has no slot for its expert")
    return phys.view_as(topk)


def physical_counts(phys_topk: torch.Tensor, num_ranks: int, num_local: int, num_slots: int) -> torch.Tensor:
    """``[R, L + B]`` rows per physical expert from one rank's physical ids."""
    n = num_ranks * (num_local + num_slots)
    return torch.bincount(phys_topk.reshape(-1), minlength=n).view(num_ranks, num_local + num_slots)
