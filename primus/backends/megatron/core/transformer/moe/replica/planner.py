###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Placement planners for dynamic redundant experts.

A plan says, for every routed expert ``e`` and every EP rank ``d``, how many of ``e``'s routed rows rank ``d``
computes (``alloc[e, d]``). Expert ``e`` is owned by rank ``e // L`` (``L`` local experts per rank); a positive
``alloc[e, d]`` with ``d`` not the owner is a replica, held in one of rank ``d``'s ``B`` slots (``slot_expert``).
Every rank computes the same plan from the same all-gathered counts, so no plan is ever communicated.

Planners
--------
``moonep``  every rank computes exactly ``cap`` rows (MoonEP: fill each receiver from one home group, then split
            the largest remaining experts). Needs ``B >= L``.
``ultraep`` quota placement (UltraEP): the smallest per-rank load ``tau`` for which overloaded ranks can shed rows to
            ranks below ``tau`` within the slot budget, a replica getting at least ``min_rows`` rows. With
            ``replica_cost`` every new replica also counts that many rows on its rank (MegaFlux's
            ``max(load_r + lambda * n_r)``: the weight push and the gradient reduce in units of expert rows).
``fixed``   re-split rows among the replicas of an existing slot table (no new replica, no weight push).

``alloc_fractions`` / ``resplit_alloc`` keep a plan's per-expert shares and apply them to new counts on the device,
so later micro-batches of a step reuse the step's slots without a host round trip.

All loops break ties toward the lower index, so plans are deterministic. The planners run on the host for every
micro-batch, so they work on Python ints (a per-element torch access costs microseconds) and return tensors.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ReplicaPlan:
    """``alloc[e, d]``: rows of expert ``e`` computed on rank ``d``; ``slot_expert[d, c]``: expert in slot ``c`` of
    rank ``d`` (-1 if empty)."""

    alloc: torch.Tensor
    slot_expert: torch.Tensor

    @property
    def num_ranks(self) -> int:
        return self.slot_expert.shape[0]

    @property
    def num_slots(self) -> int:
        return self.slot_expert.shape[1]

    @property
    def num_local_experts(self) -> int:
        return self.alloc.shape[0] // self.num_ranks

    def rank_loads(self) -> torch.Tensor:
        return self.alloc.sum(dim=0)

    def num_replicas(self) -> int:
        return int((self.slot_expert >= 0).sum())


def owner_alloc(expert_count: torch.Tensor, num_ranks: int) -> torch.Tensor:
    """The no-replica allocation: every expert on its owner."""
    E = expert_count.numel()
    L = E // num_ranks
    alloc = torch.zeros(E, num_ranks, dtype=torch.int64)
    alloc[torch.arange(E), torch.arange(E) // L] = expert_count.to(torch.int64)
    return alloc


def _alloc_tensor(own: List[int], R: int, moves: Dict[Tuple[int, int], int]) -> torch.Tensor:
    """Dense ``[E, R]`` allocation from the rows left on each owner and the moved ``(expert, rank) -> rows``."""
    alloc = owner_alloc(torch.tensor(own, dtype=torch.int64), R)
    if moves:
        idx = torch.tensor(list(moves.keys()), dtype=torch.int64)
        alloc[idx[:, 0], idx[:, 1]] = torch.tensor(list(moves.values()), dtype=torch.int64)
    return alloc


def slots_from_alloc(alloc: torch.Tensor, num_slots: int) -> torch.Tensor:
    """Slot table of an allocation: rank ``d``'s replicas in increasing expert order."""
    E, R = alloc.shape
    L = E // R
    rows = alloc.tolist()
    table = []
    for d in range(R):
        remote = [e for e in range(E) if rows[e][d] > 0 and e // L != d]
        if len(remote) > num_slots:
            raise ValueError(f"rank {d} needs {len(remote)} replica slots, only {num_slots} available")
        table.append(remote + [-1] * (num_slots - len(remote)))
    return torch.tensor(table, dtype=torch.int64).view(R, num_slots)


def _argmax(v: List[int]) -> int:
    return max(range(len(v)), key=v.__getitem__)


def _argmin(v: List[int]) -> int:
    return min(range(len(v)), key=v.__getitem__)


def plan_moonep(counts: torch.Tensor, cap: Optional[int] = None) -> torch.Tensor:
    """MoonEP's allocation. ``counts``: ``[R, E]`` rows per source rank and expert; ``cap``: rows each rank computes
    (default: total / R, which must divide evenly)."""
    R, E = counts.shape
    L = E // R
    cnt = counts.sum(dim=0).tolist()
    total = sum(cnt)
    if cap is None:
        if total % R:
            raise ValueError("moonep needs the same number of routed rows on every rank")
        cap = total // R
    balance = [sum(cnt[h * L : (h + 1) * L]) - cap for h in range(R)]
    # each receiver is filled at once from the most overloaded home group (which may overshoot and receive later),
    # so a receiver's remote experts all come from one group and fit in L slots
    quota = [[0] * R for _ in range(R)]
    while True:
        h, u = _argmax(balance), _argmin(balance)
        if balance[h] <= 0:
            break
        move = -balance[u]
        quota[h][u] = move
        balance[h] -= move
        balance[u] = 0
    own = list(cnt)
    moves: Dict[Tuple[int, int], int] = {}
    for h in range(R):
        remaining = cnt[h * L : (h + 1) * L]
        q = list(quota[h])
        while True:
            d = _argmax(q)
            if q[d] <= 0:
                break
            le = _argmax(remaining)
            take = min(remaining[le], q[d])
            e = h * L + le
            moves[(e, d)] = moves.get((e, d), 0) + take
            own[e] -= take
            remaining[le] -= take
            q[d] -= take
    return _alloc_tensor(own, R, moves)


def _shed(cnt, group, R, tau, slots, new_replicas, min_rows, block, replica_cost=0):
    """Try to bring every rank to at most ``tau`` rows. ``cnt``: rows per expert; ``group``: rows per owner rank;
    ``slots``: ``R`` lists of ``B`` already hosted replicas (-1 empty); with ``new_replicas`` a rank may also open a
    free slot for a new expert, which adds ``replica_cost`` to its load. Returns ``(own, moves, slots)`` or ``None``
    if some rank stays above ``tau``."""
    E = len(cnt)
    L = E // R
    own = list(cnt)
    moves: Dict[Tuple[int, int], int] = {}
    load = list(group)
    slots = [list(s) for s in slots]
    hosted = [set(e for e in s if e >= 0) for s in slots]
    free = [s.count(-1) for s in slots]
    over = sorted((r for r in range(R) if load[r] > tau), key=lambda r: (-(load[r] - tau), r))
    for r in over:
        exc = load[r] - tau
        experts = sorted(range(r * L, (r + 1) * L), key=lambda e: (-own[e], e))
        for e in experts:
            while exc > 0 and own[e] > 0:
                t, best = -1, None
                for c in range(R):
                    if c == r or not (e in hosted[c] or (new_replicas and free[c] > 0)):
                        continue
                    if load[c] + (0 if e in hosted[c] else replica_cost) < tau:
                        if best is None or load[c] < best:
                            t, best = c, load[c]
                if t < 0:
                    break
                opens = e not in hosted[t]
                need = max(1, min_rows) if opens else 1
                room = tau - load[t] - (replica_cost if opens else 0)
                delta = min(exc, room, own[e])
                if delta < need:
                    # shed past the excess rather than leave a remainder below min_rows (which would make the
                    # feasibility of tau non-monotone and break the binary search)
                    delta = min(need, room, own[e])
                if block > 1 and delta < own[e]:
                    delta -= delta % block
                if delta < need:
                    break
                if opens:
                    slots[t][slots[t].index(-1)] = e
                    hosted[t].add(e)
                    free[t] -= 1
                    load[t] += replica_cost
                own[e] -= delta
                moves[(e, t)] = moves.get((e, t), 0) + delta
                load[r] -= delta
                load[t] += delta
                exc -= delta
            if exc <= 0:
                break
        if exc > 0:
            return None
    return own, moves, slots


def _smallest_tau(
    expert_count: torch.Tensor,
    R: int,
    slots: torch.Tensor,
    new_replicas: bool,
    min_rows: int,
    block: int,
    replica_cost: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cnt = expert_count.tolist()
    L = len(cnt) // R
    group = [sum(cnt[h * L : (h + 1) * L]) for h in range(R)]
    table = slots.tolist()
    hi = max(group)
    lo = -(-sum(cnt) // R)
    best = _shed(cnt, group, R, hi, table, new_replicas, min_rows, block, replica_cost)
    while lo < hi:
        mid = (lo + hi) // 2
        res = _shed(cnt, group, R, mid, table, new_replicas, min_rows, block, replica_cost)
        if res is None:
            lo = mid + 1
        else:
            best, hi = res, mid
    own, moves, table = best
    return _alloc_tensor(own, R, moves), torch.tensor(table, dtype=torch.int64).view(slots.shape)


def plan_ultraep(
    counts: torch.Tensor,
    num_slots: int,
    min_rows: int = 1,
    block: int = 1,
    min_gain: float = 0.0,
    replica_cost: int = 0,
) -> ReplicaPlan:
    """UltraEP-style quota plan with ``num_slots`` replica slots per rank. With ``min_gain`` no replica is built
    unless the plan lowers the busiest rank (counting ``replica_cost`` rows per replica it hosts) by at least
    ``min_gain`` times the mean load."""
    R, E = counts.shape
    expert_count = counts.sum(dim=0).to(torch.int64)
    empty = torch.full((R, num_slots), -1, dtype=torch.int64)
    alloc, slots = _smallest_tau(expert_count, R, empty, True, min_rows, block, replica_cost)
    if min_gain > 0 or replica_cost > 0:
        L = E // R
        raw = int(expert_count.view(R, L).sum(dim=1).max())
        cost = alloc.sum(dim=0) + replica_cost * (slots >= 0).sum(dim=1)
        if raw - int(cost.max()) < max(min_gain * int(expert_count.sum()) / R, 1e-9):
            return ReplicaPlan(alloc=owner_alloc(expert_count, R), slot_expert=empty)
    return ReplicaPlan(alloc=alloc, slot_expert=slots)


def plan_fixed(counts: torch.Tensor, slot_expert: torch.Tensor, block: int = 1) -> ReplicaPlan:
    """Re-split rows among the replicas already in ``slot_expert``; no new replica."""
    R, E = counts.shape
    expert_count = counts.sum(dim=0).to(torch.int64)
    alloc, _ = _smallest_tau(expert_count, R, slot_expert, False, 1, block)
    return ReplicaPlan(alloc=alloc, slot_expert=slot_expert.clone())


def make_plan(
    planner: str,
    counts: torch.Tensor,
    num_slots: int,
    min_rows: int = 1,
    block: int = 1,
    slot_expert: Optional[torch.Tensor] = None,
    min_gain: float = 0.0,
    replica_cost: int = 0,
) -> ReplicaPlan:
    """``planner`` in ``moonep`` / ``ultraep`` / ``fixed`` (``fixed`` needs ``slot_expert``; ``min_gain`` and
    ``replica_cost`` are ultraep's)."""
    if planner == "moonep":
        alloc = plan_moonep(counts)
        return ReplicaPlan(alloc=alloc, slot_expert=slots_from_alloc(alloc, num_slots))
    if planner == "ultraep":
        return plan_ultraep(
            counts, num_slots, min_rows=min_rows, block=block, min_gain=min_gain, replica_cost=replica_cost
        )
    if planner == "fixed":
        if slot_expert is None:
            raise ValueError("planner 'fixed' needs slot_expert")
        return plan_fixed(counts, slot_expert, block=block)
    raise ValueError(f"unknown planner {planner!r}")


def resplit_if_balanced(
    counts: torch.Tensor, slot_expert: torch.Tensor, num_slots: int, block: int, replan_above: float
) -> Optional[ReplicaPlan]:
    """The ``fixed`` re-split of ``counts`` among the replicas of ``slot_expert`` if it leaves the busiest rank at
    most ``replan_above`` times the mean load, else None (plan anew)."""
    kept = make_plan("fixed", counts, num_slots, block=block, slot_expert=slot_expert)
    loads = kept.rank_loads().to(torch.float64)
    return kept if loads.max().item() <= replan_above * loads.mean().item() else None


def alloc_fractions(alloc: torch.Tensor, num_local: int) -> torch.Tensor:
    """``[E, R]`` inclusive cumulative share of every expert's rows per destination rank (owner only for an expert
    without rows)."""
    E, R = alloc.shape
    a = alloc.to(torch.float64)
    cnt = a.sum(dim=1, keepdim=True)
    owner = torch.zeros_like(a)
    owner[torch.arange(E), torch.arange(E) // num_local] = 1.0
    frac = torch.where(cnt > 0, a / cnt.clamp(min=1), owner)
    cum = frac.cumsum(dim=1)
    cum[:, -1] = 1.0
    return cum


def resplit_alloc(cum_frac: torch.Tensor, expert_count: torch.Tensor) -> torch.Tensor:
    """``[E, R]`` rows per destination: ``expert_count`` split in the shares ``cum_frac`` (exact integer sums, only
    destinations with a positive share get rows)."""
    # the epsilon keeps the first plan's own counts exact under float64 rounding of a / cnt
    bounds = torch.floor(cum_frac * expert_count.to(torch.float64).unsqueeze(1) + 1e-6).to(torch.int64)
    bounds[:, -1] = expert_count
    return torch.diff(bounds, dim=1, prepend=torch.zeros_like(bounds[:, :1]))
