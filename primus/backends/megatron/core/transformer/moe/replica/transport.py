###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Slot transports: expert weights from their owners into replica slots, slot gradients back to the owners.

A transport only moves data. Which slots to fill or return is decided by ``ReplicaSlots`` from slot tables every rank
agrees on, so every call is collective: all ranks of the EP group make it in the same order with the same tables.
A slot table is an ``[R, B]`` int64 tensor of expert ids (-1: empty slot): rank ``d`` holds the experts
``table[d]``, and expert ``e`` is owned by rank ``e // L``.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

import torch
import torch.distributed as dist


class Pending:
    """A transfer in flight; ``wait`` completes it (a second ``wait`` does nothing)."""

    def wait(self) -> None:
        pass


class SlotTransport(ABC):
    """Moves slot weights owner -> replica and slot gradients replica -> owner within one EP group."""

    def __init__(self, num_ranks: int, rank: int, num_local_experts: int):
        self.num_ranks = num_ranks
        self.rank = rank
        self.num_local_experts = num_local_experts

    def alloc(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        """A zeroed slot buffer. A transport whose peers access the slots directly allocates it where they can."""
        return torch.zeros(shape, dtype=dtype, device=device)

    @abstractmethod
    def push_weights(self, own: torch.Tensor, storage, held: torch.Tensor, table: torch.Tensor) -> Pending:
        """Copy the owners' experts ``own`` ``[L, ...]`` into ``storage.w`` for the slots of ``table`` whose expert
        differs from ``held``. ``wait`` returns once this rank's slots hold the weights."""

    @abstractmethod
    def return_grads(self, storage, held: torch.Tensor, owner_grad: Callable[[], torch.Tensor]) -> Pending:
        """Return ``storage.main_grad`` of the slots in ``held``. ``wait`` adds them into ``owner_grad()``
        ``[L, ...]`` in (source rank, slot) order and clears this rank's returned slots."""


class _NcclPending(Pending):
    def __init__(self, reqs, recv=(), owner_grad=None, clear=None):
        self.reqs = reqs
        self.recv: List[Tuple[int, int, int, torch.Tensor]] = list(recv)
        self.owner_grad = owner_grad
        self.clear = clear

    def wait(self) -> None:
        if self.reqs is None:
            return
        for req in self.reqs:
            req.wait()
        self.reqs = None
        if self.recv:
            grad = self.owner_grad()
            for _, _, local, t in sorted(self.recv, key=lambda r: (r[0], r[1])):
                grad[local] += t.to(grad.dtype)
            self.recv = []
        if self.clear is not None:
            self.clear()
            self.clear = None


class NcclTransport(SlotTransport):
    """Point-to-point through the EP group's process group: owners send weights, replicas send gradients."""

    def __init__(self, group, num_local_experts: int):
        super().__init__(group.size(), group.rank(), num_local_experts)
        self.group = group

    def _peer(self, r: int) -> int:
        return dist.get_global_rank(self.group, r)

    def push_weights(self, own, storage, held, table) -> Pending:
        L, ops = self.num_local_experts, []
        held_rows, rows = held.tolist(), table.tolist()
        for d in range(self.num_ranks):
            for c, e in enumerate(rows[d]):
                if e < 0 or e == held_rows[d][c]:
                    continue
                h = e // L
                if self.rank == h:
                    ops.append(dist.P2POp(dist.isend, own[e % L], self._peer(d), group=self.group))
                if self.rank == d:
                    ops.append(dist.P2POp(dist.irecv, storage.w[c], self._peer(h), group=self.group))
        return _NcclPending(dist.batch_isend_irecv(ops) if ops else [])

    def return_grads(self, storage, held, owner_grad) -> Pending:
        L, ops, recv = self.num_local_experts, [], []
        rows = held.tolist()
        g = storage.main_grad
        for d in range(self.num_ranks):
            for c, e in enumerate(rows[d]):
                if e < 0:
                    continue
                h = e // L
                if self.rank == d:
                    ops.append(dist.P2POp(dist.isend, g[c], self._peer(h), group=self.group))
                if self.rank == h:
                    t = torch.empty_like(g[0])
                    ops.append(dist.P2POp(dist.irecv, t, self._peer(d), group=self.group))
                    recv.append((d, c, e % L, t))
        mine = [c for c, e in enumerate(rows[self.rank]) if e >= 0]

        def clear():
            g[mine] = 0

        return _NcclPending(
            dist.batch_isend_irecv(ops) if ops else [], recv, owner_grad, clear if mine else None
        )
