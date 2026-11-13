###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-step derived norm weights, built in one launch for the whole model.

Zero-centred RMSNorm weights are applied as ``w + 1``: one tiny add kernel before
every norm call (~25 per step on MiniMax-M3), plus ``w.float() + 1.0`` (two more)
for each MSA indexer's input norm. The weights only change in the optimizer step,
so every derived weight of a step can be made at once:

* bf16 ``w + 1``: one ``torch._foreach_add(ws, 1)`` -- the same opmath add and
  bf16 rounding per element as ``w + 1``;
* fp32 ``w.float() + 1.0``: one cat of the bf16 weights, one cast, one add --
  elementwise, so batching does not change a bit.

The rebuild happens when the *leader* -- the first member ever asked for a derived
weight, i.e. the first norm of the forward -- asks again, which is once per
forward pass (also correct, just redundant, under recompute or several
micro-batches). A cached entry is returned only for the very tensor it was made
from; anything else (first step, a member registered late) is computed eagerly
by the caller, as before.

``PRIMUS_STEP_WEIGHT_CACHE=0`` turns it off.
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, List, Optional, Tuple

import torch

ENABLED = os.environ.get("PRIMUS_STEP_WEIGHT_CACHE", "1") != "0"

BF16_PLUS_ONE = "bf16+1"  # w + 1 in w's dtype
F32_PLUS_ONE = "f32+1"  # w.float() + 1.0


class _StepWeightCache:
    def __init__(self):
        self.members: Dict[Tuple[int, str], Tuple[weakref.ref, str]] = {}
        self.leader: Optional[Tuple[int, str]] = None
        self.cache: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _rebuild(self):
        self.cache = {}
        groups: Dict[Tuple[str, torch.device, torch.dtype], List[Tuple[Tuple[int, str], torch.Tensor]]] = {}
        dead = []
        for key, (ref, kind) in self.members.items():
            t = ref()
            if t is None:
                dead.append(key)
                continue
            groups.setdefault((kind, t.device, t.dtype), []).append((key, t))
        for key in dead:
            del self.members[key]
        with torch.no_grad():
            for (kind, _dev, _dt), items in groups.items():
                ts = [t.detach() for _, t in items]
                if kind == BF16_PLUS_ONE:
                    outs = torch._foreach_add(ts, 1)
                else:
                    flat = torch.cat([t.reshape(-1) for t in ts]).float()
                    flat.add_(1.0)
                    outs = [o.view(t.shape) for o, t in zip(flat.split([t.numel() for t in ts]), ts)]
                for (key, t), o in zip(items, outs):
                    self.cache[key] = (t, o)

    def get(self, tensor: torch.Tensor, kind: str) -> Optional[torch.Tensor]:
        """The step's derived weight of ``tensor``, or None (caller computes it eagerly)."""
        key = (id(tensor), kind)
        if key not in self.members:
            self.members[key] = (weakref.ref(tensor), kind)
            if self.leader is None:
                self.leader = key
            return None
        if key == self.leader:
            self._rebuild()
        entry = self.cache.get(key)
        if entry is None or entry[0] is not tensor:
            return None
        return entry[1]


_CACHE = _StepWeightCache()


class _PlusOne(torch.autograd.Function):
    """``w + 1`` whose value comes from the step cache; the gradient passes through
    unchanged, exactly as ``AddBackward0`` does."""

    @staticmethod
    def forward(ctx, w, holder):
        return holder[0].detach()

    @staticmethod
    def backward(ctx, g):
        return g, None


def plus_one(weight: torch.Tensor) -> torch.Tensor:
    """``weight + 1`` (differentiable in ``weight``)."""
    if ENABLED:
        cached = _CACHE.get(weight, BF16_PLUS_ONE)
        if cached is not None:
            return _PlusOne.apply(weight, [cached])
    return weight + 1


def float_plus_one(weight: torch.Tensor) -> torch.Tensor:
    """``weight.detach().float() + 1.0`` (no gradient)."""
    if ENABLED:
        cached = _CACHE.get(weight, F32_PLUS_ONE)
        if cached is not None:
            return cached
    return weight.detach().float() + 1.0
