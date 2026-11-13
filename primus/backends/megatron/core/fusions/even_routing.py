###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""``PrimusTopKRouter._force_even_routing`` in one forward and one backward launch (plus ATen's topk).

Eager: ``arange % E`` slots, ``zeros_like`` + ``ones_like`` + ``scatter_`` for the routing
map, ``topk`` of the scores, ``zeros_like`` + ``scatter_`` for the new scores (nine
launches); backward ``gather`` (ScatterBackward) + ``zeros`` + ``scatter`` (TopkBackward).

Here ATen's ``topk`` is kept as is (its values and, for the backward, its tie order), and
one Triton program per token row writes the new scores row (the k-th top value at slot
``(t * topk + k) % E``, zeros elsewhere) and the boolean routing-map row. The backward
writes ``grad_scores[t, idx[t, k]] = grad[t, slot(t, k)]`` and zeros elsewhere -- the same
values the gather / scatter pair produces (slots and top-k indices are distinct per row).

``PRIMUS_FUSED_EVEN_ROUTING=0`` keeps the eager code.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_EVEN_ROUTING", "1") != "0"


@triton.jit
def _fwd(VALS, NEW, MAP, CNT, E, NTOT, TOPK: tl.constexpr, BLOCK_E: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    cm = cols < E
    if t == 0:
        # tokens per expert of the new map: slots 0..T*topk-1 cycle over the experts
        q = NTOT // E
        rem = NTOT - q * E
        tl.store(CNT + cols, (q + (cols < rem).to(tl.int64)).to(tl.int64), mask=cm)
    row = tl.zeros([BLOCK_E], dtype=NEW.dtype.element_ty)
    hit = tl.zeros([BLOCK_E], dtype=tl.int1)
    base = t.to(tl.int64) * TOPK
    for k in tl.static_range(TOPK):
        v = tl.load(VALS + base + k)
        s = ((base + k) % E).to(tl.int32)
        sel = cols == s
        row = tl.where(sel, v.to(NEW.dtype.element_ty), row)
        hit = hit | sel
    out = t.to(tl.int64) * E + cols
    tl.store(NEW + out, row, mask=cm)
    tl.store(MAP + out, hit.to(tl.int8), mask=cm)


@triton.jit
def _bwd(G, IDX, GS, E, stride_g, TOPK: tl.constexpr, BLOCK_E: tl.constexpr):
    t = tl.program_id(0)
    cols = tl.arange(0, BLOCK_E)
    cm = cols < E
    row = tl.zeros([BLOCK_E], dtype=GS.dtype.element_ty)
    base = t.to(tl.int64) * TOPK
    for k in tl.static_range(TOPK):
        s = (base + k) % E
        gv = tl.load(G + t.to(tl.int64) * stride_g + s)
        ix = tl.load(IDX + base + k).to(tl.int32)
        row = tl.where(cols == ix, gv.to(GS.dtype.element_ty), row)
    tl.store(GS + t.to(tl.int64) * E + cols, row, mask=cm)


class _EvenRouting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, topk, map_dtype):
        T, E = scores.shape
        vals, idx = torch.topk(scores, topk, dim=1)
        vals = vals.contiguous()
        new = torch.empty_like(scores, memory_format=torch.contiguous_format)
        rmap = torch.empty(T, E, dtype=torch.bool, device=scores.device)
        cnt = torch.empty(E, dtype=torch.int64, device=scores.device)
        _fwd[(T,)](vals, new, rmap.view(torch.int8), cnt, E, T * topk, TOPK=topk, BLOCK_E=triton.next_power_of_2(E))
        ctx.counts = cnt
        ctx.save_for_backward(idx)
        ctx.shape = (T, E)
        ctx.topk = topk
        ctx.dtype = scores.dtype
        ctx.mark_non_differentiable(rmap)
        # no zero gradient for the boolean map (autograd would fill a [T, E] bool tensor)
        ctx.set_materialize_grads(False)
        return new, rmap

    @staticmethod
    def backward(ctx, g, _gmap):
        (idx,) = ctx.saved_tensors
        T, E = ctx.shape
        if g is None:
            return None, None, None
        if g.stride(-1) != 1:
            g = g.contiguous()
        gs = torch.empty(T, E, dtype=ctx.dtype, device=g.device)
        _bwd[(T,)](g, idx.contiguous(), gs, E, g.stride(0), TOPK=ctx.topk, BLOCK_E=triton.next_power_of_2(E))
        return gs, None, None


def eligible(scores, routing_map, topk: int) -> bool:
    return (
        ENABLED
        and scores.is_cuda
        and scores.dim() == 2
        and scores.is_contiguous()
        and scores.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and routing_map.dtype == torch.bool
        and routing_map.shape == scores.shape
        and 0 < topk <= scores.shape[1] <= 4096
    )


def force_even_routing(scores, topk):
    new, rmap = _EvenRouting.apply(scores, topk, torch.bool)
    from primus.backends.megatron.core.fusions import routing_counts

    if routing_counts.ENABLED and new.grad_fn is not None:
        routing_counts.register(rmap, new.grad_fn.counts)
    return new, rmap
