###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Embedding lookup whose weight gradient is ATen's ``embedding_dense_backward``, bit for bit,
in one launch.

ATen (more than 3072 indices, no ``scale_grad_by_freq``): ``zeros`` of the whole
``[V, H]`` gradient (two fill launches for 32-bit indexing), a stable radix sort of the
indices, segment / partial-segment offsets (partials of ``NROWS_PER_THREAD = 10`` rows),
``compute_grad_weight`` (each partial: ``acc = 0; acc += g[row]`` in FP32, rows in
position order) and ``sum_and_scatter`` (``w = 0; w += partial`` in order, cast to the
weight dtype). About 16 launches, ~0.36 ms at V = 200192, H = 6144, 4096 tokens, of which
~0.19 ms is the 2.46 GB zero fill.

Here one Triton program per (token position, column block): a position that is the first
occurrence of its id walks every later position with the same id in order and reproduces
that exact sum (FP32, the same 10-row partial grouping, the same ``0 +`` starts); the
other positions write nothing.

With ``PRIMUS_FUSED_ADAM_DIRECT_GRAD`` (one rank, one micro-batch, Adam reads
``param.grad``) the gradient tensor is kept across steps: the same launch also zeroes the
rows the previous step wrote that this step does not, so the 2.46 GB fill disappears and
the tensor still holds exactly ATen's dense gradient. The gradient is then installed as
``weight.grad`` from the backward (autograd sees ``None`` for the weight); this is only
done when ``weight.grad`` is ``None`` at that point, i.e. nothing accumulates into it.

``PRIMUS_EXACT_EMBEDDING_GRAD=0`` disables it; ``PRIMUS_EXACT_EMBEDDING_GRAD_PERSIST=0``
keeps the kernel but returns a fresh dense gradient through autograd.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_EXACT_EMBEDDING_GRAD", "1") != "0"
NROWS_PER_THREAD = 10  # ATen EmbeddingBackwardKernel.cu
_SCAN = 1024
_BLOCK_H = 2048


@triton.jit
def _emb_grad_kernel(
    IDS,  # [N] int64, this step's indices
    PREV,  # [N] int64, the previous step's indices (HAS_PREV)
    SAVE,  # [N] int64, receives IDS (HAS_PREV / SAVE_IDS)
    G,  # [N, H] grad output
    W,  # [V, H] weight grad
    MASK,  # [V] uint8 row flags (HAS_MASK): 1 iff the row may be nonzero
    N,
    H,
    stride_g,
    HAS_PREV: tl.constexpr,
    SAVE_IDS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    NROWS: tl.constexpr,
    SCAN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    i = tl.program_id(0)
    hb = tl.program_id(1)
    cols = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    cm = cols < H
    my = tl.load(IDS + i)
    if SAVE_IDS:
        if hb == 0:
            tl.store(SAVE + i, my)
    if HAS_PREV:
        prv = tl.load(PREV + i)
    else:
        prv = my
    lanes = tl.arange(0, SCAN)
    dup_before = 0
    prev_live = 0
    for c in range(0, N, SCAN):
        j = c + lanes
        jm = j < N
        ids = tl.load(IDS + j, mask=jm, other=-1)
        dup_before += tl.sum(((ids == my) & (j < i)).to(tl.int32), axis=0)
        if HAS_PREV:
            prev_live += tl.sum((ids == prv).to(tl.int32), axis=0)
    if HAS_PREV:
        if prev_live == 0:
            # row written last step, untouched this step -> zero (idempotent if repeated)
            tl.store(W + prv.to(tl.int64) * H + cols, tl.zeros([BLOCK_H], dtype=W.dtype.element_ty), mask=cm)
            if HAS_MASK:
                if hb == 0:
                    tl.store(MASK + prv, 0)
    if dup_before == 0:
        if HAS_MASK:
            if hb == 0:
                tl.store(MASK + my, 1)
        total = tl.zeros([BLOCK_H], dtype=tl.float32)
        part = tl.zeros([BLOCK_H], dtype=tl.float32)
        cnt = 0
        for c in range((i // SCAN) * SCAN, N, SCAN):
            j = c + lanes
            ids = tl.load(IDS + j, mask=j < N, other=-1)
            hit = (ids == my) & (j >= i)
            nh = tl.sum(hit.to(tl.int32), axis=0)
            while nh > 0:
                r = tl.min(tl.where(hit, j, N + SCAN), axis=0)
                g = tl.load(G + r.to(tl.int64) * stride_g + cols, mask=cm, other=0.0).to(tl.float32)
                part = part + g
                cnt += 1
                if cnt == NROWS:
                    total = total + part
                    part = tl.zeros([BLOCK_H], dtype=tl.float32)
                    cnt = 0
                hit = hit & (j != r)
                nh -= 1
        if cnt > 0:
            total = total + part
        tl.store(W + my.to(tl.int64) * H + cols, total.to(W.dtype.element_ty), mask=cm)


def _launch(ids, prev, save, g2d, w, mask=None):
    N = ids.numel()
    H = g2d.shape[1]
    grid = (N, triton.cdiv(H, _BLOCK_H))
    has_prev = prev is not None
    _emb_grad_kernel[grid](
        ids,
        prev if has_prev else ids,
        save if save is not None else ids,
        g2d,
        w,
        mask if mask is not None else ids,
        N,
        H,
        g2d.stride(0),
        HAS_PREV=has_prev,
        SAVE_IDS=save is not None,
        HAS_MASK=mask is not None,
        NROWS=NROWS_PER_THREAD,
        SCAN=_SCAN,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )


def embedding_weight_grad(ids: torch.Tensor, grad: torch.Tensor, num_weights: int) -> torch.Tensor:
    """Dense ``[num_weights, H]`` gradient, bitwise ``torch.ops.aten.embedding_dense_backward``."""
    ids = ids.reshape(-1).contiguous()
    g2d = grad.reshape(ids.numel(), grad.shape[-1]).contiguous()
    w = torch.zeros(num_weights, g2d.shape[1], dtype=grad.dtype, device=grad.device)
    _launch(ids, None, None, g2d, w)
    return w


def _persist_enabled() -> bool:
    return (
        os.environ.get("PRIMUS_EXACT_EMBEDDING_GRAD_PERSIST", "1") != "0"
        and os.environ.get("PRIMUS_FUSED_ADAM_DIRECT_GRAD", "0") == "1"
    )


class _PersistentGrad:
    """The kept gradient tensor and the ids of the step that last wrote it (ping-pong)."""

    def __init__(self):
        self.buf = None
        self.ids = [None, None]
        self.cur = 0  # self.ids[cur] holds the last step's ids; None before the first step
        self.n = None
        self.mask = None  # [V] uint8: 1 iff the row may be nonzero (rows with 0 hold exact +0)


# Persistent embedding gradients and their row flags, by gradient data_ptr: consumers that
# read the whole gradient (the fused Adam, the exact l2norm) may skip rows whose flag is 0,
# which hold exactly +0 (PRIMUS_EMB_GRAD_ROW_SKIP=0 disables).
ROW_SKIP = os.environ.get("PRIMUS_EMB_GRAD_ROW_SKIP", "1") != "0"
ZERO_ROW_MASKS = {}


def zero_row_mask(t: torch.Tensor):
    """``(mask, row_len, base_ptr, nbytes)`` if ``t`` lies in a registered persistent gradient."""
    if not ZERO_ROW_MASKS:
        return None
    p = t.data_ptr()
    for base, (mask, row_len, nbytes) in ZERO_ROW_MASKS.items():
        if base <= p < base + nbytes:
            return mask, row_len, base, nbytes
    return None


class _ExactEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ids, weight, holder):
        ctx.save_for_backward(ids)
        ctx.weight = weight
        ctx.holder = holder
        return torch.nn.functional.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad):
        (ids,) = ctx.saved_tensors
        weight = ctx.weight
        holder = ctx.holder
        ctx.weight = None
        V, H = weight.shape
        flat = ids.reshape(-1)
        if flat.dtype != torch.int64:
            flat = flat.long()
        flat = flat.contiguous()
        N = flat.numel()
        g2d = grad.reshape(N, grad.shape[-1]).contiguous()
        if holder is not None and weight.grad is None and grad.dtype == weight.dtype:
            if holder.buf is None or holder.n != N or holder.buf.shape != weight.shape:
                if holder.buf is not None:
                    ZERO_ROW_MASKS.pop(holder.buf.data_ptr(), None)
                holder.buf = torch.zeros_like(weight, memory_format=torch.contiguous_format)
                holder.mask = torch.zeros(V, dtype=torch.uint8, device=weight.device) if ROW_SKIP else None
                if holder.mask is not None:
                    ZERO_ROW_MASKS[holder.buf.data_ptr()] = (holder.mask, H, holder.buf.numel() * holder.buf.element_size())
                holder.ids = [None, torch.empty(N, dtype=torch.int64, device=weight.device)]
                holder.cur = 0
                holder.n = N
                _launch(flat, None, holder.ids[1], g2d, holder.buf, holder.mask)
            else:
                nxt = 1 - holder.cur
                if holder.ids[nxt] is None:
                    holder.ids[nxt] = torch.empty(N, dtype=torch.int64, device=weight.device)
                _launch(flat, holder.ids[holder.cur], holder.ids[nxt], g2d, holder.buf, holder.mask)
            holder.cur = 1 - holder.cur
            weight.grad = holder.buf
            return None, None, None
        w = torch.zeros(V, H, dtype=grad.dtype, device=grad.device)
        _launch(flat, None, None, g2d, w)
        return None, w, None


def eligible(ids: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        ENABLED
        and ids.is_cuda
        and ids.dtype in (torch.int64, torch.int32)
        and weight.dim() == 2
        and weight.is_contiguous()
        and ids.numel() > 3072  # ATen's sort path (smaller counts use another kernel)
        and torch.is_grad_enabled()
        and weight.requires_grad
    )


def exact_embedding(ids: torch.Tensor, weight: torch.Tensor, owner=None) -> torch.Tensor:
    holder = None
    if owner is not None and _persist_enabled():
        holder = getattr(owner, "_primus_emb_grad_holder", None)
        if holder is None:
            holder = _PersistentGrad()
            owner._primus_emb_grad_holder = holder
    return _ExactEmbedding.apply(ids, weight, holder)
