###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Word-embedding weight gradient added into ``main_grad`` for the touched rows only, bit for bit.

ATen's embedding backward writes a dense ``[vocab, hidden]`` gradient (a fill of the whole
table plus the rows the batch touched), which Megatron's DDP hook then adds into ``main_grad``:
three passes over the table per micro-batch for at most ``tokens`` non-zero rows.

Here the backward runs ATen's own ``embedding_dense_backward`` on the batch's ids renumbered by
rank among the distinct ids (sort, flag, cumsum: ATen ops, no host sync). That gives the same
rows -- the same segments in the same order, so the same sums and roundings -- in a
``[tokens, hidden]`` table, and one Triton launch adds those rows into ``main_grad`` with the
hook's addition (``main + grad`` in FP32, rounded to ``main_grad``'s dtype). Rows the batch
does not touch would add +0.0 in the dense path; ``main_grad`` starts at +0.0 after its zeroing
and a round-to-nearest sum is -0.0 only when both addends are -0.0, so the buffer never holds
-0.0 and ``x + 0.0`` is ``x``. The gradient buffer, its zeroing, the reduction across ranks and
every non-trivial addition are unchanged; the work scales with the batch's distinct tokens,
whatever their distribution.

Megatron's ``grad_added_to_main_grad`` protocol (as with TE's fused wgrad accumulation) tells the
DDP hook that the gradient is already in the buffer. The protocol drops any other gradient of
the same parameter, so the caller must know that only the embedding lookup uses the weight (no
tied output layer).

Depends on ATen's ``embedding_dense_backward`` and BF16 addition, so a torch change can break it
silently: ``bitwise_check`` compares it, before its first use for a shape, with the dense
gradient of ``F.embedding`` added by ``main_grad.add_``, and the caller keeps that path on a
mismatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions import bitwise_check

_BLOCK = 1024


@triton.jit
def _add_rows(MAIN, ROWS, COMPACT, COUNT, H, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    blk = tl.program_id(1)
    if i < tl.load(COUNT):
        cols = blk * BLOCK + tl.arange(0, BLOCK)
        m = cols < H
        dst = tl.load(ROWS + i).to(tl.int64) * H + cols
        a = tl.load(MAIN + dst, mask=m, other=0.0).to(tl.float32)
        g = tl.load(COMPACT + i.to(tl.int64) * H + cols, mask=m, other=0.0).to(tl.float32)
        tl.store(MAIN + dst, (a + g).to(MAIN.dtype.element_ty), mask=m)


def compact_ids(ids: torch.Tensor):
    """``(inverse, unique_padded, count)`` with ``ids == unique_padded[inverse]``: the ids
    renumbered by rank among the distinct ids (order-preserving), the sorted distinct ids
    padded with zeros to ``ids.numel()``, and their count as a one-element tensor."""
    flat = ids.reshape(-1)
    n = flat.numel()
    sorted_ids, perm = torch.sort(flat)
    new_seg = torch.empty(n, dtype=torch.long, device=flat.device)
    new_seg[:1] = 0
    torch.ne(sorted_ids[1:], sorted_ids[:-1], out=new_seg[1:])
    seg = torch.cumsum(new_seg, 0)
    inverse = torch.empty_like(seg).scatter_(0, perm, seg)
    unique = torch.zeros_like(seg).scatter_(0, seg, sorted_ids.long())
    count = seg[-1:] + 1
    return inverse.view(ids.shape), unique, count


def add_embedding_grad_(main_grad_2d: torch.Tensor, grad_output: torch.Tensor, ids: torch.Tensor) -> None:
    """``main_grad_2d += embedding_dense_backward(grad_output, ids, vocab)`` for the rows in
    ``ids`` only. ``main_grad_2d`` ``[vocab, hidden]`` contiguous FP32 or BF16 without -0.0 in
    the other rows; ``grad_output`` ``[*ids.shape, hidden]``."""
    inverse, unique, count = compact_ids(ids)
    n, hidden = unique.numel(), main_grad_2d.shape[1]
    compact = torch.ops.aten.embedding_dense_backward(grad_output, inverse, n, -1, False).contiguous()
    _add_rows[(n, triton.cdiv(hidden, _BLOCK))](
        main_grad_2d, unique, compact, count, hidden, BLOCK=_BLOCK, num_warps=4
    )


class _RowsGradEmbedding(torch.autograd.Function):
    """``F.embedding(ids, weight)`` whose weight gradient goes straight into ``weight.main_grad``."""

    @staticmethod
    def forward(ctx, ids, weight):
        ctx.save_for_backward(ids)
        ctx.weight = weight
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, grad_output):
        (ids,) = ctx.saved_tensors
        weight = ctx.weight
        # what the self-check covered (its key holds weight.dtype and the id count): anything
        # else is a caller this path was not written for
        if grad_output.dtype != weight.dtype or grad_output.shape != (*ids.shape, weight.shape[1]):
            raise RuntimeError(
                "embedding_rows_grad: expected a gradient of dtype "
                f"{weight.dtype} and shape {(*ids.shape, weight.shape[1])}, got {grad_output.dtype} "
                f"{tuple(grad_output.shape)}"
            )
        if getattr(weight, "_primus_held_grads", None):
            # grad_accum_group holds dense gradients of earlier micro-batches (a step that
            # changed route): they come first
            from primus.backends.megatron.core.fusions.grouped_grad_accum import (
                flush_held,
            )

            flush_held(weight)
        add_embedding_grad_(weight.main_grad.view(weight.shape), grad_output.contiguous(), ids)
        weight.grad_added_to_main_grad = True
        # Megatron's convention for a gradient already in main_grad (TE's fused wgrad
        # accumulation does the same): an uninitialised placeholder the DDP hook drops.
        return None, torch.empty(weight.shape, dtype=weight.dtype, device=weight.device)


def eligible(ids: torch.Tensor, weight: torch.Tensor) -> bool:
    """Shapes, dtypes and DDP state the row-wise path covers; decided before any work."""
    main = getattr(weight, "main_grad", None)
    return (
        torch.is_grad_enabled()
        and weight.requires_grad
        and main is not None
        and hasattr(weight, "grad_added_to_main_grad")
        and not getattr(weight, "zero_out_wgrad", False)
        and main.dtype in (torch.float32, torch.bfloat16)
        and main.is_contiguous()
        and main.numel() == weight.numel()
        and main.device == weight.device
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and weight.dim() == 2
        and ids.is_cuda
        and ids.dtype in (torch.int64, torch.int32)
        and ids.numel() > 0
    )


def _probe(n: int, hidden: int, main_dtype, weight_dtype, device):
    """Ids with repeats and with gaps, a gradient with -0.0 entries and exact cancellations, and
    a buffer that already holds other micro-batches' sums (some rows still +0.0)."""
    vocab = 2 * n + 3
    gen = torch.Generator(device=device)
    gen.manual_seed(1234)
    # Zipf-like: about half the positions repeat a few hundred frequent ids
    u = torch.rand(n, generator=gen, device=device)
    ids = torch.where(
        u < 0.5,
        (u * 2 * min(n, 300)).long(),
        torch.randint(0, vocab, (n,), generator=gen, device=device),
    )
    g = torch.randn(n, hidden, generator=gen, device=device).to(weight_dtype)
    g[1::7, : hidden // 2] = -0.0
    same = ids[0]
    ids[1] = same  # rows 0 and 1 hit one id with opposite gradients: an exact +0.0 sum
    g[1] = -g[0]
    main = torch.randn(vocab, hidden, generator=gen, device=device).to(main_dtype)
    main[::3] = 0.0
    return ids, g, main, vocab


def _run_probe(n, hidden, main_dtype, weight_dtype, device):
    ids, g, main, _ = _probe(n, hidden, main_dtype, weight_dtype, device)
    add_embedding_grad_(main, g, ids)
    return [main]


def _reference_probe(n, hidden, main_dtype, weight_dtype, device):
    ids, g, main, vocab = _probe(n, hidden, main_dtype, weight_dtype, device)
    weight = torch.zeros(vocab, hidden, dtype=weight_dtype, device=device, requires_grad=True)
    with torch.enable_grad():
        out = F.embedding(ids, weight)  # VocabParallelEmbedding's lookup on one TP rank
    (dense,) = torch.autograd.grad(out, weight, g)
    main.add_(dense)  # the DDP backward hook
    return [main]


def checked(ids: torch.Tensor, weight: torch.Tensor) -> bool:
    """The self-check's verdict for this id count, width and dtypes (ATen picks its embedding
    backward kernel by the id count). Probes are built only when the check runs."""
    n, hidden = ids.numel(), weight.shape[1]
    main_dtype, device = weight.main_grad.dtype, weight.device
    return bitwise_check.passed(
        "embedding_rows_grad",
        lambda: _run_probe(n, hidden, main_dtype, weight.dtype, device),
        lambda: _reference_probe(n, hidden, main_dtype, weight.dtype, device),
        key=(n, hidden, str(main_dtype), str(weight.dtype)),
    )


def embedding(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``F.embedding(ids, weight)`` with the row-wise gradient. Call only when ``eligible`` and
    ``checked`` hold and nothing else uses ``weight`` in the graph."""
    return _RowsGradEmbedding.apply(ids, weight)
