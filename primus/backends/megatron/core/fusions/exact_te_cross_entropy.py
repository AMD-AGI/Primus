###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer Engine's cross entropy (``cross_entropy_fusion_impl: te``) without its FP32
gradient buffer, bitwise equal to it.

TE's ``CrossEntropyFunction`` on one tensor-parallel rank runs, for logits ``[S, B, V]``:

1. ``online_softmax_kernel``: per row the max ``m`` and ``d = sum(exp(x - m))`` (reads the logits);
2. ``cross_entropy_kernel``: reads the logits again, writes the FP32 gradient
   ``exp(x - m) / d - eps`` (``x_y`` gets ``-(1 - label_smoothing)`` more) and the loss;
3. backward ``element_mul_kernel``: multiplies that FP32 buffer in place by the loss gradient,
4. and autograd casts it to the logits' BF16.

At V = 200k and 4k tokens that is a 3.3 GB FP32 buffer written once, read and written once, and
read once more for the cast. Here step 1 is TE's own kernel (so ``m`` and ``d`` are bit-identical),
the loss is computed from ``m``, ``d`` and ``x_y`` alone, and the backward recomputes the same FP32
expression from the BF16 logits, multiplies by the loss gradient and rounds to BF16 once -- the
same FP32 operations in the same order as steps 2-4, so loss and gradient are bitwise equal.
The logits (BF16) are kept for the backward instead of the FP32 gradient: half the memory.

Only the single-rank case (no tensor-parallel group, or a group of size 1) with
``reduce_loss=False`` takes this path; everything else calls TE.
"""

from __future__ import annotations

import os

import torch

_kernels = None
# PRIMUS_EXACT_CE_PIPELINED=0: TE's own online_softmax_kernel and TE's backward tile.
_PIPELINED = os.environ.get("PRIMUS_EXACT_CE_PIPELINED", "1") != "0"
_BWD_BLOCK = 8192
_BWD_WARPS = 4


def _get_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _loss_kernel(Y_ptr, MDX_ptr, LOSS_ptr, n_rows, ignore_idx, BLOCK: tl.constexpr):
        rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n_rows
        y = tl.load(Y_ptr + rows, mask=mask, other=ignore_idx)
        m = tl.load(MDX_ptr + rows * 3, mask=mask, other=0.0)
        d = tl.load(MDX_ptr + rows * 3 + 1, mask=mask, other=1.0)
        x_y = tl.load(MDX_ptr + rows * 3 + 2, mask=mask, other=0.0)
        # TE cross_entropy_kernel: loss = -(ori_X_y - m - tl.log(d)); ignored rows keep the zero
        loss = -(x_y - m - tl.log(d))
        loss = tl.where(y == ignore_idx, 0.0, loss)
        tl.store(LOSS_ptr + rows, loss, mask=mask)

    @triton.jit
    def _bwd_kernel(
        X_ptr,
        X_stride,
        G_ptr,
        G_stride,
        Y_ptr,
        MDX_ptr,
        GO_ptr,
        GO_stride,
        n_cols,
        ignore_idx,
        label_smoothing: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # Same loop shape and FP32 expression as TE's cross_entropy_kernel (with the same
        # BLOCK_SIZE / num_warps) so the compiler emits the same arithmetic; element_mul's
        # multiply and autograd's BF16 cast are folded into the store.
        program_id = tl.program_id(0).to(tl.int64)
        X_ptr += program_id * X_stride
        G_ptr += program_id * G_stride
        y = tl.load(Y_ptr + program_id)
        grad_output = tl.load(GO_ptr + program_id * GO_stride)
        if y == ignore_idx:
            for i in range(0, n_cols, BLOCK_SIZE):
                X_offsets = i + tl.arange(0, BLOCK_SIZE)
                zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                tl.store(G_ptr + X_offsets, (zero * grad_output).to(G_ptr.dtype.element_ty), mask=X_offsets < n_cols)
            return
        m = tl.load(MDX_ptr + program_id * 3)
        d = tl.load(MDX_ptr + program_id * 3 + 1)
        eps = label_smoothing / n_cols
        for i in range(0, n_cols, BLOCK_SIZE):
            X_offsets = i + tl.arange(0, BLOCK_SIZE)
            X_block = tl.load(X_ptr + X_offsets, mask=X_offsets < n_cols, other=float("-inf"))
            X_block = X_block.to(tl.float32)
            X_block = tl.exp(X_block - m) / d - eps
            X_block = tl.where(X_offsets == y, X_block + (-(1 - label_smoothing)), X_block)
            tl.store(G_ptr + X_offsets, (X_block * grad_output).to(G_ptr.dtype.element_ty), mask=X_offsets < n_cols)

    @triton.jit
    def _online_softmax_pipelined_kernel(
        X_ptr, X_stride, Y_ptr, MDX_ptr, n_cols, ignore_idx, BLOCK_SIZE: tl.constexpr
    ):
        # TE's online_softmax_kernel (same BLOCK_SIZE / num_warps, so the same load layout and
        # the same tl.max / tl.sum reduction trees, the same per-chunk FP32 update of m and d),
        # with the next chunk's load issued before the current chunk's reductions. TE's loop
        # waits for each chunk's load after the previous chunk's reductions (latency bound,
        # ~2.5 TB/s). Its n_non_ignore atomic is dropped (unused on this path).
        program_id = tl.program_id(0).to(tl.int64)
        X_ptr += program_id * X_stride
        y = tl.load(Y_ptr + program_id)
        X_y = float("-inf")
        if y >= 0:
            if y < n_cols:
                X_y = tl.load(X_ptr + y).to(tl.float32)
        m = float("-inf")
        d = 0.0
        offs = tl.arange(0, BLOCK_SIZE)
        x_next = tl.load(X_ptr + offs, mask=offs < n_cols, other=float("-inf"))
        for i in range(0, n_cols, BLOCK_SIZE):
            X_block = x_next.to(tl.float32)
            n_offs = i + BLOCK_SIZE + offs
            x_next = tl.load(X_ptr + n_offs, mask=n_offs < n_cols, other=float("-inf"))
            block_max = tl.max(X_block)
            m_new = tl.maximum(m, block_max)
            d = d * tl.exp(m - m_new) + tl.sum(tl.exp(X_block - m_new))
            m = m_new
        tl.store(MDX_ptr + program_id * 3, m)
        tl.store(MDX_ptr + program_id * 3 + 1, d)
        tl.store(MDX_ptr + program_id * 3 + 2, X_y)

    _kernels = (_loss_kernel, _bwd_kernel, _online_softmax_pipelined_kernel)
    return _kernels


class ExactTECrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, label_smoothing, ignore_idx):
        import triton
        from transformer_engine.common.triton.cross_entropy import online_softmax_kernel
        from transformer_engine.pytorch.triton import cross_entropy as te_ce

        B, SQ, V = inp.shape
        n_rows = B * SQ
        BLOCK_SIZE = min(te_ce.MAX_FUSED_SIZE, triton.next_power_of_2(V))
        if inp.stride(-1) != 1 or inp.stride(-2) != inp.shape[-1]:
            inp = inp.contiguous()
        if target.stride(-1) != 1:
            target = target.contiguous()
        if _PIPELINED:
            m_d_X_y = torch.empty(n_rows * 3, dtype=torch.float32, device=inp.device)
            _get_kernels()[2][(n_rows,)](
                inp, inp.stride(-2), target, m_d_X_y, V, ignore_idx,
                BLOCK_SIZE=BLOCK_SIZE, num_warps=te_ce.NUM_WARPS,
            )
        else:
            m_d_X_y = torch.zeros(n_rows * 3, dtype=torch.float32, device=inp.device)
            n_non_ignore = torch.zeros(1, dtype=torch.int64, device=inp.device)
            online_softmax_kernel[(n_rows,)](
                X_ptr=inp,
                X_stride=inp.stride(-2),
                Y_ptr=target,
                Y_stride=target.stride(-1),
                m_d_X_y_ptr=m_d_X_y,
                m_d_X_y_stride=m_d_X_y.stride(-1),
                rank=0,
                n_cols=V,
                ignore_idx=ignore_idx,
                n_non_ignore=n_non_ignore,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=te_ce.NUM_WARPS,
            )
        loss_1d = torch.empty(n_rows, dtype=torch.float32, device=inp.device)
        loss_kernel = _get_kernels()[0]
        loss_kernel[(triton.cdiv(n_rows, 1024),)](target, m_d_X_y, loss_1d, n_rows, ignore_idx, BLOCK=1024)
        ctx.save_for_backward(inp, target, m_d_X_y)
        ctx.label_smoothing = float(label_smoothing)
        ctx.ignore_idx = int(ignore_idx)
        return torch.reshape(loss_1d, (B, SQ))

    @staticmethod
    def backward(ctx, grad_output):
        import triton

        inp, target, m_d_X_y = ctx.saved_tensors
        B, SQ, V = inp.shape
        n_rows = B * SQ
        grad_input = torch.empty_like(inp)
        if grad_output.numel() > 1:
            go = grad_output.contiguous()
            go_stride = 1
        else:
            go = grad_output.reshape(1).to(torch.float32)
            go_stride = 0
        bwd_kernel = _get_kernels()[1]
        from transformer_engine.pytorch.triton import cross_entropy as te_ce

        bwd_kernel[(n_rows,)](
            inp,
            inp.stride(-2),
            grad_input,
            grad_input.stride(-2),
            target,
            m_d_X_y,
            go,
            go_stride,
            V,
            ctx.ignore_idx,
            label_smoothing=ctx.label_smoothing,
            # Elementwise per logit (no reduction), so the tile does not touch the bits; 8192 x 4
            # warps runs at the copy rate (TE's 32768 x 16 warps: 470 -> 398 us at 4096 x 200k).
            BLOCK_SIZE=min(_BWD_BLOCK if _PIPELINED else te_ce.MAX_FUSED_SIZE, triton.next_power_of_2(V)),
            num_warps=_BWD_WARPS if _PIPELINED else te_ce.NUM_WARPS,
        )
        return grad_input, None, None, None


def exact_te_parallel_cross_entropy(logits, labels, tp_group, is_cg_capturable=False, fallback=None):
    """Drop-in for Megatron's ``te_parallel_cross_entropy`` (label_smoothing 0, reduce_loss False,
    ignore_idx -100)."""
    single = tp_group is None or torch.distributed.get_world_size(tp_group) == 1
    if (
        single
        and not is_cg_capturable
        and logits.dim() == 3
        and logits.dtype in (torch.bfloat16, torch.float16)
        and logits.is_cuda
    ):
        return ExactTECrossEntropy.apply(logits, labels, 0.0, -100)
    return fallback(logits, labels, tp_group, is_cg_capturable)
