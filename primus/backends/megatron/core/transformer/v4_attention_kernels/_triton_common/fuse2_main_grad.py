# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""fuse2 line: weight gradients written straight into ``main_grad`` (no param.grad + DDP hook copy).

``linear_main_grad(x, w)`` (PRIMUS_FUSE2_COMP_MG=1, used by the V4 compressor's ``wkv_gate``): the same three GEMMs
as ``F.linear`` + autograd's ``MmBackward0`` (forward ``x2 @ w^T``; dgrad ``g2 @ w``; wgrad ``g2^T @ x2``, the
layout ``mm_mat2_backward`` picks for a transposed ``mat2``), with the wgrad's output being ``w.main_grad`` on the
first write of the step (wgrad_beta0 policy) and ``main_grad.add_(dW)`` otherwise -- exactly what the DDP hook does
with ``param.grad``. Requires ``main_grad`` with ``w``'s dtype / shape (bf16 main_grads in the DSv4 recipe).
"""
from __future__ import annotations

import os

import torch


def fuse2_linear_mg_ok(x: torch.Tensor, w: torch.Tensor, env: str = "PRIMUS_FUSE2_COMP_MG") -> bool:
    if os.environ.get(env, "0") != "1":
        return False
    mg = getattr(w, "main_grad", None)
    return (
        torch.is_grad_enabled()
        and isinstance(w, torch.nn.Parameter)
        and w.requires_grad
        and hasattr(w, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.shape == w.shape
        and mg.dtype == w.dtype
        and mg.is_contiguous()
        and w.is_contiguous()
        and w.dim() == 2
        and x.is_cuda
        and x.dtype == w.dtype
        and x.is_contiguous()
        and not getattr(w, "shared", False)
    )


class AttGradSlot:
    """PRIMUS_FUSE2_ATT_GADD: the attention input feeds the wq_a||wkv concat GEMM and the compressor's wkv_gate.
    The compressor backward (autograd runs it first) leaves its dX here; the concat backward's dX GEMM adds into that
    buffer in its epilogue (dsv4up accumulate=2: bf16(product) + buffer, one rounding -- autograd's bf16 add), so the
    separate add kernel goes away."""

    __slots__ = ("g", "fused")

    def __init__(self):
        self.g = self.fused = None

    def take(self, M, K, dtype):
        g, self.g = self.g, None
        if g is None or g.dtype != dtype or not g.is_contiguous() or g.numel() != M * K:
            return None
        return g.view(M, K)

    def untake(self, g):
        self.g = g


class AttFanOut(torch.autograd.Function):
    """``x -> (x_concat, x_rest)`` aliases so the two input gradients meet here, not in autograd's add."""

    @staticmethod
    def forward(ctx, x, slot):
        ctx.slot = slot
        return x.view_as(x), x.view_as(x)

    @staticmethod
    def backward(ctx, g_cat, g_rest):
        slot, ctx.slot = ctx.slot, None
        fused = slot.fused if slot is not None else None
        if slot is not None:
            slot.fused = slot.g = None
        if fused is not None:
            fp = fused.data_ptr()
            if g_cat is None or g_rest is None or g_cat.data_ptr() != fp or g_rest.data_ptr() != fp:
                # the concat's dX was added into a compressor dX that is not the whole x_rest gradient: the
                # in-place sum cannot be undone, so refuse instead of returning a wrong gradient
                raise RuntimeError("PRIMUS_FUSE2_ATT_GADD: unexpected extra consumer of the attention input")
            return g_cat, None
        if g_cat is None:
            return g_rest, None
        if g_rest is None:
            return g_cat, None
        return g_rest + g_cat, None  # autograd's order: compressor first, then the concat


def att_gadd_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSE2_ATT_GADD", "0") == "1"


class _LinearMainGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, slot=None):
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.mm(x2, w.t())
        ctx.save_for_backward(x2, w)
        ctx.wparam = w
        ctx.slot = slot
        ctx.lead = x.shape[:-1]
        w.grad_added_to_main_grad = True  # set in forward, like Turbo's fused-wgrad linears
        return out.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

        x2, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        dx = g2.mm(w).view(*ctx.lead, w.shape[1]) if ctx.needs_input_grad[0] else None
        if ctx.slot is not None and dx is not None:
            ctx.slot.g, ctx.slot = dx, None  # PRIMUS_FUSE2_ATT_GADD: the concat dX GEMM adds into this buffer
        wp, ctx.wparam = ctx.wparam, None
        mg = wp.main_grad
        if main_grad_first_write(wp):
            torch.mm(g2.t(), x2, out=mg)
        else:
            mg.add_(torch.mm(g2.t(), x2))
        return dx, _get_dummy_wgrad(list(w.shape), w.dtype), None


def linear_main_grad(x: torch.Tensor, w: torch.Tensor, slot=None) -> torch.Tensor:
    w._primus_fwd_claim = True  # grad_added_to_main_grad is flagged already in forward
    return _LinearMainGradFn.apply(x, w, slot)
