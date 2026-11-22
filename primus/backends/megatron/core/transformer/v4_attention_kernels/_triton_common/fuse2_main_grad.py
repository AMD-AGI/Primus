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


class _LinearMainGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.mm(x2, w.t())
        ctx.save_for_backward(x2, w)
        ctx.wparam = w
        ctx.lead = x.shape[:-1]
        w.grad_added_to_main_grad = True  # set in forward, like Turbo's fused-wgrad linears
        return out.view(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

        x2, w = ctx.saved_tensors
        g2 = g.reshape(-1, g.shape[-1])
        dx = g2.mm(w).view(*ctx.lead, w.shape[1]) if ctx.needs_input_grad[0] else None
        wp, ctx.wparam = ctx.wparam, None
        mg = wp.main_grad
        if main_grad_first_write(wp):
            torch.mm(g2.t(), x2, out=mg)
        else:
            mg.add_(torch.mm(g2.t(), x2))
        return dx, _get_dummy_wgrad(list(w.shape), w.dtype)


def linear_main_grad(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    w._primus_fwd_claim = True  # grad_added_to_main_grad is flagged already in forward
    return _LinearMainGradFn.apply(x, w)
