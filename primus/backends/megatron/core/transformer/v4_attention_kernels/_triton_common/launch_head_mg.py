# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""launch line, PRIMUS_LAUNCH_HEAD_MG=1 (default off): HyperHead parameter gradients written straight into their bf16
``main_grad`` (no autograd fp32 -> bf16 cast kernel + DDP hook copy per parameter).

``head_pre(mixes, scale, base, eps)`` == ``torch.sigmoid(mixes * scale + base) + eps`` with autograd's own backward ops
(sigmoid_backward, ``sum_to_size`` of the broadcast inputs, ``grad * other``), so every value is bit-identical; the
two parameter gradients go through ``main_grad.copy_(fp32)`` (one cast kernel: the same RNE as ``.to(bf16)``) on the
first write of the step (wgrad_beta0 policy), or the original ``add_(bf16)`` otherwise.
``head_weight_mg(weight)`` lets ``HcRmsLinearFn`` do the same for ``fn.weight``.
"""
from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return os.environ.get("PRIMUS_LAUNCH_HEAD_MG", "0") == "1"


def mg_param_ok(p) -> bool:
    mg = getattr(p, "main_grad", None)
    return (
        isinstance(p, torch.nn.Parameter)
        and p.requires_grad
        and p.dtype == torch.bfloat16
        and hasattr(p, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.dtype == torch.bfloat16
        and mg.is_contiguous()
        and tuple(mg.shape) == tuple(p.shape)
        and not getattr(p, "shared", False)
    )


def claim(p) -> None:
    p._primus_fwd_claim = True  # grad_added_to_main_grad is flagged already in forward
    p.grad_added_to_main_grad = True


def write_mg(p, g32: torch.Tensor):
    """``main_grad`` <- bf16(g32) (first write) or += bf16(g32); returns the dummy autograd gradient."""
    from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

    mg = p.main_grad
    if main_grad_first_write(p):
        mg.copy_(g32.view(mg.shape))
    else:
        mg.add_(g32.view(mg.shape).to(p.dtype))
    return _get_dummy_wgrad(list(p.shape), p.dtype)


class _HeadPreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mixes, scale, base, eps):
        s = torch.sigmoid(mixes * scale + base)
        ctx.save_for_backward(mixes, s)
        ctx.params = (scale, base)
        claim(scale)
        claim(base)
        return s + eps

    @staticmethod
    def backward(ctx, dpre):
        mixes, s = ctx.saved_tensors
        scale, base = ctx.params
        ctx.params = None
        dz = torch.ops.aten.sigmoid_backward(dpre, s)
        gb = write_mg(base, dz.sum_to_size(base.shape))
        # MulBackward0 order: other (scale) first, then self (mixes)
        gs = write_mg(scale, (dz * mixes).sum_to_size(scale.shape))
        dmixes = dz * scale
        return dmixes, gs, gb, None


def head_pre(mixes, scale, base, eps):
    """sigmoid(mixes * scale + base) + eps, or None when the fused-main_grad path does not apply."""
    if not (enabled() and torch.is_grad_enabled() and mixes.requires_grad and mixes.dtype == torch.float32
            and mg_param_ok(scale) and mg_param_ok(base)):
        return None
    return _HeadPreFn.apply(mixes, scale, base, eps)


def head_weight_mg(weight) -> bool:
    """True: HcRmsLinearFn writes this weight's gradient into main_grad itself (flags set here, in forward)."""
    if not (enabled() and torch.is_grad_enabled() and mg_param_ok(weight)):
        return False
    claim(weight)
    return True
