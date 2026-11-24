###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Gradients written straight into ``param.main_grad`` by the kernel that produces them.

A fused backward that owns a parameter's gradient claims it in the forward (``claim_main_grad``: flags
``grad_added_to_main_grad`` like the Primus-Turbo fused-wgrad linears, so the DDP hook does not add it again) and, in
the backward, writes the gradient into ``main_grad`` (first write of the step: overwrite, else accumulate, following
Primus-Turbo's ``main_grad_first_write`` policy) and hands autograd a dummy. ``_GradSlot`` resolves the destination
of one gradient: the main_grad when it is eligible, else a fresh tensor returned to autograd.
"""

from __future__ import annotations

import torch


def direct_main_grad(p):
    """``p.main_grad`` when the gradient of parameter ``p`` can be written into it directly, else None."""
    mg = getattr(p, "main_grad", None)
    if (
        isinstance(p, torch.nn.Parameter)
        and hasattr(p, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.is_cuda
        and mg.shape == p.shape
        and mg.is_contiguous()
        and mg.dtype in (torch.bfloat16, torch.float32)
        and p.dtype in (torch.bfloat16, torch.float32)
        and not getattr(p, "shared", False)
    ):
        return mg
    return None


def claim_main_grad(p) -> bool:
    """Forward-time claim (as rmsnorm_g90 / Turbo fused wgrad): the backward writes p.main_grad itself."""
    if p is None or direct_main_grad(p) is None:
        return False
    p.grad_added_to_main_grad = True
    p._primus_fwd_claim = True
    return True


class _GradSlot:
    """Where one parameter's gradient goes: its main_grad (direct) or a fresh tensor returned to autograd."""

    __slots__ = ("param", "out", "acc", "direct")

    def __init__(self, param, like_dtype, shape, device, claimed):
        self.param = param
        mg = direct_main_grad(param) if (claimed and param is not None) else None
        if mg is not None:
            from primus_turbo.pytorch.ops.utils import main_grad_first_write

            self.out = mg
            self.acc = not main_grad_first_write(param)
            self.direct = True
        else:
            self.out = torch.empty(shape, dtype=like_dtype, device=device)
            self.acc = False
            self.direct = False

    def grad(self, pdtype):
        if self.direct:
            from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad

            return _get_dummy_wgrad(list(self.param.shape), self.param.dtype)
        return self.out
