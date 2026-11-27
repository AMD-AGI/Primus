###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in gfx1250 FlyDSL forward GEMM for an FP32 MoE router (``moe_router_fwd_flydsl: true``).

Megatron's ``RouterGatingLinearFunction.forward`` computes the FP32 logits with
:mod:`primus.backends.megatron.core.extensions._flydsl.router_gemm` whenever the kernel takes the operands: an
FP32 router with BF16 input and weight, no bias, token count, expert count and hidden size multiples of 128.
Every other call keeps Megatron's forward and logs why once. The saved tensors are Megatron's, so the backward, and
``moe_router_bwd_bf16`` with it, is unchanged. Not installed together with ``moe_router_triton_gemm``, or off
gfx1250, or without FlyDSL.
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.gpu_arch import is_gfx1250
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_TAG = "[Patch:megatron.moe.router_fwd_flydsl]"


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_fwd_flydsl", False))


def _not_installed_reason(ctx: PatchContext):
    if getattr(get_args(ctx), "moe_router_triton_gemm", False):
        return "moe_router_triton_gemm is set (the two options are mutually exclusive)"
    if not is_gfx1250():
        return "the GPU is not a gfx1250"
    if importlib.util.find_spec("flydsl") is None:
        return "FlyDSL is not installed"
    return None


@register_patch(
    "megatron.moe.router_fwd_flydsl",
    backend="megatron",
    phase="before_train",
    description="gfx1250 FlyDSL forward GEMM for the FP32 MoE router",
    condition=_enabled,
)
def patch_router_fwd_flydsl(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    if getattr(fn, "_primus_router_fwd_flydsl_patched", False):
        return
    why = _not_installed_reason(ctx)
    if why is not None:
        warning_rank_0(f"{_TAG} not installed: {why}")
        return
    from primus.backends.megatron.core.extensions._flydsl import router_gemm

    orig_forward = fn.forward
    declined = []

    def forward(ctx_, inp, weight, bias, router_dtype):
        reason = router_gemm.router_logits_unsupported_reason(inp, weight, bias, router_dtype)
        if reason is not None:
            if not declined:
                declined.append(reason)
                log_rank_0(f"{_TAG} a call keeps Megatron's forward: {reason}")
            return orig_forward(ctx_, inp, weight, bias, router_dtype)
        ctx_.save_for_backward(inp, weight, bias)
        ctx_.router_dtype = router_dtype
        ctx_.input_dtype = inp.dtype
        ctx_.weight_dtype = weight.dtype
        return router_gemm.flydsl_router_logits(inp, weight).view(*inp.shape[:-1], -1)

    fn.forward = staticmethod(forward)
    fn._primus_router_fwd_flydsl_patched = True
    log_rank_0(f"{_TAG} FP32 router logits on the gfx1250 FlyDSL GEMM")
