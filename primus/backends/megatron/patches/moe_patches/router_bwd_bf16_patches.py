###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in BF16 backward GEMMs for an FP32 MoE router.

Keep router logits in FP32 while using BF16 operands for dX and dW when
both the saved input and weight are BF16. The dW GEMM emits FP32 before
conversion to the parameter dtype, matching the MLPerf GPT-OSS patch.
Other precision combinations and the non-TE path retain Megatron's backward.
"""

from collections.abc import Callable

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_bwd_bf16", False))


def _uses_bf16_router_backward(ctx, te_general_gemm: Callable | None) -> bool:
    return (
        te_general_gemm is not None
        and ctx.router_dtype == torch.float32
        and ctx.input_dtype == torch.bfloat16
        and ctx.weight_dtype == torch.bfloat16
    )


def _router_backward_bf16(ctx, grad_output: torch.Tensor, *, te_general_gemm: Callable):
    inp, weight, bias = ctx.saved_tensors
    inp_shape = inp.shape
    inp = inp.view(-1, inp_shape[-1])
    grad_output = grad_output.view(-1, grad_output.shape[-1])
    grad_output_bf16 = grad_output.to(torch.bfloat16)

    grad_input = te_general_gemm(weight, grad_output_bf16, ctx.input_dtype, layout="NN", grad=True)[0].to(
        ctx.input_dtype
    )
    grad_weight = te_general_gemm(inp, grad_output_bf16, ctx.router_dtype, layout="NT", grad=True)[0].to(
        ctx.weight_dtype
    )
    # Bias reduction retains the original FP32 gradient, before BF16 rounding.
    grad_bias = grad_output.sum(dim=0).to(ctx.weight_dtype) if bias is not None else None
    return grad_input.view(*inp_shape), grad_weight, grad_bias, None


@register_patch(
    "megatron.moe.router_bwd_bf16",
    backend="megatron",
    phase="before_train",
    description="Use BF16 backward GEMM operands for the FP32 MoE router",
    condition=_enabled,
)
def patch_router_bwd_bf16(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    router_function = moe_utils.RouterGatingLinearFunction
    if getattr(router_function, "_primus_router_bwd_bf16_patched", False):
        return
    original_backward = router_function.backward

    def backward(autograd_ctx, grad_output):
        te_general_gemm = moe_utils.te_general_gemm
        if not _uses_bf16_router_backward(autograd_ctx, te_general_gemm):
            return original_backward(autograd_ctx, grad_output)
        return _router_backward_bf16(autograd_ctx, grad_output, te_general_gemm=te_general_gemm)

    router_function.backward = staticmethod(backward)
    router_function._primus_router_bwd_bf16_patched = True
    log_rank_0("[Patch:megatron.moe.router_bwd_bf16] Installed BF16 router backward GEMMs")
