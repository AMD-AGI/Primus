###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in Triton GEMMs for an FP32 MoE router (``moe_router_triton_gemm: true``).

Megatron's ``RouterGatingLinearFunction`` runs its forward and backward on the kernels of
:mod:`primus.backends.megatron.core.extensions._triton.router_gemm` whenever they take the operands: an
FP32 router with BF16 input and weight, no bias, and the shapes the kernels tile exactly. Every other call
keeps Megatron's path. Not installed together with ``moe_router_bwd_bf16``.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _enabled(ctx: PatchContext) -> bool:
    return bool(getattr(get_args(ctx), "moe_router_triton_gemm", False))


@register_patch(
    "megatron.moe.router_triton_gemm",
    backend="megatron",
    phase="before_train",
    description="Triton forward/backward GEMMs for the FP32 MoE router (FP32-accurate backward)",
    condition=_enabled,
)
def patch_router_triton_gemm(ctx: PatchContext) -> None:
    from megatron.core.transformer.moe import moe_utils

    fn = moe_utils.RouterGatingLinearFunction
    if getattr(fn, "_primus_router_triton_gemm_patched", False):
        return
    if getattr(get_args(ctx), "moe_router_bwd_bf16", False):
        warning_rank_0(
            "[Patch:megatron.moe.router_triton_gemm] not installed: moe_router_bwd_bf16 is set "
            "(the two options are mutually exclusive)"
        )
        return
    # Imported here, so that importing the patch modules does not import Triton.
    from primus.backends.megatron.core.extensions._triton import router_gemm

    orig_forward, orig_backward = fn.forward, fn.backward

    def forward(ctx_, inp, weight, bias, router_dtype):
        ctx_.primus_triton = router_gemm.supported(inp, weight, bias, router_dtype)
        if not ctx_.primus_triton:
            return orig_forward(ctx_, inp, weight, bias, router_dtype)
        ctx_.save_for_backward(inp, weight)
        shape = inp.shape
        out = router_gemm.forward(inp.reshape(-1, shape[-1]).contiguous(), weight)
        return out.view(*shape[:-1], -1)

    def backward(ctx_, grad_output):
        if not ctx_.primus_triton:
            return orig_backward(ctx_, grad_output)
        inp, weight = ctx_.saved_tensors
        x = inp.reshape(-1, inp.shape[-1]).contiguous()
        g = grad_output.reshape(-1, grad_output.shape[-1]).float().contiguous()
        grad_input, grad_weight = router_gemm.backward(g, x, weight)
        return grad_input.view(inp.shape), grad_weight, None, None

    fn.forward = staticmethod(forward)
    fn.backward = staticmethod(backward)
    fn._primus_router_triton_gemm_patched = True
    log_rank_0("[Patch:megatron.moe.router_triton_gemm] FP32 router GEMMs on Triton kernels")
