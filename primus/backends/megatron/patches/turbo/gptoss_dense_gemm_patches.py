###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Opt-in GPT-OSS LM-head and router wiring for Primus-Turbo dense GEMM.

Megatron constructs the GPT output layer directly as its native
``ColumnParallelLinear`` and its router calls Transformer Engine's
``general_gemm`` directly.  Neither site consults Primus' spec provider, so
``turbo_gemm_backend`` alone cannot affect them.  This patch connects both
sites to ``primus_turbo.pytorch.ops.gemm`` when explicitly enabled by
``PRIMUS_TURBO_ROUTE_GPTOSS_DENSE_GEMM=1``.
"""

import os

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _can_route_gptoss_dense_gemm(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        _env_enabled("PRIMUS_TURBO_ROUTE_GPTOSS_DENSE_GEMM")
        and bool(getattr(args, "use_turbo_gemm", False))
        and int(getattr(args, "tensor_model_parallel_size", 1)) == 1
        and is_primus_turbo_can_patch(ctx)
    )


@register_patch(
    "megatron.turbo.gptoss_dense_gemm",
    backend="megatron",
    phase="before_train",
    description="Route GPT-OSS LM-head and router GEMMs through Primus-Turbo",
    condition=_can_route_gptoss_dense_gemm,
)
def patch_gptoss_dense_gemm(ctx: PatchContext) -> None:
    del ctx
    import torch
    import torch.nn.functional as F
    from megatron.core.models.gpt import gpt_model
    from megatron.core.transformer.moe import moe_utils, router
    from primus_turbo import pytorch as primus_turbo_torch

    from primus.backends.megatron.core.extensions.primus_turbo import (
        PrimusTurboBF16OutputColumnParallelLinear,
    )

    # GPTModel bypasses the backend spec provider for its output projection.
    # Swap the constructor only while GPTModel.__init__ runs; assigning it
    # permanently on the shared tensor_parallel module would affect unrelated
    # native ColumnParallelLinear users in the same process.
    original_gpt_init = gpt_model.GPTModel.__init__

    def _gpt_init_with_turbo_output(self, *args, **kwargs):
        original_column_parallel = gpt_model.tensor_parallel.ColumnParallelLinear
        gpt_model.tensor_parallel.ColumnParallelLinear = (
            PrimusTurboBF16OutputColumnParallelLinear
        )
        try:
            return original_gpt_init(self, *args, **kwargs)
        finally:
            gpt_model.tensor_parallel.ColumnParallelLinear = original_column_parallel

    gpt_model.GPTModel.__init__ = _gpt_init_with_turbo_output

    class _PrimusTurboRouterGatingLinearFunction(torch.autograd.Function):
        """Router GEMMs with BF16 MFMA inputs and FP32 accumulation/output."""

        @staticmethod
        def forward(ctx, inp, weight, bias, router_dtype):
            ctx.save_for_backward(inp, weight, bias)
            ctx.router_dtype = router_dtype
            ctx.input_dtype = inp.dtype
            ctx.weight_dtype = weight.dtype
            inp_shape = inp.shape
            inp_2d = inp.reshape(-1, inp_shape[-1]).contiguous()

            output = primus_turbo_torch.ops.gemm(
                inp_2d,
                weight,
                trans_a=False,
                trans_b=True,
                out_dtype=router_dtype,
            )
            if bias is not None:
                output = output + bias.to(router_dtype)
            return output.view(*inp_shape[:-1], -1)

        @staticmethod
        def backward(ctx, grad_output):
            inp, weight, bias = ctx.saved_tensors
            inp_shape = inp.shape
            grad_shape = grad_output.shape
            inp_2d = inp.reshape(-1, inp_shape[-1]).contiguous()
            grad_2d = (
                grad_output.reshape(-1, grad_shape[-1]).to(torch.bfloat16).contiguous()
            )

            # The dense FlyDSL pipeline has a minimum contraction depth of 128.
            # GPT-OSS has 32 experts, so pad only the router dgrad contraction;
            # the zero columns/rows do not change its mathematical result.
            dgrad_lhs = grad_2d
            dgrad_rhs = weight
            if grad_2d.shape[1] < 128:
                pad_k = 128 - grad_2d.shape[1]
                dgrad_lhs = F.pad(grad_2d, (0, pad_k))
                dgrad_rhs = F.pad(weight, (0, 0, 0, pad_k))

            grad_input = primus_turbo_torch.ops.gemm(
                dgrad_lhs,
                dgrad_rhs,
                trans_a=False,
                trans_b=False,
                out_dtype=ctx.router_dtype,
            ).to(ctx.input_dtype)
            grad_weight = primus_turbo_torch.ops.gemm(
                grad_2d,
                inp_2d,
                trans_a=True,
                trans_b=False,
                out_dtype=ctx.router_dtype,
            ).to(ctx.weight_dtype)
            grad_bias = (
                grad_output.reshape(-1, grad_shape[-1]).sum(dim=0).to(ctx.weight_dtype)
                if bias is not None
                else None
            )
            return grad_input.view(*inp_shape), grad_weight, grad_bias, None

    def _router_gating_linear(inp, weight, bias, router_dtype):
        return _PrimusTurboRouterGatingLinearFunction.apply(
            inp, weight, bias, router_dtype
        )

    # router.py imported the function by name during module import, so update
    # both the defining module and the consumer's already-bound alias.
    moe_utils.RouterGatingLinearFunction = _PrimusTurboRouterGatingLinearFunction
    moe_utils.router_gating_linear = _router_gating_linear
    router.router_gating_linear = _router_gating_linear

    log_rank_0(
        "[Patch:megatron.turbo.gptoss_dense_gemm] Routed GPT output and router "
        "GEMMs through Primus-Turbo dense GEMM"
    )
