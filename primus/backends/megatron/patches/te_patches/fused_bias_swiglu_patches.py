###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Fused Bias SwiGLU Patches

Patches SwiGLUFunction to use TE's fused swiglu/dswiglu kernels when
USE_TE_SWIGLU=1 is set, providing better performance on ROCm GPUs.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.te.fused_bias_swiglu",
    backend="megatron",
    phase="before_train",
    description="Use TE fused swiglu/dswiglu kernels in SwiGLUFunction forward/backward",
    condition=lambda ctx: os.getenv("USE_TE_SWIGLU", "0") == "1",
)
def patch_te_fused_bias_swiglu(ctx: PatchContext):
    """
    Patch SwiGLUFunction to use Transformer Engine's fused swiglu/dswiglu
    C++ extensions in forward and backward passes.

    Activated when USE_TE_SWIGLU=1 is set in the environment.
    """
    from transformer_engine.pytorch.cpp_extensions import (
        dswiglu as te_dswiglu,
        swiglu as te_swiglu,
    )

    from megatron.core.fusions.fused_bias_swiglu import SwiGLUFunction

    @staticmethod
    def new_forward(ctx, input, fp8_input_store, cpu_offload_input):
        input_for_backward = input.to(__import__("torch").float8_e4m3fn) if fp8_input_store else input
        if cpu_offload_input:
            input_for_backward.activation_offloading = True
        ctx.save_for_backward(input_for_backward)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return te_swiglu(input, None)

    @staticmethod
    def new_backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        return te_dswiglu(grad_output, input, None), None, None

    SwiGLUFunction.forward = new_forward
    SwiGLUFunction.backward = new_backward

    log_rank_0(
        "[Patch:megatron.te.fused_bias_swiglu] Patched SwiGLUFunction "
        "to use TE fused swiglu/dswiglu kernels (USE_TE_SWIGLU=1)"
    )
