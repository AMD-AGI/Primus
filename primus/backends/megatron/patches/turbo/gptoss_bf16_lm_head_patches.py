###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install the selective FlyDSL LM-head path for the GPT-OSS 20B workload."""

import functools

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_gptoss_bf16_lm_head_can_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return (
        ctx.model_name == "gpt_oss_20B.yaml"
        and getattr(args, "use_turbo_gemm", False)
        and getattr(args, "bf16", False)
        and getattr(args, "hidden_size", None) == 2880
        and getattr(args, "seq_length", None) == 8192
        and getattr(args, "micro_batch_size", None) == 4
        and getattr(args, "tensor_model_parallel_size", None) == 1
    )


@register_patch(
    "megatron.turbo.gptoss_bf16_lm_head",
    backend="megatron",
    phase="before_train",
    description="Route the exact GPT-OSS 20B BF16 LM-head GEMMs through Primus Turbo",
    condition=_is_gptoss_bf16_lm_head_can_patch,
)
def patch_gptoss_bf16_lm_head(ctx: PatchContext):
    """Wrap Megatron's linear entry point with the selective Primus version.

    Every non-LM-head call remains on the original Megatron implementation.
    An exact LM-head forward selects the Primus autograd function, whose
    backward routes the matching dgrad and fused beta=1 wgrad through Turbo.
    """
    import megatron.core.tensor_parallel.layers as megatron_layers

    from primus.backends.megatron.core.tensor_parallel.layers import (
        LinearWithGradAccumulationAndAsyncCommunication,
        _is_gptoss_bf16_lm_head_forward,
    )

    original_linear = megatron_layers.linear_with_grad_accumulation_and_async_allreduce

    @functools.wraps(original_linear)
    def gptoss_bf16_lm_head_linear(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer=None,
        wgrad_deferral_limit=0,
        tp_group=None,
    ):
        args = (
            input,
            weight,
            bias,
            gradient_accumulation_fusion,
            allreduce_dgrad,
            sequence_parallel,
            grad_output_buffer,
            wgrad_deferral_limit,
            tp_group,
        )
        if _is_gptoss_bf16_lm_head_forward(input, weight):
            return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)
        return original_linear(*args)

    megatron_layers.linear_with_grad_accumulation_and_async_allreduce = (
        gptoss_bf16_lm_head_linear
    )
    log_rank_0(
        "[Patch:megatron.turbo.gptoss_bf16_lm_head] Patched "
        "megatron.core.tensor_parallel.layers."
        "linear_with_grad_accumulation_and_async_allreduce for exact GPT-OSS "
        "BF16 LM-head dispatch"
    )
