###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Install the selective FlyDSL LM-head path for the GPT-OSS 20B workload."""

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
    """Replace Megatron's autograd function with the selective Primus version.

    The replacement remains a no-op for every tensor except the trace-locked
    GPT-OSS LM-head matrices; those exact forward, dgrad, and fused beta=1
    wgrad operations dispatch through Primus Turbo.
    """
    import megatron.core.tensor_parallel.layers as megatron_layers

    from primus.backends.megatron.core.tensor_parallel.layers import (
        LinearWithGradAccumulationAndAsyncCommunication,
    )

    megatron_layers.LinearWithGradAccumulationAndAsyncCommunication = (
        LinearWithGradAccumulationAndAsyncCommunication
    )
    log_rank_0(
        "[Patch:megatron.turbo.gptoss_bf16_lm_head] Patched "
        "megatron.core.tensor_parallel.layers."
        "LinearWithGradAccumulationAndAsyncCommunication for exact GPT-OSS "
        "BF16 LM-head dispatch"
    )
