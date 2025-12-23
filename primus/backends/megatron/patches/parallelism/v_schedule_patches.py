###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron V-schedule support patches for pipeline parallelism.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_v_schedule_enabled(ctx: PatchContext) -> bool:
    """Check if V-schedule is enabled in either ZeroBubble or Primus Pipeline."""
    args = get_args(ctx)
    patch_primus = getattr(args, "patch_primus_pipeline", False)
    patch_zbpp = getattr(args, "patch_zero_bubble", False)

    if not (patch_primus or patch_zbpp):
        return False

    # V-schedule is enabled if:
    # 1. Zero Bubble is enabled with V-schedule flags, OR
    # 2. Primus Pipeline is enabled with V-schedule algorithms
    if patch_zbpp:
        enable_zero_bubble = getattr(args, "enable_zero_bubble", False)
        zero_bubble_v_schedule = getattr(args, "zero_bubble_v_schedule", False)
        enable_1f1b_v = getattr(args, "enable_1f1b_v", False)
        return enable_zero_bubble and (zero_bubble_v_schedule or enable_1f1b_v)
    elif patch_primus:
        pp_algorithm = getattr(args, "pp_algorithm", None)
        return pp_algorithm in ("zbv-formatted", "v-half", "v-min")

    return False


@register_patch(
    "megatron.pp.v_schedule",
    backend="megatron",
    phase="before_train",
    description="Patch various components for V-schedule in ZeroBubble PP",
    condition=_is_v_schedule_enabled,
)
def patch_v_schedule_support(ctx: PatchContext):
    """
    Patch Megatron components to support V-schedule in ZeroBubble PP.

    Behavior:
        - Patch parallel_state functions (embedding ranks, pipeline stages)
        - Patch finalize_model_grads
        - Patch get_transformer_layer_offset
    """
    # Patch parallel_state
    import megatron.core.parallel_state as ori_parallel_state

    from primus.backends.megatron.core.parallel_state import (
        default_embedding_ranks,
        is_pipeline_last_stage,
        is_rank_in_embedding_group,
    )

    ori_parallel_state.default_embedding_ranks = default_embedding_ranks
    ori_parallel_state.is_pipeline_last_stage = is_pipeline_last_stage
    ori_parallel_state.is_rank_in_embedding_group = is_rank_in_embedding_group
    log_rank_0(
        f"[Patch:megatron.pp.v_schedule]   Patched megatron.core.parallel_state.default_embedding_ranks "
        f"-> {default_embedding_ranks.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.pp.v_schedule]   Patched megatron.core.parallel_state.is_pipeline_last_stage "
        f"-> {is_pipeline_last_stage.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.pp.v_schedule]   Patched megatron.core.parallel_state.is_rank_in_embedding_group "
        f"-> {is_rank_in_embedding_group.__name__}"
    )

    # Patch finalize_model_grads
    import megatron.core.distributed.finalize_model_grads as ori_finalize_model_grads

    from primus.backends.megatron.core.distributed.finalize_model_grad import (
        finalize_model_grads,
    )

    ori_finalize_model_grads.finalize_model_grads = finalize_model_grads
    log_rank_0(
        f"[Patch:megatron.pp.v_schedule]   Patched megatron.core.distributed.finalize_model_grads.finalize_model_grads "
        f"-> {finalize_model_grads.__name__}"
    )

    # Patch get_transformer_layer_offset
    import megatron.core.transformer.transformer_layer as ori_transformer_layer

    from primus.backends.megatron.core.transformer.transformer_layer import (
        get_transformer_layer_offset,
    )

    ori_transformer_layer.get_transformer_layer_offset = get_transformer_layer_offset
    log_rank_0(
        f"[Patch:megatron.pp.v_schedule]   Patched megatron.core.transformer.transformer_layer.get_transformer_layer_offset "
        f"-> {get_transformer_layer_offset.__name__}"
    )
