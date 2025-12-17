###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Pipeline Parallelism Patches

This module contains patches that modify Megatron's pipeline parallelism
implementation to support ZeroBubble PP and Primus Pipeline optimizations.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


def _is_zero_bubble_enabled(ctx: PatchContext) -> bool:
    """Check if ZeroBubble PP is enabled in module_config."""
    return getattr(get_args(ctx), "patch_zero_bubble", False)


@register_patch(
    "megatron.pp.zero_bubble_optimizer",
    backend="megatron",
    phase="before_train",
    description="Patch optimizer and forward_backward_func for ZeroBubble PP",
    condition=_is_zero_bubble_enabled,
)
def patch_zero_bubble_pp(ctx: PatchContext):
    """
    Patch Megatron to use ZeroBubble PP implementation.

    Behavior:
        - Replace ChainedOptimizer with ZeroBubblePPChainedOptimizer
        - Replace get_forward_backward_func with get_forward_backward_func_zbpp
    """
    log_rank_0("[Patch:megatron.pp.zero_bubble_optimizer] Patching ZeroBubble PP...")

    try:
        # Patch optimizer
        import megatron.core.optimizer as optimizer

        from primus.backends.megatron.core.optimizer.zbpp_optimizer import (
            ZeroBubblePPChainedOptimizer,
        )

        optimizer.ChainedOptimizer = ZeroBubblePPChainedOptimizer
        log_rank_0("[Patch:megatron.pp.zero_bubble_optimizer] Patched ChainedOptimizer")

        # Patch get_forward_backward_func
        import megatron.core.pipeline_parallel as ori_pp

        from primus.backends.megatron.core.pipeline_parallel.schedules import (
            get_forward_backward_func_zbpp,
        )

        ori_pp.get_forward_backward_func = get_forward_backward_func_zbpp
        log_rank_0("[Patch:megatron.pp.zero_bubble_optimizer] Patched get_forward_backward_func")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.zero_bubble_optimizer][SKIP] Failed to apply patch: {e}")


def _is_primus_pipeline_enabled(ctx: PatchContext) -> bool:
    """Check if Primus Pipeline is enabled (and ZeroBubble is not)."""
    args = get_args(ctx)
    if not getattr(args, "patch_primus_pipeline", False):
        return False
    # ZeroBubble takes precedence over Primus Pipeline
    if getattr(args, "patch_zero_bubble", False):
        return False
    return True


@register_patch(
    "megatron.pp.primus_pipeline",
    backend="megatron",
    phase="before_train",
    description="Patch forward_backward_func for Primus Pipeline",
    condition=_is_primus_pipeline_enabled,
)
def patch_primus_pipeline(ctx: PatchContext):
    """
    Patch Megatron to use Primus Pipeline implementation.

    Behavior:
        - Replace get_forward_backward_func with get_primus_pipeline_parallel_fwd_backward_func
    """
    log_rank_0("[Patch:megatron.pp.primus_pipeline] Patching Primus Pipeline...")

    try:
        import megatron.core.pipeline_parallel as ori_pp

        from primus.backends.megatron.core.pipeline_parallel.schedules import (
            get_primus_pipeline_parallel_fwd_backward_func,
        )

        ori_pp.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
        log_rank_0("[Patch:megatron.pp.primus_pipeline] Patched get_forward_backward_func")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.primus_pipeline][SKIP] Failed to apply patch: {e}")


def _is_pp_enabled(ctx: PatchContext) -> bool:
    """Check if either Primus Pipeline or ZeroBubble PP is enabled."""
    args = get_args(ctx)
    return getattr(args, "patch_primus_pipeline", False) or getattr(args, "patch_zero_bubble", False)


@register_patch(
    "megatron.pp.linear_grad_split",
    backend="megatron",
    phase="before_train",
    description="Patch Linear layer to split d_w and d_input for PP optimization",
    condition=_is_pp_enabled,
)
def patch_linear_grad_split(ctx: PatchContext):
    """
    Patch LinearWithGradAccumulationAndAsyncCommunication for gradient splitting.

    Behavior:
        - Replace LinearWithGradAccumulationAndAsyncCommunication with Primus version
          that splits weight gradient and input gradient computation
    """
    log_rank_0("[Patch:megatron.pp.linear_grad_split] Patching Linear layer for gradient splitting...")

    try:
        import megatron.core.tensor_parallel.layers as ori_layers

        from primus.backends.megatron.core.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        ori_layers.LinearWithGradAccumulationAndAsyncCommunication = (
            LinearWithGradAccumulationAndAsyncCommunication
        )
        log_rank_0(
            "[Patch:megatron.pp.linear_grad_split] Patched LinearWithGradAccumulationAndAsyncCommunication"
        )

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.linear_grad_split][SKIP] Failed to apply patch: {e}")


def _is_v_schedule_enabled(ctx: PatchContext) -> bool:
    """Check if V-schedule is enabled in either ZeroBubble or Primus Pipeline."""
    args = get_args(ctx)
    patch_primus = getattr(args, "patch_primus_pipeline", False)
    patch_zbpp = getattr(args, "patch_zero_bubble", False)

    if not (patch_primus or patch_zbpp):
        return False

    # Check if V-schedule is enabled using the same logic as is_v_schedule_enabled()
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
    "megatron.pp.v_schedule_support",
    backend="megatron",
    phase="before_train",
    description="Patch various components for V-schedule support in ZeroBubble PP",
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
    log_rank_0("[Patch:megatron.pp.v_schedule_support] Patching components for V-schedule...")

    try:
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
        log_rank_0("[Patch:megatron.pp.v_schedule_support] Patched parallel_state functions")

        # Patch finalize_model_grads
        import megatron.core.distributed.finalize_model_grads as ori_finalize_model_grads

        from primus.backends.megatron.core.distributed.finalize_model_grad import (
            finalize_model_grads,
        )

        ori_finalize_model_grads.finalize_model_grads = finalize_model_grads
        log_rank_0("[Patch:megatron.pp.v_schedule_support] Patched finalize_model_grads")

        # Patch get_transformer_layer_offset
        import megatron.core.transformer.transformer_layer as ori_transformer_layer

        from primus.backends.megatron.core.transformer.transformer_layer import (
            get_transformer_layer_offset,
        )

        ori_transformer_layer.get_transformer_layer_offset = get_transformer_layer_offset
        log_rank_0("[Patch:megatron.pp.v_schedule_support] Patched get_transformer_layer_offset")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.v_schedule_support][SKIP] Failed to apply patch: {e}")


@register_patch(
    "megatron.pp.te_wgrad_split",
    backend="megatron",
    phase="before_train",
    description="Patch Transformer Engine Linear layers for weight gradient splitting",
    condition=_is_pp_enabled,
)
def patch_te_wgrad_split(ctx: PatchContext):
    """
    Patch Transformer Engine Linear layers to split weight gradient computation.

    Behavior:
        - Replace TE _GroupedLinear with _GroupedLinearWithWGradSplit
        - Replace TE _Linear with _LinearWithWGradSplit
    """
    log_rank_0("[Patch:megatron.pp.te_wgrad_split] Patching TE layers for weight gradient splitting...")

    try:
        # Patch TE grouped_linear
        import transformer_engine.pytorch.module.grouped_linear as ori_grouped_linear

        from primus.backends.megatron.core.extensions.te_group_gemm_patch_wgrad import (
            _GroupedLinearWithWGradSplit,
        )

        ori_grouped_linear._GroupedLinear = _GroupedLinearWithWGradSplit
        log_rank_0("[Patch:megatron.pp.te_wgrad_split] Patched TE _GroupedLinear")

        # Patch TE linear
        import transformer_engine.pytorch.module.linear as ori_linear

        from primus.backends.megatron.core.extensions.te_gemm_patch_wgrad import (
            _LinearWithWGradSplit,
        )

        ori_linear._Linear = _LinearWithWGradSplit
        log_rank_0("[Patch:megatron.pp.te_wgrad_split] Patched TE _Linear")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.te_wgrad_split][SKIP] Failed to apply patch: {e}")


@register_patch(
    "megatron.pp.pipeline_parallel_layer_layout",
    backend="megatron",
    phase="before_train",
    description="Replace PipelineParallelLayerLayout with Primus implementation",
)
def patch_pipeline_parallel_layer_layout(ctx: PatchContext):
    """
    Patch Megatron to use Primus's PipelineParallelLayerLayout.

    Behavior:
        - Replace megatron.core.transformer.pipeline_parallel_layer_layout.PipelineParallelLayerLayout
          with PrimusPipelineParallelLayerLayout
    """
    log_rank_0("[Patch:megatron.pp.pipeline_parallel_layer_layout] Patching PipelineParallelLayerLayout...")

    try:
        import megatron.core.transformer.pipeline_parallel_layer_layout as orig_pipeline_parallel_layer_layout

        from primus.backends.megatron.core.transformer.pipeline_parallel_layer_layout import (
            PrimusPipelineParallelLayerLayout,
        )

        orig_pipeline_parallel_layer_layout.PipelineParallelLayerLayout = PrimusPipelineParallelLayerLayout
        log_rank_0("[Patch:megatron.pp.pipeline_parallel_layer_layout] Patched PipelineParallelLayerLayout")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.pp.pipeline_parallel_layer_layout][SKIP] Failed to apply patch: {e}")
