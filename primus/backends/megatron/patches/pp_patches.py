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

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    "megatron.pp.zero_bubble_optimizer",
    backend="megatron",
    phase="before_train",
    description="Patch optimizer and forward_backward_func for ZeroBubble PP",
)
def patch_zero_bubble_pp(ctx: PatchContext):
    """
    Patch Megatron to use ZeroBubble PP implementation.

    Behavior:
        - Replace ChainedOptimizer with ZeroBubblePPChainedOptimizer
        - Replace get_forward_backward_func with get_forward_backward_func_zbpp
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None or not getattr(params, "patch_zero_bubble", False):
        log_rank_0("[Patch:megatron.pp.zero_bubble_optimizer][SKIP] patch_zero_bubble is not enabled")
        return

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


@register_patch(
    "megatron.pp.primus_pipeline",
    backend="megatron",
    phase="before_train",
    description="Patch forward_backward_func for Primus Pipeline",
)
def patch_primus_pipeline(ctx: PatchContext):
    """
    Patch Megatron to use Primus Pipeline implementation.

    Behavior:
        - Replace get_forward_backward_func with get_primus_pipeline_parallel_fwd_backward_func
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None or not getattr(params, "patch_primus_pipeline", False):
        log_rank_0("[Patch:megatron.pp.primus_pipeline][SKIP] patch_primus_pipeline is not enabled")
        return

    # Skip if ZeroBubble is enabled (it takes precedence)
    if getattr(params, "patch_zero_bubble", False):
        log_rank_0(
            "[Patch:megatron.pp.primus_pipeline][SKIP] ZeroBubble PP is enabled, skipping Primus Pipeline"
        )
        return

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


@register_patch(
    "megatron.pp.linear_grad_split",
    backend="megatron",
    phase="before_train",
    description="Patch Linear layer to split d_w and d_input for PP optimization",
)
def patch_linear_grad_split(ctx: PatchContext):
    """
    Patch LinearWithGradAccumulationAndAsyncCommunication for gradient splitting.

    Behavior:
        - Replace LinearWithGradAccumulationAndAsyncCommunication with Primus version
          that splits weight gradient and input gradient computation
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None:
        warning_rank_0("[Patch:megatron.pp.linear_grad_split][SKIP] No params in module_config")
        return

    patch_primus = getattr(params, "patch_primus_pipeline", False)
    patch_zbpp = getattr(params, "patch_zero_bubble", False)

    if not (patch_primus or patch_zbpp):
        log_rank_0(
            "[Patch:megatron.pp.linear_grad_split][SKIP] Neither patch_primus_pipeline nor patch_zero_bubble is enabled"
        )
        return

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


@register_patch(
    "megatron.pp.v_schedule_support",
    backend="megatron",
    phase="before_train",
    description="Patch various components for V-schedule support in ZeroBubble PP",
)
def patch_v_schedule_support(ctx: PatchContext):
    """
    Patch Megatron components to support V-schedule in ZeroBubble PP.

    Behavior:
        - Patch parallel_state functions (embedding ranks, pipeline stages)
        - Patch finalize_model_grads
        - Patch get_transformer_layer_offset
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None:
        warning_rank_0("[Patch:megatron.pp.v_schedule_support][SKIP] No params in module_config")
        return

    patch_primus = getattr(params, "patch_primus_pipeline", False)
    patch_zbpp = getattr(params, "patch_zero_bubble", False)

    if not (patch_primus or patch_zbpp):
        log_rank_0("[Patch:megatron.pp.v_schedule_support][SKIP] PP patches not enabled")
        return

    # Check if V-schedule is enabled using the same logic as is_v_schedule_enabled()
    # V-schedule is enabled if:
    # 1. Zero Bubble is enabled with V-schedule flags, OR
    # 2. Primus Pipeline is enabled with V-schedule algorithms
    is_v_schedule = False
    if patch_zbpp:
        enable_zero_bubble = getattr(params, "enable_zero_bubble", False)
        zero_bubble_v_schedule = getattr(params, "zero_bubble_v_schedule", False)
        enable_1f1b_v = getattr(params, "enable_1f1b_v", False)
        is_v_schedule = enable_zero_bubble and (zero_bubble_v_schedule or enable_1f1b_v)
    elif patch_primus:
        pp_algorithm = getattr(params, "pp_algorithm", None)
        is_v_schedule = pp_algorithm in ("zbv-formatted", "v-half", "v-min")

    if not is_v_schedule:
        log_rank_0("[Patch:megatron.pp.v_schedule_support][SKIP] V-schedule is not enabled")
        return

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
)
def patch_te_wgrad_split(ctx: PatchContext):
    """
    Patch Transformer Engine Linear layers to split weight gradient computation.

    Behavior:
        - Replace TE _GroupedLinear with _GroupedLinearWithWGradSplit
        - Replace TE _Linear with _LinearWithWGradSplit
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None:
        warning_rank_0("[Patch:megatron.pp.te_wgrad_split][SKIP] No params in module_config")
        return

    patch_primus = getattr(params, "patch_primus_pipeline", False)
    patch_zbpp = getattr(params, "patch_zero_bubble", False)

    if not (patch_primus or patch_zbpp):
        log_rank_0("[Patch:megatron.pp.te_wgrad_split][SKIP] PP patches not enabled")
        return

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
