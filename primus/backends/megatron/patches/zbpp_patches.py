###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Zero-Bubble Pipeline Parallel (ZBPP) Patches

This module contains patches to enable Zero-Bubble pipeline parallelism and
related optimizer / scheduling tweaks for Megatron when requested via
module_config.
"""

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.distributed_logging import log_rank_0, warning_rank_0


@register_patch(
    "megatron.zbpp.enable",
    backend="megatron",
    phase="before_train",
    description="Enable Zero-Bubble pipeline parallel optimizer and schedules",
)
def patch_zero_bubble_pipeline(ctx: PatchContext):
    """
    Patch Megatron to enable Zero-Bubble pipeline parallelism.

    Behavior (moved from MegatronTrainer.patch_zbpp):
        - If module_config.patch_zero_bubble:
            * Replace optimizer.ChainedOptimizer with ZeroBubblePPChainedOptimizer.
            * Replace pipeline_parallel.get_forward_backward_func with
              get_forward_backward_func_zbpp.
            * Replace LinearWithGradAccumulationAndAsyncCommunication to split
              d_w and d_input.
        - If zero_bubble_v_schedule or enable_1f1b_v is set:
            * Patch embedding group helpers in parallel_state.
            * Patch finalize_model_grads to custom implementation.
            * Patch transformer_layer.get_transformer_layer_offset.
            * Patch TE grouped linear and linear modules to split wgrad.
    """
    module_config = ctx.extra.get("module_config")
    if module_config is None or not getattr(module_config, "patch_zero_bubble", False):
        return

    warning_rank_0("MegatronPatches: Patch ZeroBubble PP")

    # -------------------------------------------------------------------------
    # Optimizer and pipeline scheduling
    # -------------------------------------------------------------------------
    try:
        import megatron.core.optimizer as optimizer

        from primus.backends.megatron.patches.core.optimizer.zbpp_optimizer import (
            ZeroBubblePPChainedOptimizer,
        )

        optimizer.ChainedOptimizer = ZeroBubblePPChainedOptimizer

        import megatron.core.pipeline_parallel as ori_pp

        from primus.backends.megatron.patches.core.pipeline_parallel.schedules import (
            get_forward_backward_func_zbpp,
        )

        ori_pp.get_forward_backward_func = get_forward_backward_func_zbpp

        import megatron.core.tensor_parallel.layers as ori_layers

        from primus.backends.megatron.patches.core.tensor_parallel.layers import (
            LinearWithGradAccumulationAndAsyncCommunication,
        )

        ori_layers.LinearWithGradAccumulationAndAsyncCommunication = (
            LinearWithGradAccumulationAndAsyncCommunication
        )
    except Exception as e:
        warning_rank_0(
            "MegatronPatches: failed to apply base ZeroBubble patches: " f"{type(e).__name__}: {e}"
        )
        return

    # -------------------------------------------------------------------------
    # ZBV-related patches (optional, controlled by additional flags)
    # -------------------------------------------------------------------------
    if getattr(module_config, "zero_bubble_v_schedule", False) or getattr(
        module_config, "enable_1f1b_v", False
    ):
        try:
            import megatron.core.parallel_state as ori_parallel_state

            from primus.backends.megatron.core.parallel_state import (
                default_embedding_ranks,
                is_pipeline_last_stage,
                is_rank_in_embedding_group,
            )

            ori_parallel_state.default_embedding_ranks = default_embedding_ranks
            ori_parallel_state.is_pipeline_last_stage = is_pipeline_last_stage
            ori_parallel_state.is_rank_in_embedding_group = is_rank_in_embedding_group

            import megatron.core.distributed.finalize_model_grads as ori_finalize_model_grads

            from primus.backends.megatron.core.distributed.finalize_model_grad import (
                finalize_model_grads,
            )

            ori_finalize_model_grads.finalize_model_grads = finalize_model_grads

            import megatron.core.transformer.transformer_layer as ori_transformer_layer

            from primus.backends.megatron.core.transformer.transformer_layer import (
                get_transformer_layer_offset,
            )

            ori_transformer_layer.get_transformer_layer_offset = get_transformer_layer_offset

            # patch te_group_gemm & gemm
            import transformer_engine.pytorch.module.grouped_linear as ori_grouped_linear

            from primus.backends.megatron.core.extensions.te_group_gemm_patch_wgrad import (
                _GroupedLinearWithWGradSplit,
            )

            ori_grouped_linear._GroupedLinear = _GroupedLinearWithWGradSplit

            import transformer_engine.pytorch.module.linear as ori_linear

            from primus.backends.megatron.core.extensions.te_gemm_patch_wgrad import (
                _LinearWithWGradSplit,
            )

            ori_linear._Linear = _LinearWithWGradSplit

            log_rank_0("MegatronPatches: ZeroBubble V-schedule patches applied")
        except Exception as e:
            warning_rank_0(
                "MegatronPatches: failed to apply ZeroBubble V-schedule patches: " f"{type(e).__name__}: {e}"
            )
