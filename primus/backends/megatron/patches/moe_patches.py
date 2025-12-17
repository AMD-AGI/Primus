###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Patches

This module contains patches for Megatron's Mixture-of-Experts (MoE)
components to integrate Primus-specific behavior:

    - Deprecated MoE layer implementations
    - Primus TopKRouter
    - MoE permutation fusion with Transformer Engine
"""

import sys

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.patch.moe_layer",
    backend="megatron",
    phase="before_train",
    description=(
        "Apply Primus-specific MoE overrides: deprecated MoELayer/experts, "
        "Primus TopKRouter, and permutation fusion."
    ),
)
def patch_moe_layer(ctx: PatchContext):
    """
    Patch Megatron MoE components based on module_config flags.

    Behavior (moved from MegatronTrainer.patch_moe_layer):
        - If use_deprecated_20241209_moe_layer:
            * Replace core MoELayer / MoESubmodules / experts with deprecated versions.
            * Sync replacements into gpt.moe_module_specs.
        - If not disable_primus_topk_router:
            * Replace TopKRouter with PrimusTopKRouter (optionally via deprecated router).
        - If moe_permute_fusion:
            * Patch TE and Megatron MoE permutation helpers with Primus fused implementations.
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)

    # -------------------------------------------------------------------------
    # 1) Deprecated MoE Layer + Experts
    # -------------------------------------------------------------------------
    use_deprecated_moe = getattr(params, "use_deprecated_20241209_moe_layer", False)
    if use_deprecated_moe:
        log_rank_0("[Patch:megatron.patch.moe_layer] monkey patch MoELayer with DeprecatedMoELayer...")

        # patch module class
        from primus.backends.megatron.core.transformer.moe.deprecated_20251209.experts import (
            DeprecatedGroupedMLP,
            DeprecatedSequentialMLP,
            DeprecatedTEGroupedMLP,
        )
        from primus.backends.megatron.core.transformer.moe.deprecated_20251209.moe_layer import (
            DeprecatedMoELayer,
            DeprecatedMoESubmodules,
        )

        sys.modules["megatron.core.transformer.moe.moe_layer"].MoELayer = DeprecatedMoELayer
        sys.modules["megatron.core.transformer.moe.moe_layer"].MoESubmodules = DeprecatedMoESubmodules
        sys.modules["megatron.core.transformer.moe.experts"].GroupedMLP = DeprecatedGroupedMLP
        sys.modules["megatron.core.transformer.moe.experts"].SequentialMLP = DeprecatedSequentialMLP
        sys.modules["megatron.core.transformer.moe.experts"].TEGroupedMLP = DeprecatedTEGroupedMLP

        # patch imported module
        from megatron.core.models.gpt import moe_module_specs

        moe_module_specs.MoELayer = DeprecatedMoELayer
        moe_module_specs.MoESubmodules = DeprecatedMoESubmodules
        moe_module_specs.GroupedMLP = DeprecatedGroupedMLP
        moe_module_specs.SequentialMLP = DeprecatedSequentialMLP
        moe_module_specs.TEGroupedMLP = DeprecatedTEGroupedMLP

    # -------------------------------------------------------------------------
    # 2) Primus TopKRouter (optionally via deprecated router)
    # -------------------------------------------------------------------------
    disable_primus_topk = getattr(params, "disable_primus_topk_router", False)
    if not disable_primus_topk:
        log_rank_0("[Patch:megatron.patch.moe_layer] monkey patch TopKRouter...")

        if use_deprecated_moe:
            from primus.backends.megatron.core.transformer.moe.deprecated_20251209.router import (
                DeprecatedTopKRouter,
            )

            sys.modules["megatron.core.transformer.moe.router"].TopKRouter = DeprecatedTopKRouter

        # patch module class
        from primus.backends.megatron.core.transformer.moe.router import (
            PrimusTopKRouter,
        )

        sys.modules["megatron.core.transformer.moe.router"].TopKRouter = PrimusTopKRouter

        # patch imported module
        from megatron.core.transformer.moe import moe_layer

        moe_layer.TopKRouter = PrimusTopKRouter

        if use_deprecated_moe:
            from primus.backends.megatron.core.transformer.moe import (
                deprecated_20251209,
            )

            deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter

    # -------------------------------------------------------------------------
    # 3) MoE Permutation Fusion
    # -------------------------------------------------------------------------
    if getattr(params, "moe_permute_fusion", False):
        log_rank_0(
            "[Patch:megatron.patch.moe_layer] monkey patch permutation with latest fusion version for MoE..."
        )

        from megatron.core.extensions import (
            transformer_engine as ori_transformer_engine,
        )
        from megatron.core.transformer.moe import moe_utils as ori_moe_utils

        from primus.backends.transformer_engine.pytorch.permutation import (
            moe_permute,
            moe_permute_with_probs,
            moe_sort_chunks_by_index,
            moe_sort_chunks_by_index_with_probs,
            moe_unpermute,
        )

        ori_transformer_engine.fused_permute = moe_permute
        ori_transformer_engine.fused_permute_with_probs = moe_permute_with_probs
        ori_transformer_engine.fused_sort_chunks_by_index = moe_sort_chunks_by_index
        ori_transformer_engine.fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
        ori_transformer_engine.fused_unpermute = moe_unpermute

        ori_moe_utils.fused_permute = moe_permute
        ori_moe_utils.fused_permute_with_probs = moe_permute_with_probs
        ori_moe_utils.fused_sort_chunks_by_index = moe_sort_chunks_by_index
        ori_moe_utils.fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
        ori_moe_utils.fused_unpermute = moe_unpermute
        ori_moe_utils.HAVE_TE = True
