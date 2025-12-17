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

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# =============================================================================
# Condition Functions
# =============================================================================


def _use_deprecated_moe_layer(ctx: PatchContext) -> bool:
    """Check if deprecated MoE layer is enabled."""
    return getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)


def _use_primus_topk_router(ctx: PatchContext) -> bool:
    """Check if Primus TopKRouter should be used."""
    return not getattr(get_args(ctx), "disable_primus_topk_router", False)


def _use_moe_permute_fusion(ctx: PatchContext) -> bool:
    """Check if MoE permutation fusion is enabled."""
    return getattr(get_args(ctx), "moe_permute_fusion", False)


# =============================================================================
# Patch 1: Deprecated MoE Layer
# =============================================================================


@register_patch(
    "megatron.moe.deprecated_layer",
    backend="megatron",
    phase="before_train",
    description="Replace MoELayer/experts with deprecated 20241209 versions",
    condition=_use_deprecated_moe_layer,
)
def patch_deprecated_moe_layer(ctx: PatchContext):
    """
    Patch Megatron to use deprecated MoE layer implementations.

    Behavior:
        - Replace core MoELayer / MoESubmodules / experts with deprecated versions
        - Sync replacements into gpt.moe_module_specs
    """
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

    log_rank_0(
        f"[Patch:megatron.moe.deprecated_layer]   Patched megatron.core.models.gpt.moe_module_specs.MoELayer "
        f"-> {DeprecatedMoELayer.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_layer]   Patched megatron.core.models.gpt.moe_module_specs.MoESubmodules "
        f"-> {DeprecatedMoESubmodules.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_layer]   Patched megatron.core.models.gpt.moe_module_specs.GroupedMLP "
        f"-> {DeprecatedGroupedMLP.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_layer]   Patched megatron.core.models.gpt.moe_module_specs.SequentialMLP "
        f"-> {DeprecatedSequentialMLP.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.deprecated_layer]   Patched megatron.core.models.gpt.moe_module_specs.TEGroupedMLP "
        f"-> {DeprecatedTEGroupedMLP.__name__}"
    )


# =============================================================================
# Patch 2: Primus TopKRouter
# =============================================================================


@register_patch(
    "megatron.moe.primus_topk_router",
    backend="megatron",
    phase="before_train",
    description="Replace TopKRouter with PrimusTopKRouter",
    condition=_use_primus_topk_router,
)
def patch_primus_topk_router(ctx: PatchContext):
    """
    Patch Megatron to use Primus TopKRouter.

    Behavior:
        - Replace TopKRouter with PrimusTopKRouter
        - If deprecated MoE is enabled, also patch deprecated router
    """
    # Check if deprecated MoE is also enabled
    use_deprecated_moe = getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)

    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe.deprecated_20251209.router import (
            DeprecatedTopKRouter,
        )

        sys.modules["megatron.core.transformer.moe.router"].TopKRouter = DeprecatedTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.router.TopKRouter "
            f"-> {DeprecatedTopKRouter.__name__}"
        )

    # patch module class
    from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

    sys.modules["megatron.core.transformer.moe.router"].TopKRouter = PrimusTopKRouter
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.router.TopKRouter "
        f"-> {PrimusTopKRouter.__name__}"
    )
    # patch imported module
    from megatron.core.transformer.moe import moe_layer

    moe_layer.TopKRouter = PrimusTopKRouter

    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe import deprecated_20251209

        deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.deprecated_20251209.moe_layer.TopKRouter "
            f"-> {PrimusTopKRouter.__name__}"
        )


# =============================================================================
# Patch 3: MoE Permutation Fusion
# =============================================================================


@register_patch(
    "megatron.moe.permute_fusion",
    backend="megatron",
    phase="before_train",
    description="Patch TE and Megatron MoE with fused permutation implementations",
    condition=_use_moe_permute_fusion,
)
def patch_moe_permute_fusion(ctx: PatchContext):
    """
    Patch Transformer Engine and Megatron MoE permutation functions.

    Behavior:
        - Replace TE permutation functions with Primus fused implementations
        - Replace Megatron MoE permutation helpers
    """
    log_rank_0("[Patch:megatron.moe.permute_fusion] Patching with fused permutation implementations...")

    from megatron.core.extensions import transformer_engine as ori_transformer_engine
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
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.extensions.transformer_engine.fused_permute "
        f"-> {moe_permute.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.extensions.transformer_engine.fused_permute_with_probs "
        f"-> {moe_permute_with_probs.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.extensions.transformer_engine.fused_sort_chunks_by_index "
        f"-> {moe_sort_chunks_by_index.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.extensions.transformer_engine.fused_sort_chunks_by_index_with_probs "
        f"-> {moe_sort_chunks_by_index_with_probs.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.extensions.transformer_engine.fused_unpermute "
        f"-> {moe_unpermute.__name__}"
    )

    ori_moe_utils.fused_permute = moe_permute
    ori_moe_utils.fused_permute_with_probs = moe_permute_with_probs
    ori_moe_utils.fused_sort_chunks_by_index = moe_sort_chunks_by_index
    ori_moe_utils.fused_sort_chunks_by_index_with_probs = moe_sort_chunks_by_index_with_probs
    ori_moe_utils.fused_unpermute = moe_unpermute
    ori_moe_utils.HAVE_TE = True
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.fused_permute "
        f"-> {moe_permute.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.fused_permute_with_probs "
        f"-> {moe_permute_with_probs.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.fused_sort_chunks_by_index "
        f"-> {moe_sort_chunks_by_index.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.fused_sort_chunks_by_index_with_probs "
        f"-> {moe_sort_chunks_by_index_with_probs.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.fused_unpermute "
        f"-> {moe_unpermute.__name__}"
    )
    log_rank_0(
        f"[Patch:megatron.moe.permute_fusion]   Patched megatron.core.transformer.moe.moe_utils.HAVE_TE to True"
    )
