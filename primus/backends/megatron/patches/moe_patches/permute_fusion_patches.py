###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE Permutation Fusion Patches

Patches for replacing TE and Megatron MoE permutation functions with fused implementations.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.moe.permute_fusion",
    backend="megatron",
    phase="before_train",
    description="Patch TE and Megatron MoE with fused permutation implementations",
    condition=lambda ctx: getattr(get_args(ctx), "moe_permute_fusion", False),
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
