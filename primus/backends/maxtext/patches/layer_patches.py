###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText Layer Patches

Replaces MaxText layer implementations with Primus-optimised versions:
    - ``NANOOFp8Quantization`` → ``PrimusNANOOFp8Quantization``
    - ``AttentionOp`` → ``PrimusAttentionOp``
    - ``Attention`` → ``PrimusAttention``
    - ``RoutedMoE`` → ``PrimusRoutedMoE``
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


# ============================================================================
# Quantization
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.quantization",
    backend="maxtext",
    phase="setup",
    description="Replace NANOOFp8Quantization with Primus implementation",
    condition=lambda ctx: True,
)
def patch_quantization(ctx: PatchContext) -> None:
    """
    Replace ``MaxText.layers.quantizations.NANOOFp8Quantization`` with
    ``PrimusNANOOFp8Quantization``.
    """
    log_rank_0("[Patch:maxtext.layers.quantization] Patching NANOOFp8Quantization...")

    import MaxText.layers.quantizations as orig_quantizations

    from primus.backends.maxtext.layers.quantizations import (
        PrimusNANOOFp8Quantization,
    )

    orig_quantizations.NANOOFp8Quantization = PrimusNANOOFp8Quantization

    warning_rank_0("[Patch:maxtext.layers.quantization] NANOOFp8Quantization patched successfully.")


# ============================================================================
# Attention
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.attention",
    backend="maxtext",
    phase="setup",
    description="Replace AttentionOp and Attention with Primus implementations",
    condition=lambda ctx: True,
)
def patch_attention(ctx: PatchContext) -> None:
    """
    Replace MaxText attention classes with Primus versions:
        - ``AttentionOp`` → ``PrimusAttentionOp``
        - ``Attention`` → ``PrimusAttention``
    """
    log_rank_0("[Patch:maxtext.layers.attention] Patching Attention layers...")

    import MaxText.layers.attention_mla as orig_attention_mla
    import MaxText.layers.attention_op as orig_attention_op
    import MaxText.layers.attentions as orig_attentions

    from primus.backends.maxtext.layers.attention_op import PrimusAttentionOp
    from primus.backends.maxtext.layers.attentions import PrimusAttention

    orig_attention_op.AttentionOp = PrimusAttentionOp
    orig_attentions.AttentionOp = PrimusAttentionOp

    orig_attentions.Attention = PrimusAttention
    orig_attention_mla.Attention = PrimusAttention

    warning_rank_0("[Patch:maxtext.layers.attention] Attention layers patched successfully.")


# ============================================================================
# Mixture-of-Experts (MoE)
# ============================================================================


@register_patch(
    patch_id="maxtext.layers.moe",
    backend="maxtext",
    phase="setup",
    description="Replace RoutedMoE with Primus implementation",
    condition=lambda ctx: True,
)
def patch_moe(ctx: PatchContext) -> None:
    """
    Replace ``MaxText.layers.moe.RoutedMoE`` with ``PrimusRoutedMoE``.
    """
    log_rank_0("[Patch:maxtext.layers.moe] Patching RoutedMoE...")

    import MaxText.layers.moe as orig_moe

    from primus.backends.maxtext.layers.moe import PrimusRoutedMoE

    orig_moe.RoutedMoE = PrimusRoutedMoE

    warning_rank_0("[Patch:maxtext.layers.moe] RoutedMoE patched successfully.")
