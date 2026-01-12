###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Transformer Patches

This module contains patches that modify Megatron's transformer-related
components (configs, blocks, etc.) to integrate Primus-specific behavior.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.transformer.patch_mla_attention",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch MLA attention to use Primus PaddedMLASelfAttention "
        "when use_turbo_parallel_linear is enabled."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "use_turbo_parallel_linear", False),
)
def patch_mla_attention(ctx: PatchContext):
    """
    Patch Megatron MLA attention to support padded fusion.

    Behavior (moved from MegatronTrainer.patch_mla_attention):
        - If module_config.fused_padded_mla_attention is True, replace
          multi_latent_attention.MLASelfAttention and
          gpt_layer_specs.MLASelfAttention with PaddedMLASelfAttention.
    """
    log_rank_0("MegatronPatches: monkey patch MLA attention to support padded fusion...")

    # pad module definition
    from megatron.core.transformer import multi_latent_attention

    from primus.backends.megatron.core.transformer.multi_latent_attention import (
        PrimusMLASelfAttention,
    )

    multi_latent_attention.MLASelfAttention = PrimusMLASelfAttention
    log_rank_0(
        f"[Patch:megatron.transformer.mla_attention]   Patched megatron.core.transformer.multi_latent_attention.MLASelfAttention "
        f"-> {PrimusMLASelfAttention.__name__}"
    )

    # pad imported module
    from megatron.core.models.gpt import gpt_layer_specs

    gpt_layer_specs.MLASelfAttention = PrimusMLASelfAttention
    log_rank_0(
        f"[Patch:megatron.transformer.mla_attention]   Patched megatron.core.models.gpt.gpt_layer_specs.MLASelfAttention "
        f"-> {PrimusMLASelfAttention.__name__}"
    )
