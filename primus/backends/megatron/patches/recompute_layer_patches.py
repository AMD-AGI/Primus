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
    "megatron.transformer.custom_recompute_layer_ids",
    backend="megatron",
    phase="before_train",
    description=(
        "Monkey patch TransformerConfig and TransformerBlock to support "
        "Primus-provided recompute_layer_ids."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "recompute_layer_ids", None) is not None,
)
def patch_custom_recompute_layer_ids(ctx: PatchContext):
    """
    Patch Megatron's TransformerConfig and TransformerBlock for custom recompute.

    Behavior (moved from MegatronTrainer.patch_custom_recompute_layer_ids):
        - If module_config.recompute_layer_ids is provided, attach it to
          megatron.core.transformer.transformer_config.TransformerConfig.
        - Wrap TransformerConfig.__post_init__ to temporarily disable
          recompute_granularity during initialization.
        - Replace various model modules' TransformerBlock with
          PrimusTransformerBlock so that checkpoint_forward uses the
          recompute_layer_ids information.
    """
    args = get_args(ctx)

    import megatron.core.transformer.transformer_config as config_mod

    # 1) Attach Primus-provided recompute_layer_ids to TransformerConfig
    config_mod.TransformerConfig.recompute_layer_ids = args.recompute_layer_ids

    # 2) Wrap __post_init__ to temporarily clear recompute_granularity
    orig_post_init = config_mod.TransformerConfig.__post_init__

    def new_post_init(self):
        tmp = getattr(self, "recompute_granularity", None)
        self.recompute_granularity = None
        orig_post_init(self)
        self.recompute_granularity = tmp

    config_mod.TransformerConfig.__post_init__ = new_post_init

    # 3) Replace TransformerBlock in various Megatron models
    log_rank_0("MegatronPatches: monkey patch TransformerBlock checkpoint_forward...")

    import megatron.core.models.bert.bert_model as orig_bert_model
    import megatron.core.models.gpt.gpt_model as orig_gpt_model
    import megatron.core.models.retro.decoder_attention as orig_decoder_attention
    import megatron.core.models.T5.t5_model as orig_t5_model
    import megatron.core.models.vision.clip_vit_model as orig_clip_vit_model
    import megatron.core.models.vision.radio as orig_radio
    import megatron.core.transformer.transformer_block as orig_transformer_block

    from primus.backends.megatron.core.transformer.transformer_block import (
        PrimusTransformerBlock,
    )

    # Apply the same replacement that previously lived in MegatronTrainer
    orig_transformer_block.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.transformer.transformer_block.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_bert_model.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.bert.bert_model.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_gpt_model.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.gpt.gpt_model.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_decoder_attention.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.retro.decoder_attention.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_t5_model.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.T5.t5_model.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_clip_vit_model.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.vision.clip_vit_model.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
    orig_radio.TransformerBlock = PrimusTransformerBlock
    log_rank_0(
        f"[Patch:megatron.transformer.recompute_layer_ids]   Patched megatron.core.models.vision.radio.TransformerBlock "
        f"-> {PrimusTransformerBlock.__name__}"
    )
