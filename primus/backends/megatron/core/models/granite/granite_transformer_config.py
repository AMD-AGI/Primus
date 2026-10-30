###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer config for IBM Granite (dense).

Granite is a Llama-layout decoder with four scalar multipliers. Two of them map
to upstream Megatron MuP fields, set in the preset:

  * ``embedding_multiplier``  -> ``mup_embedding_mult`` (``LanguageModelEmbedding``)
  * ``1 / logits_scaling``    -> ``mup_output_mult``    (``LanguageModule._scale_logits``)

The other two have no settable upstream field (``softmax_scale`` is excluded from
the Megatron CLI). This config adds them:

  * ``attention_multiplier``  -> sets ``softmax_scale`` before the MuP block runs, so
    the attention scale is HF's ``attention_multiplier`` whatever ``kv_channels`` is
    (without it, MuP would fall back to ``1/kv_channels``; it is required).
  * ``residual_multiplier``   -> applied by ``GraniteTransformerLayer``.
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.models.granite.granite_validation import (
    positive_finite,
)


@dataclass
class GraniteTransformerConfig(TransformerConfig):
    """``TransformerConfig`` plus Granite's ``residual_multiplier`` and ``attention_multiplier``.

    Both are required numbers, validated like the converter does (positive, finite, not a
    bool or a string). There is no fallback scale: with ``use_mup`` Megatron's own fallback
    would be ``1/kv_channels``, which matches neither HF nor the converter, so a missing
    ``attention_multiplier`` is an error. Granite's multipliers are fixed constants of the
    checkpoint, so MuP width scaling (``mup_base_hidden_size`` != ``hidden_size``, which
    rescales init, LR and logits) is refused.
    """

    residual_multiplier: Optional[float] = None
    """Scale of each sub-block output (attention and MLP) before the residual add
    (HF ``residual_multiplier``). Required: the ``None`` default only exists because
    dataclass fields after defaulted parent fields need a default; ``__post_init__``
    refuses it. Give 1.0 explicitly for no scaling (the converter maps a missing HF key
    to HF's default 1.0)."""

    attention_multiplier: Optional[float] = None
    """Granite ``attention_multiplier``: the softmax scale (HF ``GraniteAttention.scaling``).
    It becomes ``softmax_scale``. Required."""

    def __post_init__(self):
        self.attention_multiplier = positive_finite("attention_multiplier", self.attention_multiplier)
        if self.softmax_scale is not None and self.softmax_scale != self.attention_multiplier:
            raise ValueError(
                f"softmax_scale={self.softmax_scale} conflicts with "
                f"attention_multiplier={self.attention_multiplier}"
            )
        self.softmax_scale = self.attention_multiplier
        self.residual_multiplier = positive_finite("residual_multiplier", self.residual_multiplier)
        if self.residual_multiplier != 1.0 and self.inference_fuse_tp_communication:
            # Upstream TransformerLayer skips self_attn_bda / mlp_bda on that path, so the
            # scaled residual add of GraniteTransformerLayer would be silently dropped.
            raise ValueError(
                "inference_fuse_tp_communication is not supported with Granite residual_multiplier "
                f"{self.residual_multiplier}: the fused TP inference path bypasses the scaled residual add"
            )
        if not self.use_mup and (self.mup_embedding_mult != 1.0 or self.mup_output_mult != 1.0):
            # Upstream applies mup_embedding_mult (LanguageModelEmbedding) and mup_output_mult
            # (LanguageModule._scale_logits) only when use_mup is true; without it both
            # Granite multipliers would be silently dropped.
            raise ValueError(
                f"Granite needs use_mup=True to apply mup_embedding_mult={self.mup_embedding_mult} "
                f"and mup_output_mult={self.mup_output_mult}; use_mup=False would silently drop them"
            )
        if self.mup_base_hidden_size is not None and self.mup_base_hidden_size != self.hidden_size:
            raise ValueError(
                f"Granite does not support MuP width scaling: mup_base_hidden_size="
                f"{self.mup_base_hidden_size} != hidden_size={self.hidden_size} would rescale init, "
                "LR and logits away from the checkpoint's fixed multipliers"
            )
        super().__post_init__()
        # super().__post_init__() (upstream TransformerConfig's MuP block) sets mup_width_mult.
        if self.mup_width_mult != 1.0:
            raise ValueError(f"Granite needs mup_width_mult == 1.0, got {self.mup_width_mult}")
        if self.softmax_scale != self.attention_multiplier:
            raise ValueError(
                f"softmax_scale {self.softmax_scale} != attention_multiplier {self.attention_multiplier}"
            )
