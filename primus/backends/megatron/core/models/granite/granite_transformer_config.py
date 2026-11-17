###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Transformer config for IBM Granite (dense).

Granite is a Llama-layout decoder with four scalar multipliers. Three of them map to
upstream Megatron MuP fields, set in the preset (or by the converter):

  * ``embedding_multiplier``  -> ``mup_embedding_mult`` (``LanguageModelEmbedding``)
  * ``1 / logits_scaling``    -> ``mup_output_mult``    (``LanguageModule._scale_logits``)
  * ``attention_multiplier``  -> upstream ``softmax_scale``, which the MuP block of
    ``TransformerConfig.__post_init__`` sets to
    ``sqrt(mup_base_head_dim or 1) / kv_channels ** mup_attn_scale_power``
    (1/64 = 0.015625 for granite-4.1-3b: ``kv_channels`` 64, defaults).

The fourth has no upstream field, so this config adds it:

  * ``residual_multiplier``   -> applied by the scaled bias-dropout-add in the layer
    spec (``core/fusions/scaled_bias_dropout_add.py``).
"""

from dataclasses import dataclass
from typing import Optional

from megatron.core.transformer.transformer_config import TransformerConfig

from primus.core.utils.validation import positive_finite


@dataclass
class GraniteTransformerConfig(TransformerConfig):
    """``TransformerConfig`` plus Granite's ``residual_multiplier``.

    ``residual_multiplier`` is a required number, validated like the converter does
    (positive, finite, not a bool or a string). Granite needs ``use_mup`` (it is upstream's
    only switch for the embedding and logits multipliers and for the 1/d attention scale).
    Granite's multipliers are fixed constants of the checkpoint, so MuP width scaling
    (``mup_base_hidden_size`` != ``hidden_size``, which rescales init, LR and logits) is
    refused.
    """

    residual_multiplier: Optional[float] = None
    """Scale of each sub-block output (attention and MLP) before the residual add
    (HF ``residual_multiplier``). Required: the ``None`` default only exists because
    dataclass fields after defaulted parent fields need a default; ``__post_init__``
    refuses it. Give 1.0 explicitly for no scaling (the converter maps a missing HF key
    to HF's default 1.0)."""

    def __post_init__(self):
        self.residual_multiplier = positive_finite("residual_multiplier", self.residual_multiplier)
        if self.residual_multiplier != 1.0 and self.inference_fuse_tp_communication:
            # Upstream TransformerLayer skips self_attn_bda / mlp_bda on that path, so the
            # scaled residual add would be silently dropped.
            raise ValueError(
                "inference_fuse_tp_communication is not supported with Granite residual_multiplier "
                f"{self.residual_multiplier}: the fused TP inference path bypasses the scaled residual add"
            )
        if not self.use_mup:
            # Upstream applies mup_embedding_mult, mup_output_mult and the 1/d attention
            # scale only when use_mup is true; without it the Granite multipliers are dropped.
            raise ValueError(
                "Granite needs use_mup=True: upstream applies mup_embedding_mult, mup_output_mult "
                "and the MuP softmax scale (attention_multiplier) only with use_mup"
            )
        if self.mup_base_hidden_size is not None and self.mup_base_hidden_size != self.hidden_size:
            raise ValueError(
                f"Granite does not support MuP width scaling: mup_base_hidden_size="
                f"{self.mup_base_hidden_size} != hidden_size={self.hidden_size} would rescale init, "
                "LR and logits away from the checkpoint's fixed multipliers"
            )
        super().__post_init__()
