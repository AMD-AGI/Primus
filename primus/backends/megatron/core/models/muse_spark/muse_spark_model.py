###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark text model: Megatron ``GPTModel`` with an embedding norm and a soft-capped head.

* The embedding output passes through a weightless RMSNorm (``norm_epsilon``),
  as HF ``MuseSparkTextNormedEmbedding`` does.
* The untied output logits become
  ``final_logit_softcapping * tanh(logits * output_logit_multiplier / final_logit_softcapping)``.

The decoder layers come from the layer spec (see ``muse_spark_layer_specs``);
the final norm is the plain weighted RMSNorm set in the block spec.
"""

import torch
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.gpt.gpt_model import GPTModel

__all__ = ["MuseSparkModel", "MuseSparkNormedEmbedding", "softcap_logits"]


def softcap_logits(logits: torch.Tensor, multiplier: float, cap) -> torch.Tensor:
    logits = logits * multiplier
    if cap:
        logits = torch.tanh(logits / cap) * cap
    return logits


class MuseSparkNormedEmbedding(LanguageModelEmbedding):
    """``LanguageModelEmbedding`` followed by a weightless RMSNorm (no new parameters).

    Applied to the built embedding by class swap, so the state-dict keys stay
    ``embedding.word_embeddings.weight``.
    """

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        x = out.float()
        x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + self.muse_embed_norm_eps, -0.5)
        return x.type_as(out)


class MuseSparkModel(GPTModel):
    """GPTModel with the Muse Spark embedding norm and logit soft-cap."""

    def __init__(self, config, *args, **kwargs):
        kwargs.setdefault("position_embedding_type", "rope")
        super().__init__(config, *args, **kwargs)
        if self.pre_process and getattr(config, "embedding_norm", True):
            self.embedding.__class__ = MuseSparkNormedEmbedding
            self.embedding.muse_embed_norm_eps = float(config.layernorm_epsilon)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = super()._scale_logits(logits)
        return softcap_logits(
            logits,
            float(getattr(self.config, "output_logit_multiplier", 1.0)),
            getattr(self.config, "final_logit_softcapping", None),
        )
