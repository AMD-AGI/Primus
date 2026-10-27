###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark language model: upstream ``GPTModel`` plus two model-level ops.

* The token embedding is followed by a weightless RMSNorm. It is installed as
  a forward hook so it runs exactly when the embedding does, as in the
  reference ``MuseSparkTextNormedEmbedding``.
* The LM-head logits are scaled by ``muse_output_multiplier`` and tanh
  soft-capped through ``LanguageModule._scale_logits``, which ``GPTModel``
  calls on every logits path before the loss.
"""

import torch
from megatron.core.jit import jit_fuser
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.torch_norm import L2Norm

from primus.backends.megatron.core.models.muse_spark.muse_spark_ops import (
    softcap_logits,
)
from primus.backends.megatron.core.models.muse_spark.muse_spark_transformer_config import (
    MuseSparkTransformerConfig,
)

__all__ = ["MuseSparkGPTModel"]

_fused_softcap_logits = jit_fuser(softcap_logits)


class MuseSparkGPTModel(GPTModel):
    """``GPTModel`` with Muse Spark's normalized embedding and scaled, capped logits."""

    def __init__(self, config: MuseSparkTransformerConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if self.pre_process:
            self.embedding_norm = L2Norm(hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
            self.embedding.register_forward_hook(self._normalize_embedding)

    def _normalize_embedding(self, module, inputs, output: torch.Tensor) -> torch.Tensor:
        return self.embedding_norm(output)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        logits = super()._scale_logits(logits)
        cap = self.config.final_logit_softcapping
        if cap is None:
            return logits * self.config.muse_output_multiplier
        return _fused_softcap_logits(logits, self.config.muse_output_multiplier, cap)
