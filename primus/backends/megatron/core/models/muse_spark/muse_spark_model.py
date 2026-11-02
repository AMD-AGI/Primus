###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark text model: Megatron ``GPTModel`` with an embedding norm and a soft-capped head.

* The embedding output passes through a weightless RMSNorm (eps =
  ``config.layernorm_epsilon``, i.e. Megatron's ``--norm-epsilon``), as HF
  ``MuseSparkTextNormedEmbedding`` does. Megatron's embedding applies dropout
  before it returns, so the norm runs *after* embedding dropout (HF: norm, no
  dropout). This matches HF only with ``hidden_dropout == 0`` (the presets' value);
  a non-zero ``hidden_dropout`` is rejected at build time when ``embedding_norm`` is set.
* The untied output logits become
  ``final_logit_softcapping * tanh(logits * output_logit_multiplier / final_logit_softcapping)``
  (only the multiplier when ``final_logit_softcapping`` is 0 / unset).
* With ``config.muse_audio_config`` set, the first stage has the waveform audio encoder
  (``audio_tower``). ``forward(..., input_values=, input_values_mask=)`` embeds the tokens
  with the audio placeholder ids set to 0, then writes the encoder features (after the
  embedding norm) into the placeholder rows, as HF ``MuseSparkModel.forward`` does.

The decoder layers come from the layer spec (see ``muse_spark_layer_specs``);
the final norm is the plain weighted RMSNorm set in the block spec.
"""

import torch
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from primus.backends.megatron.core.models.muse_spark.audio_encoder import (
    MuseSparkAudioEncoder,
    merge_padded_audio,
    padded_audio_features,
)
from primus.backends.megatron.core.transformer.muse_spark.norms import _rms

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
        return _rms(out.float(), self.muse_embed_norm_eps).type_as(out)


class MuseSparkModel(GPTModel):
    """GPTModel with the Muse Spark embedding norm and logit soft-cap."""

    def __init__(self, config, *args, **kwargs):
        kwargs.setdefault("position_embedding_type", "rope")
        super().__init__(config, *args, **kwargs)
        if self.pre_process and config.embedding_norm:
            if config.hidden_dropout:
                raise ValueError(
                    "Muse Spark: hidden_dropout must be 0 (the embedding norm runs after Megatron's "
                    f"embedding dropout, which HF does not have); got {config.hidden_dropout}"
                )
            self.embedding.__class__ = MuseSparkNormedEmbedding
            self.embedding.muse_embed_norm_eps = float(config.layernorm_epsilon)
        if self.pre_process and getattr(config, "muse_audio_config", None):
            self.audio_tower = MuseSparkAudioEncoder(config.muse_audio_config)

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask,
        decoder_input=None,
        *args,
        input_values=None,
        input_values_mask=None,
        **kwargs,
    ):
        """GPTModel.forward; with ``input_values`` (``[B, samples]`` waveform) the audio
        features replace the embeddings at the ``audio_token_id`` positions."""
        if input_values is not None and self.pre_process:
            if getattr(self, "audio_tower", None) is None:
                raise ValueError("Muse Spark: input_values given, but the model has no audio_tower")
            if decoder_input is not None:
                raise ValueError("Muse Spark: give input_values or decoder_input, not both")
            if self.config.sequence_parallel or self.config.context_parallel_size > 1:
                raise NotImplementedError("Muse Spark audio input with sequence / context parallelism")
            audio_mask = input_ids == self.config.audio_token_id
            embeddings = self.embedding(
                input_ids=input_ids.masked_fill(audio_mask, 0), position_ids=position_ids
            )
            # padded features + an on-device gather: no data-dependent shape, no host sync
            feats, n_tok = padded_audio_features(
                self.audio_tower, input_values, input_values_mask, float(self.config.layernorm_epsilon)
            )
            decoder_input = merge_padded_audio(embeddings, audio_mask, feats, n_tok)
        return super().forward(input_ids, position_ids, attention_mask, decoder_input, *args, **kwargs)

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Every loss path calls this: the plain output layer, the fused vocab-parallel cross
        # entropy, and Primus's chunked linear+CE (``enable_fused_linear_ce``), whose
        # _ChunkedLinearCE forward and backward call ``model._scale_logits`` per chunk.
        # So every path gets the capped logits; MTP is rejected by MuseSparkTransformerConfig.
        logits = super()._scale_logits(logits)
        return softcap_logits(
            logits,
            float(self.config.output_logit_multiplier),
            self.config.final_logit_softcapping,
        )
