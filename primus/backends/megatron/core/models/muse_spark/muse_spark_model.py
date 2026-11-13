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

The decoder layers come from the layer spec (see ``muse_spark_layer_specs``);
the final norm is the plain weighted RMSNorm set in the block spec.
"""

import torch
from megatron.core.models.common.embeddings.language_model_embedding import (
    LanguageModelEmbedding,
)
from megatron.core.models.gpt.gpt_model import GPTModel

from primus.backends.megatron.core.models.muse_spark.number_utils import parse_softcap
from primus.backends.megatron.core.transformer.muse_spark.norms import weightless_rms
from primus.core.utils.module_utils import log_rank_0

__all__ = ["MuseSparkModel", "MuseSparkNormedEmbedding", "softcap_logits", "softcap_value"]


def softcap_value(cap):
    """``final_logit_softcapping`` as a float, or None for no cap (None, 0 or ``"0"``).

    Yaml / CLI may give the cap as a numeric string (``"20"``): convert it before the truth
    test, so ``"0"`` disables the cap and ``"20"`` caps at 20.0 (not a TypeError in forward).
    An inf, NaN, negative or bool cap raises ValueError (``parse_softcap``).
    """
    return parse_softcap(cap) or None


def _apply_softcap(logits: torch.Tensor, multiplier: float, cap) -> torch.Tensor:
    """The soft-cap with an already parsed ``cap`` (a float > 0, or None for no cap)."""
    # HF's order (x * m, then / cap): pre-dividing m / cap rounds differently.
    # div_ / tanh_ run in place on the x * m copy to avoid extra logit-sized buffers.
    logits = logits * multiplier
    if cap is not None:
        logits = logits.div_(cap).tanh_() * cap
    return logits


def softcap_logits(logits: torch.Tensor, multiplier: float, cap) -> torch.Tensor:
    """``cap * tanh(logits * multiplier / cap)``; ``cap`` is parsed (:func:`softcap_value`)."""
    return _apply_softcap(logits, multiplier, softcap_value(cap))


class MuseSparkNormedEmbedding(LanguageModelEmbedding):
    """``LanguageModelEmbedding`` followed by a weightless RMSNorm (no new parameters).

    Applied to the built embedding by class swap, so the state-dict keys stay
    ``embedding.word_embeddings.weight``.
    """

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        return weightless_rms(out, self.muse_embed_norm_eps)


def built_module_summary(model) -> str:
    """The classes of the built Muse Spark pieces (the first local decoder layer), e.g.
    ``embedding=MuseSparkNormedEmbedding layer=MuseSparkTransformerLayer attention=...``. The trainer test
    checks this line, so a spec that falls back to Megatron's stock layers is caught."""
    parts = {"embedding": type(getattr(model, "embedding", None)).__name__}
    layers = getattr(getattr(model, "decoder", None), "layers", None) or []
    if layers:
        layer = layers[0]
        mlp = getattr(layer, "mlp", None)
        parts.update(
            layer=type(layer).__name__,
            attention=type(getattr(layer, "self_attention", None)).__name__,
            residual=type(getattr(layer, "mlp_bda", None)).__name__,
            mlp=type(mlp).__name__,
            router=type(getattr(mlp, "router", None)).__name__,
            experts=type(getattr(mlp, "experts", None)).__name__,
        )
        # the sliding window of the first local layer's core attention (global layer 1 is
        # sliding in every Muse Spark pattern; on later PP stages the first local layer may
        # be a full-attention layer): TE's (left, right); TE stores (-1, 0) for a causal
        # layer without a window, None if the core attention has no window_size attribute
        core = getattr(getattr(layer, "self_attention", None), "core_attention", None)
        window = getattr(core, "window_size", None)
        parts["window"] = None if window is None else tuple(int(w) for w in window)
    return " ".join(f"{k}={v}" for k, v in parts.items())


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
        log_rank_0(f"[Primus:MuseSpark] built modules: {built_module_summary(self)}")

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # Every loss path calls this: the plain output layer, the fused vocab-parallel cross
        # entropy, and Primus's chunked linear+CE (``enable_fused_linear_ce``), whose
        # _ChunkedLinearCE forward and backward call ``model._scale_logits`` per chunk.
        # So every path gets the capped logits; MTP is rejected by MuseSparkTransformerConfig.
        # The multiplier and the cap are parsed once per config value (the config already
        # validated the cap) and reused: this runs per chunk, forward and backward.
        logits = super()._scale_logits(logits)
        key = (self.config.output_logit_multiplier, self.config.final_logit_softcapping)
        cached = self.__dict__.get("_muse_logit_scale")
        if cached is None or cached[0] != key:
            cached = (key, float(key[0]), softcap_value(key[1]))
            self.__dict__["_muse_logit_scale"] = cached
        return _apply_softcap(logits, cached[1], cached[2])
