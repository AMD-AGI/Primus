###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron bench adapter.

This is the arrangement the harness grew up around, so the adapter is a
transcription rather than a translation: layers under
``language_model.decoder.layers`` (or ``encoder``, or bare ``layers``), block
halves named ``self_attention`` and ``mlp``, and hidden states carried as
``[S, B, D]`` with a boolean attention mask.

DeepSeek-V4's hybrid layers are the one departure, and they keep their existing
special case: their ``forward`` takes ``[B, S, K, D]`` plus keyword-only
``position_ids`` / ``token_ids``, so feeding them the stock inputs would either
crash or measure the wrong kernel.
"""

import inspect
from typing import Optional

import torch

from primus.core.projection.bench_harness.base import (
    ATTENTION,
    EMBEDDING,
    LAYER,
    MLP,
    MOE,
    OUTPUT,
    BenchContext,
    BenchInputs,
    BenchModelAdapter,
    LayerParts,
    ModelParts,
)


def _unwrap(module):
    """Strip DistributedDataParallel / pipeline wrappers off a model chunk."""
    return _unwrap(module.module) if hasattr(module, "module") else module


def _model_rotary(model):
    """The GPTModel's RotaryEmbedding when it feeds the layers one.

    GPTModel computes ``rotary_pos_emb`` only for plain ``rope``; multi-latent
    attention builds its own inside the attention module.
    """
    config = getattr(model, "config", None)
    if getattr(model, "position_embedding_type", None) != "rope":
        return None
    if getattr(config, "multi_latent_attention", False):
        return None
    rotary = getattr(model, "rotary_pos_emb", None)
    return rotary if callable(rotary) else None


def _forward_params(module):
    try:
        return inspect.signature(module.forward).parameters
    except (TypeError, ValueError):
        return {}


class MegatronBenchAdapter(BenchModelAdapter):
    framework = "megatron"

    def discover(self, model) -> ModelParts:
        parts = ModelParts()
        chunks = model if isinstance(model, list) else [model]

        for chunk in chunks:
            unwrapped = _unwrap(chunk)

            language_model = getattr(unwrapped, "language_model", None)
            if language_model is not None:
                if hasattr(language_model, "embedding"):
                    parts.embedding = language_model.embedding
                if hasattr(language_model, "output_layer"):
                    parts.output_layer = language_model.output_layer

                if hasattr(language_model, "encoder") and hasattr(language_model.encoder, "layers"):
                    parts.layers.extend(language_model.encoder.layers)
                elif hasattr(language_model, "decoder") and hasattr(language_model.decoder, "layers"):
                    parts.layers.extend(language_model.decoder.layers)
                elif hasattr(language_model, "layers"):
                    parts.layers.extend(language_model.layers)
                continue

            if hasattr(unwrapped, "decoder") and hasattr(unwrapped.decoder, "layers"):
                parts.layers.extend(unwrapped.decoder.layers)
                parts.layer_types.extend(getattr(unwrapped.decoder, "layer_type_list", None) or [])
            elif hasattr(unwrapped, "layers"):
                parts.layers.extend(unwrapped.layers)
            else:
                raise ValueError(f"Cannot find transformer layers in model chunk: {type(unwrapped)}")

            if hasattr(unwrapped, "embedding"):
                parts.embedding = unwrapped.embedding
            if hasattr(unwrapped, "output_layer"):
                parts.output_layer = unwrapped.output_layer
                if hasattr(unwrapped, "compute_language_model_loss"):
                    parts.loss_owner = unwrapped
                # With tied embeddings the head allocates no weight; the model passes
                # the embedding's to every forward call.
                if getattr(unwrapped.output_layer, "weight", None) is None and hasattr(
                    unwrapped, "shared_embedding_or_output_weight"
                ):
                    self._output_weight = unwrapped.shared_embedding_or_output_weight()

            rotary = _model_rotary(unwrapped)
            if rotary is not None:
                self._rotary = rotary

        return parts

    def _rotary_kwargs(self, module, seq_len: int) -> dict:
        """``rotary_pos_emb`` as GPTModel hands it to every layer, when it does.

        Without it the layer skips the RoPE kernels training runs on q and k.
        """
        rotary = getattr(self, "_rotary", None)
        if rotary is None or "rotary_pos_emb" not in _forward_params(module):
            return {}
        with torch.no_grad():
            emb = rotary(seq_len)
        if emb.shape[0] != seq_len:
            # Context-parallel slicing inside RotaryEmbedding; the layer is fed
            # the full sequence here, so leave RoPE out rather than mismatch it.
            return {}
        return {"rotary_pos_emb": emb}

    def layer_submodules(self, layer) -> LayerParts:
        return LayerParts(
            attention=getattr(layer, "self_attention", None),
            mlp=getattr(layer, "mlp", None),
        )

    def inputs(self, kind: str, module, ctx: BenchContext) -> Optional[BenchInputs]:
        from primus.core.projection.module_profilers.utils import v4_module_inputs

        hc_mult = getattr(getattr(module, "config", None), "hc_mult", 1)
        v4_kind = {LAYER: "layer", ATTENTION: "attention", MOE: "moe"}.get(kind)
        if v4_kind is not None:
            v4 = v4_module_inputs(module, ctx.batch_size, ctx.seq_len, ctx.hidden_size, hc_mult, v4_kind)
            if v4 is not None:
                shapes, kwargs = v4
                return BenchInputs(args=shapes, kwargs=kwargs or {})

        if kind == LAYER:
            return BenchInputs(
                args=[(ctx.seq_len, ctx.batch_size, ctx.hidden_size)],
                kwargs=self._rotary_kwargs(module, ctx.seq_len),
            )

        if kind == ATTENTION:
            return BenchInputs(
                args=[
                    (ctx.seq_len, ctx.batch_size, ctx.hidden_size),
                    ((1, 1, ctx.seq_len_per_cp, ctx.seq_len), torch.bool),
                ],
                kwargs=self._rotary_kwargs(module, ctx.seq_len),
            )

        if kind in (MLP, MOE):
            return BenchInputs(args=[(ctx.seq_len, ctx.batch_size, ctx.hidden_size)])

        if kind == EMBEDDING:
            return BenchInputs(args=[((ctx.batch_size, ctx.seq_len_per_cp), torch.int64)])

        if kind == OUTPUT:
            kwargs = {}
            if getattr(module, "weight", None) is None and getattr(self, "_output_weight", None) is not None:
                kwargs["weight"] = self._output_weight
            return BenchInputs(args=[(ctx.seq_len_per_cp, ctx.batch_size, ctx.hidden_size)], kwargs=kwargs)

        return None
