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
            elif hasattr(unwrapped, "layers"):
                parts.layers.extend(unwrapped.layers)
            else:
                raise ValueError(f"Cannot find transformer layers in model chunk: {type(unwrapped)}")

            if hasattr(unwrapped, "embedding"):
                parts.embedding = unwrapped.embedding
            if hasattr(unwrapped, "output_layer"):
                parts.output_layer = unwrapped.output_layer

        return parts

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
            return BenchInputs(args=[(ctx.seq_len, ctx.batch_size, ctx.hidden_size)])

        if kind == ATTENTION:
            return BenchInputs(
                args=[
                    (ctx.seq_len, ctx.batch_size, ctx.hidden_size),
                    ((1, 1, ctx.seq_len_per_cp, ctx.seq_len), torch.bool),
                ]
            )

        if kind in (MLP, MOE):
            return BenchInputs(args=[(ctx.seq_len, ctx.batch_size, ctx.hidden_size)])

        if kind == EMBEDDING:
            return BenchInputs(args=[((ctx.batch_size, ctx.seq_len_per_cp), torch.int64)])

        if kind == OUTPUT:
            return BenchInputs(args=[(ctx.seq_len_per_cp, ctx.batch_size, ctx.hidden_size)])

        return None
