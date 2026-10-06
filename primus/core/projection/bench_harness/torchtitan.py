###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""TorchTitan bench adapter.

TorchTitan is the same PyTorch and the same CUDA as Megatron, so the harness
itself carries over untouched -- what differs is only where the layers live and
what their ``forward`` wants.  Every model family TorchTitan ships (Llama 3,
Llama 4, DeepSeek V3, Qwen 3, GPT-OSS) agrees on both:

* ``Transformer.tok_embeddings`` / ``.layers`` / ``.norm`` / ``.output``, where
  ``layers`` is a ``ModuleDict`` keyed by the layer's global index as a string --
  so with pipeline parallelism a rank holds only its own stage's layers, still
  under their global keys.
* ``TransformerBlock.attention`` plus either ``.feed_forward`` (dense) or
  ``.moe``, and a forward of ``(hidden[B, S, D], rope, attention_masks,
  positions=None)``.

Two things the adapter has to supply rather than describe. The rope table is
precomputed on the model (``freqs_cis``, or ``rope_cache`` on Qwen 3 and
GPT-OSS) and its *values* decide whether the kernel is valid, so the real buffer
is handed over rather than a random tensor of the right shape.  And the input
dtype is read off the module's own parameters instead of assumed, because
TorchTitan builds some models in fp32 and reaches bf16 through autocast.

The MoE all-to-all is not decomposed here.  TorchTitan puts expert
communication in a parallelization style applied around ``MoE``, not in methods
on it, so there is no ``dispatch``/``combine`` pair to time separately -- the
whole MoE layer is measured and the EP correction is left to the analytical
model, exactly as it already is for Megatron's DeepSeek-V4 layers.
"""

from typing import List, Optional

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

# Where each model family keeps its precomputed rotary table.
_ROPE_ATTRS = ("freqs_cis", "rope_cache")


def _unwrap(module):
    """Strip DDP / activation-checkpoint / compile wrappers off a model part."""
    seen = set()
    while True:
        if id(module) in seen:
            return module
        seen.add(id(module))
        for attr in ("module", "_orig_mod", "_checkpoint_wrapped_module"):
            inner = getattr(module, attr, None)
            if inner is not None and inner is not module:
                module = inner
                break
        else:
            return module


def _param_dtype(module, default=torch.bfloat16):
    """Return the dtype of *module*'s first floating-point parameter."""
    if module is None:
        return default
    for param in module.parameters(recurse=True):
        if param.is_floating_point():
            return param.dtype
    return default


def _ordered_layers(layers) -> List[object]:
    """Return a ``ModuleDict`` of layers in global index order.

    TorchTitan keys the dict by the layer's global index as a string, so sorting
    the keys numerically is what recovers the stack order -- dict insertion order
    would happen to agree today but is not what the key means.
    """
    if hasattr(layers, "items"):
        try:
            return [module for _, module in sorted(layers.items(), key=lambda kv: int(kv[0]))]
        except (TypeError, ValueError):
            return list(layers.values())
    return list(layers)


class TorchTitanBenchAdapter(BenchModelAdapter):
    framework = "torchtitan"

    def discover(self, model) -> ModelParts:
        parts = ModelParts()
        chunks = model if isinstance(model, (list, tuple)) else [model]

        for chunk in chunks:
            unwrapped = _unwrap(chunk)

            layers = getattr(unwrapped, "layers", None)
            if layers is None:
                raise ValueError(
                    "Cannot find TorchTitan transformer layers on "
                    f"{type(unwrapped).__name__}; expected a 'layers' ModuleDict."
                )
            parts.layers.extend(_ordered_layers(layers))

            # With pipeline parallelism only the first stage carries the
            # embedding and only the last carries the head; both are set to
            # ``None`` on the stages in between.
            embedding = getattr(unwrapped, "tok_embeddings", None)
            if embedding is not None:
                parts.embedding = embedding
            output_layer = getattr(unwrapped, "output", None)
            if output_layer is not None:
                parts.output_layer = output_layer

            rope = self._rope_table(unwrapped)
            if rope is not None:
                self._rope = rope

        return parts

    def _rope_table(self, model) -> Optional[torch.Tensor]:
        for attr in _ROPE_ATTRS:
            table = getattr(model, attr, None)
            if isinstance(table, torch.Tensor):
                return table
        return None

    def layer_submodules(self, layer) -> LayerParts:
        layer = _unwrap(layer)
        # A block carries exactly one of the two; which one is what makes the
        # layer dense or MoE.
        mlp = getattr(layer, "moe", None)
        if mlp is None:
            mlp = getattr(layer, "feed_forward", None)
        return LayerParts(attention=getattr(layer, "attention", None), mlp=mlp)

    def inputs(self, kind: str, module, ctx: BenchContext) -> Optional[BenchInputs]:
        hidden = (ctx.seq_len_per_cp, ctx.hidden_size)
        dtype = _param_dtype(module)

        if kind == EMBEDDING:
            return BenchInputs(args=[((ctx.batch_size, ctx.seq_len_per_cp), torch.int64)])

        if kind == OUTPUT:
            return BenchInputs(args=[((ctx.batch_size, *hidden), dtype)])

        if kind in (MLP, MOE):
            return BenchInputs(args=[((ctx.batch_size, *hidden), dtype)])

        if kind in (LAYER, ATTENTION):
            rope = getattr(self, "_rope", None)
            if rope is None:
                # Without the model's own rotary table the attention kernel
                # would be fed noise of the wrong period; leave it to the
                # analytical estimate rather than measure something untrue.
                return None
            return BenchInputs(
                args=[((ctx.batch_size, *hidden), dtype), rope.detach()],
                # Both are positional-or-keyword on every family's forward, and
                # ``None`` selects the dense causal path the projection models.
                kwargs={"attention_masks": None, "positions": None},
            )

        return None

    def supports_moe_decomposition(self, module) -> bool:
        # TorchTitan's expert all-to-all lives in the parallelization style
        # wrapped around MoE, not in methods on it.
        return False
