###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Layer spec for IBM Granite: the upstream GPT dense layer spec (upstream
``TransformerLayer``) with both bias-dropout-adds replaced by the scaled bda
(``scaled_bias_dropout_add``), which multiplies each sub-block output by
``residual_multiplier`` before the residual add."""

import dataclasses

from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer

from primus.backends.megatron.core.fusions.scaled_bias_dropout_add import (
    BDA_FIELDS,
    residual_multiplier_of,
    scaled_bias_dropout_add,
)
from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.validation import positive_finite


def _check_bda(name, bda, multiplier):
    """Raise unless ``bda`` is a bda factory that is unscaled or scales by ``multiplier``;
    return whether it is already scaled."""
    if isinstance(bda, (ModuleSpec, type)):
        # IdentityFuncOp / IdentityOp (a spec without attention or MLP): there is no
        # residual add to scale, so the spec is not a Granite layer.
        raise ValueError(
            f"Granite: {name} is {bda!r}, not a bias-dropout-add factory such as "
            "get_bias_dropout_add; a Granite layer needs both attention and MLP with a residual add"
        )
    tag = residual_multiplier_of(bda)
    if tag is not None and tag != multiplier:
        raise ValueError(
            f"Granite: {name} already scales by {tag}, expected residual_multiplier {multiplier}"
        )
    return tag is not None


def _granite_layer(spec, multiplier):
    module = getattr(spec, "module", None)
    if not (isinstance(module, type) and issubclass(module, TransformerLayer)):
        raise TypeError(f"Granite expects a TransformerLayer spec, got module {module!r}")
    submodules = spec.submodules
    scaled = {name: _check_bda(name, getattr(submodules, name), multiplier) for name in BDA_FIELDS}
    if multiplier == 1.0:
        return spec  # nothing to scale; the checks above still ran
    bdas = {
        name: scaled_bias_dropout_add(getattr(submodules, name), multiplier)
        for name in BDA_FIELDS
        if not scaled[name]
    }
    return dataclasses.replace(spec, submodules=dataclasses.replace(submodules, **bdas))


def granite_layer_spec_from(base_spec, residual_multiplier):
    """Return ``base_spec`` with ``self_attn_bda`` and ``mlp_bda`` of each layer replaced by
    ``scaled_bias_dropout_add(bda, residual_multiplier)``.

    Two cases keep a bda as it is (after the checks below): with ``residual_multiplier ==
    1.0`` the layer spec is returned unchanged (nothing to scale), and a bda already scaled
    by the same multiplier (e.g. from ``--spec``) is not wrapped again.

    ``base_spec`` is what upstream ``gpt_builder`` accepts for ``--spec``: a ``ModuleSpec``
    for a ``TransformerLayer`` (or subclass, which is kept) or a ``TransformerBlockSubmodules``
    with ``layer_specs`` (each layer spec is converted; the block's ``layer_norm`` is kept).
    The layer module and the other submodules are not changed. A bda that is not a factory
    (``IdentityFuncOp``: a layer without attention or MLP) raises ``ValueError``.
    """
    multiplier = positive_finite("residual_multiplier", residual_multiplier)
    if isinstance(base_spec, TransformerBlockSubmodules):
        if not base_spec.layer_specs:
            raise TypeError("Granite expects a block spec with layer_specs, got none")
        return dataclasses.replace(
            base_spec, layer_specs=[_granite_layer(s, multiplier) for s in base_spec.layer_specs]
        )
    return _granite_layer(base_spec, multiplier)


def get_granite_layer_spec(args, config):
    """Granite decoder layer spec, selected the way upstream ``gpt_builder`` selects the
    dense GPT layer spec.

    It calls ``gpt_builders._get_transformer_layer_spec`` (private upstream) on purpose:
    upstream ``gpt_builder`` uses it, so Granite gets the same TE / local spec as a GPT model
    with the same args, and the Primus Turbo spec through Primus's ``TESpecProvider`` swap
    (``turbo/te_spec_provider_patches.py``), which this function reads. Primus does not patch
    this function itself. A rename upstream fails here with ``AttributeError`` at build time,
    not silently.
    """
    import gpt_builders  # third_party/Megatron-LM, on sys.path at runtime

    use_te = args.transformer_impl == "transformer_engine"
    base_spec = gpt_builders._get_transformer_layer_spec(use_te, config)
    return granite_layer_spec_from(base_spec, config.residual_multiplier)


def check_scaled_residual_layers(model, residual_multiplier):
    """Check, and log, that the built decoder layers scale their residual add.

    ``model`` is the bare ``GPTModel`` that ``granite_builder`` just built, not a wrapped
    (DDP / FSDP / Float16Module) model. A model without ``decoder`` raises ``TypeError``,
    and so does a decoder without ``layers`` (an upstream rename or a wrapped decoder),
    and so does a layer without ``self_attn_bda`` / ``mlp_bda`` (for any multiplier), so the
    check cannot pass on a model it did not look at. A pipeline stage whose
    ``layers`` is empty passes. With ``residual_multiplier == 1`` a layer may keep the plain
    bda (nothing to scale) or a bda scaled by 1.0 (accepted by ``granite_layer_spec_from``);
    a bda scaled by another value raises ``ValueError``. Otherwise a layer whose bda is not
    the scaled bda with that multiplier raises ``ValueError``. ``residual_multiplier`` is
    validated with ``positive_finite`` (as in ``granite_layer_spec_from``) after the decoder
    / layers checks. The error gives the number of bad layers and the first ones.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise TypeError(f"Granite: expected the bare GPTModel with a decoder, got {type(model).__name__}")
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise TypeError(
            f"Granite: expected the decoder to have `layers`, got {type(decoder).__name__} without it"
        )
    layers = list(layers)
    multiplier = positive_finite("residual_multiplier", residual_multiplier)
    # Same rule as _check_bda: with 1.0 a layer may keep the plain bda (tag None) or a bda
    # already scaled by 1.0 (from --spec); otherwise each bda carries the multiplier's tag.
    unscaled = multiplier == 1.0
    allowed = (None, 1.0) if unscaled else (multiplier,)
    # A layer without the bda attributes (a wrapped layer, an upstream rename, not a
    # TransformerLayer) was not inspected: an error for any multiplier, never a pass.
    missing = [i for i, layer in enumerate(layers) if not all(hasattr(layer, n) for n in BDA_FIELDS)]
    if missing:
        raise TypeError(
            f"Granite: {len(missing)} of {len(layers)} decoder layers have no {' / '.join(BDA_FIELDS)} "
            f"(first: {missing[:5]}, {type(layers[missing[0]]).__name__}); cannot check their residual add"
        )
    bad = [
        i
        for i, layer in enumerate(layers)
        if any(residual_multiplier_of(getattr(layer, n)) not in allowed for n in BDA_FIELDS)
    ]
    if bad:
        what = (
            "leave the residual add unscaled (or scaled by 1.0)"
            if unscaled
            else f"scale the residual add by {multiplier}"
        )
        raise ValueError(
            f"Granite: {len(bad)} of {len(layers)} decoder layers do not {what} " f"(first: {bad[:5]})"
        )
    if unscaled:
        log_rank_0(
            f"[Primus:Granite] built {len(layers)} decoder layers; residual_multiplier=1.0, "
            "so the residual add is not scaled"
        )
        return
    log_rank_0(
        f"[Primus:Granite] built {len(layers)} decoder layers with scaled residual add "
        f"(residual_multiplier={multiplier})"
    )


def built_model_summary(model):
    """What the built model applies, read from its modules (not from ``args``): the decoder
    layer count, the core attention class and the ``softmax_scale`` it uses (its own attribute,
    else the config's, which Megatron passes to it), the residual scale of the first layer's
    bda, the embedding's ``mup_embedding_mult`` and the model config's ``mup_output_mult``
    (what ``LanguageModule`` scales the logits by). The trainer test checks this line."""
    layers = list(model.decoder.layers)
    parts = [f"decoder_layers={len(layers)}"]
    if layers:
        core = layers[0].self_attention.core_attention
        parts.append(f"core_attention={type(core).__name__}")
        # The scale the core attention uses: its own softmax_scale when it keeps one
        # (PrimusTurboAttention passes self.softmax_scale to the kernel), else the one Megatron's
        # Attention passed to it (softmax_scale=self.config.softmax_scale; TE's
        # DotProductAttention does not keep it as an attribute).
        own = getattr(core, "softmax_scale", None)
        source = "module" if own is not None else "config"
        scale = own if own is not None else core.config.softmax_scale
        parts.append(f"core_attention.softmax_scale={scale}")
        parts.append(f"core_attention.softmax_scale_from={source}")
        parts.append(f"layer_bda_scale={residual_multiplier_of(layers[0].self_attn_bda)}")
    embedding = getattr(model, "embedding", None)
    if embedding is not None:
        parts.append(f"embedding.mup_embedding_mult={embedding.config.mup_embedding_mult}")
    if getattr(model, "post_process", False):
        parts.append(f"output.mup_output_mult={model.config.mup_output_mult}")
    return " ".join(parts)
