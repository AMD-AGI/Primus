###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-build checks of a Granite ``GPTModel`` (layer type and attention softmax scale).

Kept apart from ``granite_builders`` so it imports neither ``megatron.training`` nor the
GPU-only parts of Megatron: the CPU unit tests run it on a real built model.
"""

import math

from primus.backends.megatron.core.transformer.granite import GraniteTransformerLayer
from primus.core.utils.module_utils import log_rank_0

__all__ = ["check_built_granite_model"]


def check_built_granite_model(model, config):
    """Check the built ``model`` against ``config`` (a ``GraniteTransformerConfig``).

    The softmax scale is read back from every built core attention (TE, local or Primus
    Turbo) and must equal ``config.attention_multiplier``; with
    ``residual_multiplier != 1`` every decoder layer must be a ``GraniteTransformerLayer``.
    Raises ``TypeError`` / ``ValueError`` on a mismatch; logs what it checked.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        # granite_builder passes the bare GPTModel, which always builds a decoder
        # (TransformerBlock), even on a stage with no layers. A model without one is a wrapped
        # model (Float16Module, DDP), a list of VPP chunks, a non-GPT model or an upstream
        # rename: refuse it, so the checks below are never skipped silently.
        raise TypeError(
            f"Granite: the built model ({type(model).__name__}) has no 'decoder'; pass the bare "
            "GPTModel (not a wrapper or a list of chunks), so the Granite layer type and "
            "attention scale can be checked"
        )
    layers = getattr(decoder, "layers", None)
    if layers is None:
        # GPTModel always has decoder.layers; without it nothing below could be checked.
        raise TypeError(
            "Granite: the built GPTModel decoder has no 'layers' attribute, so the Granite "
            "layer type and attention scale cannot be checked (upstream TransformerBlock changed?)"
        )
    layers = list(layers)
    if not layers:
        # A pipeline stage may build no decoder layers (e.g. embedding/loss-only stages with
        # account_for_*_in_pipeline_split); say so instead of passing with "0 built layers".
        log_rank_0(
            "[Primus:Granite] this model chunk builds no decoder layers; the Granite layer "
            "type and attention scale are checked on the stages that build them"
        )
        return
    attention_scales = _core_attention_scales(layers)
    if len(attention_scales) != len(layers):
        kinds = sorted({type(_core_attention(layer)).__name__ for layer in layers})
        raise ValueError(
            f"cannot read the softmax scale of the Granite core attention ({', '.join(kinds)}): "
            f"found it in {len(attention_scales)} of {len(layers)} layers. Granite needs the explicit "
            f"attention_multiplier={config.attention_multiplier}; use TE, local or Primus Turbo attention. "
            "(The scale is read from core_attention.softmax_scale, or from TE's flash_attention / "
            "fused_attention / unfused_attention backends; a TE version that moved it needs "
            "_core_attention_scales updated.)"
        )
    disagree = [i for i, scale in enumerate(attention_scales) if math.isnan(scale)]
    if disagree:
        raise ValueError(
            f"Granite: the TE attention backends of layer(s) {disagree[:5]} hold different softmax "
            f"scales; expected attention_multiplier={config.attention_multiplier} in all of them"
        )
    wrong = [
        scale
        for scale in attention_scales
        if not math.isclose(scale, config.attention_multiplier, rel_tol=1e-9, abs_tol=0.0)
    ]
    if wrong:
        raise ValueError(
            f"Granite attention modules do not use attention_multiplier={config.attention_multiplier}: "
            f"softmax_scale per layer {attention_scales} ({len(layers)} layers)"
        )
    # Read back from the built attention modules, not from the config.
    log_rank_0(
        f"[Primus:Granite] attention softmax_scale in {len(layers)} built layers: "
        f"{sorted(set(attention_scales))}"
    )
    if config.residual_multiplier != 1.0:
        bad = [type(layer).__name__ for layer in layers if not isinstance(layer, GraniteTransformerLayer)]
        if bad:
            raise TypeError(
                f"Granite residual_multiplier needs GraniteTransformerLayer layers, got {bad[:3]}"
            )
        log_rank_0(
            f"[Primus:Granite] built {len(layers)} GraniteTransformerLayer decoder layers "
            f"with scaled residual add (residual_multiplier={config.residual_multiplier})"
        )


def _core_attention(layer):
    return getattr(getattr(layer, "self_attention", None), "core_attention", None)


def _core_attention_scales(layers):
    """``softmax_scale`` held by each layer's core attention.

    The builder refuses ``apply_query_key_layer_scaling`` (with it, local attention stores
    ``softmax_scale / layer_number`` and multiplies the layer number back in the softmax),
    so the stored value is the scale the kernel applies. Primus Turbo attention
    (``PrimusTurboAttention``, a TE ``DotProductAttention``) keeps it as ``softmax_scale``.
    Local attention keeps it as ``core_attention.softmax_scale``; TE's
    ``DotProductAttention`` resolves it in ``__init__`` and keeps it on its backends
    (``flash_attention`` / ``fused_attention`` / ``unfused_attention``), which must all
    agree. A layer whose core attention exposes none of these is left out (so the caller's
    count check fails).
    """
    scales = []
    for layer in layers:
        core = _core_attention(layer)
        found = set()
        for owner in (core,) + tuple(
            getattr(core, name, None) for name in ("flash_attention", "fused_attention", "unfused_attention")
        ):
            scale = getattr(owner, "softmax_scale", None)
            if scale is not None:
                found.add(float(scale))
        if len(found) == 1:
            scales.append(found.pop())
        elif found:
            scales.append(float("nan"))  # backends disagree
    return scales
