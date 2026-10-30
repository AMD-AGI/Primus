###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Granite decoder layer: a Megatron ``TransformerLayer`` with scaled residual branches.

HF ``GraniteDecoderLayer`` adds each sub-block output to the residual stream after
multiplying it by ``config.residual_multiplier``::

    h = h + residual_multiplier * self_attn(input_layernorm(h))
    h = h + residual_multiplier * mlp(post_attention_layernorm(h))

Megatron adds the sub-block output in the bias-dropout-add (bda) step:
``out = residual + dropout(x + bias)``. This layer wraps both bda callables
(``self_attn_bda`` and ``mlp_bda``) so that ``x + bias`` is multiplied by
``residual_multiplier`` before the add, inside the same (jit-fused) function.
Everything else (attention, MLP, norms, TP/SP/PP/CP handling) is the upstream ``TransformerLayer``. The scale is part of
the layer forward; the checkpoint weights are not changed.

Forward paths that do not call the bda callables would drop the scale, so they are
refused when ``residual_multiplier != 1``: upstream's fused TP inference path
(``inference_fuse_tp_communication``) raises here and in ``GraniteTransformerConfig``,
and Primus' fused residual+RMSNorm path (``fused_residual_rmsnorm._can_fuse``) skips any
layer with a residual multiplier.
"""

from typing import Callable, Optional, Tuple

import torch
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.jit import jit_fuser
from megatron.core.transformer.transformer_layer import TransformerLayer

from primus.backends.megatron.core.models.granite.granite_validation import (
    positive_finite,
)


def _scaled_bias_dropout_add_func(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    multiplier: float,
    training: bool,
) -> torch.Tensor:
    """``residual + dropout(multiplier * (x + bias))``, the scaled form of upstream
    ``_bias_dropout_add_func`` (same fp32-residual upcast), in one function so the jit
    fuser makes one kernel of the scale, the dropout and the residual add."""
    x, bias = x_with_bias
    if x.dtype != residual.dtype:
        x = x.to(residual.dtype)
        if bias is not None:
            bias = bias.to(residual.dtype)
    if bias is not None:
        x = x + bias
    out = torch.nn.functional.dropout(x * multiplier, p=prob, training=training)
    return residual + out


@jit_fuser
def _scaled_bda_fused_train(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    multiplier: float,
) -> torch.Tensor:
    return _scaled_bias_dropout_add_func(x_with_bias, residual, prob, multiplier, True)


@jit_fuser
def _scaled_bda_fused_inference(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    multiplier: float,
) -> torch.Tensor:
    return _scaled_bias_dropout_add_func(x_with_bias, residual, prob, multiplier, False)


def scaled_bias_dropout_add(bda_factory: Callable, multiplier: float) -> Callable:
    """Wrap a Megatron bda factory ``f(training, fused) -> fn((x, bias), residual, prob)``.

    The returned factory has the same signature. Its functions compute
    ``residual + dropout(multiplier * (x + bias))``. For upstream ``get_bias_dropout_add``
    (the factory of every GPT layer spec) the scale is applied inside one function, which
    is jit-fused when ``fused`` is true, so the scaled add costs no extra kernel. Any other
    factory is wrapped generically (scale ``x`` and ``bias``, then call it).
    """
    multiplier = positive_finite("residual_multiplier", multiplier)

    if bda_factory is get_bias_dropout_add:

        def factory(training, fused):
            if fused:
                fn = _scaled_bda_fused_train if training else _scaled_bda_fused_inference

                def apply(x_with_bias, residual, prob):
                    return fn(x_with_bias, residual, prob, multiplier)

            else:

                def apply(x_with_bias, residual, prob):
                    return _scaled_bias_dropout_add_func(x_with_bias, residual, prob, multiplier, training)

            return apply

    else:

        def factory(training, fused):
            inner = bda_factory(training, fused)

            def apply(x_with_bias, residual, prob):
                x, bias = x_with_bias
                x = x * multiplier
                if bias is not None:
                    bias = bias * multiplier
                return inner((x, bias), residual, prob)

            return apply

    factory.residual_multiplier = multiplier
    return factory


class GraniteTransformerLayer(TransformerLayer):
    """``TransformerLayer`` whose attention and MLP residual branches are scaled by
    ``config.residual_multiplier`` (Granite ``residual_multiplier``).

    The config must carry a valid ``residual_multiplier`` (normally a
    ``GraniteTransformerConfig``). A missing, None, bool, non-finite or non-positive value
    is an error, so a custom spec with a plain ``TransformerConfig`` cannot train unscaled.
    ``self_attn_bda`` and ``mlp_bda`` must be bda factories (functions such as upstream
    ``get_bias_dropout_add``); a module bda (``IdentityOp`` / ``IdentityFuncOp``, i.e. a
    spec without attention or MLP) is refused with ``ValueError`` for any multiplier. With
    ``residual_multiplier == 1`` nothing is scaled, so the bdas are kept as built and the
    layer behaves like ``TransformerLayer``.
    """

    def __init__(self, config, submodules, *args, **kwargs):
        if not hasattr(config, "residual_multiplier"):
            raise TypeError(
                "GraniteTransformerLayer needs a config with residual_multiplier "
                f"(GraniteTransformerConfig), got {type(config).__name__}"
            )
        # Config-only checks come before super().__init__ builds attention and MLP.
        # GraniteTransformerConfig checks the same; the layer checks again because a custom
        # spec may pass another config with a residual_multiplier field.
        multiplier = positive_finite("residual_multiplier", config.residual_multiplier)
        if multiplier != 1.0 and getattr(config, "inference_fuse_tp_communication", False):
            # TransformerLayer skips both bda calls on the fused TP inference path.
            raise ValueError(
                "GraniteTransformerLayer: inference_fuse_tp_communication bypasses the scaled "
                "residual add; it is not supported with residual_multiplier != 1"
            )
        super().__init__(config, submodules, *args, **kwargs)
        self.residual_multiplier = multiplier
        for name in ("self_attn_bda", "mlp_bda"):
            if isinstance(getattr(self, name), torch.nn.Module):
                # A module bda (IdentityOp / IdentityFuncOp: a spec without attention or MLP)
                # has no residual add to scale. Every Granite layer has both sub-blocks, so
                # such a spec is not a Granite layer, whatever the multiplier.
                raise ValueError(
                    f"GraniteTransformerLayer: {name} is a module ({type(getattr(self, name)).__name__}), "
                    "not a bias-dropout-add factory such as get_bias_dropout_add; a Granite layer "
                    "needs both attention and MLP with a residual add to scale"
                )
        if multiplier != 1.0:
            for name in ("self_attn_bda", "mlp_bda"):
                setattr(self, name, scaled_bias_dropout_add(getattr(self, name), multiplier))
