###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Scaled bias-dropout-add: ``residual + dropout(multiplier * (x + bias))``.

Megatron adds each sub-block output to the residual stream in the bias-dropout-add (bda)
step: ``out = residual + dropout(x + bias)``. Some models (IBM Granite:
``residual_multiplier``) scale the sub-block output before the add::

    h = h + residual_multiplier * self_attn(input_layernorm(h))
    h = h + residual_multiplier * mlp(post_attention_layernorm(h))

``scaled_bias_dropout_add(get_bias_dropout_add, m)`` is a bda factory with the same
signature as upstream ``get_bias_dropout_add``. A layer spec uses it as ``self_attn_bda``
and ``mlp_bda`` of an upstream ``TransformerLayer``; no layer subclass is needed. The
returned factory carries the multiplier as ``residual_multiplier``, so code that bypasses
the bda (Primus' fused residual+RMSNorm path) can see that the layer scales its residual.
"""

import functools
import inspect
from typing import Callable, Optional, Tuple

import torch
from megatron.core.fusions.fused_bias_dropout import (
    _bias_dropout_add_func,
    get_bias_dropout_add,
)
from megatron.core.jit import jit_fuser

from primus.core.utils.module_utils import log_rank_0
from primus.core.utils.validation import positive_finite

__all__ = ["BDA_FIELDS", "residual_multiplier_of", "scaled_bias_dropout_add"]

# The decoder-only TransformerLayer submodules that add a sub-block output to the residual
# (Granite has no cross-attention, so cross_attn_bda is not listed and is not scaled). A layer
# spec with a scaled residual wraps exactly these, and code that bypasses the bda (the
# fused residual+RMSNorm path) checks exactly these, so both read this one tuple.
BDA_FIELDS = ("self_attn_bda", "mlp_bda")


def _check_callable_as_bda(bda_factory):
    """Raise ``TypeError`` at build time when ``bda_factory`` cannot be called as
    ``bda_factory(training, fused)``, the way ``TransformerLayer`` calls a bda factory (for
    example a ``functools.partial`` that already binds ``training`` or ``fused``). Without
    this check the error would come only on the first forward."""
    try:
        signature = inspect.signature(bda_factory)
    except (TypeError, ValueError):  # no signature to read (e.g. some builtins): not checked
        return
    try:
        signature.bind(True, True)
    except TypeError as e:
        raise TypeError(
            f"scaled_bias_dropout_add: {bda_factory!r} is not a bda factory: TransformerLayer "
            f"calls it as bda_factory(training, fused), which fails ({e})"
        ) from None


def residual_multiplier_of(bda_factory) -> Optional[float]:
    """The ``residual_multiplier`` a bda factory scales by, or ``None`` when it does not
    scale. Looks through ``functools.partial`` layers, which drop function attributes."""
    fn = bda_factory
    while True:
        tag = getattr(fn, "residual_multiplier", None)
        if tag is not None or not isinstance(fn, functools.partial):
            return tag
        fn = fn.func


def _scaled(x_with_bias, multiplier, dtype):
    """``(multiplier * x, multiplier * bias)``, cast to ``dtype`` (the residual's) first, as
    upstream's fp32-residual upcast does, so the scale is applied at the residual's
    precision."""
    x, bias = x_with_bias
    if x.dtype != dtype:
        x = x.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)
    return x * multiplier, (bias * multiplier if bias is not None else None)


def _scaled_bias_dropout_add_func(
    x_with_bias: Tuple[torch.Tensor, Optional[torch.Tensor]],
    residual: torch.Tensor,
    prob: float,
    multiplier: float,
    training: bool,
) -> torch.Tensor:
    """``residual + dropout(multiplier * (x + bias))``.

    Casts to the residual's dtype first, then scales, then calls upstream
    ``_bias_dropout_add_func`` on the scaled sub-block output (dropout is linear, so
    scaling ``x`` and ``bias`` first is the same). Its fp32-residual upcast and its
    in-place eval path are upstream's own; the scale makes new tensors, so the in-place
    path never writes into the sub-block's output."""
    return _bias_dropout_add_func(_scaled(x_with_bias, multiplier, residual.dtype), residual, prob, training)


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
    is jit-fused when ``fused`` is true, so the scaled add costs no extra kernel (it is
    recognised by identity). Any other factory is wrapped generically: it is called as
    ``bda_factory(training, fused)``, as ``TransformerLayer`` calls it, and its function gets
    ``x`` and ``bias`` cast to the residual's dtype and scaled. Both paths scale at the
    residual's precision, so they give the same result (up to the inner factory's own math);
    building the generic path is logged.

    Raises ``TypeError`` when ``bda_factory`` cannot be called as
    ``bda_factory(training, fused)`` (``_check_callable_as_bda``). Raises ``ValueError`` when
    ``bda_factory`` is already scaled (``residual_multiplier_of`` is not ``None``, also through
    ``functools.partial``): the scales would multiply while the tag reports only one.
    """
    multiplier = positive_finite("residual_multiplier", multiplier)
    inner_multiplier = residual_multiplier_of(bda_factory)
    if inner_multiplier is not None:
        # Wrapping again would scale by both multipliers while the tag reports only the
        # outer one, and the fused-norm guard and the layer checks read that tag.
        raise ValueError(
            f"scaled_bias_dropout_add: the bda factory is already scaled "
            f"(residual_multiplier={inner_multiplier}); wrap the unscaled factory once"
        )

    _check_callable_as_bda(bda_factory)
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
        log_rank_0(
            f"scaled_bias_dropout_add: {bda_factory!r} is not megatron's get_bias_dropout_add; "
            "using the generic scaled wrapper (the scale is a separate multiply before the inner bda)"
        )

        def factory(training, fused):
            inner = bda_factory(training, fused)

            def apply(x_with_bias, residual, prob):
                return inner(_scaled(x_with_bias, multiplier, residual.dtype), residual, prob)

            return apply

    factory.residual_multiplier = multiplier
    return factory
