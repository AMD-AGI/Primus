###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark centered RMSNorm ``norm(x) * (1 + w)`` on Primus's shared fused RMSNorm.

The eager form (``MuseSparkCenteredRMSNorm``, HF ``MuseSparkTextCenteredRMSNorm``)::

    out = round(x.float() * rsqrt(mean(x.float()**2) + eps) * (1 + w.float()))

``round`` is a cast to ``x.dtype``; all math is fp32 and the result is rounded once. That
is the shared ``fused_rms_norm`` (``v4_attention_kernels/_triton_common/rmsnorm.py``)
with the weight ``1 + w.float()``, ``mid_cast=False`` and ``out_dtype=x.dtype``: its
Triton forward / backward on GPU, its eager reference elsewhere. Autograd carries
``d(1 + w)/dw = 1``, so ``w`` gets the shared kernel's weight gradient.

Deterministic mode: the shared kernel's backward adds the weight gradient of all rows
with ``tl.atomic_add``, so it changes from run to run. Under
``torch.use_deterministic_algorithms`` (Megatron's ``--deterministic-mode``)
:func:`weighted_rms_norm` runs the shared kernel without a weight (no atomics), in fp32,
and multiplies by the weight in torch: the same forward math, a deterministic gradient.
"""

from __future__ import annotations

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (
    fused_rms_norm,
)

__all__ = ["centered_rms_norm", "centered_rms_norm_reference", "weighted_rms_norm"]


def weighted_rms_norm(x, weight32, eps):
    """``round(norm(x.float()) * weight32)`` with the shared fused RMSNorm (``weight32``: fp32).

    Normally one fused kernel with the weight. Under ``torch.use_deterministic_algorithms``
    the kernel runs without the weight (its weight gradient is an atomic sum over rows) and
    the fp32 product is formed in torch, whose weight-gradient reduction is deterministic.
    """
    if torch.are_deterministic_algorithms_enabled():
        normed = fused_rms_norm(x, None, eps=eps, mid_cast=False, out_dtype=torch.float32)
        return (normed * weight32).to(x.dtype)
    return fused_rms_norm(x, weight32, eps=eps, mid_cast=False, out_dtype=x.dtype)


def centered_rms_norm_reference(x, weight, eps):
    """The eager path: fp32 ``norm(x) * (1 + w)``, rounded once to ``x.dtype``."""
    xf = x.float()
    out = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    # cast before adding 1: ``1 + w`` in bf16 would round the small learned w
    return (out * (1.0 + weight.float())).type_as(x)


def centered_rms_norm(x, weight, eps, fused: bool = True):
    """``round(norm(x.float()) * (1 + w.float()))`` over the last dim: the shared fused
    RMSNorm (Triton on GPU, its eager reference elsewhere), or the eager formula when
    ``fused`` is False."""
    if fused and x.numel():
        # the weight is cast to fp32 before adding 1 (a bf16 1 + w would round w away)
        return weighted_rms_norm(x, 1.0 + weight.float(), eps)
    return centered_rms_norm_reference(x, weight, eps)
