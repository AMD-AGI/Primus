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

On GPU, when the installed Primus-Turbo has it, both norms run on Primus-Turbo's
``rmsnorm`` instead (``zero_centered=True`` gives the weight ``1 + w`` in fp32): the same
math, rounded once to ``x.dtype``. Its backward reduces the weight gradient in two stages
in a fixed order; the shared kernel's backward adds every row's weight gradient into the
same ``D`` floats with ``tl.atomic_add``, which serialises it (two orders of magnitude
slower at ``[4096, 8192]``) and changes the sum from run to run. ``PRIMUS_RMSNORM_TURBO=0``
keeps the shared kernel.

Deterministic mode, on the shared kernel: under ``torch.use_deterministic_algorithms``
(Megatron's ``--deterministic-mode``) :func:`weighted_rms_norm` runs it without a weight
(no atomics), in fp32, and multiplies by the weight in torch: the same forward math, a
deterministic gradient.
"""

from __future__ import annotations

import functools
import inspect
import os

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_common.rmsnorm import (
    fused_rms_norm,
)

__all__ = ["centered_rms_norm", "centered_rms_norm_reference", "weighted_rms_norm"]


@functools.lru_cache(maxsize=None)
def _turbo_rmsnorm():
    """Primus-Turbo's ``rmsnorm(x, gamma, eps, zero_centered)``, or None when it is not installed."""
    try:
        from primus_turbo.pytorch.ops.normalization import rmsnorm
    except ImportError:
        return None
    return rmsnorm if "zero_centered" in inspect.signature(rmsnorm).parameters else None


def _turbo_rmsnorm_for(x, weight):
    """The Turbo ``rmsnorm`` when it takes this call (GPU tensors, ``PRIMUS_RMSNORM_TURBO`` not 0), else None."""
    if os.environ.get("PRIMUS_RMSNORM_TURBO", "1") == "0" or not (x.is_cuda and weight.is_cuda):
        return None
    return _turbo_rmsnorm()


def weighted_rms_norm(x, weight32, eps):
    """``round(norm(x.float()) * weight32)`` with the shared fused RMSNorm (``weight32``: fp32).

    Primus-Turbo's ``rmsnorm`` when available (deterministic). Otherwise normally one fused
    kernel with the weight; under ``torch.use_deterministic_algorithms`` the kernel runs
    without the weight (its weight gradient is an atomic sum over rows) and the fp32 product
    is formed in torch, whose weight-gradient reduction is deterministic.
    """
    turbo = _turbo_rmsnorm_for(x, weight32)
    if turbo is not None:
        return turbo(x, weight32, eps=eps)
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
    """``round(norm(x.float()) * (1 + w.float()))`` over the last dim: Primus-Turbo's
    ``rmsnorm`` on GPU when installed, else the shared fused RMSNorm (Triton on GPU, its eager
    reference elsewhere), or the eager formula when ``fused`` is False."""
    if fused and x.numel():
        turbo = _turbo_rmsnorm_for(x, weight)
        if turbo is not None:
            # Turbo forms 1 + w in fp32
            return turbo(x, weight, eps=eps, zero_centered=True)
        # the weight is cast to fp32 before adding 1 (a bf16 1 + w would round w away)
        return weighted_rms_norm(x, 1.0 + weight.float(), eps)
    return centered_rms_norm_reference(x, weight, eps)
