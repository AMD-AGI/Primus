###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MXFP4 linear swap for the AutoModel diffusion recipe.

Primus-Turbo ships no ``Float4Linear`` module, only the ``gemm_fp4`` op, so this
provides the module: a ``torch.nn.Linear`` subclass that routes its matmul
through MXFP4 and applies bias OUTSIDE the low-precision GEMM, mirroring how
Turbo's own ``Float8Linear`` is built and sidestepping any bias-in-GEMM kernel
gap.

Uses the same seam as every other precision -- see ``_common.py`` for the seam
and the precedence selector, ``_fp4_common.py`` for the eligibility policy, and
``mxfp4_gemm.py`` for the token padding, which is the subtle part.

WHY THIS OWNS ITS OWN MODULE WALK:
  ``_common.replace_linears`` decides a Linear's fate with one boolean and builds
  one replacement type. That fits a two-outcome swap. This one has three
  outcomes -- MXFP4, tensorwise FP8, or left in bf16 -- because of the sensitive
  band below, and the summary line has to account for all three separately to be
  worth reading. Bending the two-outcome helper into a three-outcome shape (two
  passes, or a predicate with side effects) came out less clear than the ~30
  lines here, and would have logged a skip for every layer on a pass that was
  never going to convert it.

Activation and tuning (env, so no AutoModel source or config schema change):
  PRIMUS_TURBO_FP4=1                     enable the swap (default off = no-op)
  PRIMUS_TURBO_FP4_KEEP_SENSITIVE=1      keep conditioning projections in bf16
                                         (default on)
  PRIMUS_TURBO_FP4_SR=0                  stochastically round the gradient quant
  PRIMUS_TURBO_FP4_PRESHUFFLE=1          preshuffled layout, which also selects
                                         AITER over HIPBLASLT (default on)
  PRIMUS_TURBO_FP4_BACKWARD=mxfp4        mxfp4 | fp8
  PRIMUS_TURBO_FP4_SENSITIVE_LAYERS=0    keep a band of blocks above MXFP4
  PRIMUS_TURBO_FP4_SENSITIVE_START=2     leading blocks in the band
  PRIMUS_TURBO_FP4_SENSITIVE_END=8       trailing blocks in the band
  PRIMUS_TURBO_FP4_SENSITIVE_PRECISION=tw_fp8   tw_fp8 | bf16

Recommended config pairing (drives only the swap, leaving TE autocast off):
  model.transformer_engine_linear: true
  model.transformer_engine_fp8: false
"""
from __future__ import annotations

import logging

import torch.nn as nn

from primus.backends.nemo_automodel.quantization import _common, _fp4_common, mxfp4_gemm

logger = logging.getLogger(__name__)

# Registration lives in _fp4_common so the selector works without torch; re-exported
# here because this is where a reader looks for it.
BACKEND_NAME = _fp4_common.BACKEND_NAME
_LOG_PREFIX = "[PrimusTurbo-MXFP4]"


class Float4Linear(nn.Linear):
    """Linear whose matmul runs in MXFP4, with bias applied outside the GEMM."""

    def __init__(self, in_features, out_features, bias=True, config=None, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.config = config if config is not None else _fp4_common.resolve_fp4_config()
        self.backward_precision = _fp4_common.backward_precision()
        self.fp8_backward_config = (
            _fp4_common.resolve_hybrid_fp8_config() if self.backward_precision == "fp8" else None
        )

    def forward(self, x):
        leading_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)

        if self.backward_precision == "fp8":
            out = mxfp4_gemm.mxfp4_fwd_fp8_bwd(
                x_2d, self.weight, x.dtype, self.config, self.fp8_backward_config
            )
        else:
            out = mxfp4_gemm.mxfp4_gemm(x_2d, self.weight, x.dtype, self.config)

        if self.bias is not None:
            # Outside the low-precision GEMM, in the compute dtype.
            out = out + self.bias
        return out.view(*leading_shape, self.out_features)

    def extra_repr(self):
        return f"{super().extra_repr()}, mxfp4=True, backward={self.backward_precision}"


def _band_description(block_counts) -> str:
    """Human summary of which blocks the band covers, for the one summary line."""
    if not block_counts:
        return "no transformer-block lists found"
    start, end = _fp4_common.sensitive_start(), _fp4_common.sensitive_end()
    return ", ".join(f"{name}[first {start} + last {end} of {n}]" for name, n in block_counts.items())


def replace_linears(module, module_name: str, *, fp8_safe_only: bool = False) -> int:
    """Drop-in replacement for AutoModel's TE swap, using ``Float4Linear``.

    ``fp8_safe_only`` is accepted for signature compatibility with the symbol
    being replaced and is ignored: the skip-list is always applied.

    Each Linear gets one of three outcomes. The sensitive band, when enabled,
    keeps the first N and last M blocks of every transformer-block list above
    MXFP4 -- either in tensorwise FP8 or in bf16. The band is checked before
    eligibility because it is a deliberate choice to not use MXFP4 there, which is
    a different thing from a layer being ineligible for it.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    fp4_config = _fp4_common.resolve_fp4_config()
    band_on = _fp4_common.sensitive_band_enabled()
    block_counts = _fp4_common.count_blocks(module) if band_on else {}
    band_precision = _fp4_common.sensitive_precision() if band_on else None

    Float8Linear = None
    fp8_config = None
    if band_on and band_precision == "tw_fp8":
        from primus_turbo.pytorch.modules import Float8Linear

        fp8_config = _fp4_common.resolve_hybrid_fp8_config()

    already_swapped = (Float4Linear,) + ((Float8Linear,) if Float8Linear is not None else ())
    to_mxfp4 = 0
    to_fp8 = 0
    kept_bf16 = 0

    def build(cls, linear, config):
        replacement = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            config=config,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        _common.copy_linear_params(replacement, linear)
        return replacement

    def walk(parent, prefix: str = "") -> None:
        nonlocal to_mxfp4, to_fp8, kept_bf16
        # list() because children are reassigned during the walk.
        for child_name, child in list(parent.named_children()):
            fqn = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, already_swapped):
                continue
            if not isinstance(child, nn.Linear):
                walk(child, fqn)
                continue

            if band_on and _fp4_common.is_sensitive_block(fqn, block_counts):
                if band_precision == "tw_fp8" and _is_fp8_training_safe_linear(fqn, child):
                    setattr(parent, child_name, build(Float8Linear, child, fp8_config))
                    to_fp8 += 1
                else:
                    kept_bf16 += 1
                continue

            if not _fp4_common.is_fp4_training_safe_linear(fqn, child):
                kept_bf16 += 1
                logger.debug(
                    "%s keeping %s.%s in bf16 (ineligible for MXFP4); weight=%s",
                    _LOG_PREFIX,
                    module_name,
                    fqn,
                    tuple(child.weight.shape),
                )
                continue

            setattr(parent, child_name, build(Float4Linear, child, fp4_config))
            to_mxfp4 += 1

    walk(module)

    # One summary line rather than a line per layer: on these models the per-layer
    # detail runs to hundreds of lines and buries the counts that matter. The
    # per-layer reasons are still available at DEBUG.
    logger.info(
        "%s %s: %d -> Float4Linear, %d -> Float8Linear, %d kept in bf16 "
        "(backward=%s gradient_sr=%s preshuffle=%s band=%s)",
        _LOG_PREFIX,
        module_name,
        to_mxfp4,
        to_fp8,
        kept_bf16,
        _fp4_common.backward_precision(),
        _fp4_common.gradient_sr_enabled(),
        _fp4_common.preshuffle_enabled(),
        _band_description(block_counts) if band_on else "off",
    )
    return to_mxfp4 + to_fp8


def install() -> bool:
    """Rebind AutoModel's TE swap symbol to the MXFP4 swap."""

    def probe() -> None:
        # Fail fast if the FP4 op is missing, rather than silently falling back
        # to TE -- which would look like it worked.
        from primus_turbo.pytorch.ops import gemm_fp4  # noqa: F401

    probe()
    _common.install_linear_swap(replace_linears, _LOG_PREFIX)
    return True
