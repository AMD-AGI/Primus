###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Primus-Turbo FP8 linear swap for the AutoModel diffusion recipe.

WHY NOT TRANSFORMER ENGINE:
  Config-only TE FP8 does not train the diffusion models on this path with the
  currently shipped TE and ROCm. Per-tensor delayed and current scaling have no
  hipBLASLt FP8 backward GEMM on gfx950, and MXFP8 has no bias support.
  Primus-Turbo's ``Float8Linear`` sidesteps both: it is AITER-backed
  (``gemm_fp8``), carries its own autograd forward and backward, and applies bias
  OUTSIDE the FP8 GEMM.

WHAT (NO AutoModel fork):
  Rebinds AutoModel's ``_replace_linear_with_transformer_engine`` so the existing
  config seam (``model.transformer_engine_linear: true``) swaps
  ``torch.nn.Linear`` for ``Float8Linear`` instead of TE Linear. The swap runs on
  the built transformer BEFORE FSDP2 wrapping, mirroring the TE path exactly.

  AutoModel's own ``_is_fp8_training_safe_linear`` skip-list is applied
  unconditionally, so the set of layers kept in bf16 -- timestep and text
  embeddings, norm modulation, the final projection, and any Linear whose
  dimensions are not 16-aligned -- matches the TE path rather than being a second
  opinion about numerical stability. The recipe ties its own ``fp8_safe_only``
  flag to ``transformer_engine_fp8``, which this path keeps off, so that argument
  is accepted for signature compatibility and deliberately ignored.

Recommended config pairing (drives only the swap, leaving TE autocast off):
  model.transformer_engine_linear: true
  model.transformer_engine_fp8: false

Activation and tuning (env, so no AutoModel source or config schema change):
  PRIMUS_TURBO_FP8=1                      enable the swap (default off = no-op)
  PRIMUS_TURBO_FP8_GRANULARITY=TENSORWISE ROWWISE|TENSORWISE|BLOCKWISE|MX_BLOCKWISE
  PRIMUS_TURBO_FP8_FORMAT=E4M3            E4M3|E5M2|HYBRID
"""
from __future__ import annotations

import logging
import os

from primus.backends.nemo_automodel._env import env_flag, env_int
from primus.backends.nemo_automodel.quantization import _common

logger = logging.getLogger(__name__)

BACKEND_NAME = "turbo_fp8"
_LOG_PREFIX = "[PrimusTurbo-FP8]"

_GRANULARITIES = ("ROWWISE", "TENSORWISE", "BLOCKWISE", "MX_BLOCKWISE")
_FORMATS = ("E4M3", "E5M2", "HYBRID")


def is_enabled() -> bool:
    """Whether the FP8 swap was requested. Not the same as it being active."""
    return env_flag("PRIMUS_TURBO_FP8")


# Lowest precedence of the low-precision swaps: FP8 is the fallback when a
# narrower format was also asked for. Registration happens on import; see
# _common.register_backend for why that is separate from activation.
_common.register_backend(
    BACKEND_NAME,
    precedence=10,
    is_requested=is_enabled,
    description="Primus-Turbo Float8Linear (AITER gemm_fp8)",
)


def resolve_config():
    """Build a Float8QuantConfig from env (defaults: TENSORWISE, E4M3, dynamic).

    Raises on an unrecognised value rather than falling back to the default. A
    typo in a precision knob that silently trains in a different format is worse
    than a failed launch, and the failure is otherwise invisible.

    The names are checked before primus_turbo is imported, so a typo is reported
    as a typo instead of surfacing as whatever the import happens to fail with.
    """
    gran_name = os.getenv("PRIMUS_TURBO_FP8_GRANULARITY", "TENSORWISE").upper()
    fmt_name = os.getenv("PRIMUS_TURBO_FP8_FORMAT", "E4M3").upper()
    if gran_name not in _GRANULARITIES:
        raise ValueError(
            f"PRIMUS_TURBO_FP8_GRANULARITY={gran_name!r} invalid; expected one of "
            f"{', '.join(_GRANULARITIES)}"
        )
    if fmt_name not in _FORMATS:
        raise ValueError(
            f"PRIMUS_TURBO_FP8_FORMAT={fmt_name!r} invalid; expected one of " f"{', '.join(_FORMATS)}"
        )

    from primus_turbo.pytorch.core.low_precision import (
        MXFP8_BLOCK_SIZE,
        Float8QuantConfig,
        Format,
        ScaleDtype,
        ScalingGranularity,
    )

    # Still go through getattr rather than trusting the lists above to match the
    # installed library, so version skew is a clear error and not a wrong config.
    try:
        granularity = getattr(ScalingGranularity, gran_name)
        fmt = getattr(Format, fmt_name)
    except AttributeError as exc:
        raise ValueError(
            f"granularity={gran_name!r} format={fmt_name!r} is not available in the "
            "installed primus_turbo; check the library version"
        ) from exc
    # The two blockwise granularities carry extra required fields, and Float8QuantConfig
    # asserts on them in __post_init__. Without this, the two of the four granularities
    # advertised above that need a block size fail inside primus_turbo with a bare
    # AssertionError, which reads like a library bug rather than a missing setting.
    extra = {}
    if granularity is ScalingGranularity.BLOCKWISE:
        extra["block_size"] = env_int("PRIMUS_TURBO_FP8_BLOCK_SIZE", 128)
    elif granularity is ScalingGranularity.MX_BLOCKWISE:
        # Not knobs: the MX format fixes both, and any other value is rejected. They
        # are set here rather than asked for so the granularity alone is enough.
        extra["block_size"] = MXFP8_BLOCK_SIZE
        extra["scale_dtype"] = ScaleDtype.E8M0

    return Float8QuantConfig(format=fmt, granularity=granularity, **extra)


def replace_linears(module, module_name: str, *, fp8_safe_only: bool = False) -> int:
    """Drop-in replacement for AutoModel's TE swap, using Float8Linear.

    ``fp8_safe_only`` is accepted for signature compatibility with the symbol
    being replaced and is ignored: the skip-list is always applied.
    """
    # AutoModel's own predicate, so the kept-in-bf16 set matches the TE path.
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )
    from primus_turbo.pytorch.modules import Float8Linear

    cfg = resolve_config()

    def factory(linear):
        return Float8Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            config=cfg,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

    converted, skipped = _common.replace_linears(
        module,
        module_name,
        factory=factory,
        should_convert=_is_fp8_training_safe_linear,
        already_converted=(Float8Linear,),
        log_prefix=_LOG_PREFIX,
    )
    logger.info(
        "%s replaced %d torch.nn.Linear with Float8Linear in %s; skipped=%d " "(granularity=%s format=%s)",
        _LOG_PREFIX,
        converted,
        module_name,
        skipped,
        getattr(cfg.granularity, "name", cfg.granularity),
        getattr(cfg.format, "name", cfg.format),
    )
    return converted


def install() -> bool:
    """Rebind AutoModel's TE swap symbol to the FP8 swap."""
    # Fail fast if primus_turbo is missing, so the run errors clearly rather than
    # silently falling back to TE -- which would look like it worked.
    import primus_turbo.pytorch.modules  # noqa: F401

    _common.install_linear_swap(replace_linears, _LOG_PREFIX)
    return True
