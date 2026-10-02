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
  config seam (``model.transformer_engine_linear``, which the patch turns on)
  swaps ``torch.nn.Linear`` for ``Float8Linear`` instead of TE Linear. The swap
  runs on the built transformer BEFORE FSDP2 wrapping, mirroring the TE path
  exactly.

  AutoModel's own ``_is_fp8_training_safe_linear`` skip-list is applied
  unconditionally. That predicate names the FLUX module tree: ``time_text_embed.``,
  ``norm_out.``, and the adaptive-norm modulation linears (``.norm``, ``.norm1``,
  ``.norm1_context``), plus any Linear whose dimensions are not 16-aligned.
  Wan's matching layers are not in that tree. They live under
  ``condition_embedder.`` (the timestep embedder, the text embedder, and the
  projection that produces every block's shift, scale and gate), so the predicate
  would convert them. They are kept in bf16 here as well. The MXFP4 swap already
  excludes the same prefix, they run on one vector per sample or on the text
  tokens, and leaving them out does not change the validated FLUX recipe:
  ``proj_out``, ``x_embedder`` and ``context_embedder`` stay eligible. The recipe
  ties its own ``fp8_safe_only`` flag to ``transformer_engine_fp8``, which this
  path keeps off, so that argument is accepted for signature compatibility and
  deliberately ignored.

Leave ``model.transformer_engine_fp8`` off: it would wrap the swapped layers in
TE's FP8 autocast as well.

Settings (module config, ``primus_turbo:`` section):
  fp8_linear: false                   enable the swap (default off = no-op)
  fp8_linear_granularity: tensorwise  rowwise|tensorwise|blockwise|mx_blockwise
  fp8_linear_format: e4m3             e4m3|e5m2|hybrid
  fp8_linear_block_size: 128          blockwise only
"""
from __future__ import annotations

import logging

from primus.backends.nemo_automodel import options
from primus.backends.nemo_automodel.quantization import _common

logger = logging.getLogger(__name__)

BACKEND_NAME = "turbo_fp8"
_LOG_PREFIX = "[PrimusTurbo-FP8]"

_GRANULARITIES = ("ROWWISE", "TENSORWISE", "BLOCKWISE", "MX_BLOCKWISE")
_FORMATS = ("E4M3", "E5M2", "HYBRID")

# See the module docstring. Same prefix the MXFP4 swap excludes, for the same
# reason: AutoModel's predicate does not know this name.
_EXTRA_BF16_PREFIXES = ("condition_embedder.",)


def should_convert(name: str, linear) -> bool:
    """Whether a Linear is eligible for the FP8 swap.

    AutoModel's predicate first, then the Wan conditioning prefix it does not
    name. A layer the predicate already keeps in bf16 stays in bf16.
    """
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    if name.startswith(_EXTRA_BF16_PREFIXES):
        return False
    return _is_fp8_training_safe_linear(name, linear)


def is_enabled() -> bool:
    """Whether the FP8 swap was requested. Not the same as it being active."""
    return options.flag("primus_turbo.fp8_linear")


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
    """Build a Float8QuantConfig from the settings (defaults: TENSORWISE, E4M3, dynamic).

    Raises on an unrecognised value rather than falling back to the default. A
    typo in a precision knob that silently trains in a different format is worse
    than a failed launch, and the failure is otherwise invisible.

    The names are checked before primus_turbo is imported, so a typo is reported
    as a typo instead of surfacing as whatever the import happens to fail with.
    """
    gran_name = options.string("primus_turbo.fp8_linear_granularity", "TENSORWISE").upper()
    fmt_name = options.string("primus_turbo.fp8_linear_format", "E4M3").upper()
    if gran_name not in _GRANULARITIES:
        raise ValueError(
            f"primus_turbo.fp8_linear_granularity={gran_name.lower()!r} invalid; expected one of "
            f"{', '.join(g.lower() for g in _GRANULARITIES)}"
        )
    if fmt_name not in _FORMATS:
        raise ValueError(
            f"primus_turbo.fp8_linear_format={fmt_name.lower()!r} invalid; expected one of "
            f"{', '.join(f.lower() for f in _FORMATS)}"
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
        extra["block_size"] = options.integer("primus_turbo.fp8_linear_block_size", 128)
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
        should_convert=should_convert,
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


def install(model_config=None) -> bool:
    """Rebind AutoModel's TE swap symbol to the FP8 swap (see ``_common.install_linear_swap``)."""
    # Fail fast if primus_turbo is missing, so the run errors clearly rather than
    # silently falling back to TE -- which would look like it worked.
    import primus_turbo.pytorch.modules  # noqa: F401

    _common.install_linear_swap(replace_linears, _LOG_PREFIX, model_config)
    return True
