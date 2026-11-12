###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Policy and configuration for the AutoModel MXFP6 linear path.

MXFP6 training uses the gfx950 A6W6 kernels exposed by Primus-Turbo.  Those
kernels require every GEMM dimension to be a multiple of 256.  Feature
dimensions are static and checked here; the runtime token dimension is padded
in :mod:`mxfp6_gemm`.

Kept free of torch at import time so patch selection and unit tests do not need
a GPU stack.
"""
from __future__ import annotations

from primus.backends.automodel import options
from primus.backends.automodel.quantization import _common

MXFP6_TILE_SIZE = 256
MXFP6_SENSITIVE_PREFIXES = ("condition_embedder.",)
BACKEND_NAME = "turbo_mxfp6"


def is_enabled() -> bool:
    """Whether the MXFP6 swap was requested."""
    return options.flag("primus_turbo.mxfp6_linear")


def keep_sensitive_bf16() -> bool:
    """Keep conditioning projections in BF16 by default."""
    return options.flag("primus_turbo.mxfp6_keep_sensitive", default=True)


def is_fp6_training_safe_linear(name: str, linear) -> bool:
    """Whether ``linear`` satisfies both model and A6W6 requirements."""
    from nemo_automodel._diffusers.auto_diffusion_pipeline import (
        _is_fp8_training_safe_linear,
    )

    if not _is_fp8_training_safe_linear(name, linear):
        return False
    if keep_sensitive_bf16() and name.startswith(MXFP6_SENSITIVE_PREFIXES):
        return False
    return linear.weight.shape[0] % MXFP6_TILE_SIZE == 0 and linear.weight.shape[1] % MXFP6_TILE_SIZE == 0


def resolve_fp6_config():
    """Build Turbo's fixed E2M3, block-of-32 forward recipe.

    Importing here is intentional.  The patch can register on a host without
    the kernel stack, but a requested model build fails instead of falling back
    to BF16 when the installed Turbo lacks MXFP6.
    """
    from primus_turbo.pytorch.core.low_precision import Float6QuantConfig

    return Float6QuantConfig()


def resolve_mxfp4_backward_config():
    """Build the fixed MXFP4 backward recipe required by Issue 364.

    Preshuffle selects Turbo's AITER path.  Stochastic rounding stays off until
    the matched-loss gate establishes a reason to change the frozen recipe.
    """
    from primus_turbo.pytorch.core.low_precision import Float4QuantConfig

    required = {"use_preshuffle", "use_gradient_sr"}
    missing = required - Float4QuantConfig.__dataclass_fields__.keys()
    if missing:
        raise RuntimeError(
            "The installed primus_turbo Float4QuantConfig cannot express the "
            f"MXFP4 backward recipe; missing fields: {', '.join(sorted(missing))}"
        )
    return Float4QuantConfig(use_preshuffle=True, use_gradient_sr=False)


def pad_multiple(rows: int, multiple: int = MXFP6_TILE_SIZE) -> int:
    """Round ``rows`` up to the next A6W6 tile."""
    return ((rows + multiple - 1) // multiple) * multiple


# Six bits outranks the existing four- and eight-bit requests.  Concurrent
# precision requests are a configuration error in spirit, but selecting one
# deterministically and warning is safer than depending on patch import order.
_common.register_backend(
    BACKEND_NAME,
    precedence=25,
    is_requested=is_enabled,
    description="Primus-Turbo MXFP6-forward/MXFP4-backward Float6Linear",
)
