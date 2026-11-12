###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Primus-Turbo MXFP4 linear swap.

Gated on ``primus_turbo.mxfp4_linear``, and conditioned on ``_common.is_active``
rather than on the setting alone: only one precision can own AutoModel's swap symbol, so
the selector decides and a losing request is logged instead of silently dropped.
MXFP4 outranks FP8, so asking for both gives four bits.

Shares a priority with the FP8 swap because the two are mutually exclusive by
construction -- at most one can be active, so their relative order never comes
up.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, get_param, register_patch

# primus_te.mxfp4_linear reads mxfp4_keep_sensitive too.
options.declare(
    "primus_turbo",
    "mxfp4_linear",
    "mxfp4_backward",
    "mxfp4_gradient_sr",
    "mxfp4_preshuffle",
    "mxfp4_keep_sensitive",
    "mxfp4_sensitive_layers",
    "mxfp4_sensitive_start",
    "mxfp4_sensitive_end",
    "mxfp4_sensitive_precision",
)


def _active(ctx: PatchContext) -> bool:
    # _fp4_common, not mxfp4_linear: the latter needs torch to define Float4Linear,
    # and a patch condition should be answerable without importing a kernel stack.
    from primus.backends.automodel.quantization import _common, _fp4_common

    return _common.is_active(_fp4_common.BACKEND_NAME)


@register_patch(
    "automodel.quantization.mxfp4_linear",
    backend="automodel",
    phase="before_train",
    description="Swap nn.Linear for Primus-Turbo MXFP4 Float4Linear (primus_turbo.mxfp4_linear)",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.quantization import mxfp4_linear

    mxfp4_linear.install(get_param(ctx, "model"))
