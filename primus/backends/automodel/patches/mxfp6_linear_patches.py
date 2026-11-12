###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Primus-Turbo MXFP6 linear swap."""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, get_param, register_patch

options.declare(
    "primus_turbo",
    "mxfp6_linear",
    "mxfp6_keep_sensitive",
)


def _active(ctx: PatchContext) -> bool:
    from primus.backends.automodel.quantization import _common, _fp6_common

    return _common.is_active(_fp6_common.BACKEND_NAME)


@register_patch(
    "automodel.quantization.mxfp6_linear",
    backend="automodel",
    phase="before_train",
    description="Swap nn.Linear for Primus-Turbo MXFP6-forward/MXFP4-backward Float6Linear",
    condition=_active,
    priority=20,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.quantization import mxfp6_linear

    mxfp6_linear.install(get_param(ctx, "model"))
