###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the FLUX single-stream adaLN as one FlyDSL op each way.

Gated on ``primus_perf.flux_fused_adaln_flydsl`` for FLUX runs; see ``models/flux/adaln_flydsl.py``.
Runs after
the fused GELU (priority 70), whose single-stream forward it re-implements, and before the
un-aliased block outputs (80) and the image-only last block (85), which wrap it.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_fused_adaln_flydsl")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import adaln_flydsl

    return _is_flux(ctx) and adaln_flydsl.is_enabled()


@register_patch(
    "automodel.models.flux.fused_adaln_flydsl",
    backend="automodel",
    phase="before_train",
    description="FLUX single-stream adaLN as one FlyDSL op each way (primus_perf.flux_fused_adaln_flydsl)",
    condition=_enabled,
    priority=75,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import adaln_flydsl

    adaln_flydsl.install()
