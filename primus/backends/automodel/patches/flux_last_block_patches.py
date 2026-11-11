###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the image-tokens-only last FLUX single-stream block.

Gated on ``primus_perf.flux_last_block_image_only`` for FLUX runs; see ``models/flux/last_block.py``.
Runs after the fused GELU (priority 70) and un-aliased block outputs (priority 80), whose
single-stream forward it wraps for every block but the last.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_last_block_image_only")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import last_block

    return _is_flux(ctx) and last_block.is_enabled()


@register_patch(
    "automodel.models.flux.last_block_image_only",
    backend="automodel",
    phase="before_train",
    description="Last FLUX single-stream block computes its image tokens only (primus_perf.flux_last_block_image_only)",
    condition=_enabled,
    priority=85,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import last_block

    last_block.install()
