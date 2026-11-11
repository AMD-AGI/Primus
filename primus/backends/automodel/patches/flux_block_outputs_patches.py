###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the un-aliased FLUX single-stream block outputs.

Gated on ``primus_perf.flux_unalias_block_outputs`` for FLUX runs; see
``models/flux/block_outputs.py``.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_unalias_block_outputs")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import block_outputs

    return _is_flux(ctx) and block_outputs.is_enabled()


@register_patch(
    "automodel.models.flux.unalias_block_outputs",
    backend="automodel",
    phase="before_train",
    description="FLUX single-stream blocks return fresh tensors (primus_perf.flux_unalias_block_outputs)",
    condition=_enabled,
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import block_outputs

    block_outputs.install()
