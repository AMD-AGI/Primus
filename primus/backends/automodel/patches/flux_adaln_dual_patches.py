###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the FLUX dual-stream blocks' gated residual + second adaLN as one Primus-Turbo op.

Gated on ``primus_perf.flux_fused_adaln_dual`` for FLUX runs; see ``models/flux/adaln_dual.py``. It
replaces diffusers' ``FluxTransformerBlock.forward``, which no other patch wraps; the attention
processor and the MLPs it calls keep their own patches.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_fused_adaln_dual")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import adaln_dual

    return _is_flux(ctx) and adaln_dual.is_enabled()


@register_patch(
    "automodel.models.flux.fused_adaln_dual",
    backend="automodel",
    phase="before_train",
    description="FLUX dual-stream gated residual + second adaLN as one Primus-Turbo op (primus_perf.flux_fused_adaln_dual)",
    condition=_enabled,
    priority=77,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import adaln_dual

    adaln_dual.install()
