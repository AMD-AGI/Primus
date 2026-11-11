###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the FLUX MLP's up-projection and GELU as one op.

Gated on ``primus_perf.flux_fused_gelu`` for FLUX runs; see ``models/flux/mlp_gelu.py``. Runs before
the un-aliased block outputs (priority 80), which then wrap the fused single-stream forward.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "flux_fused_gelu", "flux_fused_gelu_bias_grad")

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import mlp_gelu

    return _is_flux(ctx) and mlp_gelu.is_enabled()


@register_patch(
    "automodel.models.flux.fused_gelu",
    backend="automodel",
    phase="before_train",
    description="FLUX MLP up-projection + GELU as one op (primus_perf.flux_fused_gelu)",
    condition=_enabled,
    priority=70,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import mlp_gelu

    mlp_gelu.install()
