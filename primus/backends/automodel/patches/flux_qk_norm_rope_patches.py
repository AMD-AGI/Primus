###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register FLUX attention's fused q/k RMSNorm + RoPE.

Gated on ``primus_perf.flux_fused_qk_norm_rope`` for FLUX runs; see ``models/flux/qk_norm_rope.py``.
Must run before the model is built, which ``before_train`` guarantees.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch

options.declare(
    "primus_perf",
    "flux_fused_qk_norm_rope",
    "flux_fused_qk_norm_rope_bias_grad",
    "flux_fused_qk_norm_rope_flydsl_bwd",
)

_is_flux = transformer_is("FluxTransformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.models.flux import qk_norm_rope

    return _is_flux(ctx) and qk_norm_rope.is_enabled()


@register_patch(
    "automodel.models.flux.fused_qk_norm_rope",
    backend="automodel",
    phase="before_train",
    description="FLUX attention's q/k RMSNorm, stream concat and RoPE as one op (primus_perf.flux_fused_qk_norm_rope)",
    condition=_enabled,
    priority=70,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import qk_norm_rope

    qk_norm_rope.install()
