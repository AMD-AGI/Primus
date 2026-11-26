###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Make the FLUX.1 adapter honour ``use_guidance_embeds``, so FLUX.1-schnell trains.

Not gated by a setting, since with the default ``use_guidance_embeds: true``
nothing changes; skipped when the config names a transformer other than FLUX.
"""

from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "automodel.models.flux.guidance",
    backend="automodel",
    phase="before_train",
    description="FluxAdapter honours use_guidance_embeds (FLUX.1-schnell has no guidance embedder)",
    condition=transformer_is("FluxTransformer2DModel"),
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import guidance

    guidance.install()
