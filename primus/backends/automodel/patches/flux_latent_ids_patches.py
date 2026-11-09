###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Build the FLUX.1 adapter's latent position ids on the GPU (``models/flux/latent_ids``).

Not gated by a setting, since the ids have the same values either way; skipped
when the config names a transformer other than FLUX.
"""

from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "automodel.models.flux.latent_ids",
    backend="automodel",
    phase="before_train",
    description="FluxAdapter builds its latent image ids on device instead of copying them from the CPU",
    condition=transformer_is("FluxTransformer2DModel"),
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.flux import latent_ids

    latent_ids.install()
