###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the fused gradient-clip scaling (``training/grad_clip.py``).

Gated on ``primus_train.fast_grad_clip``. Expressed as the patch's ``condition``
so that a disabled patch is reported as "skipped (condition not met)".
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare("primus_train", "fast_grad_clip")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.training import grad_clip

    return grad_clip.is_enabled()


@register_patch(
    "automodel.training.fast_grad_clip",
    backend="automodel",
    phase="before_train",
    description="Scale clipped gradients with one fused multiply (primus_train.fast_grad_clip)",
    condition=_enabled,
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.training import grad_clip

    grad_clip.install()
