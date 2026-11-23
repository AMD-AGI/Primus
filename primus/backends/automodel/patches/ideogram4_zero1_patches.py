###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register ZeRO-1 optimizer sharding for Ideogram-4 DDP runs.

Opt-in, gated on ``primus_ideogram4.zero1``, so a default run and every FSDP run
are untouched. The optimizer patch is process-wide, so it is also gated on the
run training Ideogram-4: the setting in another model's config changes nothing.
"""

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import (
    checkpoint_enabled,
    transformer_is,
)
from primus.core.patches import PatchContext, register_patch

options.declare("primus_ideogram4", "zero1")

_is_ideogram4 = transformer_is("Ideogram4Transformer2DModel")


def _enabled(ctx: PatchContext) -> bool:
    if not _is_ideogram4(ctx):
        return False
    # zero1 imports only the settings module, so this stays answerable during
    # discovery without torch or AutoModel.
    from primus.backends.automodel.models.ideogram4 import zero1

    return zero1.is_zero1_enabled()


@register_patch(
    "automodel.models.ideogram4.zero1",
    backend="automodel",
    phase="before_train",
    description="ZeRO-1 optimizer sharding on the DDP path (primus_ideogram4.zero1)",
    condition=_enabled,
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.ideogram4 import zero1

    zero1.install(checkpoint_enabled=checkpoint_enabled(ctx))
