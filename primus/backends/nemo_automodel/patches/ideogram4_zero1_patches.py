###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register ZeRO-1 optimizer sharding for DDP runs.

Opt-in, gated on ``primus_ideogram4.zero1``, so a default run and every FSDP run
are untouched.
"""

from primus.core.patches import PatchContext, get_param, register_patch


def _enabled(ctx: PatchContext) -> bool:
    # zero1 imports only the settings module, so this stays answerable during
    # discovery without torch or AutoModel.
    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    return zero1.is_zero1_enabled()


@register_patch(
    "nemo_automodel.models.ideogram4.zero1",
    backend="nemo_automodel",
    phase="before_train",
    description="ZeRO-1 optimizer sharding on the DDP path (primus_ideogram4.zero1)",
    condition=_enabled,
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    # AutoModel's checkpoint config defaults to enabled.
    zero1.install(checkpoint_enabled=bool(get_param(ctx, "checkpoint.enabled", True)))
