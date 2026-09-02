###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the Ideogram-4 DDP and ZeRO-1 patches.

Gated on ``PRIMUS_IDEOGRAM_DDP`` or ``PRIMUS_IDEOGRAM_ZERO1``, so a default run
and every FSDP run are untouched.

Runs BEFORE the FSDP parallelization strategy, at a priority the two do not
share. The ordering does not matter for correctness -- the strategies apply to
disjoint paths, and only one of them can be in use -- but a fixed order means the
log reads the same way every time, which is the point of the priorities.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    # zero1 imports only the environment helper, the shared AC helper and the
    # sibling strategy module for its stride parser, none of which pull in torch or
    # AutoModel, so this stays answerable during discovery.
    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    return zero1.is_ddp_enabled() or zero1.is_zero1_enabled()


@register_patch(
    "nemo_automodel.models.ideogram4.zero1",
    backend="nemo_automodel",
    phase="before_train",
    description=(
        "DDP activation checkpointing and ZeRO-1 optimizer sharding for Ideogram-4 "
        "(PRIMUS_IDEOGRAM_DDP, PRIMUS_IDEOGRAM_ZERO1)"
    ),
    condition=_enabled,
    # Before the FSDP strategy at priority 50. Both are inert on the other's path,
    # so this fixes the log order rather than resolving a dependency.
    priority=45,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.ideogram4 import zero1

    zero1.install()
