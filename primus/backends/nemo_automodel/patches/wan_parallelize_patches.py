###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the repaired Wan 2.2 parallelization strategy.

Gated on ``PRIMUS_WAN_PARALLELIZE_FIX``. The repair depends on the FSDP2 reshard
repair in ``distributed/fsdp2_reshard.py`` -- fixing either half alone changes
nothing -- so this runs after it. The strategy logs an error rather than raising
if it finds the other half missing, since by then a training run is already under
way and failing it would be the worse outcome.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.models.wan import parallelize

    return parallelize.is_enabled()


@register_patch(
    "nemo_automodel.models.wan.parallelize",
    backend="nemo_automodel",
    phase="before_train",
    description="Honor selective AC and reshard_after_forward in the Wan strategy "
    "(PRIMUS_WAN_PARALLELIZE_FIX)",
    condition=_enabled,
    # After the reshard repair (priority 10), which this depends on.
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.wan import parallelize

    parallelize.install()
