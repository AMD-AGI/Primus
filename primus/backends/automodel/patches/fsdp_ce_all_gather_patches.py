###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register FSDP2 all-gathers on RCCL's copy engines.

Gated on ``primus_perf.fsdp_ce_all_gather``; see ``distributed/ce_all_gather.py``.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare("primus_perf", "fsdp_ce_all_gather", "fsdp_ce_all_gather_modules")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.distributed import ce_all_gather

    return ce_all_gather.is_enabled()


@register_patch(
    "automodel.distributed.fsdp_ce_all_gather",
    backend="automodel",
    phase="before_train",
    description="FSDP2 all-gathers on RCCL's copy engines (primus_perf.fsdp_ce_all_gather)",
    condition=_enabled,
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.distributed import ce_all_gather

    ce_all_gather.install()
