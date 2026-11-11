###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register FSDP2 reduce-scatters on RCCL's copy engines.

Gated on ``primus_perf.fsdp_ce_reduce_scatter``; see ``distributed/ce_reduce_scatter.py``.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare(
    "primus_perf",
    "fsdp_ce_reduce_scatter",
    "fsdp_ce_reduce_scatter_modules",
    "fsdp_ce_reduce_scatter_debug_check",
)


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.distributed import ce_reduce_scatter

    return ce_reduce_scatter.is_enabled()


@register_patch(
    "automodel.distributed.fsdp_ce_reduce_scatter",
    backend="automodel",
    phase="before_train",
    description="FSDP2 reduce-scatters on RCCL's copy engines (primus_perf.fsdp_ce_reduce_scatter)",
    condition=_enabled,
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.distributed import ce_reduce_scatter

    ce_reduce_scatter.install()
