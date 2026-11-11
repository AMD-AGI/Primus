###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the train loop's logged step metrics computed locally (``training/local_log_metrics.py``).

Gated on ``primus_train.local_log_metrics``. Expressed as the patch's ``condition``
rather than an early return in the body so that a disabled patch is reported as
"skipped (condition not met)" in the run log instead of being invisible.
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare("primus_train", "local_log_metrics")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.training import local_log_metrics

    return local_log_metrics.is_enabled()


@register_patch(
    "automodel.training.local_log_metrics",
    backend="automodel",
    phase="before_train",
    description="Log step metrics without a device sync or collectives (primus_train.local_log_metrics)",
    condition=_enabled,
    # Replaces methods on the recipe class, so it must run before the profiler
    # wrapper (priority 90) takes its reference to the train loop.
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.training import local_log_metrics

    local_log_metrics.install()
