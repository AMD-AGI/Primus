###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the deferred loss check and SIGTERM poll (``training/step_syncs.py``).

Gated on ``primus_train.defer_step_syncs``. Expressed as the patch's
``condition`` so that a disabled patch is reported as "skipped (condition not
met)".
"""

from primus.backends.automodel import options
from primus.core.patches import PatchContext, register_patch

options.declare("primus_train", "defer_step_syncs")


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.automodel.training import step_syncs

    return step_syncs.is_enabled()


@register_patch(
    "automodel.training.defer_step_syncs",
    backend="automodel",
    phase="before_train",
    description="Defer the train step's loss check and SIGTERM poll (primus_train.defer_step_syncs)",
    condition=_enabled,
    priority=80,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.training import step_syncs

    step_syncs.install()
