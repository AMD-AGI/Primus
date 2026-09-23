###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the real-AC FLUX parallelization strategy.

Gated on ``PRIMUS_FLUX_REAL_AC``. Shares a priority band with the other per-model
strategies: after the FSDP2 reshard repair, which they all read, and before the
profiler, which wraps the train loop.
"""

from primus.core.patches import PatchContext, register_patch


def _enabled(ctx: PatchContext) -> bool:
    from primus.backends.nemo_automodel.models.flux import parallelize

    return parallelize.is_enabled()


@register_patch(
    "nemo_automodel.models.flux.parallelize",
    backend="nemo_automodel",
    phase="before_train",
    description="Make fsdp.activation_checkpointing take effect for FLUX (PRIMUS_FLUX_REAL_AC)",
    condition=_enabled,
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.models.flux import parallelize

    parallelize.install()
