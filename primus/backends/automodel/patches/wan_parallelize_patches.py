###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the Wan 2.2 sidecar that honours selective AC and reshard_after_forward.

Not gated by a setting, since with ``activation_checkpointing: true`` and no
``reshard_after_forward`` it behaves exactly like upstream's Wan sidecar; skipped
when the config names a transformer other than Wan.
"""

from primus.backends.automodel.patches._conditions import transformer_is
from primus.core.patches import PatchContext, register_patch


@register_patch(
    "automodel.models.wan.parallelize",
    backend="automodel",
    phase="before_train",
    description="Honor selective AC and reshard_after_forward in the Wan parallelizer",
    condition=transformer_is("WanTransformer3DModel"),
    priority=50,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.models.wan import parallelize

    parallelize.install()
