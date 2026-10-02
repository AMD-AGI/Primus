###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the ``ddp.activation_checkpointing`` forwarding repair.

Ungated: it restores a value the user already set, and is a no-op when they did
not set one or when upstream already forwards it.
"""

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "nemo_automodel.distributed.ddp_activation_checkpointing",
    backend="nemo_automodel",
    phase="before_train",
    description="Forward ddp.activation_checkpointing, which the AutoModel parser drops for DDP",
    priority=10,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.distributed import ddp_activation_checkpointing

    ddp_activation_checkpointing.install()
