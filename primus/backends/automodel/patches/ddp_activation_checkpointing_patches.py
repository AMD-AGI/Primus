###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Register the ``ddp.activation_checkpointing`` forwarding repair.

Gated on ``ddp.activation_checkpointing``: the diffusion recipe takes the DDP
path only for a config with a ``ddp:`` section, and the repair has something to
forward only when that section asks for checkpointing. An FSDP2 run, or a DDP
run without checkpointing, logs it as skipped. Where upstream already forwards
the flag, the repair finds nothing to copy.
"""

from primus.core.patches import PatchContext, get_param, register_patch


def _ddp_checkpointing_requested(ctx: PatchContext) -> bool:
    return ctx is not None and bool(get_param(ctx, "ddp.activation_checkpointing", False))


@register_patch(
    "automodel.distributed.ddp_activation_checkpointing",
    backend="automodel",
    phase="before_train",
    description="Forward ddp.activation_checkpointing, which the AutoModel parser drops for DDP",
    condition=_ddp_checkpointing_requested,
    priority=10,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.distributed import ddp_activation_checkpointing

    ddp_activation_checkpointing.install()
