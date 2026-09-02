###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register the FSDP2 ``reshard_after_forward`` repair.

The implementation lives in ``distributed/fsdp2_reshard.py`` rather than here
because the per-model parallelization strategies import that module by reference
to read its provenance globals -- they need to know the repair is in place before
forwarding the value themselves. Importing runtime state *from a patch module*
would be a strange contract, so registration and implementation are kept apart.

Deliberately not env-gated: the repair restores a value the user already set in
YAML, and is a no-op when they did not set one.
"""

from primus.core.patches import PatchContext, register_patch


@register_patch(
    "nemo_automodel.distributed.fsdp2_reshard",
    backend="nemo_automodel",
    phase="before_train",
    description="Re-apply fsdp.reshard_after_forward, which the AutoModel manager whitelist drops",
    # Runs early: the parallelization strategies check this repair is present, so
    # it must be applied before any of them.
    priority=10,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.nemo_automodel.distributed import fsdp2_reshard

    fsdp2_reshard.install()
