###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Register Megatron-FSDP for the diffusion recipe.

Both patches act on settings the config already has: ``fsdp.backend:
megatron_fsdp``, which the diffusion recipe otherwise drops, and
``fsdp.average_in_collective``, which crashes megatron-fsdp 0.5.0 on a model
without expert parallelism. The one Primus setting is
``primus_mfsdp.no_param_reshard``.

A run that asks for Megatron-FSDP and cannot have it stops in the condition:
the patch runner logs a failing patch body and carries on, and here that would
train on FSDP2 under a config that says otherwise.
"""

import importlib.util

from primus.backends.automodel import options
from primus.core.patches import PatchContext, get_param, register_patch

options.declare("primus_mfsdp", "no_param_reshard")

_TRUE = {"1", "true", "yes", "on"}


def _megatron_fsdp_requested(ctx: PatchContext) -> bool:
    from primus.backends.automodel.distributed import megatron_fsdp

    if not megatron_fsdp.is_megatron_fsdp_backend(get_param(ctx, "fsdp.backend", None)):
        return False
    if options.flag("primus_ideogram4.zero1"):
        raise ValueError(
            "primus_ideogram4.zero1 shards the optimizer on the DDP path and cannot be combined with "
            "fsdp.backend: megatron_fsdp, which shards it already."
        )
    if importlib.util.find_spec("megatron_fsdp") is None:
        raise ImportError(
            "fsdp.backend: megatron_fsdp needs the megatron-fsdp package, which AutoModel pins "
            "(pip install megatron-fsdp==0.5.0)."
        )
    return True


def _average_in_collective_requested(ctx: PatchContext) -> bool:
    value = get_param(ctx, "fsdp.average_in_collective", False)
    return str(value).strip().lower() in _TRUE and _megatron_fsdp_requested(ctx)


@register_patch(
    "automodel.distributed.megatron_fsdp",
    backend="automodel",
    phase="before_train",
    description="Route fsdp.backend: megatron_fsdp to Megatron-FSDP (ZeRO-1/2/3) and shard the optimizer",
    condition=_megatron_fsdp_requested,
    priority=40,
)
def apply(ctx: PatchContext) -> None:
    from primus.backends.automodel.distributed import megatron_fsdp

    megatron_fsdp.install()
    if str(get_param(ctx, "checkpoint.enabled", True)).strip().lower() in _TRUE:
        megatron_fsdp.logger.warning(
            "[PrimusMegatronFSDP] checkpoint saving is untested with fsdp.backend: megatron_fsdp "
            "on the diffusion path."
        )


@register_patch(
    "automodel.distributed.mfsdp_average_in_collective",
    backend="automodel",
    phase="before_train",
    description="Megatron-FSDP average_in_collective without an expert-parallel group",
    condition=_average_in_collective_requested,
    priority=39,
)
def apply_average_in_collective(ctx: PatchContext) -> None:
    from primus.backends.automodel.distributed import mfsdp_average_in_collective

    mfsdp_average_in_collective.install()
