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

A run that asks for Megatron-FSDP and cannot have it stops: in the condition
when megatron-fsdp is missing, and with ``SystemExit`` when a patch fails to
install. The patch runner logs any other exception from a patch body and
carries on, and here that would train on FSDP2, or with an unsharded
optimizer, under a config that says otherwise.
"""

import importlib.util

from primus.backends.automodel import options
from primus.backends.automodel.patches._conditions import (
    as_automodel_reads_it,
    checkpoint_enabled,
)
from primus.core.patches import PatchContext, get_param, register_patch

options.declare("primus_mfsdp", "no_param_reshard")


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
        raise ImportError(megatron_fsdp.MISSING_PACKAGE)
    return True


def _average_in_collective_requested(ctx: PatchContext) -> bool:
    # Read as AutoModel hands it on: whatever reaches megatron-fsdp as on needs the repair.
    value = as_automodel_reads_it(get_param(ctx, "fsdp.average_in_collective", False))
    return bool(value) and _megatron_fsdp_requested(ctx)


def _install_or_exit(install, what: str) -> None:
    from primus.backends.automodel.distributed.megatron_fsdp import LOG_PREFIX

    # SystemExit is a BaseException, which the patch runner does not catch.
    try:
        install()
    except Exception as exc:
        raise SystemExit(f"{LOG_PREFIX} {what} failed to install: {exc}") from exc


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

    _install_or_exit(megatron_fsdp.install, "Megatron-FSDP for the diffusion recipe")
    if checkpoint_enabled(ctx):
        megatron_fsdp.logger.warning(
            "%s checkpoint saving is untested with fsdp.backend: megatron_fsdp on the diffusion path.",
            megatron_fsdp.LOG_PREFIX,
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

    _install_or_exit(mfsdp_average_in_collective.install, "the average_in_collective repair")
