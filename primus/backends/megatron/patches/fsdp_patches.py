###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron FSDP Patches

This module contains patches that modify Megatron's FSDP integration to use
Primus-specific implementations when requested via module_config.
"""


from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _patch_and_log(target_module, attribute_name: str, new_value, patch_name: str):
    """
    Patch a module attribute and log the operation.

    Args:
        target_module: The module to patch
        attribute_name: The attribute name to replace
        new_value: The new value to assign
        patch_name: Patch identifier for logging
    """
    setattr(target_module, attribute_name, new_value)
    log_rank_0(
        f"[Patch:{patch_name}]   Patched {target_module.__name__}.{attribute_name} "
        f"-> {new_value.__name__}"
    )


def _is_fsdp2_enabled(ctx: PatchContext) -> bool:
    """Check if FSDP2 is enabled in backend_args."""
    return getattr(get_args(ctx), "use_torch_fsdp2", False)


@register_patch(
    "megatron.patch.torch_fsdp2",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace Megatron's TorchFullyShardedDataParallel with Primus implementation "
        "when use_torch_fsdp2 is enabled."
    ),
    condition=_is_fsdp2_enabled,
)
def patch_torch_fsdp(ctx: PatchContext):
    """
    Patch Megatron to use Primus's TorchFullyShardedDataParallel wrapper.

    Behavior (moved from MegatronTrainer.patch_torch_fsdp):
        - If backend_args.use_torch_fsdp2 is True:
            * Patch megatron.core.distributed.torch_fully_sharded_data_parallel.
            * Patch megatron.training.training.torch_FSDP reference.
    """

    # Import custom FSDP wrapper
    import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp_module
    from megatron.training import training

    from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
        PrimusTorchFullyShardedDataParallel,
    )

    # Patch Megatron's internal reference to FSDP2 class
    _patch_and_log(
        torch_fsdp_module,
        "TorchTorchFullyShardedDataParallel",
        PrimusTorchFullyShardedDataParallel,
        "megatron.fsdp.torch_fsdp2",
    )

    # Patch training code reference
    _patch_and_log(
        training,
        "torch_FSDP",
        PrimusTorchFullyShardedDataParallel,
        "megatron.fsdp.torch_fsdp2",
    )
