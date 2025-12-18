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


@register_patch(
    "megatron.fsdp.torch_fsdp2",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace Megatron's TorchFullyShardedDataParallel with Primus implementation "
        "when use_torch_fsdp2 is enabled."
    ),
    condition=lambda ctx: getattr(get_args(ctx), "use_torch_fsdp2", False),
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
    # Patch Megatron's internal reference to FSDP2 class
    import megatron.core.distributed.torch_fully_sharded_data_parallel as torch_fsdp_module

    from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
        PrimusTorchFullyShardedDataParallel,
    )

    torch_fsdp_module.TorchTorchFullyShardedDataParallel = PrimusTorchFullyShardedDataParallel
    log_rank_0(
        "[Patch:megatron.fsdp.torch_fsdp2]   Patched "
        "megatron.core.distributed.torch_fully_sharded_data_parallel.TorchTorchFullyShardedDataParallel "
        f"-> {PrimusTorchFullyShardedDataParallel.__name__}"
    )

    # Patch training code reference
    from megatron.training import training

    training.torch_FSDP = PrimusTorchFullyShardedDataParallel
    log_rank_0(
        f"[Patch:megatron.fsdp.torch_fsdp2]   Patched megatron.training.training.torch_FSDP "
        f"-> {PrimusTorchFullyShardedDataParallel.__name__}"
    )
