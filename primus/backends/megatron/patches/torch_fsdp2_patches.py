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
    # Megatron Core 0.16 may pass new kwargs (e.g., force_all_reduce) into
    # model_chunk.finish_grad_sync(). Keep FSDP2 path forward-compatible by
    from megatron.core.distributed import data_parallel_base

    original_start_grad_sync = data_parallel_base._BaseDataParallel.start_grad_sync
    original_finish_grad_sync = data_parallel_base._BaseDataParallel.finish_grad_sync

    if not getattr(original_start_grad_sync, "_primus_grad_sync_compat", False):

        def _patched_start_grad_sync(self, *unused, **unused_kwargs):
            return original_start_grad_sync(self, *unused)

        def _patched_finish_grad_sync(self, *unused, **unused_kwargs):
            return original_finish_grad_sync(self)

        setattr(_patched_start_grad_sync, "_primus_grad_sync_compat", True)
        setattr(_patched_finish_grad_sync, "_primus_grad_sync_compat", True)
        data_parallel_base._BaseDataParallel.start_grad_sync = _patched_start_grad_sync
        data_parallel_base._BaseDataParallel.finish_grad_sync = _patched_finish_grad_sync

        log_rank_0(
            "[Patch:megatron.fsdp.torch_fsdp2]   Patched _BaseDataParallel "
            "start_grad_sync/finish_grad_sync to accept extra kwargs "
        )
