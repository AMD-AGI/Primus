###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

    # Patch get_data_parallel_group_if_dtensor to handle 2D HSDP meshes.
    # The upstream implementation calls tensor.device_mesh.get_group() without mesh_dim,
    # which fails for 2D meshes. For HSDP (dp_replicate, dp_shard), we return the
    # shard group (innermost dim) since that's where parameters are actually sharded;
    # replicas have identical gradients so we must not all-reduce across them.
    import megatron.core.optimizer.clip_grads as clip_grads_module
    import megatron.core.utils as mcore_utils
    from megatron.training import utils as training_utils

    try:
        from torch.distributed._tensor import DTensor

        HAVE_DTENSOR = True
    except ImportError:
        HAVE_DTENSOR = False

    def _get_data_parallel_group_if_dtensor(tensor, data_parallel_group=None):
        if HAVE_DTENSOR and isinstance(tensor, DTensor):
            mesh = tensor.device_mesh
            if mesh.ndim > 1:
                current_group = mesh.get_group(mesh_dim=-1)
            else:
                current_group = mesh.get_group()
            if data_parallel_group is not None and current_group != data_parallel_group:
                raise RuntimeError("DTensor mesh group does not match the expected data_parallel_group")
            return current_group
        return None

    mcore_utils.get_data_parallel_group_if_dtensor = _get_data_parallel_group_if_dtensor
    clip_grads_module.get_data_parallel_group_if_dtensor = _get_data_parallel_group_if_dtensor
    if hasattr(training_utils, "get_data_parallel_group_if_dtensor"):
        training_utils.get_data_parallel_group_if_dtensor = _get_data_parallel_group_if_dtensor
    log_rank_0(
        "[Patch:megatron.fsdp.torch_fsdp2]   Patched get_data_parallel_group_if_dtensor "
        "for 2D HSDP mesh support"
    )
