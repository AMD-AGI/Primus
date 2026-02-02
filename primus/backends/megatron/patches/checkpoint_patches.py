###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Checkpoint / FileSystemWriter Patches

This module contains patches that modify Megatron's checkpointing strategies
to use Primus-specific implementations.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.patch.load_checkpoint_distributed_fix",
    backend="megatron",
    phase="before_train",
    description=(
        "Fix load_checkpoint to handle torch_dist checkpoints where model weights "
        "are already loaded via dist_checkpointing.load() and state_dict has no 'model' key."
    ),
)
def patch_load_checkpoint_for_distributed(ctx: PatchContext):
    """
    Patch load_checkpoint to handle KeyError: 'model' for torch_dist checkpoints.
    
    For distributed checkpoints, model weights are loaded in-place during
    _load_base_checkpoint, so state_dict doesn't contain 'model' key.
    """
    log_rank_0("[Patch:megatron.checkpoint.load_checkpoint] Patching for distributed checkpoint support...")
    
    from megatron.training import checkpointing, training
    
    original_load_checkpoint = checkpointing.load_checkpoint
    
    def patched_load_checkpoint(ddp_model, optimizer, opt_param_scheduler, load_arg='load', 
                                strict=True, checkpointing_context=None, skip_load_to_model_and_opt=False):
        """Wrapper that handles missing 'model' key for torch_dist checkpoints in finetune mode."""
        try:
            return original_load_checkpoint(
                ddp_model, optimizer, opt_param_scheduler, load_arg,
                strict, checkpointing_context, skip_load_to_model_and_opt
            )
        except KeyError as e:
            if str(e) == "'model'":
                # For torch_dist checkpoints in finetune mode, model weights are already loaded
                # via dist_checkpointing.load(), so 'model' key doesn't exist in state_dict
                log_rank_0("[Patch] KeyError: 'model' - this is expected for torch_dist finetune mode")
                log_rank_0("[Patch] Model weights already loaded via dist_checkpointing.load()")
                return 0, 0  # Return (iteration=0, num_floating_point_operations_so_far=0)
            raise
    
    # Patch both the module and the imported reference in training.py
    checkpointing.load_checkpoint = patched_load_checkpoint
    training.load_checkpoint = patched_load_checkpoint
    
    log_rank_0("[Patch:megatron.checkpoint.load_checkpoint] Patch applied successfully.")


@register_patch(
    "megatron.patch.filesystem_writer_async",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace Megatron's FileSystemWriterAsync with Primus implementation "
        "to support ROCm-aware checkpointing behavior."
    ),
)
def patch_filesystem_writer_async(ctx: PatchContext):
    """
    Patch Megatron's FileSystemWriterAsync to use Primus implementation.

    Behavior (moved from MegatronTrainer.patch_file_system_writer):
        - Always attempts to replace
          megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync
          with PrimusFileSystemWriterAsync.
    """
    log_rank_0("[Patch:megatron.checkpoint.filesystem_writer_async] Patching FileSystemWriterAsync...")

    import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

    from primus.backends.megatron.core.dist_checkpointing.strategies.filesystem_async import (
        PrimusFileSystemWriterAsync,
    )

    filesystem_async_module.FileSystemWriterAsync = PrimusFileSystemWriterAsync

    log_rank_0(
        "[Patch:megatron.checkpoint.filesystem_writer_async] Patch FileSystemWriterAsync successfully."
    )
