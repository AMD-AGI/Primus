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


@register_patch(
    "megatron.checkpoint.save_checkpoint",
    backend="megatron",
    phase="setup",
    description="Wrap save_checkpoint to skip saving at the last iteration",
)
def patch_save_checkpoint(ctx: PatchContext):
    """
    Wrap Megatron's save_checkpoint to skip saving at the last iteration

    This patch monkey-patches the save_checkpoint function in
    megatron.training.training module to check if:
    1. disable_last_saving is True
    2. Current iteration equals train_iters (final iteration)

    If both conditions are met, the checkpoint save is skipped.
    """
    try:
        import megatron.training.training as training_module
        from megatron.training import get_args
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.checkpoint.save_checkpoint] Skip patch (Megatron not available): {e}")
        return

    # Save original function
    original_save_checkpoint = training_module.save_checkpoint

    def wrapped_save_checkpoint(
        iteration,
        model,
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        checkpointing_context,
        non_persistent_ckpt=False,
        train_data_iterator=None,
    ):
        args = get_args()

        if args.disable_last_saving and iteration == args.train_iters:
            log_rank_0(
                f"[Patch:megatron.checkpoint.save_checkpoint] Skip saving at the last iteration: {iteration}"
            )
            return

        return original_save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far,
            checkpointing_context,
            non_persistent_ckpt=False,
            train_data_iterator=None,
        )

    training_module.save_checkpoint = wrapped_save_checkpoint
    log_rank_0("[Patch:megatron.checkpoint.save_checkpoint] Patch save_checkpoint successfully.")
