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
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    "megatron.checkpoint.filesystem_writer_async",
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
    try:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from primus.backends.megatron.core.dist_checkpointing.strategies.filesystem_async import (
            PrimusFileSystemWriterAsync,
        )

        filesystem_async_module.FileSystemWriterAsync = PrimusFileSystemWriterAsync
    except Exception as e:  # pragma: no cover - best-effort logging
        warning_rank_0(
            f"[Patch:megatron.checkpoint.filesystem_writer_async] Patch FileSystemWriterAsync failed: {e}"
        )
    else:
        log_rank_0(
            "[Patch:megatron.checkpoint.filesystem_writer_async] Patch FileSystemWriterAsync successfully."
        )
