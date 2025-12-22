###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron pipeline parallelism patches.

This package splits the original PP patch definitions from ``pp_patches.py``
into focused modules under ``patches.parallelism``:

    - ZeroBubble optimizer and schedule patches
    - Primus Pipeline schedule patches
    - Linear / TE weight gradient split patches
    - V-schedule support patches
    - PipelineParallelLayerLayout replacement

The patch IDs and behavior are preserved; only the code organization changes.
"""

from primus.backends.megatron.patches.parallelism import (  # noqa: F401
    linear_grad_split_patches,
    pipeline_parallel_layout_patches,
    schedule_patches,
    te_wgrad_split_patches,
    v_schedule_patches,
    zero_bubble_patches,
)

__all__ = [
    "zero_bubble_patches",
    "schedule_patches",
    "linear_grad_split_patches",
    "te_wgrad_split_patches",
    "v_schedule_patches",
    "pipeline_parallel_layout_patches",
]
