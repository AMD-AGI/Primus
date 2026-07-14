###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Skip redundant model-parallel synchronization when TP=1 and PP=1.

When running pure data-parallel (TP=1, PP=1), the model-parallel process
group has size 1.  The upstream Megatron ``train_step`` and ``training_log``
unconditionally call ``logical_and_across_model_parallel_group`` and
``reduce_max_stat_across_model_parallel_group`` which each issue an
``all_reduce`` + ``.item()`` pair.  With a size-1 group these are no-ops
that still force a GPU-to-CPU synchronization barrier, draining the
asynchronous GPU pipeline.

This patch replaces those two module-level names inside
``megatron.training.training`` with lightweight pass-through functions
that preserve the return-type contract (``bool`` and ``float | None``)
without issuing any collective or ``.item()`` call.
"""

import torch

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_pure_dp(ctx: PatchContext) -> bool:
    """True when TP=1 and PP=1 -- the MP group has size 1."""
    args = get_args(ctx)
    if args is None:
        return False
    tp = getattr(args, "tensor_model_parallel_size", 1)
    pp = getattr(args, "pipeline_model_parallel_size", 1)
    return tp == 1 and pp == 1


@register_patch(
    "megatron.training.skip_redundant_mp_sync",
    backend="megatron",
    phase="before_train",
    description="Skip redundant model-parallel all-reduces when TP=1 and PP=1",
    condition=_is_pure_dp,
    priority=35,
)
def patch_skip_redundant_mp_sync(ctx: PatchContext):
    import megatron.training.training as megatron_training

    def _passthrough_logical_and(val):
        return val

    def _passthrough_reduce_max(val):
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            return val.item()
        return val

    megatron_training.logical_and_across_model_parallel_group = _passthrough_logical_and
    megatron_training.reduce_max_stat_across_model_parallel_group = _passthrough_reduce_max
    log_rank_0("[Patch:skip_redundant_mp_sync] " "Replaced MP sync functions with passthrough (TP=1, PP=1)")
