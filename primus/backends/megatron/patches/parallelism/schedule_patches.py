###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Primus Pipeline parallelism patches.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.pp.schedule",
    backend="megatron",
    phase="before_train",
    description="Patch forward_backward_func for Primus Pipeline schedule",
    condition=lambda ctx: (
        getattr(get_args(ctx), "patch_primus_pipeline", False)
        and not getattr(get_args(ctx), "patch_zero_bubble", False)
    ),
)
def patch_pipeline_schedule(ctx: PatchContext):
    """
    Patch Megatron to use Primus Pipeline schedule.

    Behavior:
        - Replace get_forward_backward_func with get_primus_pipeline_parallel_fwd_backward_func
    """

    import megatron.core.pipeline_parallel as ori_pp

    from primus.backends.megatron.core.pipeline_parallel.schedules import (
        get_primus_pipeline_parallel_fwd_backward_func,
    )

    ori_pp.get_forward_backward_func = get_primus_pipeline_parallel_fwd_backward_func
    log_rank_0(
        f"[Patch:megatron.pp.schedule]   Patched megatron.core.pipeline_parallel.get_forward_backward_func "
        f"-> {get_primus_pipeline_parallel_fwd_backward_func.__name__}"
    )
