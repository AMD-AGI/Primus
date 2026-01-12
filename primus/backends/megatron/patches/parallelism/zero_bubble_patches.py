###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron ZeroBubble pipeline parallelism patches.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.pp.zero_bubble_optimizer",
    backend="megatron",
    phase="before_train",
    description="Patch optimizer and forward_backward_func for ZeroBubble PP",
    condition=lambda ctx: getattr(get_args(ctx), "patch_zero_bubble", False),
)
def patch_zero_bubble_pp(ctx: PatchContext):
    """
    Patch Megatron to use ZeroBubble PP implementation.

    Behavior:
        - Replace ChainedOptimizer with ZeroBubblePPChainedOptimizer
        - Replace get_forward_backward_func with get_forward_backward_func_zbpp
    """

    # Patch optimizer
    import megatron.core.optimizer as optimizer

    from primus.backends.megatron.core.optimizer.zbpp_optimizer import (
        ZeroBubblePPChainedOptimizer,
    )

    optimizer.ChainedOptimizer = ZeroBubblePPChainedOptimizer
    log_rank_0(
        f"[Patch:megatron.pp.zero_bubble_optimizer]   Patched megatron.core.optimizer.ChainedOptimizer "
        f"-> {ZeroBubblePPChainedOptimizer.__name__}"
    )

    # Patch get_forward_backward_func
    import megatron.core.pipeline_parallel as ori_pp

    from primus.backends.megatron.core.pipeline_parallel.schedules import (
        get_forward_backward_func_zbpp,
    )

    ori_pp.get_forward_backward_func = get_forward_backward_func_zbpp
    log_rank_0(
        f"[Patch:megatron.pp.zero_bubble_optimizer]   Patched megatron.core.pipeline_parallel.get_forward_backward_func "
        f"-> {get_forward_backward_func_zbpp.__name__}"
    )
