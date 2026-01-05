###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Linear gradient split patches for pipeline parallelism.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.pp.linear_grad_split",
    backend="megatron",
    phase="before_train",
    description="Patch Linear layer to split d_w and d_input for PP optimization",
    condition=lambda ctx: (
        getattr(get_args(ctx), "patch_primus_pipeline", False)
        or getattr(get_args(ctx), "patch_zero_bubble", False)
    ),
)
def patch_linear_grad_split(ctx: PatchContext):
    """
    Patch LinearWithGradAccumulationAndAsyncCommunication for gradient splitting.

    Behavior:
        - Replace LinearWithGradAccumulationAndAsyncCommunication with Primus version
          that splits weight gradient and input gradient computation
    """
    import megatron.core.tensor_parallel.layers as ori_layers

    from primus.backends.megatron.core.tensor_parallel.layers import (
        LinearWithGradAccumulationAndAsyncCommunication,
    )

    ori_layers.LinearWithGradAccumulationAndAsyncCommunication = (
        LinearWithGradAccumulationAndAsyncCommunication
    )
    log_rank_0(
        f"[Patch:megatron.pp.linear_grad_split]   Patched megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication "
        f"-> {LinearWithGradAccumulationAndAsyncCommunication.__name__}"
    )
