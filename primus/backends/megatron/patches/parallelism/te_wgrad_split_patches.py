###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Transformer Engine weight gradient split patches for pipeline parallelism.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.pp.te_wgrad_split",
    backend="megatron",
    phase="before_train",
    description="Patch Transformer Engine Linear layers for weight gradient splitting",
    condition=lambda ctx: (
        getattr(get_args(ctx), "patch_primus_pipeline", False)
        or getattr(get_args(ctx), "patch_zero_bubble", False)
    ),
)
def patch_te_wgrad_split(ctx: PatchContext):
    """
    Patch Transformer Engine Linear layers to split weight gradient computation.

    Behavior:
        - Replace TE _GroupedLinear with _GroupedLinearWithWGradSplit
        - Replace TE _Linear with _LinearWithWGradSplit
    """
    # Patch TE grouped_linear
    import transformer_engine.pytorch.module.grouped_linear as ori_grouped_linear

    from primus.backends.megatron.core.extensions.te_group_gemm_patch_wgrad import (
        _GroupedLinearWithWGradSplit,
    )

    ori_grouped_linear._GroupedLinear = _GroupedLinearWithWGradSplit
    log_rank_0(
        f"[Patch:megatron.pp.te_wgrad_split]   Patched transformer_engine.pytorch.module.grouped_linear._GroupedLinear "
        f"-> {_GroupedLinearWithWGradSplit.__name__}"
    )

    # Patch TE linear
    import transformer_engine.pytorch.module.linear as ori_linear

    from primus.backends.megatron.core.extensions.te_gemm_patch_wgrad import (
        _LinearWithWGradSplit,
    )

    ori_linear._Linear = _LinearWithWGradSplit
    log_rank_0(
        f"[Patch:megatron.pp.te_wgrad_split]   Patched transformer_engine.pytorch.module.linear._Linear "
        f"-> {_LinearWithWGradSplit.__name__}"
    )
