###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Transformer Engine weight gradient split patches for pipeline parallelism.

Uses TE's native delay_wgrad_compute mechanism with a patched WeightGradStore
that redirects wgrad closures to Primus's pipeline scheduler, instead of
monkey-patching TE's entire _Linear / _GroupedLinear autograd Functions.

This approach is TE-version-agnostic: TE's own backward handles all the
FP8 quantization, communication overlap, and GEMM logic; we only intercept
the final wgrad closure via WeightGradStore.put().
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.pp.te_wgrad_split",
    backend="megatron",
    phase="before_train",
    description="Patch TE WeightGradStore to redirect wgrad to Primus pipeline scheduling",
    condition=lambda ctx: (
        getattr(get_args(ctx), "patch_primus_pipeline", False)
        or getattr(get_args(ctx), "patch_zero_bubble", False)
    ),
)
def patch_te_wgrad_split(ctx: PatchContext):
    """
    Patch TE's WeightGradStore.put/pop methods so that delayed wgrad closures
    are forwarded to Primus's WGradRunningCache or zero-bubble WeightGradStore.

    Also sets delay_wgrad_compute=True in Megatron's TransformerConfig so that
    TE layers created later will use the delayed wgrad path.
    """
    from primus.backends.megatron.core.extensions.te_wgrad_store import (
        patch_weight_grad_store,
    )

    patch_weight_grad_store()
    log_rank_0(
        "[Patch:megatron.pp.te_wgrad_split] " "Patched TE WeightGradStore.put/pop for Primus wgrad scheduling"
    )
