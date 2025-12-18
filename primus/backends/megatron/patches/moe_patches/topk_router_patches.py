###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE TopKRouter Patches

Patches for replacing Megatron's TopKRouter with PrimusTopKRouter.
"""

import sys

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.moe.primus_topk_router",
    backend="megatron",
    phase="before_train",
    description="Replace TopKRouter with PrimusTopKRouter",
    condition=lambda ctx: not getattr(get_args(ctx), "disable_primus_topk_router", False),
)
def patch_primus_topk_router(ctx: PatchContext):
    """
    Patch Megatron to use Primus TopKRouter.

    Behavior:
        - Replace TopKRouter with PrimusTopKRouter
        - If deprecated MoE is enabled, also patch deprecated router
    """
    # Check if deprecated MoE is also enabled
    use_deprecated_moe = getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)

    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe.deprecated_20251209.router import (
            DeprecatedTopKRouter,
        )

        sys.modules["megatron.core.transformer.moe.router"].TopKRouter = DeprecatedTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.router.TopKRouter "
            f"-> {DeprecatedTopKRouter.__name__}"
        )

    # patch module class
    from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

    sys.modules["megatron.core.transformer.moe.router"].TopKRouter = PrimusTopKRouter
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.router.TopKRouter "
        f"-> {PrimusTopKRouter.__name__}"
    )
    # patch imported module
    from megatron.core.transformer.moe import moe_layer

    moe_layer.TopKRouter = PrimusTopKRouter

    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe import deprecated_20251209

        deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.deprecated_20251209.moe_layer.TopKRouter "
            f"-> {PrimusTopKRouter.__name__}"
        )
