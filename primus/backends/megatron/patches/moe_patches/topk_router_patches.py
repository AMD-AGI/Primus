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
        - Replace TopKRouter with PrimusTopKRouter in core modules
        - If deprecated MoE is enabled, also patch deprecated_20251209.moe_layer
    """
    from primus.backends.megatron.core.transformer.moe.router import PrimusTopKRouter

    # Patch sys.modules for core megatron router
    sys.modules["megatron.core.transformer.moe.router"].TopKRouter = PrimusTopKRouter
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.router.TopKRouter "
        f"-> {PrimusTopKRouter.__name__}"
    )

    # Patch imported moe_layer module
    from megatron.core.transformer.moe import moe_layer

    moe_layer.TopKRouter = PrimusTopKRouter
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.moe_layer.TopKRouter "
        f"-> {PrimusTopKRouter.__name__}"
    )

    # If deprecated MoE is enabled, also patch the deprecated module
    use_deprecated_moe = getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)
    if use_deprecated_moe:
        from primus.backends.megatron.core.transformer.moe import deprecated_20251209

        deprecated_20251209.moe_layer.TopKRouter = PrimusTopKRouter
        log_rank_0(
            f"[Patch:megatron.moe.primus_topk_router]   Patched megatron.core.transformer.moe.deprecated_20251209.moe_layer.TopKRouter "
            f"-> {PrimusTopKRouter.__name__}"
        )
