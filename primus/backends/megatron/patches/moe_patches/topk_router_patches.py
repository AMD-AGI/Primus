###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE TopKRouter Patches

Patches for replacing Megatron's TopKRouter with PrimusTopKRouter.
"""

from primus.backends.megatron.moe_adapter.patching import patch_megatron_topk_router
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

    use_deprecated_moe = getattr(get_args(ctx), "use_deprecated_20241209_moe_layer", False)
    patch_megatron_topk_router(PrimusTopKRouter, use_deprecated_moe=use_deprecated_moe)
    log_rank_0(
        f"[Patch:megatron.moe.primus_topk_router] Patched TopKRouter -> {PrimusTopKRouter.__name__} "
        f"(deprecated_moe={use_deprecated_moe})"
    )
