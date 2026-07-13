###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus Turbo Mega MoE Patches

Replace Megatron's ``MoELayer`` with the fully fused PrimusTurbo ``MegaMoE``
adapter. EP-only (TP==1) + bf16.
"""


from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_turbo_mega_moe_can_patch(ctx: PatchContext) -> bool:
    """
    Check if the PrimusTurbo fused Mega MoE layer is enabled.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
      - use_turbo_mega_moe == True
    """
    args = get_args(ctx)
    use_turbo_mega_moe = bool(getattr(args, "use_turbo_mega_moe", False))

    return use_turbo_mega_moe and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.mega_moe",
    backend="megatron",
    phase="before_train",
    description="Replace MoELayer with the fully fused PrimusTurbo MegaMoE layer",
    condition=_is_turbo_mega_moe_can_patch,
)
def patch_mega_moe(ctx: PatchContext):
    """
    Patch Megatron to use the fully fused PrimusTurbo MegaMoE layer.

    Replaces ``MoELayer`` in both ``moe_layer`` and ``gpt.moe_module_specs`` so the
    ``== MoELayer`` identity check in ``transformer_layer.py`` stays consistent.
    """
    from megatron.core.models.gpt import moe_module_specs
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.extensions.mega_moe import (
        PrimusTurboMegaMoELayer,
    )

    log_rank_0("[Patch:megatron.turbo.mega_moe] Patching MoELayer with fused MegaMoE...")

    moe_layer.MoELayer = PrimusTurboMegaMoELayer
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe]   Patched "
        f"megatron.core.transformer.moe.moe_layer.MoELayer -> {PrimusTurboMegaMoELayer.__name__}"
    )

    moe_module_specs.MoELayer = PrimusTurboMegaMoELayer
    log_rank_0(
        "[Patch:megatron.turbo.mega_moe]   Patched "
        f"megatron.core.models.gpt.moe_module_specs.MoELayer -> {PrimusTurboMegaMoELayer.__name__}"
    )
