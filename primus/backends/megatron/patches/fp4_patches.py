###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron enums Patches
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.core.enums",
    backend="megatron",
    phase="before_train",
    description="Override Megatron enums to use Primus implementation when fp4 is enabled",
    condition=lambda ctx: getattr(get_args(ctx), "fp4", False),
)
def patch_enums(ctx: PatchContext):
    from megatron.core import enums

    from primus.backends.megatron.core.enums import Fp4Recipe

    log_rank_0("[Patch:megatron.core.enums] Overriding enums for fp4=True")

    enums.Fp4Recipe = Fp4Recipe
    log_rank_0(f"[Patch:megatron.core.enums]   Patched {enums.__name__}.Fp4Recipe")


@register_patch(
    "megatron.core.fp4_utils",
    backend="megatron",
    phase="before_train",
    description="Override Megatron get_fp4_context to use Primus implementation when fp4 is enabled",
    condition=lambda ctx: getattr(get_args(ctx), "fp4", False),
)
def patch_fp4_context(ctx: PatchContext):
    """
    Patch Megatron's get_fp4_context functions to use Primus implementation.

    Behavior (moved from MegatronTrainer.patch_fp4_context):
        - When module_config.fp4 is True, replace get_fp4_context in:
            * megatron.core.transformer.transformer_block
            * megatron.core.ssm.mamba_block
            * megatron.core.transformer.multi_token_prediction
            * megatron.core.fp4_utils
          with Primus's ROCm-friendly get_fp4_context.
    """
    from megatron.core import fp4_utils
    from megatron.core.ssm import mamba_block
    from megatron.core.transformer import multi_token_prediction, transformer_block

    from primus.backends.megatron.core.fp4_utils import get_fp4_context

    log_rank_0("[Patch:megatron.fp4.context] Overriding get_fp4_context for fp4=True")

    # Patch get_fp4_context in all relevant modules
    modules_to_patch = [
        transformer_block,
        mamba_block,
        multi_token_prediction,
        fp4_utils,
    ]

    for module in modules_to_patch:
        module.get_fp4_context = get_fp4_context
        log_rank_0(f"[Patch:megatron.fp4.context]   Patched {module.__name__}.get_fp4_context")
