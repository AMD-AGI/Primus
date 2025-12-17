###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron FP8 Patches

This module contains patches that modify Megatron's FP8 context handling
to use Primus-specific implementations for better ROCm compatibility.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_fp8_enabled(ctx: PatchContext) -> bool:
    """Check if FP8 is enabled in module_config."""
    return getattr(get_args(ctx), "fp8", False)


@register_patch(
    "megatron.patch.fp8_context",
    backend="megatron",
    phase="before_train",
    description="Override Megatron get_fp8_context to use Primus implementation when fp8 is enabled",
    condition=_is_fp8_enabled,
)
def patch_fp8_context(ctx: PatchContext):
    """
    Patch Megatron's get_fp8_context functions to use Primus implementation.

    Behavior (moved from MegatronTrainer.patch_fp8_context):
        - When module_config.fp8 is True, replace get_fp8_context in:
            * megatron.core.transformer.transformer_block
            * megatron.core.ssm.mamba_block
            * megatron.core.transformer.multi_token_prediction
            * megatron.core.fp8_utils
          with Primus's ROCm-friendly get_fp8_context.
    """
    from megatron.core import fp8_utils
    from megatron.core.ssm import mamba_block
    from megatron.core.transformer import multi_token_prediction, transformer_block

    from primus.backends.megatron.patches.core.fp8_utils import get_fp8_context

    log_rank_0("[Patch:megatron.fp8.get_fp8_context] Overriding get_fp8_context for fp8=True")

    # Patch get_fp8_context in all relevant modules
    modules_to_patch = [
        transformer_block,
        mamba_block,
        multi_token_prediction,
        fp8_utils,
    ]

    for module in modules_to_patch:
        module.get_fp8_context = get_fp8_context
        log_rank_0(f"[Patch:megatron.fp8.get_fp8_context]   Patched {module.__name__}.get_fp8_context")
