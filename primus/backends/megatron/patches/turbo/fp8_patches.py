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

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


def _is_fp8_can_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    fp8 = bool(getattr(args, "fp8", False))

    return fp8 and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.fp8.context",
    backend="megatron",
    phase="before_train",
    description="Override Megatron get_fp8_context to use Primus implementation when fp8 is enabled",
    condition=_is_fp8_can_patch,
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

    from primus.backends.megatron.core.fp8_utils import get_fp8_context

    log_rank_0("[Patch:megatron.fp8.context] Overriding get_fp8_context for fp8=True")

    # Patch get_fp8_context in all relevant modules
    modules_to_patch = [
        transformer_block,
        mamba_block,
        multi_token_prediction,
        fp8_utils,
    ]

    for module in modules_to_patch:
        module.get_fp8_context = get_fp8_context
        log_rank_0(f"[Patch:megatron.fp8.context]   Patched {module.__name__}.get_fp8_context")


@register_patch(
    "transformer_engine.pytorch.fp8",
    backend="megatron",
    phase="before_train",
    description="Override Transformer Engine's check_fp8_block_scaling_support to use Primus implementation",
    condition=_is_fp8_can_patch,
)
def patch_fp8_block_scaling_support(ctx: PatchContext):
    """
    Patch Transformer Engine's check_fp8_block_scaling_support to use Primus implementation.
    """
    from transformer_engine.pytorch import fp8

    from primus.backends.transformer_engine.pytorch.fp8 import (
        check_fp8_block_scaling_support,
    )

    log_rank_0("[Patch:transformer_engine.pytorch.fp8] Overriding check_fp8_block_scaling_support")

    fp8.check_fp8_block_scaling_support = check_fp8_block_scaling_support
