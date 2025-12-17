###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine get_extra_te_kwargs Patches

This module contains patches that override Megatron's _get_extra_te_kwargs
to customize Transformer Engine layer initialization parameters.
"""

import inspect

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# =============================================================================
# Helper Functions
# =============================================================================


def _make_get_extra_te_kwargs_with_override(original_func, **overrides):
    """Create a wrapped version of _get_extra_te_kwargs with custom overrides."""

    def _wrapped(config):
        kwargs = original_func(config)
        kwargs.update(overrides)
        return kwargs

    return _wrapped


def _has_parameter(cls, param):
    """Check if a class __init__ has a specific parameter."""
    try:
        return param in inspect.signature(cls.__init__).parameters
    except Exception:
        return False


# =============================================================================
# Patches
# =============================================================================


@register_patch(
    "megatron.te.te_linear_fp8_cache",
    backend="megatron",
    phase="before_train",
    description="Disable FP8 weight transpose cache in TELinear to reduce memory usage",
    condition=lambda ctx: getattr(get_args(ctx), "no_fp8_weight_transpose_cache", False),
)
def patch_te_linear_fp8_cache(ctx: PatchContext):
    """
    Patch TELinear to disable FP8 weight transpose cache.

    This reduces memory usage at the cost of some performance by setting
    keep_fp8_weight_transpose_cache=False during layer initialization.

    Config:
        no_fp8_weight_transpose_cache: true  # Enable FP8 cache disabling
    """
    import transformer_engine as te
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import TELinear

    assert _has_parameter(
        te.pytorch.Linear, "keep_fp8_weight_transpose_cache"
    ), "Current Transformer-Engine does not support keep_fp8_weight_transpose_cache parameter"

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs
    orig_init = TELinear.__init__

    def new_init(self, *args, **kwargs):
        # Temporarily override the TE kwargs with our custom flag
        te_ext._get_extra_te_kwargs = _make_get_extra_te_kwargs_with_override(
            original_get_extra_te_kwargs, keep_fp8_weight_transpose_cache=False
        )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            # Always restore the original function after init
            te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

    TELinear.__init__ = new_init
    log_rank_0(
        "[Patch:megatron.te.te_linear_fp8_cache] Patched TELinear.__init__ "
        "to disable keep_fp8_weight_transpose_cache"
    )


@register_patch(
    "megatron.te.te_layernorm_linear_fp8_cache",
    backend="megatron",
    phase="before_train",
    description="Disable FP8 weight transpose cache in TELayerNormColumnParallelLinear to reduce memory usage",
    condition=lambda ctx: getattr(get_args(ctx), "no_fp8_weight_transpose_cache", False),
)
def patch_te_layernorm_linear_fp8_cache(ctx: PatchContext):
    """
    Patch TELayerNormColumnParallelLinear to disable FP8 weight transpose cache.

    This reduces memory usage at the cost of some performance by setting
    keep_fp8_weight_transpose_cache=False during layer initialization.

    Config:
        no_fp8_weight_transpose_cache: true  # Enable FP8 cache disabling
    """
    import transformer_engine as te
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import (
        TELayerNormColumnParallelLinear,
    )

    assert _has_parameter(
        te.pytorch.LayerNormLinear, "keep_fp8_weight_transpose_cache"
    ), "Current Transformer-Engine does not support keep_fp8_weight_transpose_cache parameter"

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs
    orig_init = TELayerNormColumnParallelLinear.__init__

    def new_init(self, *args, **kwargs):
        # Temporarily override the TE kwargs with our custom flag
        te_ext._get_extra_te_kwargs = _make_get_extra_te_kwargs_with_override(
            original_get_extra_te_kwargs, keep_fp8_weight_transpose_cache=False
        )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            # Always restore the original function after init
            te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

    TELayerNormColumnParallelLinear.__init__ = new_init
    log_rank_0(
        "[Patch:megatron.te.te_layernorm_linear_fp8_cache] Patched TELayerNormColumnParallelLinear.__init__ "
        "to disable keep_fp8_weight_transpose_cache"
    )


@register_patch(
    "megatron.te.te_delayed_scaling_reduce_amax",
    backend="megatron",
    phase="before_train",
    description="Disable reduce_amax in TEDelayedScaling for FP8 training",
)
def patch_te_delayed_scaling_reduce_amax(ctx: PatchContext):
    """
    Patch TEDelayedScaling to disable reduce_amax.

    This customizes the DelayedScaling recipe behavior by setting
    reduce_amax=False during initialization.

    Note: This patch is always applied if the TE version supports reduce_amax parameter.
    """
    import transformer_engine as te
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    if not _has_parameter(te.common.recipe.DelayedScaling, "reduce_amax"):
        log_rank_0(
            "[Patch:megatron.te.te_delayed_scaling_reduce_amax][SKIP] "
            "Current Transformer-Engine does not support reduce_amax parameter"
        )
        return

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs
    orig_init = TEDelayedScaling.__init__

    def new_init(self, *args, **kwargs):
        # Temporarily override the TE kwargs with our custom flag
        te_ext._get_extra_te_kwargs = _make_get_extra_te_kwargs_with_override(
            original_get_extra_te_kwargs, reduce_amax=False
        )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            # Always restore the original function after init
            te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

    TEDelayedScaling.__init__ = new_init
    log_rank_0(
        "[Patch:megatron.te.te_delayed_scaling_reduce_amax] Patched TEDelayedScaling.__init__ "
        "to disable reduce_amax"
    )
