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


def _no_fp8_weight_transpose_cache_enabled(ctx: PatchContext) -> bool:
    """Check if no_fp8_weight_transpose_cache is enabled in module_config."""
    return getattr(get_args(ctx), "no_fp8_weight_transpose_cache", False)


@register_patch(
    "megatron.patch.get_extra_te_kwargs",
    backend="megatron",
    phase="before_train",
    description="Override _get_extra_te_kwargs to customize TE layer initialization parameters",
    condition=_no_fp8_weight_transpose_cache_enabled,
)
def patch_get_extra_te_kwargs(ctx: PatchContext):
    """
    Patch Transformer Engine _get_extra_te_kwargs for custom layer initialization.

    This patch wraps Megatron's _get_extra_te_kwargs function to inject custom
    parameters during TE layer initialization. Currently supports:

    1. Disabling FP8 weight transpose cache (keep_fp8_weight_transpose_cache=False)
       to reduce memory usage at the cost of some performance
    2. Disabling reduce_amax for DelayedScaling recipe

    Patched Classes:
        - TELinear: Disables keep_fp8_weight_transpose_cache
        - TELayerNormColumnParallelLinear: Disables keep_fp8_weight_transpose_cache
        - TEDelayedScaling: Disables reduce_amax

    Config:
        no_fp8_weight_transpose_cache: true  # Enable FP8 cache disabling
    """
    import transformer_engine as te
    from megatron.core.extensions import transformer_engine as te_ext

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs

    # Create a wrapped version of _get_extra_te_kwargs with custom overrides
    def make_get_extra_te_kwargs_with_override(**overrides):
        def _wrapped(config):
            kwargs = original_get_extra_te_kwargs(config)
            kwargs.update(overrides)
            return kwargs

        return _wrapped

    def has_parameter(cls, param):
        """Check if a class __init__ has a specific parameter."""
        try:
            return param in inspect.signature(cls.__init__).parameters
        except Exception:
            return False

    patches_applied = []

    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)

    # Patch TELinear
    def patch_TELinear():
        from megatron.core.extensions.transformer_engine import TELinear

        if not getattr(params, "no_fp8_weight_transpose_cache", False):
            log_rank_0(
                "[Patch:megatron.patch.get_extra_te_kwargs] no_fp8_weight_transpose_cache is not enabled"
            )
            return False

        assert has_parameter(
            te.pytorch.Linear, "keep_fp8_weight_transpose_cache"
        ), "Current Transformer-Engine not support this feature"

        orig_init = TELinear.__init__

        def new_init(self, *args, **kwargs):
            # Temporarily override the TE kwargs with our custom flag
            te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
                keep_fp8_weight_transpose_cache=False
            )
            try:
                orig_init(self, *args, **kwargs)
            finally:
                # Always restore the original function after init
                te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

        TELinear.__init__ = new_init
        return True

    # Patch TELayerNormColumnParallelLinear
    def patch_TELayerNormColumnParallelLinear():
        from megatron.core.extensions.transformer_engine import (
            TELayerNormColumnParallelLinear,
        )

        if not getattr(params, "no_fp8_weight_transpose_cache", False):
            log_rank_0(
                "[Patch:megatron.patch.get_extra_te_kwargs] no_fp8_weight_transpose_cache is not enabled"
            )
            return False

        assert has_parameter(
            te.pytorch.LayerNormLinear, "keep_fp8_weight_transpose_cache"
        ), "Current Transformer-Engine not support this feature"

        orig_init = TELayerNormColumnParallelLinear.__init__

        def new_init(self, *args, **kwargs):
            # Temporarily override the TE kwargs with our custom flag
            te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
                keep_fp8_weight_transpose_cache=False
            )
            try:
                orig_init(self, *args, **kwargs)
            finally:
                # Always restore the original function after init
                te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

        TELayerNormColumnParallelLinear.__init__ = new_init
        return True

    # Patch TEDelayedScaling
    def patch_TEDelayedScaling():
        from megatron.core.extensions.transformer_engine import TEDelayedScaling

        if not has_parameter(te.common.recipe.DelayedScaling, "reduce_amax"):
            return False

        orig_init = TEDelayedScaling.__init__

        def new_init(self, *args, **kwargs):
            # Temporarily override the TE kwargs with our custom flag
            te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(reduce_amax=False)
            try:
                orig_init(self, *args, **kwargs)
            finally:
                # Always restore the original function after init
                te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

        TEDelayedScaling.__init__ = new_init
        return True

    # Apply all patches
    if patch_TELinear():
        patches_applied.append("TELinear")
    if patch_TELayerNormColumnParallelLinear():
        patches_applied.append("TELayerNormColumnParallelLinear")
    if patch_TEDelayedScaling():
        patches_applied.append("TEDelayedScaling")

    if patches_applied:
        log_rank_0(
            f"[Patch:megatron.patch.get_extra_te_kwargs] "
            f"Patched {len(patches_applied)} TE classes: {', '.join(patches_applied)}"
        )
    else:
        log_rank_0("[Patch:megatron.patch.get_extra_te_kwargs] No TE classes patched")
