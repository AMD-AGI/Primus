###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Patches

Patches for Transformer Engine (TE) integration with Megatron.
Handles FP8 configurations and memory optimizations.
"""

import inspect

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.distributed_logging import log_rank_0


# ============================================================================
# FP8 Weight Transpose Cache Patches
# ============================================================================


@register_patch(
    "megatron.te.disable_fp8_weight_transpose_cache",
    backend="megatron",
    phase="before_train",
    description="Disable FP8 weight transpose cache to reduce memory usage",
    condition=lambda ctx: ctx.extra.get("module_config", {}).params.get("no_fp8_weight_transpose_cache", False)
    if hasattr(ctx.extra.get("module_config", {}), "params")
    else False,
)
def disable_fp8_weight_transpose_cache(ctx: PatchContext):
    """
    Patch Transformer Engine classes to disable FP8 weight transpose cache.

    This patch modifies the initialization of TE layers to disable the
    keep_fp8_weight_transpose_cache parameter, which can save significant
    memory at the cost of some performance.

    Patches:
        - TELinear
        - TELayerNormColumnParallelLinear
        - TEDelayedScaling

    Config:
        no_fp8_weight_transpose_cache: true  # Enable this patch
    """
    try:
        import transformer_engine as te
        from megatron.core.extensions import transformer_engine as te_ext
    except ImportError:
        log_rank_0("[Patch:megatron.te.disable_fp8_weight_transpose_cache][SKIP] Transformer Engine not available")
        return

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

    # Patch TELinear
    def patch_TELinear():
        try:
            from megatron.core.extensions.transformer_engine import TELinear

            if not has_parameter(te.pytorch.Linear, "keep_fp8_weight_transpose_cache"):
                log_rank_0(
                    "[Patch:megatron.te.disable_fp8_weight_transpose_cache][WARN] "
                    "Transformer Engine version does not support keep_fp8_weight_transpose_cache"
                )
                return False

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
        except ImportError:
            return False

    # Patch TELayerNormColumnParallelLinear
    def patch_TELayerNormColumnParallelLinear():
        try:
            from megatron.core.extensions.transformer_engine import (
                TELayerNormColumnParallelLinear,
            )

            if not has_parameter(te.pytorch.LayerNormLinear, "keep_fp8_weight_transpose_cache"):
                return False

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
        except ImportError:
            return False

    # Patch TEDelayedScaling
    def patch_TEDelayedScaling():
        try:
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
        except ImportError:
            return False

    # Apply all patches
    if patch_TELinear():
        patches_applied.append("TELinear")
    if patch_TELayerNormColumnParallelLinear():
        patches_applied.append("TELayerNormColumnParallelLinear")
    if patch_TEDelayedScaling():
        patches_applied.append("TEDelayedScaling")

    if patches_applied:
        log_rank_0(
            f"[Patch:megatron.te.disable_fp8_weight_transpose_cache] "
            f"Patched {len(patches_applied)} TE classes: {', '.join(patches_applied)}"
        )
    else:
        log_rank_0("[Patch:megatron.te.disable_fp8_weight_transpose_cache][WARN] No TE classes patched")

