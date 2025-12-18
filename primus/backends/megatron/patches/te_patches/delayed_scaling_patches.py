###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Delayed Scaling Patches

Patches for customizing TEDelayedScaling behavior.
"""

from megatron.core.extensions import transformer_engine as te_ext
from megatron.core.extensions.transformer_engine import TEDelayedScaling

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _make_get_extra_te_kwargs_with_override(original_func, **overrides):
    """Create a wrapped version of _get_extra_te_kwargs with custom overrides."""

    def _wrapped(config):
        kwargs = original_func(config)
        kwargs.update(overrides)
        return kwargs

    return _wrapped


@register_patch(
    "megatron.te.delayed_scaling_reduce_amax",
    backend="megatron",
    phase="before_train",
    description="Disable reduce_amax in TEDelayedScaling for FP8 training",
)
def patch_te_delayed_scaling_reduce_amax(ctx: PatchContext):
    """
    Patch TEDelayedScaling to disable reduce_amax.

    This customizes the DelayedScaling recipe behavior by setting
    reduce_amax=False during initialization.
    """
    log_rank_0("[Patch:megatron.te.delayed_scaling_reduce_amax] Patching TEDelayedScaling...")

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
        "[Patch:megatron.te.delayed_scaling_reduce_amax]   Patched TEDelayedScaling.__init__ "
        "to disable reduce_amax"
    )
