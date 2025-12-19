###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Linear FP8 Cache Patches

Patches for disabling FP8 weight transpose cache in TELinear layers.
"""


from primus.backends.megatron.patches.te_patches.utils import (
    make_get_extra_te_kwargs_with_override,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.te.linear_fp8_cache",
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
    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import TELinear

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs
    orig_init = TELinear.__init__

    def new_init(self, *args, **kwargs):
        """Wrapper around TELinear.__init__ that temporarily disables the FP8
        weight transpose cache by overriding TE kwargs during initialization."""
        # Temporarily override the TE kwargs with our custom flag
        te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
            original_get_extra_te_kwargs, keep_fp8_weight_transpose_cache=False
        )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            # Always restore the original function after init
            te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

    TELinear.__init__ = new_init
    log_rank_0(
        "[Patch:megatron.te.linear_fp8_cache]   Patched TELinear.__init__ "
        "to disable keep_fp8_weight_transpose_cache"
    )
