###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine Delayed Scaling Patches

Patches for customizing TEDelayedScaling behavior.
"""

from primus.backends.megatron.patches.te_patches.utils import (
    is_te_min_version,
    make_get_extra_te_kwargs_with_override,
)
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.te.delayed_scaling_reduce_amax",
    backend="megatron",
    phase="before_train",
    description="Set reduce_amax in TEDelayedScaling based on TP/SP requirements",
    condition=lambda ctx: (
        getattr(get_args(ctx), "fp8", False) and getattr(get_args(ctx), "fp8_recipe", "delayed") == "delayed"
    ),
)
def patch_te_delayed_scaling_reduce_amax(ctx: PatchContext):
    """
    Patch TEDelayedScaling to choose reduce_amax based on parallelism mode.

        - reduce_amax=False was introduced as an FP8 optimization to avoid
            cross-rank amax synchronization in delayed scaling.
        - FP8 + sequence parallelism requires amax reduction across the tensor
            parallel group so all ranks use compatible scaling factors.

    """
    args = get_args(ctx)
    tp_size = getattr(args, "tensor_model_parallel_size", 1) or 1
    sequence_parallel = bool(getattr(args, "sequence_parallel", False))
    reduce_amax = bool(sequence_parallel and tp_size > 1)

    from megatron.core.extensions import transformer_engine as te_ext
    from megatron.core.extensions.transformer_engine import TEDelayedScaling

    # Save the original _get_extra_te_kwargs function
    original_get_extra_te_kwargs = te_ext._get_extra_te_kwargs
    orig_init = TEDelayedScaling.__init__

    def new_init(self, *args, **kwargs):
        """Wrapper around TEDelayedScaling.__init__ that temporarily overrides
        Transformer Engine kwargs to set reduce_amax during initialization."""
        # Temporarily override the TE kwargs with our custom flag
        te_ext._get_extra_te_kwargs = make_get_extra_te_kwargs_with_override(
            original_get_extra_te_kwargs, reduce_amax=reduce_amax
        )
        try:
            orig_init(self, *args, **kwargs)
        finally:
            # Always restore the original function after init
            te_ext._get_extra_te_kwargs = original_get_extra_te_kwargs

    TEDelayedScaling.__init__ = new_init
    log_rank_0(
        "[Patch:megatron.te.delayed_scaling_reduce_amax]   Patched TEDelayedScaling.__init__ "
        f"to set reduce_amax={reduce_amax} "
    )


@register_patch(
    "megatron.te.delayed_scaling_save_original_input",
    backend="megatron",
    phase="before_train",
    description="Fix DelayedScaling incompatibility with save_original_input in TE >= 2.6.0",
    condition=lambda ctx: (
        getattr(get_args(ctx), "fp8", False)
        and is_te_min_version("2.6.0dev0")
        and getattr(get_args(ctx), "fp8_recipe", "delayed") == "delayed"
    ),
)
def patch_te_delayed_scaling_save_original_input(ctx: PatchContext):
    """
    Patch SharedExpertMLP to skip set_save_original_input when using DelayedScaling.

    Background:
        - In TE >= 2.6.0, SharedExpertMLP automatically calls set_save_original_input()
          on linear_fc1 for memory optimization
        - DelayedScaling recipe requires saving quantized tensors for statistics
        - save_original_input=True saves original unquantized inputs
        - These two are incompatible and cause AssertionError

    Solution:
        When using delayed scaling with TE >= 2.6.0, skip the set_save_original_input
        call to avoid the conflict.
    """
    from megatron.core.transformer.moe import shared_experts

    original_init = shared_experts.SharedExpertMLP.__init__

    def patched_init(self, config, submodules=None, gate=False, pg_collection=None):
        original_init(self, config, submodules, gate, pg_collection)

        if hasattr(self.linear_fc1, "save_original_input"):
            if config.fp8 and config.fp8_recipe == "delayed":
                self.linear_fc1.save_original_input = False
                log_rank_0(
                    "[Patch:megatron.te.delayed_scaling_save_original_input] Disabled save_original_input on SharedExpertMLP.linear_fc1 for compatibility with DelayedScaling"
                )

    shared_experts.SharedExpertMLP.__init__ = patched_init
    log_rank_0(
        "[Patch:megatron.te.delayed_scaling_save_original_input] Patched SharedExpertMLP.__init__ to fix DelayedScaling + save_original_input conflict"
    )
