###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron torch.compile Patches

This module contains patches that modify Megatron's setup_model_and_optimizer
to apply torch.compile after model setup.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.training.torch_compile",
    backend="megatron",
    phase="before_train",
    description="Patch setup_model_and_optimizer to apply torch.compile after model setup",
    priority=60,  # Higher than optimizer patch to wrap last
    condition=lambda ctx: (
        getattr(get_args(ctx), "torch_compile", None) is not None
        and getattr(getattr(get_args(ctx), "torch_compile", None), "enable", False)
    )
    or getattr(get_args(ctx), "enable_torch_compile", False),
)
def patch_setup_model_and_optimizer_for_torch_compile(ctx: PatchContext):
    """
    Patch Megatron's setup_model_and_optimizer to apply torch.compile after model setup.

    Behavior:
        - Wraps setup_model_and_optimizer() to call apply_torch_compile_if_enabled()
          after model is created and wrapped
        - If compilation fails, exception propagates (fails entire setup)
        - Works for both FSDP2 and non-FSDP2 paths
    """
    try:
        from megatron.training import training

        from primus.backends.megatron.core.utils import (
            apply_torch_compile_if_enabled,
            apply_torch_compile_to_optimizer_if_enabled,
        )

        # Save original function
        original_setup_model_and_optimizer = training.setup_model_and_optimizer

        def patched_setup_model_and_optimizer(*args, **kwargs):
            """Patched setup_model_and_optimizer that applies torch.compile after model setup."""
            from megatron.training import get_args

            result = original_setup_model_and_optimizer(*args, **kwargs)

            # Extract model from result tuple
            model, optimizer, opt_param_scheduler = result

            # Apply torch.compile (raises exception on failure)
            megatron_args = get_args()
            apply_torch_compile_if_enabled(model, megatron_args)
            apply_torch_compile_to_optimizer_if_enabled(optimizer, megatron_args)

            # Return original result
            return result

        # Apply the patch
        training.setup_model_and_optimizer = patched_setup_model_and_optimizer
        log_rank_0(
            "[Patch:megatron.training.torch_compile] "
            "Patched setup_model_and_optimizer to apply torch.compile after model setup"
        )

    except Exception as e:
        log_rank_0(
            f"[Patch:megatron.training.torch_compile] "
            f"WARNING: Failed to patch setup_model_and_optimizer: {type(e).__name__}: {e}"
        )
        import traceback

        log_rank_0(f"Traceback: {traceback.format_exc()}")
