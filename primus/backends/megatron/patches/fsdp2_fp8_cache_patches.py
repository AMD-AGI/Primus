###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron train_step patch for FP8 all-gather cache refresh.

The FP8 all-gather cache (populated by precompute_fp8_scales_for_fsdp) must be
refreshed after every optimizer.step() so that the cached FP8 data reflects the
updated weights.  Without this patch the cache is only populated once during
FSDP setup and goes stale after the first iteration.

This patch wraps Megatron's standalone train_step() to call
precompute_fp8_scales_for_fsdp(model[0]) after the original train_step returns.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _needs_fp8_cache_update(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return args is not None and getattr(args, "use_fsdp2_fp8_all_gather", False)


@register_patch(
    "megatron.training.train_step_fp8_cache_update",
    backend="megatron",
    phase="before_train",
    description="Patch train_step to refresh FP8 all-gather cache after optimizer.step()",
    condition=_needs_fp8_cache_update,
    priority=45,
)
def patch_train_step_fp8_cache(ctx: PatchContext):
    import megatron.training.training as megatron_training

    from primus.backends.megatron.core.distributed.fsdp2_fp8_all_gather import (
        precompute_fp8_scales_for_fsdp,
    )
    from primus.backends.megatron.patches._patch_guard import is_patched, mark_patched

    _PATCH_KEY = "megatron.training.train_step_fp8_cache_update"
    if is_patched(megatron_training, _PATCH_KEY):
        log_rank_0("[Patch:train_step_fp8_cache_update] Already applied; skipping re-wrap.")
        return

    args = get_args(ctx)
    cache_data = getattr(args, "fp8_precompute_data_cache", True)
    use_cpp = getattr(args, "use_cpp_fp8_quantize", False)
    stochastic_rounding = getattr(args, "fp8_all_gather_stochastic_rounding", False)

    _original_train_step = megatron_training.train_step

    def _patched_train_step(
        forward_step_func,
        data_iterator,
        model,
        optimizer,
        opt_param_scheduler,
        config,
        forward_backward_func,
        iteration=None,
    ):
        result = _original_train_step(
            forward_step_func,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            config,
            forward_backward_func,
            iteration=iteration,
        )
        if _refresh_scales:
            precompute_fp8_scales_for_fsdp(
                model[0],
                cache_data=cache_data,
                use_cpp_quantize=use_cpp,
                stochastic_rounding=stochastic_rounding,
            )
        return result

    # When using the C++ quantize kernel without data caching, scales were
    # already set once during FSDP setup and the C++ kernel uses them
    # directly -- no per-step refresh needed (matches run_35 behavior).
    _refresh_scales = cache_data or not use_cpp

    megatron_training.train_step = _patched_train_step
    mark_patched(megatron_training, _PATCH_KEY)
    log_rank_0(
        "[Patch:train_step_fp8_cache_update] "
        f"Patched train_step to refresh FP8 all-gather cache after optimizer.step() "
        f"(cache_data={cache_data}, use_cpp_quantize={use_cpp}, "
        f"stochastic_rounding={stochastic_rounding}, "
        f"refresh_scales={_refresh_scales})"
    )
