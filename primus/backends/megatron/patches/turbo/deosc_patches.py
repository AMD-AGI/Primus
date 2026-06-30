###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Patch ``get_megatron_optimizer`` to install MXFP4 weight de-oscillation.

When MXFP4 training is driven through the Primus-Turbo FP4 autocast path and
``weight_deosc`` is enabled, this patch wraps every
``DistributedOptimizer.step_with_ready_grads`` so the de-oscillation detector
runs right after the optimizer updates the fp32 master and all-gathers the bf16
model weight. See ``primus.backends.megatron.core.optimizer.weight_deosc`` for
the algorithm and rationale.
"""

from primus.backends.megatron.patches.turbo.utils import is_primus_turbo_can_patch
from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_weight_deosc_can_patch(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    fp4 = bool(getattr(args, "fp4", False))
    use_turbo_fp4 = bool(getattr(args, "use_turbo_fp4_autocast", False))
    deosc = bool(getattr(args, "weight_deosc", False))
    return fp4 and use_turbo_fp4 and deosc and is_primus_turbo_can_patch(ctx)


@register_patch(
    "megatron.turbo.weight_deosc",
    backend="megatron",
    phase="before_train",
    description="Install MXFP4 weight de-oscillation on the distributed optimizer.",
    condition=_is_weight_deosc_can_patch,
)
def patch_get_megatron_optimizer_weight_deosc(ctx: PatchContext) -> None:
    try:
        import megatron.training.training as training_module
    except ImportError as e:
        log_rank_0(f"[Patch:megatron.turbo.weight_deosc] Skip (Megatron not available): {e}")
        return

    from primus.backends.megatron.core.optimizer.weight_deosc import (
        WeightDeOscConfig,
        install_weight_deosc,
    )

    args = get_args(ctx)
    config = WeightDeOscConfig(
        enable=True,
        period=int(getattr(args, "weight_deosc_period", 200)),
        ratio_threshold=float(getattr(args, "weight_deosc_ratio", 4.0)),
        start_step=int(getattr(args, "weight_deosc_start_step", 0)),
        log_freq=int(getattr(args, "weight_deosc_log_freq", 0)),
    )

    original_get_megatron_optimizer = training_module.get_megatron_optimizer

    if getattr(original_get_megatron_optimizer, "_primus_weight_deosc_wrapper", False):
        return

    def _patched_get_megatron_optimizer(*func_args, **func_kwargs):
        optimizer = original_get_megatron_optimizer(*func_args, **func_kwargs)
        try:
            install_weight_deosc(optimizer, config)
        except Exception as exc:  # never block optimizer construction
            log_rank_0(f"[Patch:megatron.turbo.weight_deosc] install failed, skipped: {exc}")
        return optimizer

    setattr(_patched_get_megatron_optimizer, "_primus_weight_deosc_wrapper", True)
    training_module.get_megatron_optimizer = _patched_get_megatron_optimizer
    log_rank_0(
        "[Patch:megatron.turbo.weight_deosc] Patched get_megatron_optimizer to install "
        f"MXFP4 weight de-oscillation (period={config.period}, ratio={config.ratio_threshold}, "
        f"start_step={config.start_step})."
    )
