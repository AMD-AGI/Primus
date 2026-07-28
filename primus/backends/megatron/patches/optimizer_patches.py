###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron Optimizer Patches

This module contains patches that modify Megatron's optimizer creation to use
Primus-specific implementations when requested.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.optimizer.fsdp2_fp32_param",
    backend="megatron",
    phase="before_train",
    description="Patch get_megatron_optimizer for FSDP2 FP32 param optimizer (TorchTitan-style)",
    priority=50,
    condition=lambda ctx: (
        getattr(get_args(ctx), "use_fsdp2_fp32_param_optimizer", False)
        and getattr(get_args(ctx), "use_torch_fsdp2", False)
        and getattr(get_args(ctx), "bf16", False)
    ),
)
def patch_fsdp2_fp32_optimizer(ctx: PatchContext):
    """Patch Megatron to use FSDP2 FP32 optimizer.

    This optimizer uses FP32 parameters + FP32 optimizer states with FSDP2's
    MixedPrecisionPolicy for BF16 forward/backward. Eliminates the stale
    weights problem of BF16 optimizer states.
    """
    try:
        import megatron.core.optimizer as optimizer_module
        from megatron.training import training

        from primus.backends.megatron.core.optimizer.fsdp2_fp32_optimizer import (
            get_fsdp2_fp32_optimizer,
        )

        args = get_args(ctx)
        use_foreach = getattr(args, "optimizer_foreach", False)

        _MEGATRON_ONLY_KWARGS = {
            "config_overrides",
            "use_gloo_process_groups",
            "dump_param_to_param_group_map",
        }

        def patched_get_megatron_optimizer(config, model_chunks, *args, **kwargs):
            log_rank_0("=" * 80)
            log_rank_0("[Using FSDP2 FP32 Param Optimizer (TorchTitan-style)]")
            log_rank_0("  FP32 params + FP32 optimizer states")
            log_rank_0("  FSDP2 MixedPrecisionPolicy handles BF16 compute")
            log_rank_0("  DTensor-native gradient clipping")
            log_rank_0(f"  AdamW mode: {'foreach' if use_foreach else 'fused'}")
            log_rank_0("=" * 80)

            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in _MEGATRON_ONLY_KWARGS}

            return get_fsdp2_fp32_optimizer(
                config=config,
                model_chunks=model_chunks,
                use_foreach=use_foreach,
                **filtered_kwargs,
            )

        patched_count = 0
        if hasattr(training, "get_megatron_optimizer"):
            training.get_megatron_optimizer = patched_get_megatron_optimizer
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_fp32_param] " "Patched training.get_megatron_optimizer"
            )
            patched_count += 1

        if hasattr(optimizer_module, "get_megatron_optimizer"):
            optimizer_module.get_megatron_optimizer = patched_get_megatron_optimizer
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_fp32_param] "
                "Patched optimizer_module.get_megatron_optimizer"
            )
            patched_count += 1

        if patched_count == 0:
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_fp32_param] "
                "WARNING: get_megatron_optimizer not found in either location!"
            )
        else:
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_fp32_param] "
                f"Patched get_megatron_optimizer in {patched_count} location(s) "
                "to use FSDP2FP32Optimizer"
            )

    except Exception as e:
        log_rank_0(
            f"[Patch:megatron.optimizer.fsdp2_fp32_param] "
            f"WARNING: Failed to patch get_megatron_optimizer: {type(e).__name__}: {e}"
        )
        import traceback

        log_rank_0(f"Traceback: {traceback.format_exc()}")


@register_patch(
    "megatron.optimizer.fsdp2_bf16_master_weight",
    backend="megatron",
    phase="before_train",
    description="Patch get_megatron_optimizer for FSDP2 BF16 master weight optimizer",
    priority=50,
    condition=lambda ctx: (
        getattr(get_args(ctx), "use_fsdp2_bf16_master_weight_optimizer", False)
        and getattr(get_args(ctx), "use_torch_fsdp2", False)
        and getattr(get_args(ctx), "bf16", False)
    ),
)
def patch_fsdp2_bf16_master_weight_optimizer(ctx: PatchContext):
    """Patch Megatron to use FSDP2 BF16 master weight optimizer.

    This optimizer keeps model parameters in BF16 (no FP32->BF16 cast in
    forward all-gather) and maintains FP32 master copies for optimizer
    precision, matching Megatron's Float16OptimizerWithFloat16Params pattern.
    """
    try:
        import megatron.core.optimizer as optimizer_module
        from megatron.training import training

        from primus.backends.megatron.core.optimizer.fsdp2_bf16_master_weight_optimizer import (
            get_fsdp2_bf16_master_weight_optimizer,
        )

        args = get_args(ctx)
        use_foreach = getattr(args, "optimizer_foreach", True)

        _MEGATRON_ONLY_KWARGS = {
            "config_overrides",
            "use_gloo_process_groups",
            "dump_param_to_param_group_map",
        }

        def patched_get_megatron_optimizer(config, model_chunks, *args, **kwargs):
            log_rank_0("=" * 80)
            log_rank_0("[Using FSDP2 BF16 Master Weight Optimizer]")
            log_rank_0("  BF16 model params + FP32 master weights")
            log_rank_0("  No FP32->BF16 cast in forward all-gather")
            log_rank_0("  DTensor-native gradient clipping")
            log_rank_0(f"  Foreach batching: {use_foreach}")
            log_rank_0("=" * 80)

            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in _MEGATRON_ONLY_KWARGS}

            return get_fsdp2_bf16_master_weight_optimizer(
                config=config,
                model_chunks=model_chunks,
                use_foreach=use_foreach,
                **filtered_kwargs,
            )

        patched_count = 0
        if hasattr(training, "get_megatron_optimizer"):
            training.get_megatron_optimizer = patched_get_megatron_optimizer
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_bf16_master_weight] "
                "Patched training.get_megatron_optimizer"
            )
            patched_count += 1

        if hasattr(optimizer_module, "get_megatron_optimizer"):
            optimizer_module.get_megatron_optimizer = patched_get_megatron_optimizer
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_bf16_master_weight] "
                "Patched optimizer_module.get_megatron_optimizer"
            )
            patched_count += 1

        if patched_count == 0:
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_bf16_master_weight] "
                "WARNING: get_megatron_optimizer not found in either location!"
            )
        else:
            log_rank_0(
                "[Patch:megatron.optimizer.fsdp2_bf16_master_weight] "
                f"Patched get_megatron_optimizer in {patched_count} location(s) "
                "to use FSDP2BF16MasterWeightOptimizer"
            )

    except Exception as e:
        log_rank_0(
            f"[Patch:megatron.optimizer.fsdp2_bf16_master_weight] "
            f"WARNING: Failed to patch get_megatron_optimizer: {type(e).__name__}: {e}"
        )
        import traceback

        log_rank_0(f"Traceback: {traceback.format_exc()}")


@register_patch(
    "megatron.optimizer.precision_aware_fp8_tensorwise",
    backend="megatron",
    phase="before_train",
    description=(
        "Override precision-aware optimizer flag for local spec + tensorwise FP8, "
        "enabling BF16 decoupled_grad path (no BF16-to-FP32 gradient cast)."
    ),
    priority=35,
    condition=lambda ctx: (
        getattr(get_args(ctx), "use_precision_aware_optimizer", False)
        and getattr(get_args(ctx), "fp8_recipe", None) == "tensorwise"
        and getattr(get_args(ctx), "transformer_impl", None) == "local"
    ),
)
def patch_precision_aware_for_tensorwise(ctx: PatchContext):
    """Enable precision-aware optimizer's decoupled_grad path for local spec + tensorwise FP8.

    With local spec, parameters are BF16 (not Float8Tensor) -- FP8 quantization
    happens inside per-module autograd Functions, transparent to the optimizer.
    The upstream condition incorrectly excludes tensorwise FP8 because it was
    designed for TE spec where Float8Tensor storage requires special handling.
    """
    try:
        from megatron.core.optimizer.optimizer_config import OptimizerConfig

        original_post_init = OptimizerConfig.__post_init__

        def patched_post_init(self):
            original_post_init(self)
            if self.use_precision_aware_optimizer and self.fp8_recipe == "tensorwise":
                self.use_precision_aware_optimizer_no_fp8_or_ds_fp8 = True

        OptimizerConfig.__post_init__ = patched_post_init

        log_rank_0(
            "[Patch:megatron.optimizer.precision_aware_fp8_tensorwise] "
            "Patched OptimizerConfig.__post_init__ to enable "
            "decoupled_grad path for local spec + tensorwise FP8"
        )

    except Exception as e:
        log_rank_0(
            f"[Patch:megatron.optimizer.precision_aware_fp8_tensorwise] "
            f"WARNING: Failed to patch OptimizerConfig: {type(e).__name__}: {e}"
        )
        import traceback

        log_rank_0(f"Traceback: {traceback.format_exc()}")
