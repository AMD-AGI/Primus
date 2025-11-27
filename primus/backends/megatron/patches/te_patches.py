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
    condition=lambda ctx: (
        ctx.extra.get("module_config", {}).params.get("no_fp8_weight_transpose_cache", False)
        if hasattr(ctx.extra.get("module_config", {}), "params")
        else False
    ),
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
        log_rank_0(
            "[Patch:megatron.te.disable_fp8_weight_transpose_cache][SKIP] Transformer Engine not available"
        )
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


# ============================================================================
# FP8 Context Patches
# ============================================================================


@register_patch(
    "megatron.fp8.get_fp8_context",
    backend="megatron",
    phase="before_train",
    description="Override Megatron get_fp8_context to use Primus implementation when fp8 is enabled",
)
def patch_fp8_context(ctx: PatchContext):
    """
    Patch Megatron's get_fp8_context functions to use Primus implementation.

    Behavior (moved from MegatronTrainer.patch_fp8_context):
        - When module_config.fp8 is True, replace:
            * megatron.core.transformer.transformer_block.get_fp8_context
            * megatron.core.ssm.mamba_block.get_fp8_context
            * megatron.core.transformer.multi_token_prediction.get_fp8_context
            * megatron.core.fp8_utils.get_fp8_context
          with Primus's ROCm-friendly get_fp8_context.
    """
    module_config = ctx.extra.get("module_config")
    if module_config is None or not getattr(module_config, "fp8", False):
        return

    try:
        from megatron.core import fp8_utils
        from megatron.core.ssm import mamba_block
        from megatron.core.transformer import multi_token_prediction, transformer_block

        from primus.backends.megatron.patches.core.fp8_utils import get_fp8_context

        log_rank_0("[Patch:megatron.fp8.get_fp8_context] Overriding get_fp8_context for fp8=True")

        transformer_block.get_fp8_context = get_fp8_context
        mamba_block.get_fp8_context = get_fp8_context
        multi_token_prediction.get_fp8_context = get_fp8_context
        fp8_utils.get_fp8_context = get_fp8_context

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.fp8.get_fp8_context][SKIP] Import failed: {e}")
    except AttributeError as e:
        log_rank_0(f"[Patch:megatron.fp8.get_fp8_context][WARN] Attribute error: {e}")


# ============================================================================
# Primus Turbo Backend Patches
# ============================================================================


@register_patch(
    "megatron.te.primus_turbo_backend",
    backend="megatron",
    phase="before_train",
    description=(
        "Configure Transformer Engine to use PrimusTurbo backend when "
        "enable_primus_turbo is set and tensor_model_parallel_size == 1."
    ),
)
def patch_primus_turbo_backend(ctx: PatchContext):
    """
    Patch Transformer Engine integration to use PrimusTurbo backend.

    Behavior (moved from MegatronTrainer.patch_pt_replace_te):
        - If primus_turbo is installed AND tensor_model_parallel_size == 1 AND
          args.enable_primus_turbo is True:
            * Patch TESpecProvider to PrimusTurboSpecProvider.
            * Optionally replace GPT output layer, MoE dispatcher, and RMSNorm
              with PrimusTurbo implementations based on args flags.
        - Otherwise, fallback to TE backend and log the decision.
    """
    args = ctx.extra.get("args")
    if args is None:
        return

    # Check if primus_turbo package is available
    if importlib.util.find_spec("primus_turbo") is None:
        log_rank_0("[Patch:megatron.te.primus_turbo_backend] primus_turbo not found, use TE backend...")
        return

    tp_size = getattr(args, "tensor_model_parallel_size", 1)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))

    if tp_size != 1:
        if enable_primus_turbo:
            log_rank_0(
                "[Patch:megatron.te.primus_turbo_backend] "
                "Primus Turbo does not support TP; using TE backend instead..."
            )
        else:
            log_rank_0("[Patch:megatron.te.primus_turbo_backend] TP > 1; using TE backend...")
        return

    if not enable_primus_turbo:
        log_rank_0("[Patch:megatron.te.primus_turbo_backend] enable_primus_turbo=False; using TE backend...")
        return

    # At this point we know:
    #   - primus_turbo is installed
    #   - tensor_model_parallel_size == 1
    #   - enable_primus_turbo == True
    try:
        import megatron.core.extensions as meg_ext
        from megatron.core.extensions import (
            transformer_engine as transformer_engine_spec_provider,
        )
        from megatron.core.models.gpt import (
            gpt_layer_specs,
            gpt_model,
            moe_module_specs,
        )
        from megatron.core.transformer import multi_token_prediction
        from megatron.core.transformer.moe import moe_layer, token_dispatcher

        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboColumnParallelLinearTorch,
            PrimusTurboDeepEPTokenDispatcher,
            PrimusTurboRMSNorm,
        )
        from primus.backends.megatron.core.extensions.transformer_engine_spec_provider import (
            PrimusTurboSpecProvider,
        )

        log_rank_0(
            "[Patch:megatron.te.primus_turbo_backend] "
            "Patch TESpecProvider to PrimusTurboSpecProvider; PrimusTurbo backend enabled"
        )

        assert (
            meg_ext.transformer_engine.HAVE_TE
        ), "PrimusTurboSpecProvider patch failed, can't find transformer_engine"

        # Replace TESpecProvider in all relevant locations
        transformer_engine_spec_provider.TESpecProvider = PrimusTurboSpecProvider
        gpt_layer_specs.TESpecProvider = PrimusTurboSpecProvider
        moe_module_specs.TESpecProvider = PrimusTurboSpecProvider
        multi_token_prediction.TESpecProvider = PrimusTurboSpecProvider

        # Optional: patch GPT output layer
        if getattr(args, "use_turbo_parallel_linear", False):
            gpt_model.tensor_parallel.ColumnParallelLinear = PrimusTurboColumnParallelLinearTorch

        # Optional: patch MoE dispatcher
        if getattr(args, "use_turbo_deepep", False):
            # use PrimusTurboDeepEPTokenDispatcher will auto-enable:
            #   moe_enable_deepep=True, moe_token_dispatcher_type='flex'
            args.moe_enable_deepep = True
            args.moe_token_dispatcher_type = "flex"
            token_dispatcher.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher
            moe_layer.MoEFlexTokenDispatcher = PrimusTurboDeepEPTokenDispatcher

        # Optional: patch RMSNorm
        if getattr(args, "use_turbo_rms_norm", False):
            import transformer_engine as te

            te.pytorch.RMSNorm = PrimusTurboRMSNorm

        log_rank_0("[Patch:megatron.te.primus_turbo_backend] Using PrimusTurbo backend (PT)")

    except ImportError as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][SKIP] Import failed: {e}")
    except AssertionError as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][WARN] Assertion failed: {e}")
    except Exception as e:
        log_rank_0(f"[Patch:megatron.te.primus_turbo_backend][ERROR] Unexpected error: {e}")
