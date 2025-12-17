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

import functools

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

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


# ============================================================================
# TP Overlap Patches
# ============================================================================


@register_patch(
    "megatron.patch.tp_te_overlap",
    backend="megatron",
    phase="before_train",
    description="Enable Transformer Engine tensor-parallel communication overlap with Primus patches",
)
def patch_tp_te_overlap(ctx: PatchContext):
    """
    Patch Transformer Engine to enable TP communication overlap.

    Behavior (moved from MegatronTrainer.patch_te_tp_overlap):
        - Guarded by module_config.tp_comm_overlap.
        - For FP8, disallow rs / bulk overlap combinations and raise if enabled.
        - Patch transformer_engine / transformer_engine_torch general_gemm or
          gemm/fp8_gemm and workspace helpers to Primus implementations.
    """
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None:
        warning_rank_0("[Patch:megatron.patch.tp_te_overlap][SKIP] No params in module_config")
        return

    if not getattr(params, "tp_comm_overlap", False):
        warning_rank_0("[Patch:megatron.patch.tp_te_overlap][SKIP] tp_comm_overlap is not enabled in params")
        return

    if getattr(params, "fp8", False):
        if (
            getattr(params, "tp_comm_overlap_rs", False)
            or getattr(params, "tp_comm_bulk_dgrad", False)
            or getattr(params, "tp_comm_bulk_wgrad", False)
        ):
            warning_rank_0(
                "[Patch:megatron.patch.tp_te_overlap][WARN] FP8 Async-tp not support for rs, bulk overlap! "
                "Please set tp_comm_overlap_rs=False, "
                "tp_comm_bulk_dgrad=False, tp_comm_bulk_wgrad=False"
            )
            return

    try:
        import transformer_engine as te
        import transformer_engine_torch as tex
        from megatron.core.utils import is_te_min_version

        from primus.backends.transformer_engine import transformer_engine_torch as ptex
        from primus.backends.transformer_engine.pytorch.module.base import (
            get_workspace,
            initialize_ub,
        )

        log_rank_0("[Patch:megatron.patch.tp_te_overlap] Patching transformer_engine TP overlap...")

        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType

        if is_te_min_version("2.0"):
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                general_gemm,
            )

            prev_general_gemm = te.pytorch.cpp_extensions.general_gemm
            te.pytorch.cpp_extensions.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
            te.pytorch.module.layernorm_linear.general_gemm = functools.partial(
                general_gemm, orig_func=prev_general_gemm
            )
        else:
            from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
                fp8_gemm,
                gemm,
            )

            prev_gemm = te.pytorch.cpp_extensions.gemm
            prev_fp8_gemm = te.pytorch.cpp_extensions.fp8_gemm

            tex.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.CommOverlapAlgo = ptex.CommOverlapAlgo
            te.pytorch.cpp_extensions.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.module.linear.gemm = functools.partial(gemm, orig_func=prev_gemm)
            te.pytorch.cpp_extensions.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
            te.pytorch.module.linear.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)

        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace
        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.patch.tp_te_overlap][SKIP] Failed to apply patch: {e}")
