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
import importlib.util

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


def _check_tp_overlap_conditions(ctx: PatchContext) -> bool:
    """Helper to check basic TP overlap conditions."""
    module_config = ctx.extra.get("module_config")
    params = getattr(module_config, "params", None)
    if params is None:
        return False

    if not getattr(params, "tp_comm_overlap", False):
        return False

    # Check FP8 incompatible settings
    if getattr(params, "fp8", False):
        if (
            getattr(params, "tp_comm_overlap_rs", False)
            or getattr(params, "tp_comm_bulk_dgrad", False)
            or getattr(params, "tp_comm_bulk_wgrad", False)
        ):
            return False

    return True


def _is_te_v2_or_above(ctx: PatchContext) -> bool:
    """Check if TE version is 2.0 or above."""
    if not _check_tp_overlap_conditions(ctx):
        return False
    try:
        from megatron.core.utils import is_te_min_version

        return is_te_min_version("2.0")
    except Exception:
        return False


def _is_te_below_v2(ctx: PatchContext) -> bool:
    """Check if TE version is below 2.0."""
    if not _check_tp_overlap_conditions(ctx):
        return False
    try:
        from megatron.core.utils import is_te_min_version

        return not is_te_min_version("2.0")
    except Exception:
        return False


@register_patch(
    "megatron.patch.tp_te_overlap.v2",
    backend="megatron",
    phase="before_train",
    description="Enable TE TP communication overlap for TE >= 2.0 (using general_gemm)",
    condition=_is_te_v2_or_above,
)
def patch_tp_te_overlap_v2(ctx: PatchContext):
    """
    Patch Transformer Engine TP overlap for TE >= 2.0.

    This version uses general_gemm API introduced in TE 2.0.
    """
    try:
        import transformer_engine as te
        import transformer_engine_torch as tex

        from primus.backends.transformer_engine import transformer_engine_torch as ptex
        from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
            general_gemm,
        )
        from primus.backends.transformer_engine.pytorch.module.base import (
            get_workspace,
            initialize_ub,
        )

        log_rank_0("[Patch:megatron.patch.tp_te_overlap.v2] Patching TE TP overlap (TE >= 2.0)...")

        # Patch CommOverlap types
        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType

        # Patch general_gemm
        prev_general_gemm = te.pytorch.cpp_extensions.general_gemm
        te.pytorch.cpp_extensions.general_gemm = functools.partial(general_gemm, orig_func=prev_general_gemm)
        te.pytorch.module.linear.general_gemm = functools.partial(general_gemm, orig_func=prev_general_gemm)
        te.pytorch.module.layernorm_linear.general_gemm = functools.partial(
            general_gemm, orig_func=prev_general_gemm
        )

        # Patch workspace helpers
        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace
        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

        log_rank_0("[Patch:megatron.patch.tp_te_overlap.v2] Successfully patched TE TP overlap")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.patch.tp_te_overlap.v2][SKIP] Failed to apply patch: {e}")


@register_patch(
    "megatron.patch.tp_te_overlap.v1",
    backend="megatron",
    phase="before_train",
    description="Enable TE TP communication overlap for TE < 2.0 (using gemm/fp8_gemm)",
    condition=_is_te_below_v2,
)
def patch_tp_te_overlap_v1(ctx: PatchContext):
    """
    Patch Transformer Engine TP overlap for TE < 2.0.

    This version uses gemm and fp8_gemm APIs for older TE versions.
    """
    try:
        import transformer_engine as te
        import transformer_engine_torch as tex

        from primus.backends.transformer_engine import transformer_engine_torch as ptex
        from primus.backends.transformer_engine.pytorch.cpp_extensions.gemm import (
            fp8_gemm,
            gemm,
        )
        from primus.backends.transformer_engine.pytorch.module.base import (
            get_workspace,
            initialize_ub,
        )

        log_rank_0("[Patch:megatron.patch.tp_te_overlap.v1] Patching TE TP overlap (TE < 2.0)...")

        # Patch CommOverlap types
        tex.CommOverlap = ptex.CommOverlap
        tex.CommOverlapP2P = ptex.CommOverlapP2P
        tex.CommOverlapType = ptex.CommOverlapType
        tex.CommOverlapAlgo = ptex.CommOverlapAlgo

        # Patch gemm and fp8_gemm
        prev_gemm = te.pytorch.cpp_extensions.gemm
        prev_fp8_gemm = te.pytorch.cpp_extensions.fp8_gemm

        te.pytorch.cpp_extensions.CommOverlapAlgo = ptex.CommOverlapAlgo
        te.pytorch.cpp_extensions.gemm = functools.partial(gemm, orig_func=prev_gemm)
        te.pytorch.module.linear.gemm = functools.partial(gemm, orig_func=prev_gemm)
        te.pytorch.cpp_extensions.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)
        te.pytorch.module.linear.fp8_gemm = functools.partial(fp8_gemm, orig_func=prev_fp8_gemm)

        # Patch workspace helpers
        te.pytorch.module.base.initialize_ub = initialize_ub
        te.pytorch.module.base.get_workspace = get_workspace
        te.pytorch.cpp_extensions.CommOverlapType = ptex.CommOverlapType

        log_rank_0("[Patch:megatron.patch.tp_te_overlap.v1] Successfully patched TE TP overlap")

    except Exception as e:
        warning_rank_0(f"[Patch:megatron.patch.tp_te_overlap.v1][SKIP] Failed to apply patch: {e}")
