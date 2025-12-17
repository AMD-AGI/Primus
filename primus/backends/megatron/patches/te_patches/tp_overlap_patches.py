###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Transformer Engine TP Communication Overlap Patches

Patches for enabling tensor-parallel communication overlap in Transformer Engine.
Handles different TE versions (< 2.0 and >= 2.0) with appropriate APIs.
"""

import functools

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


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
