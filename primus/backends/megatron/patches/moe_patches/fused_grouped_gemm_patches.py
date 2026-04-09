###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Fused BF16 Grouped GEMM Patch

Replaces ``grouped_gemm.ops.GroupedGemm`` with a fused Triton BF16 grouped
GEMM that computes **all E experts in a single kernel launch**.

Forward and dA use the fused Triton kernel; dW uses hipBLASLt (faster for
weight gradients with small M).

Controlled by ``PRIMUS_FUSED_GROUPED_GEMM`` (default ``"0"``).
Set to ``"1"`` to enable.
"""

import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


def _fused_gg_enabled() -> bool:
    return os.getenv("PRIMUS_FUSED_GROUPED_GEMM", "0") == "1"


@register_patch(
    "megatron.moe.fused_grouped_gemm",
    backend="megatron",
    phase="before_train",
    description="Replace grouped_gemm with fused Triton BF16 single-launch kernel",
    condition=lambda ctx: _fused_gg_enabled(),
)
def patch_fused_grouped_gemm(ctx: PatchContext):
    log_rank_0(
        "[Patch:megatron.moe.fused_grouped_gemm] "
        "Patching grouped_gemm with fused Triton BF16 kernel ..."
    )

    try:
        import grouped_gemm
    except ImportError:
        log_rank_0(
            "[Patch:megatron.moe.fused_grouped_gemm] "
            "grouped_gemm not installed; skipping."
        )
        return

    from primus.backends.megatron.core.fusions.bf16_fused_grouped_gemm import (
        FusedBF16GroupedGemm,
        fused_gmm,
    )

    original_cls = grouped_gemm.ops.GroupedGemm
    grouped_gemm.ops.GroupedGemm = FusedBF16GroupedGemm
    grouped_gemm.ops.gmm = fused_gmm

    from megatron.core.transformer.moe import grouped_gemm_util as gg_util
    if gg_util.ops is not None:
        gg_util.ops.GroupedGemm = FusedBF16GroupedGemm
        gg_util.ops.gmm = fused_gmm

    log_rank_0(
        "[Patch:megatron.moe.fused_grouped_gemm] "
        f"  Replaced {original_cls.__name__} -> FusedBF16GroupedGemm"
    )
    log_rank_0(
        "[Patch:megatron.moe.fused_grouped_gemm] "
        "  FWD+dA: single-launch Triton | dW: hipBLASLt"
    )
