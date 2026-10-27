###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: enlarge the TE hipBLASLt GEMM workspace on ROCm.

On gfx950, hipBLASLt picks split-K algorithms for the large-K / small-N GEMMs of
the latent MoE (expert fc2 4096 -> latent, post_expert_proj, router wgrad) whose
workspace exceeds TE's 64 MiB default and the GEMM fails with
"HIPBLASLT Error: 6". Same fix as the GLM-5.3 port.
"""

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.core.utils.module_utils import log_rank_0

_MODEL_TYPES = ("muse_spark", "muse_spark_text")


def _is_muse_spark(ctx: PatchContext) -> bool:
    try:
        if getattr(get_args(ctx), "model_type", None) in _MODEL_TYPES:
            return True
    except AssertionError:
        pass
    return getattr(ctx.extra.get("backend_args"), "model_type", None) in _MODEL_TYPES


@register_patch(
    "megatron.muse_spark.te_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Muse Spark: enlarge the TE hipBLASLt GEMM workspace (64 MiB is too small for the latent MoE GEMMs).",
    condition=_is_muse_spark,
)
def patch_muse_spark_te_gemm_workspace(ctx: PatchContext):
    del ctx
    import torch

    if torch.version.hip is None:
        return
    import transformer_engine.pytorch.cpp_extensions.gemm as te_gemm

    size = int(os.environ.get("PRIMUS_TE_GEMM_WORKSPACE_MB", "512")) * 1024 * 1024
    if te_gemm.get_cublas_workspace_size_bytes() >= size:
        return
    te_gemm.get_cublas_workspace_size_bytes = lambda: size
    te_gemm.get_cublas_workspace.cache_clear()
    log_rank_0(f"[Patch:megatron.muse_spark.te_gemm_workspace] TE GEMM workspace -> {size >> 20} MiB")
