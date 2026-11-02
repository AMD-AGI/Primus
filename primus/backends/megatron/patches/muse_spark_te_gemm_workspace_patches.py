###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: enlarge the TE hipBLASLt GEMM workspace on ROCm.

On gfx950, hipBLASLt picks split-K algorithms for the large-K / small-N GEMMs of
the latent MoE (expert fc2 4096 -> latent, post_expert_proj, router wgrad) whose
workspace exceeds TE's default and the GEMM fails with "HIPBLASLT Error: 6".
The work is done by the ``core.extensions.te_gemm_workspace`` helper, which handles
both TE layouts.
"""

from primus.backends.megatron.core.extensions.te_gemm_workspace import (
    apply_rocm_te_gemm_workspace,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, get_args, register_patch


@register_patch(
    "megatron.muse_spark.te_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Muse Spark: enlarge the TE hipBLASLt GEMM workspace (the default is too small for the latent MoE GEMMs).",
    condition=is_muse_spark_model_type,
)
def patch_muse_spark_te_gemm_workspace(ctx: PatchContext):
    # size: the ``te_gemm_workspace_mb`` config key (MiB, muse_spark_base.yaml: 512)
    apply_rocm_te_gemm_workspace("[Patch:megatron.muse_spark.te_gemm_workspace]", get_args(ctx))
