###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: enlarge the TE hipBLASLt GEMM workspace on ROCm.

The large-K / small-N GEMMs of the latent MoE (expert fc2 4096 -> latent,
post_expert_proj, router wgrad) need more split-K workspace than TE's default on gfx950
("HIPBLASLT Error: 6"). The shared helper ``core/extensions/te_gemm_workspace.py`` (also
used by GLM-5.3) does it: ``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default 512, from 1 to 65536;
another value raises ``ValueError`` naming the variable), ROCm only, a larger TE size kept.
"""

from primus.backends.megatron.core.extensions.te_gemm_workspace import (
    enlarge_te_gemm_workspace,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, register_patch

PATCH_NAME = "megatron.muse_spark.te_gemm_workspace"


@register_patch(
    PATCH_NAME,
    backend="megatron",
    phase="before_train",
    description="Muse Spark: enlarge the TE hipBLASLt GEMM workspace (PRIMUS_TE_GEMM_WORKSPACE_MB).",
    condition=is_muse_spark_model_type,
)
def patch_muse_spark_te_gemm_workspace(ctx: PatchContext):
    del ctx
    enlarge_te_gemm_workspace(PATCH_NAME)
