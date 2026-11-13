###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: enlarge the TE hipBLASLt GEMM workspace on ROCm.

The large-K / small-N GEMMs of the latent MoE (expert fc2 4096 -> latent,
post_expert_proj, router wgrad) need more split-K workspace than TE's default on gfx950
("HIPBLASLT Error: 6"). The shared helper ``core/extensions/te_gemm_workspace.py`` (also
used by GLM-5.3) does it: ``PRIMUS_TE_GEMM_WORKSPACE_MB`` (default 512 MiB, from 1 to
``te_gemm_workspace.MAX_WORKSPACE_MB`` = 4096 MiB per workspace; another value raises
``ValueError`` naming the variable), ROCm only, a larger TE size kept. The
``after_train`` patch restores TE's own size, so a later model in the same process does not
keep the enlarged workspaces.
"""

from primus.backends.megatron.core.extensions.te_gemm_workspace import (
    enlarge_te_gemm_workspace,
    enlarged_by,
    restore_te_gemm_workspace,
)
from primus.backends.megatron.core.models.muse_spark.model_types import (
    fail_fast,
    is_muse_spark_model_type,
)
from primus.core.patches import PatchContext, register_patch

PATCH_NAME = "megatron.muse_spark.te_gemm_workspace"
RESTORE_PATCH_NAME = "megatron.muse_spark.te_gemm_workspace_restore"


@register_patch(
    PATCH_NAME,
    backend="megatron",
    phase="before_train",
    description="Muse Spark: enlarge the TE hipBLASLt GEMM workspace (PRIMUS_TE_GEMM_WORKSPACE_MB).",
    condition=is_muse_spark_model_type,
)
@fail_fast(PATCH_NAME)
def patch_muse_spark_te_gemm_workspace(ctx: PatchContext):
    del ctx
    enlarge_te_gemm_workspace(PATCH_NAME)


def _muse_spark_enlarged_it(ctx) -> bool:
    """Gated on Muse Spark's own enlargement, not on ``model_type``: by ``after_train``
    Megatron has replaced ``args.model_type`` with its ModelType enum, so the model_type
    check of the enlarge patch would not match. The enlarge patch (condition
    ``is_muse_spark_model_type``) marks TE's getter with its name; only that mark restores,
    so a GLM-5.3 enlargement is left alone."""
    del ctx
    return enlarged_by() == PATCH_NAME


@register_patch(
    RESTORE_PATCH_NAME,
    backend="megatron",
    phase="after_train",
    description="Muse Spark: restore TE's own GEMM workspace size after training.",
    condition=_muse_spark_enlarged_it,
)
def patch_muse_spark_te_gemm_workspace_restore(ctx: PatchContext):
    """A later model in the same process gets TE's default workspace back (not fail-fast:
    training is over, a restore error is only logged by the patch runner)."""
    del ctx
    restore_te_gemm_workspace(RESTORE_PATCH_NAME)
