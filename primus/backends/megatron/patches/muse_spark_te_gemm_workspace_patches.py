###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark: enlarge the TE hipBLASLt GEMM workspace on ROCm.

The large-K / small-N GEMMs of the latent MoE (expert fc2 4096 -> latent,
post_expert_proj, router wgrad) need more split-K workspace than TE's default on gfx950
("HIPBLASLT Error: 6"). GLM-5.3's patch already does this (``PRIMUS_TE_GEMM_WORKSPACE_MB``,
its default when unset; ROCm only; a larger TE size is kept), and this patch runs it on
Muse Spark. On ROCm only (where GLM-5.3's body acts; elsewhere it does nothing), it first
checks a set ``PRIMUS_TE_GEMM_WORKSPACE_MB`` by the rule GLM-5.3's ``int()`` applies,
narrowed to ASCII digits with optional surrounding whitespace, from 1 to 65536 MiB.
Anything else (blank, ``1.5``, ``1e3``, ``+5``, ``0``, non-ASCII digits, a typo such as
``99999999999``) raises ``ValueError`` naming the variable instead of GLM-5.3's bare one,
a silent tiny workspace or an allocation failure far from the config. The environment is not changed:
a value this accepts is one GLM-5.3's ``int()`` reads as the same number. It also logs its
own line before GLM-5.3's.
"""

import os
import re

from primus.backends.megatron.core.models.muse_spark.model_types import (
    is_muse_spark_model_type,
)
from primus.backends.megatron.patches.glm5_next_patches import (
    patch_glm5_next_te_gemm_workspace,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


@register_patch(
    "megatron.muse_spark.te_gemm_workspace",
    backend="megatron",
    phase="before_train",
    description="Muse Spark: enlarge the TE hipBLASLt GEMM workspace (GLM-5.3's patch; PRIMUS_TE_GEMM_WORKSPACE_MB).",
    condition=is_muse_spark_model_type,
)
def patch_muse_spark_te_gemm_workspace(ctx: PatchContext):
    import torch

    if torch.version.hip is None:  # GLM-5.3's body acts on ROCm only: nothing to do or log
        return
    value = os.environ.get(WORKSPACE_ENV)
    if value is not None:  # unset: GLM-5.3's default
        validate_workspace_mb(value)  # GLM-5.3's int() then reads the same number
    # GLM-5.3's body logs its own line when it enlarges the workspace (it keeps a larger TE
    # size); this line names the Muse Spark patch that ran it
    log_rank_0("[Patch:megatron.muse_spark.te_gemm_workspace] running GLM-5.3's TE GEMM workspace patch")
    patch_glm5_next_te_gemm_workspace(ctx)


WORKSPACE_ENV = "PRIMUS_TE_GEMM_WORKSPACE_MB"
# 64 GiB: above any GPU's spare memory for a GEMM workspace; a larger value is a typo
MAX_WORKSPACE_MB = 64 * 1024
_ASCII_DIGITS = re.compile(r"[0-9]+")


def validate_workspace_mb(value):
    """Check a set ``PRIMUS_TE_GEMM_WORKSPACE_MB``: ASCII digits with optional surrounding
    whitespace (a value Python's ``int()`` reads the same way), from 1 to
    ``MAX_WORKSPACE_MB``. Anything else raises ``ValueError`` naming the variable. GLM-5.3's
    body reads the variable itself, so nothing is returned."""
    text = str(value).strip()
    # at most len(str(MAX_WORKSPACE_MB)) digits after leading zeros: no int() of a long
    # string (Python's own 4300-digit limit would raise a bare ValueError)
    digits = text.lstrip("0") or "0"
    ok = _ASCII_DIGITS.fullmatch(text) and len(digits) <= len(str(MAX_WORKSPACE_MB))
    mb = int(digits) if ok else 0
    if not 1 <= mb <= MAX_WORKSPACE_MB:
        raise ValueError(
            f"{WORKSPACE_ENV} must be an integer number of MiB from 1 to {MAX_WORKSPACE_MB}, got {value!r}"
        )
