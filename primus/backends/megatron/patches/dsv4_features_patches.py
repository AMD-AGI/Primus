###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Logs the DeepSeek-V4 fast-path gates once on rank 0 (see ``core/dsv4_features.py``)."""

from primus.backends.megatron.core import dsv4_features
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _active(ctx: PatchContext) -> bool:
    del ctx
    return dsv4_features.is_gfx1250()


@register_patch(
    "megatron.dsv4_features.summary",
    backend="megatron",
    phase="before_train",
    description="Log the DeepSeek-V4 gfx1250 feature gates (arch, Triton / Gluon, master switches).",
    condition=_active,
    priority=1,
)
def patch_dsv4_features_summary(ctx: PatchContext) -> None:
    del ctx
    log_rank_0(dsv4_features.describe())
