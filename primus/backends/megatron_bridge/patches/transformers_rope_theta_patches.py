###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge transformers RoPE compatibility patches.

transformers 5.x removed the flat ``Qwen3Config.rope_theta`` field (and the
same for other model configs). The vendored Megatron-Bridge still reads
``hf_config.rope_theta`` during HF->Megatron conversion and provider setup.

These patches install a ``PretrainedConfig`` attribute shim so existing bridge
code keeps working without editing ``third_party/Megatron-Bridge``.
"""

from primus.backends.megatron_bridge.utils.rope_config import (
    install_transformers_rope_theta_shim,
)
from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0


def _apply_rope_theta_shim(_ctx: PatchContext, phase: str) -> None:
    if install_transformers_rope_theta_shim():
        log_rank_0(
            f"[Patch:megatron_bridge.transformers_rope_theta] "
            f"Installed transformers 5.x rope_theta shim ({phase})"
        )


@register_patch(
    "megatron_bridge.convert.transformers_rope_theta",
    backend="megatron_bridge",
    phase="convert",
    priority=5,
    description=(
        "Shim PretrainedConfig.rope_theta for transformers 5.x during "
        "Megatron-Bridge HF->Megatron checkpoint conversion."
    ),
)
def patch_transformers_rope_theta_convert(ctx: PatchContext) -> None:
    """Install the rope_theta shim for Megatron-Bridge checkpoint conversion."""
    _apply_rope_theta_shim(ctx, "convert")


@register_patch(
    "megatron_bridge.train.transformers_rope_theta",
    backend="megatron_bridge",
    phase="before_train",
    priority=5,
    description=(
        "Shim PretrainedConfig.rope_theta for transformers 5.x during " "Megatron-Bridge training startup."
    ),
)
def patch_transformers_rope_theta_train(ctx: PatchContext) -> None:
    """Install the rope_theta shim before Megatron-Bridge training starts."""
    _apply_rope_theta_shim(ctx, "before_train")
