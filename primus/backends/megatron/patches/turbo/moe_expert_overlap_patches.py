###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus MoE Expert ↔ Communication Overlap Patch

Patches MoELayer.forward to use chunked expert computation that overlaps
expert GEMM with AlltoAll communication via CUDA stream pipelining.

This patch is activated when:
    - enable_primus_turbo == True
    - use_turbo_deepep == True
    - turbo_moe_expert_comm_overlap == True

The number of pipeline chunks is controlled by ``turbo_moe_overlap_num_chunks``
(default: 2).
"""

import importlib.util

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_expert_comm_overlap_enabled(ctx: PatchContext) -> bool:
    """
    Check if expert-communication overlap is enabled.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True
      - use_turbo_deepep == True
      - turbo_moe_expert_comm_overlap == True
    """
    if importlib.util.find_spec("primus_turbo") is None:
        return False

    args = get_args(ctx)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))
    use_turbo_deepep = bool(getattr(args, "use_turbo_deepep", False))
    expert_comm_overlap = bool(getattr(args, "turbo_moe_expert_comm_overlap", False))

    return enable_primus_turbo and use_turbo_deepep and expert_comm_overlap


@register_patch(
    "megatron.turbo.moe_expert_comm_overlap",
    backend="megatron",
    phase="before_train",
    description="Overlap expert GEMM with AlltoAll communication in MoE layers",
    condition=_is_expert_comm_overlap_enabled,
)
def patch_moe_expert_comm_overlap(ctx: PatchContext):
    """
    Patch MoELayer.forward to use chunked expert–communication overlap.

    This replaces MoELayer.forward with a version that splits dispatched tokens
    into chunks and pipelines expert computation with communication operations
    on separate CUDA streams.
    """
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.transformer.moe.expert_comm_overlap import (
        make_overlapped_forward,
    )

    args = get_args(ctx)
    num_chunks = int(getattr(args, "turbo_moe_overlap_num_chunks", 2))

    log_rank_0(
        f"[Patch:megatron.turbo.moe_expert_comm_overlap] "
        f"Patching MoELayer.forward with {num_chunks}-chunk expert↔comm overlap..."
    )

    # Store original forward for reference
    original_forward = moe_layer.MoELayer.forward

    # Create and apply patched forward
    patched_forward = make_overlapped_forward(original_forward, num_chunks=num_chunks)
    moe_layer.MoELayer.forward = patched_forward

    log_rank_0(
        f"[Patch:megatron.turbo.moe_expert_comm_overlap] "
        f"Successfully patched MoELayer.forward with {num_chunks}-chunk overlap pipeline"
    )
