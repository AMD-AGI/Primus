###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Primus MoE Expert ↔ Communication Overlap Patch

Patches MoELayer.forward to overlap dispatch AlltoAll communication with
shared expert computation using separate CUDA streams.

The overlap relies on DeepEP's async dispatch (comm stream) running
concurrently with shared expert GEMM (compute stream). This gives a
meaningful speedup for MoE models with shared experts (e.g., DeepSeek-V2/V3)
where dispatch latency can be hidden behind shared expert computation.

This patch is activated when:
    - enable_primus_turbo == True
    - use_turbo_deepep == True
    - turbo_moe_expert_comm_overlap == True

The patch automatically enables ``turbo_deepep_use_comm_stream=True`` so
that DeepEP runs AlltoAll on a separate comm stream.
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
    description="Overlap dispatch AlltoAll with shared expert compute in MoE layers",
    condition=_is_expert_comm_overlap_enabled,
)
def patch_moe_expert_comm_overlap(ctx: PatchContext):
    """
    Patch MoELayer.forward to overlap dispatch communication with shared
    expert computation via CUDA stream pipelining.

    Also auto-enables DeepEP comm stream for proper async AlltoAll.
    """
    from megatron.core.transformer.moe import moe_layer

    from primus.backends.megatron.core.transformer.moe.expert_comm_overlap import (
        make_overlapped_forward,
    )

    args = get_args(ctx)

    # Auto-enable DeepEP comm stream for async AlltoAll
    if not getattr(args, "turbo_deepep_use_comm_stream", False):
        args.turbo_deepep_use_comm_stream = True
        log_rank_0(
            "[Patch:moe_expert_comm_overlap] Auto-enabled turbo_deepep_use_comm_stream=True"
        )

    # Auto-disable Megatron's own shared_expert_overlap to avoid conflict
    if getattr(args, "moe_shared_expert_overlap", False):
        log_rank_0(
            "[Patch:moe_expert_comm_overlap] moe_shared_expert_overlap already enabled, "
            "Primus overlap will take precedence"
        )

    log_rank_0(
        "[Patch:moe_expert_comm_overlap] "
        "Patching MoELayer.forward with dispatch↔shared-expert overlap..."
    )

    original_forward = moe_layer.MoELayer.forward
    patched_forward = make_overlapped_forward(original_forward)
    moe_layer.MoELayer.forward = patched_forward

    log_rank_0(
        "[Patch:moe_expert_comm_overlap] "
        "Successfully patched MoELayer.forward with comm↔compute overlap"
    )
