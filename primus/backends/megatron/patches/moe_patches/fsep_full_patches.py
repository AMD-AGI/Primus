###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
Full FSEP Patches (Phase 1-3).

Activated when moe_fsep_full_mode: true AND moe_fsep_sharding_degree > 1.

Patches applied:
  1. Initialize FSEPState (global_expert_locations, inverse_expert_map)
  2. Replace token dispatcher with FSEPAlltoAllTokenDispatcher
  3. Attach FSEPLoadPlanner to FSEPState
  4. Register backward hook for FSEPRelayoutExecutor

Note: The static FSEPGroupedMLP (from fsep_patches.py) is still used for
Expert GEMM + ReduceScatter. The full FSEP extends the dispatch logic.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_full_fsep_enabled(ctx: PatchContext) -> bool:
    """Check if full FSEP (dynamic dispatch + Load Planner) is enabled."""
    args = get_args(ctx)
    return (
        getattr(args, "moe_fsep_sharding_degree", 0) > 1
        and getattr(args, "moe_fsep_full_mode", False)
    )


@register_patch(
    "megatron.moe.fsep_state_init",
    backend="megatron",
    phase="before_train",
    description="Initialize FSEP global state (expert placement tracking)",
    condition=_is_full_fsep_enabled,
)
def patch_fsep_state_init(ctx: PatchContext):
    """
    Initialize the FSEP global state.

    Creates FSEPState with uniform initial placement and attaches
    FSEPLoadPlanner for dynamic monitoring.
    """
    import torch.distributed as dist
    from megatron.core import parallel_state

    args = get_args(ctx)
    S = args.moe_fsep_sharding_degree
    num_experts = getattr(args, "num_experts", None) or getattr(args, "num_moe_experts", 256)
    ep_size = getattr(args, "expert_model_parallel_size", 8)

    from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
        init_fsep_state,
    )
    from primus.backends.megatron.core.transformer.moe.load_planner import (
        FSEPLoadPlanner,
    )

    ep_group = parallel_state.get_expert_model_parallel_group()
    fsep_state = init_fsep_state(
        num_experts=num_experts,
        ep_size=ep_size,
        sharding_degree=S,
        ep_group=ep_group,
    )

    # Expand to full FSEP (all GPUs can serve all experts)
    fsep_state.expand_to_full_fsep()

    # Attach Load Planner
    check_interval = getattr(args, "moe_fsep_relayout_interval", 50)
    threshold = getattr(args, "moe_fsep_imbalance_threshold", 1.5)
    fsep_state.load_planner = FSEPLoadPlanner(
        num_experts=num_experts,
        ep_size=ep_size,
        sharding_degree=S,
        check_interval=check_interval,
        imbalance_threshold=threshold,
    )

    log_rank_0(
        f"[Patch:megatron.moe.fsep_state_init] FSEP State initialized: "
        f"N_E={num_experts}, EP={ep_size}, S={S}, "
        f"relayout_interval={check_interval}, threshold={threshold}"
    )
