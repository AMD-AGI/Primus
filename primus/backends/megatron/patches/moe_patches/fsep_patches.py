###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE FSEP (Fully Sharded Expert Parallel) Patches.

When moe_fsep_sharding_degree > 1, replaces PrimusTurboGroupedMLP with
FSEPGroupedMLP, which uses ReduceScatter instead of All-Reduce for output
aggregation to enable load-balanced expert computation.

The PrimusTurboDeepEPTokenDispatcher already handles the FSEP AllGather in
combine_preprocess (patched directly in primus_turbo.py).
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_fsep_enabled(ctx: PatchContext) -> bool:
    """Check if FSEP is enabled via moe_fsep_sharding_degree > 1."""
    args = get_args(ctx)
    return getattr(args, "moe_fsep_sharding_degree", 0) > 1


@register_patch(
    "megatron.moe.fsep_grouped_mlp",
    backend="megatron",
    phase="before_train",
    description=(
        "Replace PrimusTurboGroupedMLP with FSEPGroupedMLP for "
        "load-balanced expert sharding via ReduceScatter"
    ),
    condition=_is_fsep_enabled,
)
def patch_fsep_grouped_mlp(ctx: PatchContext):
    """
    Patch GroupedMLP to use FSEP (ReduceScatter output aggregation).

    Behavior:
        - Replace GroupedMLP in megatron.core.transformer.moe.experts
          with FSEPGroupedMLP
        - Replace in moe_module_specs (for spec-based model building)
        - Replace PrimusTurboGroupedMLP in primus_turbo if already patched
    """
    import sys

    from primus.backends.megatron.core.transformer.moe.fsep_experts import (
        FSEPGroupedMLP,
    )

    # Patch megatron core experts module
    import megatron.core.transformer.moe.experts as meg_experts
    meg_experts.GroupedMLP = FSEPGroupedMLP
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp]   Patched "
        f"megatron.core.transformer.moe.experts.GroupedMLP "
        f"-> {FSEPGroupedMLP.__name__}"
    )

    # Patch moe_module_specs (used by GPT model spec builder)
    from megatron.core.models.gpt import moe_module_specs
    moe_module_specs.GroupedMLP = FSEPGroupedMLP
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp]   Patched "
        f"megatron.core.models.gpt.moe_module_specs.GroupedMLP "
        f"-> {FSEPGroupedMLP.__name__}"
    )

    # If primus_turbo module is already loaded, patch its GroupedMLP import too
    turbo_module_key = "primus.backends.megatron.core.extensions.primus_turbo"
    if turbo_module_key in sys.modules:
        turbo_mod = sys.modules[turbo_module_key]
        if hasattr(turbo_mod, "GroupedMLP"):
            turbo_mod.GroupedMLP = FSEPGroupedMLP
            log_rank_0(
                f"[Patch:megatron.moe.fsep_grouped_mlp]   Patched "
                f"primus_turbo.GroupedMLP -> {FSEPGroupedMLP.__name__}"
            )

    args = get_args(ctx)
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp] FSEP enabled: "
        f"sharding_degree={args.moe_fsep_sharding_degree}, "
        f"expert_model_parallel_size={getattr(args, 'expert_model_parallel_size', 1)}"
    )
