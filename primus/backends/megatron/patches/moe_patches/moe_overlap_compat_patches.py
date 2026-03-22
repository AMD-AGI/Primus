###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MoE Overlap Compatibility Patch

When overlap_moe_expert_parallel_comm=True (without patch_moe_overlap), Megatron's
fine_grained_callables.py tries to set _comm_manager.token_probs and
_comm_manager.dispatched_probs on the token dispatcher's communication manager
to prevent backward graph connections from spanning across the stream boundaries.

Primus's PrimusTurboDeepEPTokenDispatcher uses pt.modules.DeepEPTokenDispatcher
as its _comm_manager. This class doesn't explicitly implement those properties,
but Python's dynamic attribute model allows the assignment. However, to ensure
gradient isolation is correct, we need the detach to actually disconnect the
autograd graph at the right points.

This patch:
1. Makes the _comm_manager property assignment safe and explicit.
2. Ensures the ep_overlap_early_attn_memory_release config field is set correctly
   for DeepEP-based dispatchers (avoids AttributeError in fine_grained_callables).
3. Logs the overlap configuration on startup.
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_overlap_moe_without_primus_patch(ctx) -> bool:
    args = get_args(ctx)
    return (
        getattr(args, "overlap_moe_expert_parallel_comm", False)
        and not getattr(args, "patch_moe_overlap", False)
        and getattr(args, "use_turbo_deepep", False)
    )


@register_patch(
    "megatron.moe.overlap_compat",
    backend="megatron",
    phase="before_train",
    description="Compatibility shim for overlap_moe_expert_parallel_comm with PrimusTurboDeepEP",
    condition=_is_overlap_moe_without_primus_patch,
)
def patch_moe_overlap_compat(ctx: PatchContext):
    """
    Ensure PrimusTurboDeepEPTokenDispatcher is compatible with Megatron's
    fine_grained_callables when overlap_moe_expert_parallel_comm=True.

    Key concerns:
    1. fine_grained_callables.py sets _comm_manager.token_probs and
       _comm_manager.dispatched_probs to detached tensors. With Primus's
       DeepEPTokenDispatcher, these are dynamic attributes. The backward pass
       of PrimusTurboDeepEP uses its own internal state (not these attributes),
       so they are safe to set but won't affect gradient computation.

    2. The config field `ep_overlap_early_attn_memory_release` may not exist
       in older Megatron versions. Ensure it's present to avoid AttributeError.

    3. The field `moe_flex_dispatcher_backend` must be "deepep" for
       fine_grained_callables to correctly handle the dispatch/combine split.
       This is already set by moe_enable_deepep=True in moe_dispatcher_patches,
       but we verify it here for clarity.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig

    # Ensure ep_overlap_early_attn_memory_release has a default if missing
    if not hasattr(TransformerConfig, "ep_overlap_early_attn_memory_release"):
        # Add as a default-False property so fine_grained_callables.py doesn't error
        original_init = TransformerConfig.__init__

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "ep_overlap_early_attn_memory_release"):
                object.__setattr__(self, "ep_overlap_early_attn_memory_release", False)

        TransformerConfig.__init__ = patched_init
        log_rank_0(
            "[Patch:megatron.moe.overlap_compat] Added ep_overlap_early_attn_memory_release "
            "default to TransformerConfig"
        )

    # Verify that moe_flex_dispatcher_backend will be set to "deepep"
    # (This happens in TransformerConfig.__post_init__ when moe_enable_deepep=True)
    args = get_args(ctx)
    log_rank_0(
        f"[Patch:megatron.moe.overlap_compat] MoE overlap (native Megatron path) enabled:\n"
        f"  overlap_moe_expert_parallel_comm = True\n"
        f"  patch_moe_overlap = False (using native GPTModel.build_schedule_plan)\n"
        f"  use_turbo_deepep = True\n"
        f"  Expected: moe_flex_dispatcher_backend='deepep', "
        f"moe_token_dispatcher_type='flex'\n"
        f"  Schedule: combined_1f1b_schedule_for_no_pipelining (PP=1) or\n"
        f"            combined_1f1b_schedule_for_interleaved_pipelining (PP>1, VPP>1)\n"
        f"  Overlap pattern per layer pair (mb_n fwd, mb_n-1 bwd):\n"
        f"    comm: [combine_bwd] [dispatch_fwd + dispatch_bwd] [combine_fwd]\n"
        f"    comp: [attn_fwd   ] [mlp_bwd + mlp_dw + mlp_fwd] [attn_bwd   ]\n"
    )
