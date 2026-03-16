###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
Full FSEP Patches (Phase 1-3).

Activated when moe_fsep_full_mode: true AND moe_fsep_sharding_degree > 1.

Patches applied:
  1. Initialize FSEPState (global_expert_locations, inverse_expert_map)
  2. Attach FSEPLoadPlanner to FSEPState
  3. Hook FSEPRelayoutExecutor into MoE layer forward/backward
  4. Integrate smart routing into PrimusTurboDeepEPTokenDispatcher

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
    gpus_per_node = getattr(args, "moe_fsep_gpus_per_node", 8)

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
        gpus_per_node=gpus_per_node,
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
        gpus_per_node=gpus_per_node,
    )

    log_rank_0(
        f"[Patch:megatron.moe.fsep_state_init] FSEP State initialized: "
        f"N_E={num_experts}, EP={ep_size}, S={S}, "
        f"relayout_interval={check_interval}, threshold={threshold}"
    )


@register_patch(
    "megatron.moe.fsep_smart_routing_hook",
    backend="megatron",
    phase="before_train",
    description="Hook smart routing into MoE layer dispatch for load-balanced token distribution",
    condition=_is_full_fsep_enabled,
)
def patch_fsep_smart_routing(ctx: PatchContext):
    """
    Patch the MoE layer to use smart routing in the dispatch path.

    This hooks into PrimusTurboDeepEPTokenDispatcher.dispatch_preprocess to:
      1. Collect global token-per-expert statistics
      2. Compute smart routing allocation (Algorithm 3: Lite Routing)
      3. Remap routing_map to new_routing_map (tokens go to balanced slots)
      4. Update tokens_per_slot for capacity-aware assignment

    The patched preprocess feeds the new routing_map into DeepEP's dispatch.
    """
    args = get_args(ctx)

    from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
        get_fsep_state,
    )
    from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
        smart_routing_map,
        compute_smart_routing,
    )

    # Monkey-patch the DeepEP dispatcher's preprocess to inject smart routing
    import primus.backends.megatron.core.extensions.primus_turbo as turbo_mod

    OriginalDispatcher = turbo_mod.PrimusTurboDeepEPTokenDispatcher
    original_dispatch_preprocess = OriginalDispatcher.dispatch_preprocess

    def _smart_dispatch_preprocess(self, hidden_states, routing_map, probs):
        """
        Enhanced dispatch_preprocess with FSEP smart routing.

        Before calling the original DeepEP dispatch, remap routing_map
        to balance tokens across expert replicas.
        """
        fsep_state = get_fsep_state()

        if fsep_state is not None and self.fsep_sharding_degree > 1:
            import torch
            import torch.distributed as dist

            T, N_E = routing_map.shape

            # Step 1: Collect global token distribution
            num_local_tokens = routing_map.long().sum(dim=0)  # [N_E]
            num_global_tokens = num_local_tokens.clone()
            dist.all_reduce(num_global_tokens, group=self.ep_group)

            # Step 2: Compute smart routing allocation (Lite Routing - Algorithm 3)
            ep_rank = dist.get_rank(group=self.ep_group)
            tokens_per_slot = smart_routing_map(
                num_global_tokens,
                fsep_state.global_expert_locations,
                fsep_state.num_local_experts,
                ep_rank=ep_rank,
                gpus_per_node=fsep_state.gpus_per_node,
            )

            # Scale to local rank's share
            ep_size = dist.get_world_size(group=self.ep_group)
            tokens_per_slot_local = (tokens_per_slot // ep_size).long()

            # Step 3: Remap routing_map → new_routing_map
            new_routing_map, new_probs = compute_smart_routing(
                routing_map,
                probs,
                fsep_state.global_expert_locations,
                fsep_state.inverse_expert_map,
                tokens_per_slot_local,
            )

            # Step 4: Feed load to planner
            if fsep_state.load_planner is not None:
                fsep_state.load_planner.update(num_global_tokens)

            # Use the smart-routed map for dispatch
            routing_map = new_routing_map
            probs = new_probs

        return original_dispatch_preprocess(self, hidden_states, routing_map, probs)

    OriginalDispatcher.dispatch_preprocess = _smart_dispatch_preprocess

    log_rank_0(
        "[Patch:megatron.moe.fsep_smart_routing_hook] Smart routing hooked into "
        "PrimusTurboDeepEPTokenDispatcher.dispatch_preprocess"
    )


@register_patch(
    "megatron.moe.fsep_relayout_hook",
    backend="megatron",
    phase="before_train",
    description="Hook FSEPRelayoutExecutor into training loop for dynamic expert re-layout",
    condition=_is_full_fsep_enabled,
)
def patch_fsep_relayout_hook(ctx: PatchContext):
    """
    Hook the FSEPRelayoutExecutor into the training loop.

    Two integration points:
      1. After each MoE layer forward: check if Load Planner wants a relayout
      2. At start of each training step: finalize any pending relayout

    The relayout executor is attached to FSEPState and triggered by the planner.
    """
    args = get_args(ctx)

    from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
        get_fsep_state,
    )
    from primus.backends.megatron.core.transformer.moe.relayout_executor import (
        FSEPRelayoutExecutor,
    )

    # Hook into MoE layer to check relayout after forward pass
    import megatron.core.transformer.moe.moe_layer as moe_layer_mod

    OriginalMoELayer = moe_layer_mod.MoELayer
    original_forward = OriginalMoELayer.forward

    def _moe_forward_with_relayout_check(self, hidden_states):
        """MoE forward with post-forward relayout check."""
        output = original_forward(self, hidden_states)

        # Check if relayout is needed (non-blocking)
        fsep_state = get_fsep_state()
        if fsep_state is not None and fsep_state.load_planner is not None:
            plan = fsep_state.load_planner.plan(fsep_state)
            if plan is not None and fsep_state.relayout_executor is not None:
                fsep_state.relayout_executor.schedule_relayout(plan)

        return output

    OriginalMoELayer.forward = _moe_forward_with_relayout_check

    log_rank_0(
        "[Patch:megatron.moe.fsep_relayout_hook] Relayout check hooked into "
        "MoELayer.forward (post-forward)"
    )

    # Hook into training step to finalize relayout
    # This integrates with Primus's pipeline schedule
    import primus.backends.megatron.core.models.common.model_chunk_schedule_plan as schedule_mod

    if hasattr(schedule_mod, "execute_overlapped_1f1b"):
        original_execute = schedule_mod.execute_overlapped_1f1b

        def _execute_with_relayout_finalize(f_layer, b_layer, f_input=None, b_grad=None, is_last_layer_in_bwd=False):
            """Execute overlapped 1F1B with relayout finalization at backward boundary."""
            result = original_execute(f_layer, b_layer, f_input, b_grad, is_last_layer_in_bwd)

            # Finalize any pending relayout at the end of backward
            if is_last_layer_in_bwd:
                fsep_state = get_fsep_state()
                if fsep_state is not None and fsep_state.relayout_executor is not None:
                    fsep_state.relayout_executor.finalize_relayout()

            return result

        schedule_mod.execute_overlapped_1f1b = _execute_with_relayout_finalize
        log_rank_0(
            "[Patch:megatron.moe.fsep_relayout_hook] Relayout finalize hooked into "
            "execute_overlapped_1f1b (at backward boundary)"
        )


@register_patch(
    "megatron.moe.fsep_executor_attach",
    backend="megatron",
    phase="before_train",
    description="Attach FSEPRelayoutExecutor to FSEPState after model is built",
    condition=_is_full_fsep_enabled,
)
def patch_fsep_executor_attach(ctx: PatchContext):
    """
    Attach FSEPRelayoutExecutor to FSEPState.

    This runs after the model is built, so we can access the Expert module
    to create the executor with a reference to the weight tensors.

    We hook into the first MoE layer's __init__ to lazily attach the executor.
    """
    args = get_args(ctx)

    from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
        get_fsep_state,
    )
    from primus.backends.megatron.core.transformer.moe.relayout_executor import (
        FSEPRelayoutExecutor,
    )

    # Hook into GroupedMLP (or FSEPGroupedMLP) to capture expert module
    import megatron.core.transformer.moe.experts as experts_mod

    OriginalGroupedMLP = experts_mod.GroupedMLP
    original_init = OriginalGroupedMLP.__init__

    _executor_attached = [False]  # Use list to allow mutation in closure

    def _init_with_executor(self, *args_inner, **kwargs_inner):
        original_init(self, *args_inner, **kwargs_inner)

        if not _executor_attached[0]:
            fsep_state = get_fsep_state()
            if fsep_state is not None and fsep_state.relayout_executor is None:
                try:
                    executor = FSEPRelayoutExecutor(
                        experts=self,
                        fsep_state=fsep_state,
                        ep_group=fsep_state.ep_group,
                        num_experts=fsep_state.num_experts,
                    )
                    fsep_state.relayout_executor = executor
                    _executor_attached[0] = True
                    log_rank_0(
                        "[Patch:megatron.moe.fsep_executor_attach] "
                        "FSEPRelayoutExecutor attached to FSEPState"
                    )
                except Exception as e:
                    log_rank_0(
                        f"[Patch:megatron.moe.fsep_executor_attach] "
                        f"Failed to attach executor: {e}"
                    )

    OriginalGroupedMLP.__init__ = _init_with_executor

    log_rank_0(
        "[Patch:megatron.moe.fsep_executor_attach] "
        "Executor attach hook registered on GroupedMLP.__init__"
    )
