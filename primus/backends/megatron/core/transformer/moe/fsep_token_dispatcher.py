###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP AlltoAll Token Dispatcher.

Implements the full FSEP dispatch with smart routing:
  1. Collect global token-per-expert statistics
  2. Compute smart routing: redistribute tokens across expert replicas (load balance)
  3. A2A dispatch using new routing map (tokens go to slots, not fixed owners)
  4. Expert GEMM + ReduceScatter (via FSEPGroupedMLP)
  5. A2A gather to return results to original token owners

Key difference from traditional EP A2A:
  - Traditional EP: token for Expert e → always goes to GPU owning e
  - FSEP: token for Expert e → goes to whichever GPU has capacity for e's replica
           (load-balanced across all S GPUs holding e's shards)

Reference: laer_moe/galvatron/core/runtime/moe/smart_routing.py
  ::MoEAlltoAllSmartTokenDispatcher
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel import (
    all_to_all,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_utils import (
    permute,
    unpermute,
    sort_chunks_by_idxs,
)

from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
    FSEPState,
    get_fsep_state,
)
from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
    compute_smart_routing,
    smart_routing_map,
)


def _log(msg):
    try:
        from primus.modules.module_utils import log_rank_0
        log_rank_0(msg)
    except Exception:
        try:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(msg)
        except Exception:
            print(msg)


class FSEPAlltoAllTokenDispatcher:
    """
    AlltoAll-based token dispatcher for full FSEP.

    Replaces the traditional EP A2A with smart-routing-aware dispatch:
    tokens are redistributed across expert replicas to balance GPU load.

    The workflow follows the reference implementation in
    laer_moe/galvatron/core/runtime/moe/smart_routing.py::MoEAlltoAllSmartTokenDispatcher:

      token_permutation():
        1. Collect global token stats → compute smart routing allocation
        2. Remap routing_map from [T, N_E] to [T, N_slots] (via compute_smart_routing)
        3. Permute tokens based on new_routing_map
        4. A2A dispatch with computed splits
        5. Sort received tokens by local expert

      token_unpermutation():
        1. Unsort tokens by local expert
        2. A2A reverse (tokens go back to original GPU)
        3. Unpermute tokens and weight-sum by probs
    """

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TransformerConfig,
        fsep_state: FSEPState,
        ep_group: Optional[dist.ProcessGroup] = None,
    ):
        self.config = config
        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.fsep_state = fsep_state

        # Process groups
        self.ep_group = ep_group or parallel_state.get_expert_model_parallel_group()
        self.ep_size = dist.get_world_size(group=self.ep_group)
        self.ep_rank = dist.get_rank(group=self.ep_group)

        self.num_experts = config.num_moe_experts

        # N_slots = ep_size * num_local_experts
        self.num_slots = self.ep_size * self.num_local_experts

        # Cached metadata
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape = None
        self.hidden_shape_before_permute = None
        self.probs = None
        self.routing_map = None
        self.new_routing_map = None
        self.new_probs = None
        self.reversed_local_input_permutation_mapping = None
        self.num_out_tokens = 0
        self.num_global_tokens_per_local_expert = None

        # Pre-compute sort indices for local expert reordering
        expert_capacity = self.num_local_experts * self.ep_size
        input_chunk_idxs = torch.arange(expert_capacity)
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

        _log(
            f"[FSEP] FSEPAlltoAllTokenDispatcher initialized: "
            f"num_experts={self.num_experts}, EP={self.ep_size}, "
            f"num_local_experts={self.num_local_experts}, "
            f"sharding_degree={fsep_state.sharding_degree}"
        )

    def preprocess(
        self,
        routing_map: torch.Tensor,  # [T, N_E] bool
        probs: torch.Tensor,        # [T, N_E] float
    ) -> torch.Tensor:
        """
        Preprocess: compute smart routing and A2A metadata.

        Steps:
          1. Gather global token distribution across EP ranks
          2. Apply smart routing (Algorithm 3: Lite Routing) to get per-slot allocation
          3. Remap routing_map → new_routing_map using capacity-aware assignment
          4. Compute A2A input_splits and output_splits

        Returns:
            tokens_per_local_expert: [num_local_experts] token counts
        """
        fsep = self.fsep_state
        T, N_E = routing_map.shape

        # ── Step 1: Global token distribution ──
        num_local_tokens_per_expert = routing_map.sum(dim=0).int()  # [N_E]

        # AllGather → [ep_size, N_E]
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
        num_global_tokens_per_expert = (
            gather_from_sequence_parallel_region(
                num_local_tokens_per_expert, group=self.ep_group
            )
            .reshape(self.ep_size, N_E)
        )
        # num_global_tokens_per_expert: [ep_size, N_E]

        # ── Step 2: Smart routing allocation (Algorithm 3) ──
        # Expand to [1, ep_size, N_E] for 3D API
        tokens_3d = num_global_tokens_per_expert.unsqueeze(0)
        slot_allocation = smart_routing_map(
            tokens_3d,
            fsep.global_expert_locations,
            self.num_local_experts,
            gpus_per_node=fsep.gpus_per_node,
        )
        # slot_allocation: [1, ep_size, N_slots]

        # ── Step 3: Remap routing → new routing ──
        tokens_per_slot_local = slot_allocation[0, self.ep_rank, :].long()

        new_routing_map, new_probs = compute_smart_routing(
            routing_map,
            probs,
            fsep.global_expert_locations,
            fsep.inverse_expert_map,
            tokens_per_slot_local,
        )
        self.new_routing_map = new_routing_map
        self.new_probs = new_probs
        self.num_out_tokens = routing_map.size(0) * self.config.moe_router_topk

        # ── Step 4: Compute A2A splits ──
        # input_splits[r] = tokens sent from this rank to rank r
        self.input_splits = num_global_tokens_per_expert[self.ep_rank].reshape(
            self.ep_size, self.num_local_experts
        ).sum(dim=1)

        # output_splits[r] = tokens received by this rank from rank r
        # = num_global_tokens_per_expert[r, local_expert_indices].sum()
        local_start = self.local_expert_indices[0]
        local_end = self.local_expert_indices[-1] + 1
        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, local_start:local_end
        ].contiguous()
        self.output_splits = num_global_tokens_per_local_expert.sum(dim=1)

        # Save for sort in token_permutation
        self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert

        # tokens_per_local_expert = sum over all source ranks
        tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        # Feed load to planner
        if fsep.load_planner is not None:
            global_load = num_global_tokens_per_expert.sum(dim=0).float()
            fsep.load_planner.update(global_load, num_global_tokens_per_expert)

        return tokens_per_local_expert

    def token_permutation(
        self,
        hidden_states: torch.Tensor,  # [B, S, H] or [T, H]
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to expert slots using smart-routed A2A.

        Follows the reference: MoEAlltoAllSmartTokenDispatcher.token_permutation

        Steps:
          1. Preprocess → compute smart routing and splits
          2. Permutation 1: group tokens by target EP rank
          3. EP AlltoAll: send tokens to destination GPUs
          4. Permutation 2: sort received tokens by local expert

        Returns:
            global_input_tokens: [T_local, H] tokens grouped by local expert
            tokens_per_expert: [num_local_experts] token counts
        """
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        self.routing_map = routing_map
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Preprocess
        tokens_per_expert = self.preprocess(routing_map, probs)

        # Permutation 1: permute tokens for A2A
        self.hidden_shape_before_permute = hidden_states.shape
        permuted_local, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            self.new_routing_map,
            num_out_tokens=self.num_out_tokens,
        )

        # Move splits to CPU for A2A
        input_splits_cpu = self.input_splits.cpu().numpy()
        output_splits_cpu = self.output_splits.cpu().numpy()

        # EP AlltoAll
        global_input_tokens = all_to_all(
            self.ep_group, permuted_local, output_splits_cpu, input_splits_cpu,
        )

        # Permutation 2: sort by local expert
        if self.num_local_experts > 1:
            global_input_tokens = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
            )

        tokens_per_expert = tokens_per_expert.cpu()
        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,  # [T_local, H] expert outputs
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the FSEP dispatch: gather expert outputs back to original tokens.

        Steps:
          1. Unpermutation 2: unsort by local expert
          2. EP AlltoAll reverse: send outputs back to source GPUs
          3. Unpermutation 1: restore original token order and weight by probs

        Returns:
            output: [B, S, H] final output
        """
        assert bias is None, "Bias not supported in FSEPAlltoAllTokenDispatcher"

        # Unpermutation 2: unsort by local expert
        if self.num_local_experts > 1:
            hidden_states = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
            )

        # Reverse A2A
        input_splits_cpu = self.input_splits.cpu().numpy()
        output_splits_cpu = self.output_splits.cpu().numpy()

        permuted_local = all_to_all(
            self.ep_group, hidden_states, input_splits_cpu, output_splits_cpu,
        )

        # Unpermutation 1: restore original order and apply probs
        output = unpermute(
            permuted_local,
            self.reversed_local_input_permutation_mapping,
            restore_shape=self.hidden_shape_before_permute,
            probs=self.new_probs,
            routing_map=self.new_routing_map,
        )

        output = output.view(self.hidden_shape)
        return output, None
