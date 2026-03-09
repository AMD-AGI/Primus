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
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
    FSEPState,
    get_fsep_state,
)
from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
    compute_smart_routing,
    smart_routing_map,
)
from primus.modules.module_utils import log_rank_0


class FSEPAlltoAllTokenDispatcher:
    """
    AlltoAll-based token dispatcher for full FSEP.

    Replaces the traditional EP A2A with smart-routing-aware dispatch:
    tokens are redistributed across expert replicas to balance GPU load.

    Workflow:
      token_permutation():
        1. Collect global token stats → compute smart routing allocation
        2. Remap routing_map from [T, N_E] to [T, N_slots] (via compute_smart_routing)
        3. A2A dispatch with new routing map
        4. Return permuted tokens for Expert GEMM

      token_unpermutation():
        1. Reverse A2A (tokens go back to original GPU)
        2. Unpermute and weight-sum by probs
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

        # Cached metadata from preprocess
        self.input_splits = None
        self.output_splits = None
        self.hidden_shape = None
        self.hidden_shape_before_permute = None
        self.probs = None
        self.new_routing_map = None
        self.new_probs = None
        self.reversed_local_input_permutation_mapping = None

        log_rank_0(
            f"[FSEP] FSEPAlltoAllTokenDispatcher initialized: "
            f"num_experts={self.num_experts}, EP={self.ep_size}, "
            f"sharding_degree={fsep_state.sharding_degree}"
        )

    def _compute_splits(
        self,
        new_routing_map: torch.Tensor,  # [T, N_slots]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute input_splits and output_splits for A2A communication.

        In FSEP, N_slots == N_E (Mode B), so the split computation is similar
        to traditional EP but uses the NEW routing map (smart-routed).

        Returns:
            input_splits: [ep_size] tokens sent to each EP rank
            output_splits: [ep_size] tokens received from each EP rank
        """
        T = new_routing_map.shape[0]
        N_slots = new_routing_map.shape[1]

        # Count tokens going to each slot → aggregate to EP rank
        tokens_per_slot = new_routing_map.long().sum(dim=0)  # [N_slots]

        # Each slot maps to an EP rank: slot s → rank = s // num_local_experts
        num_local_experts = self.num_experts // self.ep_size
        tokens_per_rank = torch.zeros(self.ep_size, dtype=torch.long, device=tokens_per_slot.device)
        for s in range(N_slots):
            rank = s // num_local_experts
            tokens_per_rank[rank] += tokens_per_slot[s]

        input_splits = tokens_per_rank.cpu().numpy()

        # Gather output splits (how many tokens we receive from each rank)
        input_splits_tensor = tokens_per_rank.unsqueeze(0)  # [1, ep_size]
        gathered = torch.zeros(self.ep_size, self.ep_size, dtype=torch.long,
                               device=tokens_per_slot.device)
        dist.all_gather_into_tensor(
            gathered, input_splits_tensor.expand(self.ep_size, -1),
            group=self.ep_group
        )
        output_splits_tensor = gathered[:, self.ep_rank]
        output_splits = output_splits_tensor.cpu().numpy()

        return input_splits, output_splits

    def preprocess(
        self,
        routing_map: torch.Tensor,  # [T, N_E] bool
        probs: torch.Tensor,        # [T, N_E] float
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Preprocess routing map: compute smart routing allocation.

        Steps:
          1. Gather global token-per-expert distribution
          2. Compute smart routing: how many tokens each slot should receive
          3. Remap routing_map → new_routing_map via compute_smart_routing
          4. Compute A2A splits

        Returns:
            new_routing_map: [T, N_slots] bool
            new_probs: [T, N_slots] float
            num_out_tokens: total tokens to dispatch
        """
        T, N_E = routing_map.shape
        fsep = self.fsep_state

        # ── Step 1: Global token-per-expert stats ──
        num_local_tokens_per_expert = routing_map.long().sum(dim=0)  # [N_E]

        # Gather from all EP ranks → global distribution
        num_global_tokens_per_expert = num_local_tokens_per_expert.clone()
        dist.all_reduce(
            num_global_tokens_per_expert,
            group=self.ep_group,
        )
        # num_global_tokens_per_expert: [N_E] total tokens per expert across all ranks

        # ── Step 2: Smart routing allocation ──
        tokens_per_slot = smart_routing_map(
            num_global_tokens_per_expert,
            fsep.global_expert_locations,
            self.num_local_experts,
        )
        # tokens_per_slot[slot] = how many tokens (from this rank) should go to slot

        # Scale down: we care about tokens FROM THIS RANK only
        # (smart_routing_map currently returns global counts; divide by ep_size)
        tokens_per_slot_local = (tokens_per_slot // self.ep_size).long()

        # ── Step 3: Remap routing → new_routing_map ──
        new_routing_map, new_probs = compute_smart_routing(
            routing_map,
            probs,
            fsep.global_expert_locations,
            fsep.inverse_expert_map,
            tokens_per_slot_local,
        )

        # ── Step 4: Compute A2A splits ──
        self.input_splits, self.output_splits = self._compute_splits(new_routing_map)

        num_out_tokens = int(new_routing_map.long().sum().item())

        return new_routing_map, new_probs, num_out_tokens

    def token_permutation(
        self,
        hidden_states: torch.Tensor,  # [B, S, H] or [T, H]
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dispatch tokens to expert slots using smart-routed A2A.

        Args:
            hidden_states: Input tokens [B, S, H] or [T, H]
            probs: Routing probabilities [T, N_E]
            routing_map: Token-expert assignment [T, N_E] bool

        Returns:
            permuted_tokens: Tokens grouped by local expert [T_local, H]
            tokens_per_expert: [num_local_experts] token counts
        """
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])  # [T, H]

        # ── Preprocess: compute smart routing ──
        new_routing_map, new_probs, num_out_tokens = self.preprocess(
            routing_map, probs
        )
        self.new_routing_map = new_routing_map
        self.new_probs = new_probs

        # ── Permutation 1: group tokens by target EP rank ──
        self.hidden_shape_before_permute = hidden_states.shape
        T, H = hidden_states.shape
        N_slots = new_routing_map.shape[1]

        # Permute: each token that goes to a slot is ordered by slot index
        # (similar to traditional EP permute but using new_routing_map)
        permutation_indices = []
        slot_assignments = []  # which slot each permuted token goes to

        for slot in range(N_slots):
            token_ids = new_routing_map[:, slot].nonzero(as_tuple=True)[0]
            for tid in token_ids:
                permutation_indices.append(tid.item())
                slot_assignments.append(slot)

        if len(permutation_indices) > 0:
            perm_idx = torch.tensor(permutation_indices, dtype=torch.long,
                                    device=hidden_states.device)
            permuted = hidden_states[perm_idx]  # [num_out_tokens, H]
        else:
            permuted = hidden_states.new_zeros((0, H))

        self.slot_assignments = slot_assignments
        self.permutation_indices = permutation_indices

        # ── A2A dispatch ──
        torch.cuda.current_stream().synchronize()
        global_input_tokens = all_to_all(
            self.ep_group,
            permuted,
            self.output_splits,
            self.input_splits,
        )

        # ── Sort by local expert ──
        # global_input_tokens contains tokens for our local experts,
        # sorted by source EP rank. Re-sort by local expert index.
        num_local = self.num_local_experts
        num_global_tokens_per_local_expert = torch.zeros(
            num_local, dtype=torch.long, device=hidden_states.device
        )

        # Count tokens received per local expert
        # (from output_splits, we know how many tokens came from each source rank;
        # each source rank sent tokens grouped by our local experts)
        # Simple approach: count using the routing info from all_gather
        local_start = self.ep_rank * num_local
        local_end = local_start + num_local

        # Get global token counts per expert from preprocess
        # Use a gather to get full picture
        num_local_tokens_per_expert = self.new_routing_map.long().sum(dim=0)  # [N_slots]
        local_slots = list(range(local_start, local_end))
        tokens_per_expert_local = torch.zeros(
            num_local, dtype=torch.long, device=hidden_states.device
        )
        for i, slot in enumerate(local_slots):
            if slot < N_slots:
                tokens_per_expert_local[i] = num_local_tokens_per_expert[slot]

        # Gather from all ranks to get complete picture
        gathered = torch.zeros(
            self.ep_size * num_local, dtype=torch.long, device=hidden_states.device
        )
        dist.all_gather_into_tensor(
            gathered.view(self.ep_size, num_local),
            tokens_per_expert_local.unsqueeze(0),
            group=self.ep_group,
        )
        # gathered[rank, local_expert] = tokens from rank going to local_expert
        tokens_per_expert = gathered.view(self.ep_size, num_local).sum(dim=0).cpu()

        return global_input_tokens, tokens_per_expert

    def token_unpermutation(
        self,
        hidden_states: torch.Tensor,  # [T_local, H] expert outputs
        bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reverse the FSEP dispatch: gather expert outputs back to original tokens.

        Note: In full FSEP, the expert outputs from FSEPGroupedMLP are already
        [T/S, H] shards (after ReduceScatter). The AllGather in
        PrimusTurboDeepEPTokenDispatcher.combine_preprocess restores [T, H].
        Here we use a simpler AlltoAll-based unpermutation.

        Returns:
            output: [B, S, H] final output
        """
        assert bias is None, "Bias not supported in FSEPAlltoAllTokenDispatcher"

        # ── Reverse A2A: send expert outputs back to token origins ──
        permutated_local = all_to_all(
            self.ep_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
        )

        # ── Unpermute: scatter results back to original token positions ──
        T, H = self.hidden_shape[0] * (self.hidden_shape[1] if len(self.hidden_shape) > 2 else 1), self.hidden_shape[-1]
        T = 1
        for d in self.hidden_shape[:-1]:
            T *= d

        output = torch.zeros((T, H), dtype=hidden_states.dtype,
                             device=hidden_states.device)

        if len(self.permutation_indices) > 0:
            # new_probs[token_id, slot] = routing probability
            N_slots = self.new_routing_map.shape[1]
            # Map permuted token index back to original token id
            for perm_idx, (orig_tid, slot) in enumerate(
                zip(self.permutation_indices, self.slot_assignments)
            ):
                if perm_idx < permutated_local.shape[0]:
                    prob = self.new_probs[orig_tid, slot].item()
                    output[orig_tid] += prob * permutated_local[perm_idx]

        output = output.view(self.hidden_shape)
        return output, None
