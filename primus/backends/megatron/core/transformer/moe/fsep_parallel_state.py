###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP (Fully Sharded Expert Parallel) Global State Management.

Maintains the expert placement state required for full FSEP:
  - global_expert_locations: [N_E, max_S] - for each expert, which GPU slots hold its shards
    (-1 means invalid / not used)
  - inverse_expert_map: [N_slots] - for each slot, which original expert it maps to
  - expert_capacity: [N_slots] - current token capacity available at each slot

This state is shared across all MoE layers and updated by the Load Planner
during dynamic Re-layout.

Reference: laer_moe/galvatron/core/runtime/moe/smart_routing.py
  - global_expert_indices: [N_E, S]
  - global_expert_locations: same
  - inverse_expert_map: [N_slots]
"""

from typing import Optional
import torch
import torch.distributed as dist

# Module-level singleton state
_FSEP_STATE: Optional["FSEPState"] = None


def get_fsep_state() -> Optional["FSEPState"]:
    """Return the current FSEP state, or None if not initialized."""
    return _FSEP_STATE


def set_fsep_state(state: "FSEPState") -> None:
    """Set the global FSEP state."""
    global _FSEP_STATE
    _FSEP_STATE = state


def init_fsep_state(
    num_experts: int,
    ep_size: int,
    sharding_degree: int,
    ep_group: dist.ProcessGroup,
) -> "FSEPState":
    """
    Initialize the FSEP state with uniform expert placement.

    For static FSEP with S == EP (Mode B):
      - Every GPU holds 1/S shard of ALL N_E experts
      - global_expert_locations[e, s] = s  (slot s is on GPU s)
      - Each GPU slot uniquely maps to one GPU
      - N_slots = N_E (each expert has exactly one "canonical" slot per GPU,
        but all S GPUs participate in computing it)

    For dynamic FSEP (after Re-layout):
      - global_expert_locations[e, :] may point to different sets of GPUs
      - Updated by FSEPLoadPlanner + FSEPRelayoutExecutor

    Args:
        num_experts: Total number of experts (N_E)
        ep_size: Expert parallel group size
        sharding_degree: FSEP sharding degree S (S == EP for Mode B)
        ep_group: The expert parallel process group
    """
    global _FSEP_STATE

    state = FSEPState(
        num_experts=num_experts,
        ep_size=ep_size,
        sharding_degree=sharding_degree,
        ep_group=ep_group,
    )
    _FSEP_STATE = state
    return state


class FSEPState:
    """
    Mutable FSEP expert placement state.

    In the paper's LAER-MoE:
      global_expert_locations[e, j] = physical slot index for the j-th replica of expert e
      inverse_expert_map[slot] = original expert index for this slot
      expert_capacity[slot] = max tokens this slot can receive per dispatch

    Slot indexing for uniform S=EP placement:
      slot = gpu_rank * num_local_experts + local_expert_idx
      (where num_local_experts = num_experts / ep_size)

    In Mode B (S==EP, ETP=1), there is a 1-to-1 mapping between slots and
    (GPU, local_expert) pairs, giving N_slots = N_E.
    """

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        sharding_degree: int,
        ep_group: dist.ProcessGroup,
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.sharding_degree = sharding_degree  # S
        self.ep_group = ep_group
        self.ep_rank = dist.get_rank(group=ep_group)

        # Number of local experts per GPU (traditional EP assignment)
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        )
        self.num_local_experts = num_experts // ep_size

        # Total slots = N_E (Mode B uniform: each expert has one slot per GPU,
        # but all GPUs participate via ReduceScatter)
        # For Mode B with S=EP, the "slots" are identified with the original
        # expert indices (no replication in terms of routing slots).
        self.num_slots = num_experts  # N_E in traditional EP view

        # ── global_expert_locations[e, j] ──────────────────────────────
        # [N_E, max_S] tensor, GPU-side.
        # For uniform S=EP placement: expert e → slot = e (traditional EP owner)
        # -1 means unused (when S < max_S).
        device = torch.cuda.current_device()
        max_S = sharding_degree

        # Start with traditional EP placement (1 replica per expert = S=1 semantics)
        # For full FSEP (S=EP), we expand: expert e can be served by any of the
        # EP GPUs, but we start conservative with S=1 replica (traditional owner).
        # The Load Planner will update this to S>1 as needed.
        self.global_expert_locations = torch.full(
            (num_experts, max_S),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        # Default: expert e → its traditional owner slot (slot = e)
        for e in range(num_experts):
            self.global_expert_locations[e, 0] = e  # one replica at traditional owner

        # ── inverse_expert_map[slot] ───────────────────────────────────
        # [N_slots] tensor: for each slot, which original expert it maps to.
        # In traditional EP: slot e → expert e (trivial identity mapping)
        self.inverse_expert_map = torch.arange(
            num_experts, dtype=torch.long, device=device
        )

        # ── expert_capacity[slot] ──────────────────────────────────────
        # How many tokens each slot can handle per dispatch step.
        # Initialized to "unlimited" (large number); Load Planner refines this.
        self.expert_capacity = torch.full(
            (num_experts,),
            fill_value=1_000_000,
            dtype=torch.long,
            device=device,
        )

    def expand_to_full_fsep(self) -> None:
        """
        Expand placement to full FSEP: each expert replicated on all S=EP GPUs.

        After this call:
          global_expert_locations[e, s] = e  (for s in range(S))
          → each expert can be served by ANY of the S GPUs

        This enables the smart routing to redistribute tokens across all GPUs.
        The Load Planner can then fine-tune capacity allocation.
        """
        S = self.sharding_degree
        device = self.global_expert_locations.device

        # Uniform full replication: all S entries point to the same expert slot
        # (the slot identity equals expert index in Mode B)
        for e in range(self.num_experts):
            for s in range(S):
                self.global_expert_locations[e, s] = e

        # Capacity: split evenly across S replicas initially
        # (will be refined by smart_routing_map based on actual load)
        # Leave capacity as large number for now; smart routing handles allocation

    def update_placement(
        self,
        new_global_expert_locations: torch.Tensor,
        new_inverse_expert_map: torch.Tensor,
    ) -> None:
        """
        Atomically update the placement state after a Re-layout.

        Called by FSEPRelayoutExecutor after parameter migration completes.
        """
        self.global_expert_locations.copy_(new_global_expert_locations)
        self.inverse_expert_map.copy_(new_inverse_expert_map)

    def get_local_expert_indices(self):
        """
        Return the list of expert indices that this EP rank owns.
        Same as traditional EP: [rank * num_local, (rank+1) * num_local)
        """
        start = self.ep_rank * self.num_local_experts
        return list(range(start, start + self.num_local_experts))
