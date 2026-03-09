###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP Expert Re-layout Executor: async parameter migration.

Implements the Expert Re-layout component of LAER-MoE:
  - During backward pass of step T, asynchronously migrate Expert parameter
    shards from old GPU layout to new GPU layout
  - At start of step T+1, atomically switch to new layout
  - Uses double buffer + NCCL point-to-point for zero-downtime migration

Key design:
  - Migration runs on a dedicated CUDA stream (relayout_stream)
  - Doesn't block the main forward/backward streams
  - Double buffer: old and new param shards coexist during migration
  - After migration, FSEPState is updated atomically

Reference: laer_moe/galvatron/core/runtime/hybrid_parallel_model.py
  (FSDP parameter migration logic)
  laer_moe/galvatron/core/runtime/moe/fused_kernel.py
  ::triton_all_to_all_forward (Expert weights A2A)
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.experts import GroupedMLP

from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
    FSEPState,
    get_fsep_state,
)
from primus.modules.module_utils import log_rank_0


class FSEPRelayoutExecutor:
    """
    Asynchronous Expert parameter migration for dynamic FSEP Re-layout.

    Usage:
      executor = FSEPRelayoutExecutor(experts, fsep_state, ep_group)

      # Triggered during backward pass:
      executor.schedule_relayout(placement_plan)

      # Called at start of next step (after backward completes):
      executor.finalize_relayout()  # blocks until migration done, switches state
    """

    def __init__(
        self,
        experts: GroupedMLP,  # The Expert module (contains weight1, weight2)
        fsep_state: FSEPState,
        ep_group: dist.ProcessGroup,
        num_experts: int,
    ):
        self.experts = experts
        self.fsep_state = fsep_state
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(group=ep_group)
        self.ep_rank = dist.get_rank(group=ep_group)
        self.num_experts = num_experts
        self.num_local_experts = num_experts // self.ep_size

        # Dedicated stream for async migration
        self.relayout_stream = torch.cuda.Stream()

        # Double buffer: new param shards received during migration
        self.new_weight1_buffer: Optional[torch.Tensor] = None
        self.new_weight2_buffer: Optional[torch.Tensor] = None

        # Pending placement (set by schedule_relayout, cleared by finalize_relayout)
        self.pending_plan: Optional[Dict] = None
        self.migration_scheduled = False

    def schedule_relayout(self, plan: Dict) -> None:
        """
        Schedule asynchronous parameter migration.

        Should be called during backward pass (to overlap with computation).
        Uses non-blocking NCCL point-to-point communication.

        Args:
            plan: Dict from FSEPLoadPlanner.compute_new_placement() containing:
                  - new_global_expert_locations [N_E, max_S]
                  - new_inverse_expert_map [N_E]
        """
        if self.migration_scheduled:
            log_rank_0("[FSEP RelayoutExecutor] Previous migration still pending, skip")
            return

        self.pending_plan = plan
        new_locations = plan["new_global_expert_locations"]

        log_rank_0(
            f"[FSEP RelayoutExecutor] Scheduling relayout on stream "
            f"(expert params migration)"
        )

        with torch.cuda.stream(self.relayout_stream):
            self._async_migrate_params(new_locations)

        self.migration_scheduled = True

    def _async_migrate_params(self, new_locations: torch.Tensor) -> None:
        """
        Asynchronously migrate Expert parameter shards using All-to-All.

        For each expert e:
          - Compute which GPU currently holds its shard (old layout)
          - Compute which GPU should hold its shard (new layout)
          - If different: initiate async send/recv

        Uses NCCL groupStart/groupEnd for efficient batched P2P.

        Reference: laer_moe/galvatron/core/runtime/moe/fused_kernel.py
          ::triton_all_to_all_forward
        """
        device = torch.cuda.current_device()
        old_locations = self.fsep_state.global_expert_locations  # [N_E, max_S]
        N_E = self.num_experts
        ep_size = self.ep_size
        ep_rank = self.ep_rank

        # Build migration plan: {(src_rank, dst_rank): [expert_ids]}
        # For each expert, if its new location differs from old location
        migrations = []  # (src_rank, dst_rank, expert_id)

        for e in range(N_E):
            old_slot = old_locations[e, 0].item()
            new_slot = new_locations[e, 0].item()

            if old_slot < 0 or new_slot < 0:
                continue

            old_rank = old_slot // self.num_local_experts
            new_rank = new_slot // self.num_local_experts

            if old_rank != new_rank:
                migrations.append((old_rank, new_rank, e))

        if not migrations:
            log_rank_0("[FSEP RelayoutExecutor] No parameter migrations needed")
            return

        log_rank_0(f"[FSEP RelayoutExecutor] {len(migrations)} parameter migrations")

        # Allocate receive buffers
        w1_shape = self.experts.weight1.shape  # [H, F/S per local expert]
        w2_shape = self.experts.weight2.shape  # [F/S, H per local expert]

        # Count incoming experts
        incoming = [(src, dst, e) for src, dst, e in migrations if dst == ep_rank]
        if incoming:
            # Pre-allocate buffers for incoming expert shards
            # (simplified: one buffer per incoming expert)
            self.incoming_experts = incoming

        # ── Phase 1: Non-blocking sends ──
        for src_rank, dst_rank, expert_id in migrations:
            if ep_rank == src_rank:
                # Get local expert index
                local_idx = expert_id - src_rank * self.num_local_experts

                if 0 <= local_idx < self.num_local_experts:
                    # Extract this expert's parameters
                    # weight1 is [H, F/S * num_local_experts] → extract expert's slice
                    F_per_expert = w1_shape[1] // self.num_local_experts
                    H = w1_shape[0]

                    w1_slice = self.experts.weight1[
                        :, local_idx * F_per_expert: (local_idx + 1) * F_per_expert
                    ].contiguous()
                    w2_slice = self.experts.weight2[
                        local_idx * F_per_expert: (local_idx + 1) * F_per_expert, :
                    ].contiguous()

                    # Non-blocking send to destination rank
                    dist.isend(w1_slice, dst=dst_rank, group=self.ep_group,
                               tag=expert_id * 2)
                    dist.isend(w2_slice, dst=dst_rank, group=self.ep_group,
                               tag=expert_id * 2 + 1)

        # ── Phase 2: Non-blocking receives ──
        recv_handles = []
        recv_buffers = {}

        for src_rank, dst_rank, expert_id in migrations:
            if ep_rank == dst_rank:
                F_per_expert = w1_shape[1] // self.num_local_experts
                H = w1_shape[0]

                recv_w1 = torch.empty(H, F_per_expert, dtype=self.experts.weight1.dtype,
                                      device=device)
                recv_w2 = torch.empty(F_per_expert, H, dtype=self.experts.weight2.dtype,
                                      device=device)

                h1 = dist.irecv(recv_w1, src=src_rank, group=self.ep_group,
                                 tag=expert_id * 2)
                h2 = dist.irecv(recv_w2, src=src_rank, group=self.ep_group,
                                 tag=expert_id * 2 + 1)

                recv_handles.append((h1, h2))
                recv_buffers[expert_id] = (recv_w1, recv_w2)

        self.recv_handles = recv_handles
        self.recv_buffers = recv_buffers

    def finalize_relayout(self) -> bool:
        """
        Wait for async migration to complete and switch to new layout.

        Should be called at the start of the next training step
        (after backward pass completes).

        Returns:
            True if relayout was applied, False if nothing was pending.
        """
        if not self.migration_scheduled:
            return False

        # Wait for relayout stream to complete
        torch.cuda.current_stream().wait_stream(self.relayout_stream)

        # Wait for all receives to complete
        for h1, h2 in getattr(self, 'recv_handles', []):
            h1.wait()
            h2.wait()

        # Apply received parameters
        for expert_id, (recv_w1, recv_w2) in getattr(self, 'recv_buffers', {}).items():
            new_local_idx = expert_id - self.ep_rank * self.num_local_experts
            if 0 <= new_local_idx < self.num_local_experts:
                F_per_expert = self.experts.weight1.shape[1] // self.num_local_experts
                # Update parameters in-place
                with torch.no_grad():
                    self.experts.weight1.data[
                        :, new_local_idx * F_per_expert: (new_local_idx + 1) * F_per_expert
                    ].copy_(recv_w1)
                    self.experts.weight2.data[
                        new_local_idx * F_per_expert: (new_local_idx + 1) * F_per_expert, :
                    ].copy_(recv_w2)

        # Atomically update FSEPState
        if self.pending_plan is not None:
            self.fsep_state.update_placement(
                self.pending_plan["new_global_expert_locations"],
                self.pending_plan["new_inverse_expert_map"],
            )
            log_rank_0(
                f"[FSEP RelayoutExecutor] Relayout finalized: "
                f"new placement active at next forward pass"
            )

        # Clean up
        self.pending_plan = None
        self.migration_scheduled = False
        self.recv_handles = []
        self.recv_buffers = {}

        return True
