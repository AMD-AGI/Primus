###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP Load Planner: monitor expert load and plan optimal placement.

Implements the Load-Adaptive component of LAER-MoE:
  1. Monitor per-expert token counts (via router statistics)
  2. Detect load imbalance (max/avg ratio threshold)
  3. Run Algorithm 4 (Replica Allocation) to determine #replicas per expert
  4. Run Greedy Placement to assign expert replicas to GPUs
  5. Signal FSEPRelayoutExecutor to migrate parameters

The planner runs entirely in Python (no C++ dependency required).
For maximum performance, the C++ greedy_balancer from third_party/laer_moe/csrc
can be compiled and used as a drop-in replacement.

Reference: laer_moe/galvatron/core/runtime/moe/prefetch/solver.py::MoEOptimizer
           laer_moe/csrc/greedy_balancer.cpp
"""

from typing import Dict, List, Optional, Tuple
import heapq

import torch
import torch.distributed as dist

from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
    FSEPState,
    get_fsep_state,
)


def _log(msg):
    """Safe log that works with or without initialized Primus logger."""
    try:
        from primus.modules.module_utils import log_rank_0
        log_rank_0(msg)
    except Exception:
        try:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(msg)
        except Exception:
            print(msg)


# ---------------------------------------------------------------------------
# Algorithm 4: Replica Allocation (Priority Queue)
# Reference: laer_moe/csrc/greedy_balancer.cpp::allocate_expert_replicas_precise
# ---------------------------------------------------------------------------

def allocate_expert_replicas(
    expert_loads: torch.Tensor,    # [N_E] total tokens per expert
    n_device: int,
    capacity_per_device: int,
    gpus_per_node: int = 8,
) -> List[int]:
    """
    Algorithm 4 from LAER-MoE: Replica Allocation using a priority queue.

    Determines how many replicas each expert should have, proportional to
    its load. Uses a max-heap keyed by (load / #replicas) — the expert
    with the highest average-load-per-replica gets the next replica slot.

    Ports: laer_moe/csrc/greedy_balancer.cpp::allocate_expert_replicas_precise

    Args:
        expert_loads: [N_E] tensor of total token counts per expert.
        n_device: Total number of GPU devices in EP group.
        capacity_per_device: Max expert slots per device (num_local_experts).
        gpus_per_node: GPUs per physical node.

    Returns:
        List of length N_E: number of replicas for each expert.
    """
    total_capacity = n_device * capacity_per_device
    N_E = expert_loads.shape[0]
    node_num = max(1, n_device // gpus_per_node)

    loads = expert_loads.float().cpu().tolist()

    # Initialize: each expert gets at least 1 replica
    # If capacity allows, start with node_num replicas per expert (one per node)
    if node_num * N_E <= total_capacity:
        initial_replicas = node_num
    else:
        initial_replicas = 1

    # Max-heap: (-average_load, expert_id, replicas)
    # Python heapq is min-heap, so negate to get max-heap behavior
    heap = []
    now_capacity = initial_replicas * N_E

    for i in range(N_E):
        avg_load = loads[i] / initial_replicas if initial_replicas > 0 else loads[i]
        heapq.heappush(heap, (-avg_load, i, initial_replicas))

    result = [0] * N_E

    while now_capacity < total_capacity and heap:
        neg_avg, expert_id, replicas = heapq.heappop(heap)

        # Try to add replicas in node-sized chunks when aligned
        if replicas >= node_num and replicas % node_num == 0:
            if now_capacity + node_num <= total_capacity:
                new_replicas = replicas + node_num
                new_avg = loads[expert_id] / new_replicas
                heapq.heappush(heap, (-new_avg, expert_id, new_replicas))
                now_capacity += node_num
            else:
                result[expert_id] = replicas
        else:
            new_replicas = replicas + 1
            new_avg = loads[expert_id] / new_replicas
            heapq.heappush(heap, (-new_avg, expert_id, new_replicas))
            now_capacity += 1

    # Drain remaining items from heap
    while heap:
        _, expert_id, replicas = heapq.heappop(heap)
        result[expert_id] = replicas

    return result


# ---------------------------------------------------------------------------
# Greedy Placement: assign expert replicas to GPUs
# Reference: laer_moe/csrc/greedy_balancer.cpp::get_greedy_placement
# ---------------------------------------------------------------------------

def greedy_placement(
    expert_replicas: List[int],     # [N_E] number of replicas per expert
    expert_loads: torch.Tensor,     # [N_E] total load per expert
    n_device: int,
    n_expert: int,
    capacity_per_device: int,
    gpus_per_node: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Greedy placement: assign expert replicas to GPUs minimizing max GPU load.

    Ports: laer_moe/csrc/greedy_balancer.cpp::get_greedy_placement

    Algorithm:
      1. Generate all (expert, load_per_replica) pairs
      2. Sort by load descending
      3. For each pair, find the GPU with:
         a) Available capacity
         b) Preference for nodes where this expert isn't yet placed (spread)
         c) Minimum current load (load balance)

    Returns:
        global_expert_locations: [N_E, max_replicas] slot indices
        inverse_expert_map: [N_slots] expert index for each slot
        A: [N_E, n_device] placement matrix (A[e][d] = #replicas of e on d)
    """
    node_num = max(1, n_device // gpus_per_node)
    loads = expert_loads.float().cpu().tolist()

    # A[expert][device] = number of replicas of expert on device
    A = [[0] * n_device for _ in range(n_expert)]
    device_expert_count = [0] * n_device
    device_loads = [0.0] * n_device

    # Generate all (expert, load_per_replica) pairs sorted by load desc
    expert_list = []  # [(expert_id, load_per_replica)]
    for e in range(n_expert):
        replicas = expert_replicas[e]
        if replicas <= 0:
            continue
        load_per_replica = loads[e] / replicas
        for _ in range(replicas):
            expert_list.append((e, load_per_replica))

    expert_list.sort(key=lambda x: x[1], reverse=True)

    for expert_id, load in expert_list:
        # Find available devices (capacity not full)
        available = [d for d in range(n_device) if device_expert_count[d] < capacity_per_device]
        if not available:
            raise RuntimeError(
                f"[FSEP LoadPlanner] No capacity left for expert {expert_id}. "
                f"Increase capacity_per_device or reduce total replicas."
            )

        # Count existing nodes for this expert
        existing_nodes = [0] * node_num
        for d in range(n_device):
            if A[expert_id][d] > 0:
                existing_nodes[d // gpus_per_node] += A[expert_id][d]

        min_node_cnt = min(existing_nodes)

        # Prefer devices on nodes where this expert has fewest replicas (spread)
        new_node_devs = [
            d for d in available
            if existing_nodes[d // gpus_per_node] == min_node_cnt
        ]

        if new_node_devs:
            best_device = min(new_node_devs, key=lambda d: device_loads[d])
        else:
            best_device = min(available, key=lambda d: device_loads[d])

        A[expert_id][best_device] += 1
        device_loads[best_device] += load
        device_expert_count[best_device] += 1

    # Convert A to global_expert_locations and inverse_expert_map
    N_slots = n_device * capacity_per_device

    # Count max replicas across all experts
    max_replicas = max(sum(A[e]) for e in range(n_expert)) if n_expert > 0 else 1

    global_expert_locations = torch.full(
        (n_expert, max_replicas), -1, dtype=torch.long,
    )
    inverse_expert_map = torch.zeros(N_slots, dtype=torch.long)

    # Build slot assignments
    expert_replica_count = [0] * n_expert
    for d in range(n_device):
        slot_offset = d * capacity_per_device
        local_slot = 0
        for e in range(n_expert):
            for _ in range(A[e][d]):
                slot_idx = slot_offset + local_slot
                if slot_idx < N_slots:
                    global_expert_locations[e, expert_replica_count[e]] = slot_idx
                    inverse_expert_map[slot_idx] = e
                    expert_replica_count[e] += 1
                    local_slot += 1

    max_load = max(device_loads)
    return global_expert_locations, inverse_expert_map, torch.tensor(A, dtype=torch.long)


# ---------------------------------------------------------------------------
# FSEPLoadPlanner: main entry point
# ---------------------------------------------------------------------------

class FSEPLoadPlanner:
    """
    Periodic load monitor and placement optimizer for FSEP.

    Complete algorithm pipeline (per check_interval steps):
      1. Compute exponential moving average (EMA) of expert load
      2. Check if max_load / avg_load > threshold → detect imbalance
      3. Run Algorithm 4 (allocate_expert_replicas) → #replicas per expert
      4. Run greedy_placement → assign replicas to GPUs
      5. Compute expected improvement (skip if negligible)
      6. Signal FSEPRelayoutExecutor with new placement plan

    Reference: laer_moe/galvatron/core/runtime/moe/prefetch/solver.py
    """

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        sharding_degree: int,
        check_interval: int = 50,         # Check every K steps
        imbalance_threshold: float = 1.5, # Trigger if max/avg > threshold
        ema_decay: float = 0.9,           # EMA smoothing factor
        min_improvement: float = 0.05,    # Min expected improvement to trigger
        gpus_per_node: int = 8,
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.S = sharding_degree
        self.check_interval = check_interval
        self.imbalance_threshold = imbalance_threshold
        self.ema_decay = ema_decay
        self.min_improvement = min_improvement
        self.gpus_per_node = gpus_per_node
        self.capacity_per_device = num_experts // ep_size  # num_local_experts

        self.step_count = 0

        # EMA of per-expert token counts
        self.ema_load: Optional[torch.Tensor] = None  # [N_E]

        # Pending relayout plan (set when relayout is needed)
        self.pending_placement: Optional[Dict] = None

        # History for variance-based detection
        self.load_history: List[torch.Tensor] = []
        self.history_window = max(10, check_interval // 5)

        # Per-rank load history for full routing computation
        self.per_rank_load_history: List[torch.Tensor] = []

    def update(self, load: torch.Tensor, per_rank_load: Optional[torch.Tensor] = None) -> None:
        """
        Update load statistics with new per-expert token counts.

        Args:
            load: [N_E] tensor of token counts per expert (global, all-reduced)
            per_rank_load: Optional [ep_size, N_E] per-rank token distribution
        """
        load = load.float().detach().cpu()

        # EMA update
        if self.ema_load is None:
            self.ema_load = load.clone()
        else:
            self.ema_load = self.ema_decay * self.ema_load + (1 - self.ema_decay) * load

        # History window
        self.load_history.append(load)
        if len(self.load_history) > self.history_window:
            self.load_history.pop(0)

        # Per-rank history
        if per_rank_load is not None:
            self.per_rank_load_history.append(per_rank_load.float().detach().cpu())
            if len(self.per_rank_load_history) > self.history_window:
                self.per_rank_load_history.pop(0)

        self.step_count += 1

    def should_relayout(self) -> bool:
        """
        Check if a relayout is needed.

        Criteria:
          - Enough history collected
          - Not already have a pending relayout
          - Load imbalance above threshold
        """
        if self.step_count % self.check_interval != 0:
            return False
        if self.pending_placement is not None:
            return False  # Previous relayout not yet applied
        if self.ema_load is None or len(self.load_history) < self.history_window // 2:
            return False

        # Use stable average over recent history
        avg_load = torch.stack(self.load_history).mean(0)
        mean = avg_load.mean().item()
        if mean <= 0:
            return False

        max_val = avg_load.max().item()
        ratio = max_val / mean

        if ratio > self.imbalance_threshold:
            _log(
                f"[FSEP LoadPlanner] Imbalance detected: "
                f"max/avg = {ratio:.2f} > threshold {self.imbalance_threshold:.2f} "
                f"(step {self.step_count})"
            )
            return True

        return False

    def compute_new_placement(
        self,
        fsep_state: FSEPState,
    ) -> Optional[Dict]:
        """
        Compute new expert placement using the full LAER-MoE pipeline:
          Step 1: Algorithm 4 → allocate_expert_replicas (priority queue)
          Step 2: Greedy Placement → assign replicas to GPUs

        Returns:
            dict with new placement info, or None if no improvement expected
        """
        if self.ema_load is None:
            return None

        avg_load = (
            torch.stack(self.load_history).mean(0) if self.load_history
            else self.ema_load
        )
        N_E = self.num_experts
        ep_size = self.ep_size
        capacity = self.capacity_per_device

        # Current imbalance
        mean = avg_load.mean().item()
        max_val = avg_load.max().item()
        current_ratio = max_val / mean if mean > 0 else 1.0

        # ── Step 1: Algorithm 4 — Replica Allocation ──
        expert_replicas = allocate_expert_replicas(
            avg_load, ep_size, capacity, self.gpus_per_node,
        )

        # ── Step 2: Greedy Placement ──
        try:
            new_locations, new_inverse_map, A = greedy_placement(
                expert_replicas, avg_load, ep_size, N_E, capacity, self.gpus_per_node,
            )
        except RuntimeError as e:
            _log(f"[FSEP LoadPlanner] Placement failed: {e}")
            return None

        # ── Step 3: Estimate new imbalance ──
        # Compute expected per-GPU load under new placement
        gpu_loads = torch.zeros(ep_size)
        for e in range(N_E):
            replicas = expert_replicas[e]
            if replicas <= 0:
                continue
            load_per_replica = avg_load[e].item() / replicas
            for d in range(ep_size):
                gpu_loads[d] += A[e, d].item() * load_per_replica

        expected_max = gpu_loads.max().item()
        expected_mean = gpu_loads.mean().item()
        expected_ratio = expected_max / expected_mean if expected_mean > 0 else 1.0

        improvement = (current_ratio - expected_ratio) / current_ratio if current_ratio > 0 else 0.0
        if improvement < self.min_improvement:
            _log(
                f"[FSEP LoadPlanner] Improvement too small: "
                f"{improvement:.1%} < {self.min_improvement:.1%}. Skipping relayout."
            )
            return None

        _log(
            f"[FSEP LoadPlanner] Relayout planned: "
            f"expected improvement {improvement:.1%} "
            f"(ratio {current_ratio:.2f} → {expected_ratio:.2f}), "
            f"replicas={expert_replicas}"
        )

        return {
            "new_global_expert_locations": new_locations,
            "new_inverse_expert_map": new_inverse_map,
            "expert_replicas": expert_replicas,
            "expected_ratio": expected_ratio,
            "improvement": improvement,
        }

    def plan(self, fsep_state: FSEPState) -> Optional[Dict]:
        """
        Main entry point: check if relayout needed, compute plan if so.

        Should be called every step (after router statistics are updated).

        Returns:
            Placement plan dict if relayout needed, else None.
        """
        if not self.should_relayout():
            return None

        plan = self.compute_new_placement(fsep_state)
        if plan is not None:
            self.pending_placement = plan

        return plan

    def ack_relayout(self) -> None:
        """Acknowledge that the pending relayout has been applied."""
        self.pending_placement = None
        _log(f"[FSEP LoadPlanner] Relayout applied at step {self.step_count}")
