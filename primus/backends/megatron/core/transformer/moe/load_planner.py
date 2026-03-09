###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP Load Planner: monitor expert load and plan optimal placement.

Implements the Load-Adaptive component of LAER-MoE:
  1. Monitor per-expert token counts (via router statistics)
  2. Detect load imbalance (max/avg ratio threshold)
  3. Call greedy balancing algorithm to compute new Expert placement
  4. Signal FSEPRelayoutExecutor to migrate parameters

Reference: laer_moe/galvatron/core/runtime/moe/prefetch/solver.py::MoEOptimizer
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
    FSEPState,
    get_fsep_state,
)
from primus.modules.module_utils import log_rank_0


class FSEPLoadPlanner:
    """
    Periodic load monitor and placement optimizer for FSEP.

    Algorithm:
      Every K steps:
        1. Compute exponential moving average (EMA) of expert load
        2. Check if max_load / avg_load > threshold
        3. If so: run greedy_placement() to find new expert allocation
        4. Compute expected improvement (skip if negligible)
        5. Signal FSEPRelayoutExecutor with new placement plan

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
    ):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.S = sharding_degree
        self.check_interval = check_interval
        self.imbalance_threshold = imbalance_threshold
        self.ema_decay = ema_decay
        self.min_improvement = min_improvement

        self.step_count = 0

        # EMA of per-expert token counts
        self.ema_load: Optional[torch.Tensor] = None  # [N_E]

        # Pending relayout plan (set when relayout is needed)
        self.pending_placement: Optional[Dict] = None

        # History for variance-based detection
        self.load_history: List[torch.Tensor] = []
        self.history_window = max(10, check_interval // 5)

    def update(self, load: torch.Tensor) -> None:
        """
        Update load statistics with new per-expert token counts.

        Args:
            load: [N_E] tensor of token counts per expert (global, all-reduced)
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
            log_rank_0(
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
        Compute new expert placement using greedy load balancing.

        Algorithm (simplified Greedy from LAER-MoE paper):
          1. Sort experts by load (descending)
          2. Assign each heavy expert to the GPU pair with minimum load
          3. Output: new global_expert_locations [N_E, max_S]

        Full algorithm: port from laer_moe/csrc/greedy_balancer.cpp
        (C++ implementation with Expert replication support)

        Returns:
            dict with new placement info, or None if no improvement expected
        """
        if self.ema_load is None:
            return None

        avg_load = torch.stack(self.load_history).mean(0) if self.load_history else self.ema_load
        N_E = self.num_experts
        ep_size = self.ep_size
        S = self.S

        # Current imbalance
        mean = avg_load.mean().item()
        max_val = avg_load.max().item()
        current_ratio = max_val / mean if mean > 0 else 1.0

        # ── Greedy placement algorithm ──
        # Each GPU has capacity = total_tokens / ep_size (per-step average)
        total_tokens = avg_load.sum().item()
        gpu_load = torch.zeros(ep_size)  # current load on each GPU

        # Start with traditional EP placement
        new_locations = torch.full((N_E, S), fill_value=-1, dtype=torch.long)
        for e in range(N_E):
            new_locations[e, 0] = e  # traditional owner

        # Sort experts by load descending
        sorted_experts = avg_load.argsort(descending=True).tolist()

        # Try to replicate heavy experts onto underloaded GPUs
        # (S > 1 gives more replicas to absorb load)
        for e_idx in sorted_experts:
            load_e = avg_load[e_idx].item()
            traditional_owner = e_idx // (N_E // ep_size)
            gpu_load[traditional_owner] += load_e / S

            if S > 1 and load_e > mean * self.imbalance_threshold:
                # This expert is overloaded: distribute across S GPUs
                # Find S GPUs with minimum current load
                best_gpus = gpu_load.argsort()[:S].tolist()
                for s, gpu in enumerate(best_gpus):
                    new_locations[e_idx, s] = e_idx  # slot = expert index (Mode B)
                    gpu_load[gpu] += load_e / S

        # Build inverse map
        new_inverse_map = torch.arange(N_E, dtype=torch.long)

        # Expected new imbalance
        expected_max = gpu_load.max().item()
        expected_mean = gpu_load.mean().item()
        expected_ratio = expected_max / expected_mean if expected_mean > 0 else 1.0

        improvement = (current_ratio - expected_ratio) / current_ratio
        if improvement < self.min_improvement:
            log_rank_0(
                f"[FSEP LoadPlanner] Improvement too small: "
                f"{improvement:.1%} < {self.min_improvement:.1%}. Skipping relayout."
            )
            return None

        log_rank_0(
            f"[FSEP LoadPlanner] Relayout planned: "
            f"expected improvement {improvement:.1%} "
            f"(ratio {current_ratio:.2f} → {expected_ratio:.2f})"
        )

        return {
            "new_global_expert_locations": new_locations,
            "new_inverse_expert_map": new_inverse_map,
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
        log_rank_0(f"[FSEP LoadPlanner] Relayout applied at step {self.step_count}")
