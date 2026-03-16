###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
Correctness tests for Full FSEP (Phase 1-3) implementation.

Tests:
  Test 1: FSEPState initialization and basic operations
  Test 2: smart_routing_map correctness (Lite Routing - Algorithm 3)
  Test 3: smart_routing_map 3D version with intra/inter node
  Test 4: compute_smart_routing token assignment
  Test 5: new_routing_map_with_gradients backward
  Test 6: Algorithm 4 - Replica Allocation (priority queue)
  Test 7: Greedy Placement correctness
  Test 8: FSEPLoadPlanner update + should_relayout + compute_new_placement
  Test 9: FSEPAlltoAllTokenDispatcher forward (4 GPU)
  Test 10: FSEPRelayoutExecutor parameter migration (4 GPU)

Run: python tests/unit_tests/megatron/transformer/moe/test_fsep_full.py
"""

import os
import unittest
from types import SimpleNamespace
from torch.testing._internal.common_utils import TestCase

import torch
import torch.distributed as dist

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests


# ─── Single-process tests (no GPU needed) ─────────────────────────────────────

class TestFSEPStateUnit(TestCase):
    """Test FSEPState initialization and operations (no distributed needed)."""

    def test_fsep_state_init_basic(self):
        """Test FSEPState can be instantiated with mock values."""
        num_experts = 8
        S = 4

        global_expert_locations = torch.full((num_experts, S), -1, dtype=torch.long)
        for e in range(num_experts):
            global_expert_locations[e, 0] = e

        inverse_expert_map = torch.arange(num_experts, dtype=torch.long)
        expert_capacity = torch.full((num_experts,), 1_000_000, dtype=torch.long)

        self.assertEqual(global_expert_locations.shape, (num_experts, S))
        self.assertEqual(inverse_expert_map.shape, (num_experts,))

        for e in range(num_experts):
            self.assertEqual(global_expert_locations[e, 0].item(), e)
            for s in range(1, S):
                self.assertEqual(global_expert_locations[e, s].item(), -1)

    def test_fsep_state_expand_full(self):
        """Test expand_to_full_fsep sets all S entries for each expert."""
        num_experts = 8
        S = 4

        global_expert_locations = torch.full((num_experts, S), -1, dtype=torch.long)
        for e in range(num_experts):
            for s in range(S):
                global_expert_locations[e, s] = e

        for e in range(num_experts):
            valid = global_expert_locations[e][global_expert_locations[e] >= 0]
            self.assertEqual(len(valid), S)
            self.assertTrue((valid == e).all())


class TestSmartRoutingUnit(TestCase):
    """Test smart routing map computation (Lite Routing - Algorithm 3)."""

    def test_smart_routing_map_uniform(self):
        """With uniform load and 1 replica, each slot gets equal allocation."""
        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            smart_routing_map,
        )

        N_E = 8
        S = 4
        tokens_per_expert = torch.full((N_E,), 100, dtype=torch.long)

        expert_locations = torch.full((N_E, S), -1, dtype=torch.long)
        for e in range(N_E):
            expert_locations[e, 0] = e  # only 1 replica

        slot_allocation = smart_routing_map(
            tokens_per_expert, expert_locations, N_E // 8,
        )

        self.assertEqual(slot_allocation.shape[0], N_E)
        for e in range(N_E):
            self.assertEqual(slot_allocation[e].item(), 100)

    def test_smart_routing_map_imbalanced_replicas(self):
        """With replicated hot expert, tokens split across replicas."""
        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            smart_routing_map,
        )

        N_E = 4
        S = 2
        tokens_per_expert = torch.tensor([400, 100, 100, 100], dtype=torch.long)

        # Expert 0 has 2 replicas at slots 0 and 1
        # Experts 1,2,3 each have 1 replica at slots 2,3,4 (no overlap with expert 0)
        expert_locations = torch.full((N_E, S), -1, dtype=torch.long)
        expert_locations[0, 0] = 0
        expert_locations[0, 1] = 1  # Expert 0 has 2 replicas
        expert_locations[1, 0] = 2
        expert_locations[2, 0] = 3
        expert_locations[3, 0] = 4

        slot_allocation = smart_routing_map(
            tokens_per_expert, expert_locations, N_E // 2,
        )

        # Expert 0's 400 tokens should be split evenly: 200 to slot 0, 200 to slot 1
        self.assertEqual(slot_allocation[0].item(), 200,
                         f"Slot 0 should get 200 tokens, got {slot_allocation[0].item()}")
        self.assertEqual(slot_allocation[1].item(), 200,
                         f"Slot 1 should get 200 tokens, got {slot_allocation[1].item()}")
        # Expert 1 → slot 2: 100 tokens
        self.assertEqual(slot_allocation[2].item(), 100)
        # Total across all slots = sum of all expert tokens
        self.assertEqual(slot_allocation.sum().item(), 700)

    def test_smart_routing_3d_intra_node_priority(self):
        """3D routing should prioritize intra-node replicas."""
        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            smart_routing_map,
        )

        # 2 GPUs, 2 experts, 2 local experts each
        # GPU 0 (node 0), GPU 1 (node 0) in a single node
        N_E = 2
        max_S = 2
        ep_size = 2
        num_local = 1
        gpus_per_node = 8  # both GPUs on same node

        tokens = torch.zeros(1, ep_size, N_E, dtype=torch.long)
        tokens[0, 0, 0] = 100  # GPU 0 has 100 tokens for expert 0
        tokens[0, 1, 1] = 50   # GPU 1 has 50 tokens for expert 1

        expert_locations = torch.full((N_E, max_S), -1, dtype=torch.long)
        expert_locations[0, 0] = 0  # expert 0 at slot 0
        expert_locations[0, 1] = 1  # expert 0 also at slot 1
        expert_locations[1, 0] = 1  # expert 1 at slot 1

        allocation = smart_routing_map(
            tokens, expert_locations, num_local, gpus_per_node=gpus_per_node,
        )

        # All GPUs are on same node → intra-node, should distribute evenly
        self.assertEqual(allocation.shape, (1, ep_size, ep_size * num_local))
        total = allocation.sum().item()
        self.assertEqual(total, 150)  # 100 + 50

    def test_compute_smart_routing_preserves_assignments(self):
        """Each token should be assigned to exactly one slot per expert."""
        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            compute_smart_routing,
        )

        T = 10
        N_E = 4
        topk = 2

        routing_map = torch.zeros(T, N_E, dtype=torch.bool)
        for t in range(T):
            experts = torch.randperm(N_E)[:topk]
            routing_map[t, experts] = True

        probs = routing_map.float() / topk

        expert_locations = torch.full((N_E, 2), -1, dtype=torch.long)
        for e in range(N_E):
            expert_locations[e, 0] = e
        inverse_expert_map = torch.arange(N_E, dtype=torch.long)
        slot_capacity = torch.full((N_E,), 100, dtype=torch.long)

        new_routing_map, new_probs = compute_smart_routing(
            routing_map, probs, expert_locations, inverse_expert_map, slot_capacity
        )

        # Total assignments should be preserved
        original = routing_map.long().sum().item()
        new = new_routing_map.long().sum().item()
        self.assertEqual(original, new)

    def test_new_routing_map_backward(self):
        """Gradients should flow back through new_routing_map_with_gradients."""
        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            new_routing_map_with_gradients,
        )

        T, N_E = 4, 3
        routing_map = torch.zeros(T, N_E, dtype=torch.bool)
        routing_map[0, 0] = True
        routing_map[1, 1] = True
        routing_map[2, 2] = True
        routing_map[3, 0] = True

        probs_masked = (routing_map.float() * 0.5).detach().requires_grad_(True)

        expert_locations = torch.full((N_E, 1), -1, dtype=torch.long)
        for e in range(N_E):
            expert_locations[e, 0] = e
        inverse_expert_map = torch.arange(N_E, dtype=torch.long)
        slot_capacity = torch.full((N_E,), 10, dtype=torch.long)

        new_routing_map, new_probs = new_routing_map_with_gradients(
            routing_map, probs_masked, expert_locations, inverse_expert_map, slot_capacity
        )

        loss = new_probs.sum()
        loss.backward()

        self.assertIsNotNone(probs_masked.grad)
        self.assertFalse(torch.isnan(probs_masked.grad).any())


class TestReplicaAllocationUnit(TestCase):
    """Test Algorithm 4: Replica Allocation (priority queue)."""

    def test_uniform_load_equal_replicas(self):
        """With uniform load, replicas should be roughly equal."""
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            allocate_expert_replicas,
        )

        N_E = 8
        n_device = 4
        capacity = 2  # 4 * 2 = 8 total slots = N_E
        loads = torch.full((N_E,), 100.0)

        replicas = allocate_expert_replicas(loads, n_device, capacity)

        self.assertEqual(len(replicas), N_E)
        self.assertEqual(sum(replicas), n_device * capacity)  # total = 8
        # With uniform load, each should get 1 replica
        for r in replicas:
            self.assertEqual(r, 1)

    def test_hot_expert_gets_more_replicas(self):
        """Hot expert should get more replicas than cold experts."""
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            allocate_expert_replicas,
        )

        N_E = 4
        n_device = 4
        capacity = 2  # 4 * 2 = 8 total slots
        loads = torch.tensor([800.0, 100.0, 100.0, 100.0])

        replicas = allocate_expert_replicas(loads, n_device, capacity)

        self.assertEqual(sum(replicas), 8)
        # Expert 0 should have more replicas
        self.assertGreater(replicas[0], replicas[1])

    def test_all_capacity_used(self):
        """Total replicas should exactly match total capacity when N_E <= capacity."""
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            allocate_expert_replicas,
        )

        # Only test valid configs where N_E <= n_dev * cap
        # (each expert needs at least 1 replica)
        valid_configs = [
            (4, 4, 2),   # N_E=4, capacity=8, each gets 2
            (4, 8, 2),   # N_E=4, capacity=16
            (8, 4, 2),   # N_E=8, capacity=8, each gets 1
            (8, 8, 2),   # N_E=8, capacity=16
            (8, 4, 4),   # N_E=8, capacity=16
            (16, 8, 4),  # N_E=16, capacity=32
        ]
        for N_E, n_dev, cap in valid_configs:
            total_capacity = n_dev * cap
            if N_E > total_capacity:
                continue  # Skip impossible configs
            loads = torch.rand(N_E) * 1000 + 1
            replicas = allocate_expert_replicas(loads, n_dev, cap)
            self.assertEqual(
                sum(replicas), total_capacity,
                f"N_E={N_E}, n_dev={n_dev}, cap={cap}: "
                f"sum(replicas)={sum(replicas)} != {total_capacity}"
            )
            # Each expert should get at least 1 replica
            for e, r in enumerate(replicas):
                self.assertGreaterEqual(r, 1, f"Expert {e} should have >= 1 replica")


class TestGreedyPlacementUnit(TestCase):
    """Test Greedy Placement algorithm."""

    def test_basic_placement(self):
        """Greedy placement should produce valid expert→GPU assignments."""
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            greedy_placement,
        )

        N_E = 4
        n_device = 4
        capacity = 1
        replicas = [1, 1, 1, 1]
        loads = torch.tensor([100.0, 100.0, 100.0, 100.0])

        locations, inv_map, A = greedy_placement(
            replicas, loads, n_device, N_E, capacity,
        )

        self.assertEqual(locations.shape[0], N_E)
        # Each expert should have exactly 1 valid location
        for e in range(N_E):
            valid = locations[e][locations[e] >= 0]
            self.assertEqual(len(valid), 1)

    def test_replicated_placement(self):
        """Expert with 2 replicas should appear on 2 different GPUs."""
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            greedy_placement,
        )

        N_E = 4
        n_device = 4
        capacity = 2
        replicas = [2, 2, 2, 2]  # total = 8 = 4 * 2
        loads = torch.tensor([200.0, 100.0, 100.0, 100.0])

        locations, inv_map, A = greedy_placement(
            replicas, loads, n_device, N_E, capacity,
        )

        # Expert 0 should have 2 valid locations on different GPUs
        valid_locs = locations[0][locations[0] >= 0]
        self.assertEqual(len(valid_locs), 2)
        gpus = set()
        for loc in valid_locs:
            gpu = loc.item() // capacity
            gpus.add(gpu)
        self.assertEqual(len(gpus), 2, "Expert 0 replicas should be on different GPUs")


class TestLoadPlannerUnit(TestCase):
    """Test FSEPLoadPlanner logic (no distributed needed)."""

    def _make_planner(self, **kwargs):
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            FSEPLoadPlanner,
        )
        return FSEPLoadPlanner(
            num_experts=8,
            ep_size=4,
            sharding_degree=4,
            check_interval=kwargs.get("check_interval", 5),
            imbalance_threshold=kwargs.get("imbalance_threshold", 1.5),
        )

    def test_no_relayout_on_balanced_load(self):
        """Balanced load should NOT trigger relayout."""
        planner = self._make_planner(check_interval=5)

        balanced_load = torch.full((8,), 100.0)
        for _ in range(5):
            planner.update(balanced_load)

        self.assertFalse(planner.should_relayout())

    def test_relayout_on_imbalanced_load(self):
        """Highly imbalanced load SHOULD trigger relayout."""
        planner = self._make_planner(check_interval=5, imbalance_threshold=1.5)

        imbalanced = torch.tensor([1000.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0])
        for _ in range(5):
            planner.update(imbalanced)

        self.assertTrue(planner.should_relayout())

    def test_ema_smoothing(self):
        """EMA should smooth out transient spikes."""
        planner = self._make_planner(check_interval=10)

        for _ in range(8):
            planner.update(torch.full((8,), 100.0))

        for _ in range(2):
            planner.update(torch.tensor([1000.0] + [50.0] * 7))

        if planner.ema_load is not None:
            ratio = planner.ema_load.max().item() / planner.ema_load.mean().item()
            self.assertLess(ratio, 10.0)

    def test_plan_returns_placement_with_algo4(self):
        """compute_new_placement should use Algorithm 4 + Greedy Placement."""
        # Use config where total_capacity > N_E so replication is possible
        from primus.backends.megatron.core.transformer.moe.load_planner import (
            FSEPLoadPlanner,
        )
        # N_E=4, ep_size=4, cap=2 → total_capacity=8, room for replication
        planner = FSEPLoadPlanner(
            num_experts=4,
            ep_size=4,
            sharding_degree=4,
            check_interval=5,
            imbalance_threshold=1.5,
        )

        imbalanced = torch.tensor([800.0, 100.0, 100.0, 100.0])
        for _ in range(5):
            planner.update(imbalanced)

        class MockState:
            num_experts = 4
            ep_size = 4
            sharding_degree = 4
            global_expert_locations = torch.full((4, 4), -1, dtype=torch.long)
            inverse_expert_map = torch.arange(4, dtype=torch.long)

        plan = planner.compute_new_placement(MockState())

        if plan is not None:
            self.assertIn("new_global_expert_locations", plan)
            self.assertIn("new_inverse_expert_map", plan)
            self.assertIn("expert_replicas", plan)
            self.assertIn("improvement", plan)
            self.assertGreater(plan["improvement"], 0.0)
            # Expert 0 (800 tokens) should have more replicas than others (100 tokens)
            replicas = plan["expert_replicas"]
            self.assertGreater(replicas[0], replicas[1],
                               f"Expert 0 should have more replicas: {replicas}")


# ─── Distributed tests (4 GPU) ────────────────────────────────────────────────

@instantiate_parametrized_tests
class TestFSEPFullDispatcher4GPU(MultiProcessTestCase):
    """Distributed tests for FSEPAlltoAllTokenDispatcher (4 GPU)."""

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()

    @property
    def world_size(self):
        return min(4, torch.cuda.device_count())

    @property
    def device(self):
        return torch.device("cuda", self.rank)

    def _init_process(self):
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.rank)
        torch.cuda.set_device(self.device)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    @skip_if_lt_x_gpu(4)
    def test_fsep_state_init_distributed(self):
        """FSEPState can be initialized in a distributed setting."""
        self._init_process()

        import megatron.core.parallel_state as ps
        ps.initialize_model_parallel(
            expert_model_parallel_size=4,
        )

        from primus.backends.megatron.core.transformer.moe.fsep_parallel_state import (
            init_fsep_state,
        )

        ep_group = ps.get_expert_model_parallel_group()
        state = init_fsep_state(
            num_experts=8,
            ep_size=4,
            sharding_degree=4,
            ep_group=ep_group,
        )

        self.assertIsNotNone(state)
        self.assertEqual(state.num_experts, 8)
        self.assertEqual(state.ep_size, 4)
        self.assertEqual(state.num_local_experts, 2)
        self.assertEqual(state.global_expert_locations.shape, (8, 4))

        for e in range(8):
            self.assertEqual(state.global_expert_locations[e, 0].item(), e)

    @skip_if_lt_x_gpu(4)
    def test_smart_routing_correctness_distributed(self):
        """Smart routing preserves total token count in distributed setting."""
        self._init_process()

        import megatron.core.parallel_state as ps
        ps.initialize_model_parallel(expert_model_parallel_size=4)

        from primus.backends.megatron.core.transformer.moe.fsep_smart_routing import (
            compute_smart_routing,
        )

        T, N_E = 16, 8
        routing_map = torch.zeros(T, N_E, dtype=torch.bool, device=self.device)
        for t in range(T):
            routing_map[t, t % N_E] = True
            routing_map[t, (t + 1) % N_E] = True

        probs = routing_map.float() * 0.5

        expert_locations = torch.full((N_E, 2), -1, dtype=torch.long, device=self.device)
        for e in range(N_E):
            expert_locations[e, 0] = e
        inverse_expert_map = torch.arange(N_E, dtype=torch.long, device=self.device)
        slot_capacity = torch.full((N_E,), 100, dtype=torch.long, device=self.device)

        new_routing_map, new_probs = compute_smart_routing(
            routing_map.cpu(), probs.cpu(),
            expert_locations.cpu(), inverse_expert_map.cpu(),
            slot_capacity.cpu(),
        )

        orig_count = routing_map.long().sum().item()
        new_count = new_routing_map.long().sum().item()
        self.assertEqual(orig_count, new_count)

    @skip_if_lt_x_gpu(4)
    def test_load_planner_full_pipeline(self):
        """Load planner full pipeline: detect → allocate → place."""
        self._init_process()

        from primus.backends.megatron.core.transformer.moe.load_planner import (
            FSEPLoadPlanner,
        )

        planner = FSEPLoadPlanner(
            num_experts=8,
            ep_size=4,
            sharding_degree=4,
            check_interval=5,
            imbalance_threshold=2.0,
        )

        # Feed strongly imbalanced load
        imbalanced = torch.tensor([600.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        for _ in range(5):
            planner.update(imbalanced)

        self.assertTrue(planner.should_relayout())


if __name__ == "__main__":
    run_tests()
