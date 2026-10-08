###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Run directly with Python; the planner requires neither Megatron nor GPUs."""

import importlib.util
import random
import sys
import unittest
from math import prod
from pathlib import Path
from types import SimpleNamespace

_PATH = (
    Path(__file__).resolve().parents[3] / "primus/backends/megatron/core/distributed/mxfp4_shard_layout.py"
)
_SPEC = importlib.util.spec_from_file_location("mxfp4_shard_layout_under_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
plan_weight_regions = _MODULE.plan_weight_regions


class TestShardLayout(unittest.TestCase):
    def test_random_shards_cover_weight_exactly(self):
        rng = random.Random(42)
        for _ in range(250):
            shape = (rng.randint(1, 5), 32 * rng.randint(1, 20), 32 * rng.randint(1, 20))
            world = rng.randint(1, 16)
            start = rng.randint(0, 10000)
            end = start + prod(shape)
            size = (end + rng.randint(0, 10000) + world - 1) // world * world
            regions = plan_weight_regions(shape, start, size, world)
            cursor = start
            for region in regions:
                self.assertEqual(region.start, cursor)
                self.assertGreater(region.end, region.start)
                self.assertEqual(region.rank, region.start // (size // world))
                self.assertEqual(region.rank, (region.end - 1) // (size // world))
                if region.complete_tiles:
                    self.assertEqual((region.start - start) % (32 * shape[-1]), 0)
                    self.assertEqual((region.end - start) % (32 * shape[-1]), 0)
                cursor = region.end
            self.assertEqual(cursor, end)

    def test_aligned_and_split_tiles(self):
        regions = plan_weight_regions((2, 128, 128), 0, 32768, 8)
        self.assertTrue(all(region.complete_tiles for region in regions))
        regions = plan_weight_regions((1, 32, 32), 0, 1024, 8)
        self.assertFalse(any(region.complete_tiles for region in regions))

    def test_reject_invalid_layout(self):
        for args in (
            ((1, 33, 32), 0, 2048, 8),
            ((1, 32, 32), 1, 1024, 8),
            ((1, 32, 32), 0, 1025, 8),
            ((1, 32, 32), 0, 1024, 0),
        ):
            with self.assertRaises(ValueError):
                plan_weight_regions(*args)

    def test_audit_uses_bucket_relative_offsets(self):
        class Param:
            ndim = 3
            shape = (1, 64, 32)

        param = Param()
        buffer = SimpleNamespace(
            param_index_map={param: (4096 + 128, 4096 + 128 + 2048, 0)},
            buckets=[SimpleNamespace(offset=4096, param_data=SimpleNamespace(numel=lambda: 4096))],
            data_parallel_world_size=2,
        )
        reports = _MODULE.audit_buffer(buffer, {param: "decoder.layers.0.mlp.experts.linear_fc1.weights"})
        self.assertEqual(reports[0]["complete_tile_elements"], 1024)
        self.assertEqual(reports[0]["boundary_elements"], 1024)
        self.assertEqual(_MODULE.audit_buffer(buffer, {param: "unrelated.weight"}), [])

    def test_strip_ownership_and_boundary_exchange(self):
        spec = importlib.util.spec_from_file_location(
            "mxfp4_training_layout_test", _PATH.with_name("mxfp4_training.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        rng = random.Random(1701)
        for _ in range(200):
            shape = (rng.randint(1, 4), 32 * rng.randint(1, 12), 32 * rng.randint(1, 8))
            start, world = rng.randint(0, 4096), rng.randint(1, 16)
            size = (start + prod(shape) + world - 1) // world * world
            owners, boundaries = module.strip_ownership(shape, start, size, world)
            cursor, strip = 0, 32 * shape[-1]
            for rank, (first, count) in enumerate(owners):
                self.assertEqual(first, cursor)
                for index in range(first, first + count):
                    tile_start = start + index * strip
                    self.assertEqual(tile_start // (size // world), rank)
                    if (tile_start + strip - 1) // (size // world) != rank:
                        self.assertIn((tile_start, tile_start + strip), boundaries)
                cursor += count
            self.assertEqual(cursor, shape[0] * shape[1] // 32)

    def test_restore_plan_preserves_wire_offsets_and_skips_unneeded_values(self):
        import torch

        spec = importlib.util.spec_from_file_location(
            "mxfp4_restore_plan_test", _PATH.with_name("mxfp4_training.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        state = module.PackedExpertBucket.__new__(module.PackedExpertBucket)
        state.rank = 1
        state.bucket = SimpleNamespace(param_data=torch.full((32,), -1, dtype=torch.bfloat16))
        state.bucket.param_data[8:16] = torch.arange(8, 16, dtype=torch.bfloat16)
        ranges = [[(2, 5), (7, 8)], [(8, 11), (14, 16)], [(16, 20), (21, 24)], [(24, 26)]]
        rows = torch.zeros((4, 32), dtype=torch.uint8)
        for rank, pieces in enumerate(ranges):
            values = torch.cat([torch.arange(start, end, dtype=torch.bfloat16) for start, end in pieces])
            rows[rank, : values.numel() * 2] = values.view(torch.uint8)
        plan = state._restore_plan(ranges, [(14, 20), (22, 25)])
        state._restore_bf16_ranges(SimpleNamespace(rows=rows), plan)
        expected = torch.full((32,), -1, dtype=torch.bfloat16)
        expected[8:20] = torch.arange(8, 20, dtype=torch.bfloat16)
        expected[22:25] = torch.arange(22, 25, dtype=torch.bfloat16)
        torch.testing.assert_close(state.bucket.param_data, expected, rtol=0, atol=0)
        self.assertEqual(plan, [(2, 0, 16, 20), (2, 10, 22, 24), (3, 0, 24, 25)])


class TestSharedWorkspace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "mxfp4_workspace_test", _PATH.with_name("mxfp4_training.py")
        )
        cls.runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.runtime
        spec.loader.exec_module(cls.runtime)

    def test_capacity_bound_covers_uneven_experts_and_bf16_gaps(self):
        rng = random.Random(4811)
        for _ in range(250):
            world, cursor, weights = rng.randint(1, 16), 0, []
            for _ in range(rng.randint(1, 5)):
                shape = (rng.randint(1, 4), 32 * rng.randint(1, 9), 32 * rng.randint(1, 8))
                start = cursor + rng.randint(0, 3000)
                cursor = start + prod(shape)
                weights.append((shape, start, cursor))
            size = (cursor + rng.randint(0, 30000) + world - 1) // world * world
            bucket = SimpleNamespace(
                param_data=SimpleNamespace(numel=lambda: size),
                params=[
                    SimpleNamespace(shape=shape, _primus_mxfp4_comm_candidate=True) for shape, _, _ in weights
                ],
            )
            capacity = self.runtime.packed_workspace_width_bound(bucket, world)
            capacity = (capacity + 255) // 256 * 256
            for rank in range(world):
                begin, end = rank * (size // world), (rank + 1) * (size // world)
                expert_elements = sum(
                    max(0, min(end, stop) - max(begin, start)) for _, start, stop in weights
                )
                payload = 2 * (size // world - expert_elements)
                for shape, start, _ in weights:
                    ownership, _ = self.runtime.strip_ownership(shape, start, size, world)
                    payload += ownership[rank][1] * 32 * shape[-1] * 17 // 16
                self.assertLessEqual((payload + 255) // 256 * 256, capacity)

    def test_pool_selection_preserves_process_group_ownership(self):
        from unittest import mock

        import torch

        class Param:
            shape = (1, 32, 32)
            _primus_mxfp4_comm_candidate = True

        class Group:
            def __init__(self, process_group, bucket):
                self.intra_distributed_optimizer_instance_group = process_group
                self.buckets = [bucket]

        class Workspace:
            rank = 1

            def __init__(self, width, group, device, use_sdma):
                self.width, self.group, self.device = width, group, device

        a, b = object(), object()
        params = [Param() for _ in range(3)]
        buckets = [
            SimpleNamespace(params=[p], param_data=torch.empty(2048 * (i + 1), dtype=torch.bfloat16))
            for i, p in enumerate(params)
        ]
        mapping = {p: Group(group, bucket) for p, group, bucket in zip(params, (a, a, b), buckets)}
        with mock.patch.object(self.runtime.dist, "get_world_size", return_value=2), mock.patch.object(
            self.runtime, "ByteAllGather", Workspace
        ):
            workspaces = self.runtime.assign_shared_workspaces(mapping)
        self.assertEqual(len(workspaces), 2)
        self.assertIs(buckets[0]._primus_mxfp4_workspace, buckets[1]._primus_mxfp4_workspace)
        self.assertIsNot(buckets[0]._primus_mxfp4_workspace, buckets[2]._primus_mxfp4_workspace)
        self.assertEqual(workspaces[0].width, self.runtime.packed_workspace_width_bound(buckets[1], 2))


if __name__ == "__main__":
    unittest.main()
