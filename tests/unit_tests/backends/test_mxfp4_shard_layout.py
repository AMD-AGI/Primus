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


if __name__ == "__main__":
    unittest.main()
