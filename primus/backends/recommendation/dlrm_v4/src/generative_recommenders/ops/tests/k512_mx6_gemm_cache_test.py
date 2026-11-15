#!/usr/bin/env python3

# pyre-unsafe

"""DLRM_HSTU_FP8_FWD_MXFP6: the MX6 weight-shadow CACHE of ops/hip/k512_mx6_gemm.py (CPU only, no GPU, no build).

mx6_weight must never serve a stale shadow after the weight changed: the cache entry is valid only for the same
tensor object (weakref identity), the same data_ptr() and the same _version, per (shape, stride, dtype, layout).
The refresh itself (one kernel launch on a GPU) is replaced by the module's torch reference, which runs on the CPU;
ops/tests/k512_mx6_gemm_test.py covers the kernel. Run: python -m pytest -q generative_recommenders/ops/tests/
k512_mx6_gemm_cache_test.py
"""

from __future__ import annotations

import gc
import unittest
from unittest.mock import patch

import torch

from generative_recommenders.ops.hip import k512_mx6_gemm as mx6

VQK = ((512, 2048),)
VK = ((512, 1024), (1536, 2048))


def _same(a, b) -> bool:
    return all(mx6.bitwise_equal(x, y) for x, y in zip(a, b))


class K512Mx6ShadowCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        mx6.clear_shadow_cache()
        self.calls = []

        def fake_refresh(weight, layout, **kw):
            self.calls.append((id(weight), tuple(layout), int(weight._version)))
            return mx6.mx6_weight_reference(weight, layout)

        # new= (a plain function): a MagicMock would keep every weight alive through its call records
        self.patcher = patch.object(mx6, "refresh_weight_shadow", new=fake_refresh)
        self.patcher.start()
        g = torch.Generator().manual_seed(7)
        self.w = torch.nn.Parameter(torch.randn((512, 2048), generator=g) * 0.02)

    def tearDown(self) -> None:
        self.patcher.stop()
        mx6.clear_shadow_cache()

    def test_hit_returns_cached_tensors(self) -> None:
        a = mx6.mx6_weight(self.w, VQK)
        b = mx6.mx6_weight(self.w, VQK)
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(all(x is y for x, y in zip(a, b)))
        self.assertEqual(mx6.shadow_cache_stats()["hits"], 1)
        self.assertEqual(tuple(a[0].shape), (6, 2, 4, 12288))
        self.assertEqual(tuple(a[1].shape), (48, 4, 4))
        self.assertEqual(tuple(a[2].shape), (1, 1))

    def test_version_bump_refreshes(self) -> None:
        a = mx6.mx6_weight(self.w, VQK)
        v0 = self.w._version
        with torch.no_grad():
            self.w.mul_(1.25)  # in place: bumps _version
        self.assertGreater(self.w._version, v0)
        b = mx6.mx6_weight(self.w, VQK)
        self.assertEqual(len(self.calls), 2, "a version bump must refresh the shadow")
        self.assertIsNot(a[0], b[0])
        self.assertTrue(_same(b, mx6.mx6_weight_reference(self.w, VQK)))
        self.assertFalse(mx6.bitwise_equal(a[2], b[2]))  # the scale followed the update
        # the same values under a new version still refresh (the version is part of the key)
        with torch.no_grad():
            self.w.add_(0.0)
        c = mx6.mx6_weight(self.w, VQK)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(_same(c, b))

    def test_optimizer_step_refreshes(self) -> None:
        opt = torch.optim.Adam([self.w], lr=1e-2)
        a = mx6.mx6_weight(self.w, VQK)
        self.w.grad = torch.full_like(self.w, 0.01)
        opt.step()
        b = mx6.mx6_weight(self.w, VQK)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(_same(a, b))
        self.assertTrue(_same(b, mx6.mx6_weight_reference(self.w, VQK)))

    def test_storage_swap_and_identity(self) -> None:
        a = mx6.mx6_weight(self.w, VQK)
        with torch.no_grad():
            self.w.data = self.w.data.clone() * 2.0  # new storage (data_ptr changes)
        b = mx6.mx6_weight(self.w, VQK)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(_same(b, mx6.mx6_weight_reference(self.w, VQK)))
        twin = self.w.detach().clone()  # equal values, another tensor: its own entry
        c = mx6.mx6_weight(twin, VQK)
        self.assertEqual(len(self.calls), 3)
        self.assertTrue(_same(c, b))
        self.assertIsNot(c[0], b[0])
        del a

    def test_layouts_and_dtypes_are_separate_entries(self) -> None:
        a = mx6.mx6_weight(self.w, VQK)
        b = mx6.mx6_weight(self.w, VK)
        c = mx6.mx6_weight(self.w, [[512, 2048]])  # same layout spelled as lists: a hit
        self.assertEqual(len(self.calls), 2)
        self.assertIs(c[0], a[0])
        self.assertEqual(tuple(b[0].shape), (4, 2, 4, 12288))
        wb = self.w.detach().to(torch.bfloat16)
        mx6.mx6_weight(wb, VQK)
        self.assertEqual(len(self.calls), 3)

    def test_dead_weight_entries_are_dropped(self) -> None:
        tmp = torch.randn((512, 2048)) * 0.02
        mx6.mx6_weight(tmp, VQK)
        del tmp
        gc.collect()
        mx6.mx6_weight(self.w, VQK)  # a refresh purges entries of freed weights
        self.assertEqual(mx6.shadow_cache_stats()["entries"], 1)

    def test_freed_id_reuse_cannot_hit(self) -> None:
        # a new tensor can reuse a freed tensor's id(); the weakref identity check refuses the old entry
        tmp = torch.zeros((512, 2048))
        mx6.mx6_weight(tmp, VQK)
        key_id = id(tmp)
        del tmp
        gc.collect()
        for _ in range(64):
            other = torch.ones((512, 2048))
            if id(other) == key_id:
                break
        got = mx6.mx6_weight(other, VQK)
        self.assertTrue(_same(got, mx6.mx6_weight_reference(other, VQK)))


if __name__ == "__main__":
    unittest.main()
