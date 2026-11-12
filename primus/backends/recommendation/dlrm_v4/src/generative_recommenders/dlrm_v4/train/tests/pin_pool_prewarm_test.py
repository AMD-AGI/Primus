# pyre-strict
"""CPU tests for the pinned-pool pre-grow (dlrm_v4/train/utils.pin_pool_prewarm, yambda_5b.gin item (121)): one
block per in-flight batch of every pre-forked pinned pool (+1), in the pool's power-of-two class; kill switch; no
pools -> no-op; and the free-memory check before pinning (_pin_free_check): enough free memory passes, a short NUMA
node is reclaimed and re-checked, memory that stays short stops the run before RUN_START."""

import collections
import os
import types
import unittest
from typing import Any, List
from unittest import mock

import torch

from generative_recommenders.dlrm_v4.train import utils


class _DL:
    def __init__(self, nw: int, pf: int, bs: Any, pin: bool = True) -> None:
        self.num_workers = nw
        self.prefetch_factor = pf
        self.batch_size = bs
        self.pin_memory = pin


class PinPoolPrewarmTest(unittest.TestCase):
    def _run(self, bundle: Any, env: str = "1") -> List[int]:
        calls: List[int] = []
        real_empty = torch.empty

        def fake_empty(n: int, dtype: Any = None, pin_memory: bool = False) -> torch.Tensor:
            self.assertTrue(pin_memory)
            calls.append(n)
            return real_empty(1, dtype=dtype)

        with mock.patch.object(utils, "_PRESTART_BUNDLE", bundle), \
                mock.patch.object(utils.torch.cuda, "is_available", lambda: True), \
                mock.patch.object(utils.torch, "empty", fake_empty), \
                mock.patch.object(utils, "_pin_free_check", lambda *a, **k: None), \
                mock.patch.dict(os.environ, {utils._PIN_PREWARM_ENV: env}):
            utils.pin_pool_prewarm(0)
        return calls

    def test_yambda_plan(self) -> None:
        bundle = {"prefetcher": types.SimpleNamespace(_dls=[_DL(6, 8, None)]), "eval_dl": _DL(8, 8, 1024)}
        c = collections.Counter(self._run(bundle))
        self.assertEqual(c, {128 << 20: 49, 256 << 20: 65})

    def test_classes_scale_with_batch(self) -> None:
        bundle = {"prefetcher": types.SimpleNamespace(_dls=[_DL(2, 4, 512), _DL(2, 4, 512)]),
                  "eval_dl": _DL(4, 2, 512)}
        c = collections.Counter(self._run(bundle))
        self.assertEqual(c, {64 << 20: 17, 128 << 20: 9})

    def test_unpinned_pools_and_kill_switch(self) -> None:
        bundle = {"prefetcher": types.SimpleNamespace(_dls=[_DL(6, 8, None, pin=False)]),
                  "eval_dl": _DL(8, 8, 1024, pin=False)}
        self.assertEqual(self._run(bundle), [])
        bundle = {"prefetcher": types.SimpleNamespace(_dls=[_DL(6, 8, None)]), "eval_dl": _DL(8, 8, 1024)}
        self.assertEqual(self._run(bundle, env="0"), [])
        self.assertEqual(self._run(None), [])


G = 1 << 30


class PinFreeCheckTest(unittest.TestCase):
    """_pin_free_check with /sys meminfo and the reclaim balloon mocked: per-node FREE memory vs share + headroom."""

    def _check(self, free_seq: List[dict], reclaim: Any = 1, headroom_gib: Any = 10, per_rank_gib: int = 2,
               local: int = 8) -> List[int]:
        # free_seq: successive {node: free_gib} returned by _numa_free_bytes (before, after reclaim, ...)
        seq = [{n: (f * G, 0, 0) for n, f in d.items()} for d in free_seq]
        calls: List[int] = []

        def fake_free() -> dict:
            return seq.pop(0) if len(seq) > 1 else seq[0]

        def fake_reclaim(target: int) -> str:
            calls.append(target)
            return "reclaimed"

        with mock.patch.object(utils, "_numa_free_bytes", fake_free), \
                mock.patch.object(utils, "_pin_reclaim", fake_reclaim), \
                mock.patch.dict(os.environ, {"LOCAL_WORLD_SIZE": str(local)}):
            utils._pin_free_check(0, per_rank_gib * G, reclaim=reclaim, headroom_gib=headroom_gib)
        return calls

    def test_enough_free_memory_passes_without_reclaim(self) -> None:
        # share = 2 GiB x 8 ranks / 2 nodes = 8 GiB; + 10 headroom = 18 GiB needed per node
        self.assertEqual(self._check([{0: 18, 1: 100}]), [])

    def test_short_node_is_reclaimed_then_passes(self) -> None:
        calls = self._check([{0: 5, 1: 100}, {0: 50, 1: 100}])
        self.assertEqual(calls, [18 * G])

    def test_still_short_after_reclaim_stops_the_run(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"NUMA node\(s\) 0 .*even after reclaiming page cache.*before RUN_START"):
            self._check([{0: 5, 1: 100}, {0: 6, 1: 100}])

    def test_reclaim_off_stops_without_reclaiming(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"Page-cache reclaim is off"):
            self._check([{0: 5, 1: 100}], reclaim=0)

    def test_headroom_is_honoured(self) -> None:
        self.assertEqual(self._check([{0: 18, 1: 18}], headroom_gib=10), [])
        with self.assertRaises(RuntimeError):
            self._check([{0: 18, 1: 18}], headroom_gib=11, reclaim=0)

    def test_no_numa_info_is_a_no_op(self) -> None:
        with mock.patch.object(utils, "_numa_free_bytes", lambda: {}):
            utils._pin_free_check(0, 100 * G, reclaim=0, headroom_gib=10)

    def test_prewarm_passes_gin_values_and_kill_switch_skips_the_check(self) -> None:
        seen: List[Any] = []
        bundle = {"prefetcher": types.SimpleNamespace(_dls=[_DL(6, 8, None)]), "eval_dl": _DL(8, 8, 1024)}
        with mock.patch.object(utils, "_PRESTART_BUNDLE", bundle), \
                mock.patch.object(utils.torch.cuda, "is_available", lambda: True), \
                mock.patch.object(utils.torch, "empty", lambda n, dtype=None, pin_memory=False: torch.empty(1)), \
                mock.patch.object(utils, "_pin_free_check", lambda r, b, **k: seen.append((b, k))):
            utils.pin_pool_prewarm(0, enabled=1, reclaim=0, headroom_gib=64)
            utils.pin_pool_prewarm(0, enabled=0)
        self.assertEqual(seen, [((49 * 128 + 65 * 256) << 20, {"reclaim": 0, "headroom_gib": 64})])


if __name__ == "__main__":
    unittest.main()
