#!/usr/bin/env python3
# pyre-strict
"""DL_PRESTART: the pre-forked pools read nothing until released, then stream
exactly what the legacy build streams, with the same global RNG stream.

Uses real persistent fork workers on a toy dataset (CPU only)."""

import multiprocessing as mp
import time
import unittest
from unittest import mock

import numpy as np
import torch

from generative_recommenders.dlrm_v4.train import dl_prestart as dlp
from generative_recommenders.dlrm_v4.train import utils as tu

_READS = mp.get_context("fork").Value("i", 0)


class _Toy:
    is_eval = False

    def train_window_indices(self, ts: int) -> np.ndarray:
        return np.arange(ts * 100, ts * 100 + 64, dtype=np.int64)

    def load_query_samples(self, idx) -> None:  # pyre-ignore[2]
        with _READS.get_lock():
            _READS.value += len(idx)

    def get_sample(self, i: int) -> int:
        return int(i)

    def unload_query_samples(self, idx) -> None:  # pyre-ignore[2]
        pass


def _ints(xs):  # pyre-ignore[2,3]
    return [int(x) for x in xs]


class DlPrestartTest(unittest.TestCase):
    def _prefetcher(self) -> "tu._PrefetchingWindowLoader":
        dlp._reset_for_tests()
        base = tu.HammerToTorchDataset(_Toy())  # pyre-ignore[6]
        probed = tu._ReadProbedDataset(base, dlp.new_read_slot("train"))
        return tu._PrefetchingWindowLoader(
            dataset=base,
            sampler_factory=lambda: tu.StreamingWindowSampler(rank=0, world_size=2),
            dl_factory=lambda s: tu.make_persistent_streaming_dataloader(
                dataset=probed, sampler=s, batch_size=8, num_workers=2,
                prefetch_factor=2, pin_memory=False,
            ),
        )

    def _run(self, prestart: bool):  # pyre-ignore[3]
        _READS.value = 0
        torch.manual_seed(1234)
        pf = self._prefetcher()
        if prestart:
            before = torch.get_rng_state().clone()
            pf.prestart()
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            time.sleep(1.0)
            self.assertEqual(_READS.value, 0, "sample read before release")
            self.assertEqual(dlp.read_stamps(), {})
        release = time.time()
        out = [(ts, nb, list(it)) for ts, it, nb in pf.stream([0, 1, 2])]
        return out, torch.rand(4), release, dlp.read_stamps()

    def test_prestart_reads_nothing_then_matches_legacy(self) -> None:
        with mock.patch.object(tu, "collate_fn", _ints):
            ref, ref_rng, _, _ = self._run(prestart=False)
            got, got_rng, release, stamps = self._run(prestart=True)
        self.assertEqual(got, ref)
        self.assertTrue(torch.equal(got_rng, ref_rng), "RNG stream changed")
        self.assertGreaterEqual(stamps["train"], release)

    def test_report_flags_reads_before_run_start(self) -> None:
        dlp._reset_for_tests()
        slot = dlp.new_read_slot("train")
        dlp.note_read(slot)
        time.sleep(0.01)
        dlp.set_run_start()
        self.assertEqual(len(dlp.compliance_violations()), 1)
        line = dlp.report(rank=0, prestarted=True)
        self.assertIn("COMPLIANCE VIOLATION", line or "")


if __name__ == "__main__":
    unittest.main()
