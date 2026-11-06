# pyre-strict
"""CPU tests for the per-stack BLAS first-use replay table (blas_warmup.replay_blas_table) and its generator
(scripts/gen_blas_replay_table.py). The GPU behaviour (every row count of the dedup chain fast after the replay)
is checked by the generator's `verify` mode on the stack the table was recorded on."""

import importlib.util
import json
import os
import unittest
from unittest import mock

import torch

from generative_recommenders.dlrm_v4.train import blas_warmup

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRAIN = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_TRAIN)))
_GEN = os.path.join(_ROOT, "scripts", "gen_blas_replay_table.py")


def _load_gen():  # pyre-ignore[3]
    spec = importlib.util.spec_from_file_location("gen_blas_replay_table", _GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # pyre-ignore[16]
    return mod


class BlasReplayTableTest(unittest.TestCase):
    def test_tables_well_formed(self) -> None:
        for name in blas_warmup._BLAS_REPLAY_TABLES:
            path = os.path.join(_TRAIN, name)
            self.assertTrue(os.path.exists(path), path)
            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
            hdr, rows = lines[0], lines[1:]
            self.assertEqual(hdr["kind"], "blas_replay_table")
            for k in blas_warmup._BLAS_REPLAY_KEYS:
                self.assertIn(k, hdr["stack"]["validators"])
            self.assertIsInstance(hdr["stack"]["cu"], int)
            self.assertTrue(rows)
            pairs = {tuple(p) for p in hdr["pairs"]}
            for r in rows:
                self.assertIn((r["K"], r["N"]), pairs)
                self.assertIn(r["op"], ("fwd", "dgrad", "wgrad"))
                self.assertGreater(r["M"], 0)
                # every row is a row count the dedup chain can produce (train 512 grid, eval 256 grid)
                self.assertEqual(r["M"] % 256, 0)
                self.assertLessEqual(r["M"], 512 * 8192)
            ms = [r["M"] for r in rows]
            self.assertEqual(ms, sorted(ms))

    def test_generator_grid_covers_dedup_row_counts(self) -> None:
        gen = _load_gen()
        pts = {}
        for part in range(8):
            for m, ops in gen.grid(part, 8):
                self.assertNotIn(m, pts)  # each point in exactly one part
                pts[m] = ops
        for k in range(1, 8193):
            self.assertEqual(pts[512 * k], ["dgrad", "fwd", "wgrad"])
            self.assertIn("fwd", pts[256 * k])
        shifted = {m for part in range(8) for m, _ in gen.grid(part, 8, shift=1)}
        self.assertEqual(shifted, set(pts))
        # the pairs are the dedup chain's FusedLinearNorm pairs (blas_warmup's production shapes)
        self.assertEqual(set(gen.PAIRS), {(blas_warmup._D, blas_warmup._PRE_HIDDEN),
                                          (blas_warmup._ADDL_IN, blas_warmup._PRE_HIDDEN),
                                          (blas_warmup._PRE_HIDDEN, blas_warmup._D)})

    def test_kill_switch_and_cpu(self) -> None:
        with mock.patch.dict(os.environ, {blas_warmup._BLAS_REPLAY_ENV: "0"}):
            self.assertEqual(blas_warmup.replay_blas_table(torch.device("cpu"))["state"], "off")
        with mock.patch.dict(os.environ, {blas_warmup._BLAS_REPLAY_ENV: "1"}):
            self.assertEqual(blas_warmup.replay_blas_table(torch.device("cpu"))["state"], "no_cuda")

    def test_replay_view_is_contiguous_prefix(self) -> None:
        big = torch.arange(64, dtype=torch.float32)
        v = blas_warmup._replay_view(big, 8, 3)
        self.assertEqual(tuple(v.shape), (3, 8))
        self.assertTrue(v.is_contiguous())
        self.assertEqual(v.data_ptr(), big.data_ptr())


if __name__ == "__main__":
    unittest.main()
