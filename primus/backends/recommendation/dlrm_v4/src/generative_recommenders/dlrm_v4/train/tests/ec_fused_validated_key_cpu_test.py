"""EC_REDUCE_FUSED offline-validated key is geometry-aware (CPU only, no kernel launch).

_FUSED_VALIDATED was validated offline at BLOCK_S=32 / SPLIT_ABOVE=1024 (rccl iter 72).
Its key carries both constexprs, so the default synthetic-derived geometry (BLOCK_S=16,
SPLIT_ABOVE=8192, s15p1b run 70728) must take the one-time raw-bit self-test instead of
reusing the offline entry, while the incumbent geometry still reuses it with no probe.
GPU pieces (_even_odd_ok, the self-test inputs and both launches) are stubbed.
"""

import unittest

import torch

try:
    from generative_recommenders.dlrm_v4.train import ec_gpu_plan as gpu
    from generative_recommenders.dlrm_v4.train import ec_segment_reduce as ec
except ImportError as exc:  # e.g. triton missing
    raise unittest.SkipTest(f"needs ec_gpu_plan: {exc}")

DTYPES = (torch.bfloat16, torch.bfloat16)
STRIDES = (512, 512)  # unit-stride False, % 16 True: the offline-validated class


class FusedValidatedKeyTest(unittest.TestCase):
    def setUp(self):
        self._saved = {n: getattr(gpu, n) for n in
                       ("_even_odd_ok", "_fused_selftest_members",
                        "_segment_reduce_rows_lpt", "_ecf_launch", "_ecf_print")}
        self._fused_ok = dict(gpu._FUSED_OK)
        gpu._FUSED_OK.clear()
        self.launches = []
        gpu._even_odd_ok = lambda *a, **k: True
        gpu._fused_selftest_members = lambda w, d, s, dev: ([(None, None, None)] * 2, 4)
        gpu._segment_reduce_rows_lpt = lambda g, o, f, rows, dt, acc: torch.zeros(rows, 1)

        def launch(members, rows, eo, src0):
            self.launches.append(src0)
            # only src0=False reproduces the per-member bits in this stub
            return torch.zeros(rows, 1) if not src0 else torch.ones(rows, 1)

        gpu._ecf_launch = launch
        gpu._ecf_print = lambda msg: None

    def tearDown(self):
        for n, v in self._saved.items():
            setattr(gpu, n, v)
        gpu._FUSED_OK.clear()
        gpu._FUSED_OK.update(self._fused_ok)
        ec.configure_segment_reduce()

    def test_module_defaults_are_synthetic_pick(self):
        ec.configure_segment_reduce()
        self.assertEqual((ec._SPLIT_ABOVE, ec._BLOCK_S, ec._NUM_WARPS, ec._BLOCK_W_CAP),
                         (8192, 16, 2, 512))

    def test_validated_entries_tagged_with_incumbent_geometry(self):
        self.assertTrue(gpu._FUSED_VALIDATED)
        for key in gpu._FUSED_VALIDATED:
            self.assertEqual(key[:2], (32, 1024))

    def test_new_geometry_runs_selftest(self):
        ec.configure_segment_reduce()
        got = gpu._fused_src0(512, 512, 1, "cpu", DTYPES, STRIDES)
        self.assertEqual(self.launches, [True, False])  # probed, offline entry not reused
        self.assertIs(got, False)
        self.assertIn(("cpu", 16, 8192, 512, 512) + (DTYPES, (False, False), (True, True),
                                                      (True, True)), gpu._FUSED_OK)

    def test_block_s_alone_changes_the_key(self):
        ec.configure_segment_reduce(split_above=1024, block_s=16)
        gpu._fused_src0(512, 512, 1, "cpu", DTYPES, STRIDES)
        self.assertEqual(self.launches, [True, False])

    def test_incumbent_geometry_reuses_offline_entry(self):
        ec.configure_segment_reduce(split_above=1024, block_s=32)
        for dtypes, strides in ((DTYPES, STRIDES), (DTYPES[:1], STRIDES[:1])):
            self.assertIs(gpu._fused_src0(512, 512, 1, "cpu", dtypes, strides), True)
        self.assertEqual(self.launches, [])  # no runtime probe


if __name__ == "__main__":
    unittest.main()
