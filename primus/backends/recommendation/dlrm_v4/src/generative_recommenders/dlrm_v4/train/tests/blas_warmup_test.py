#!/usr/bin/env python3
# pyre-strict
"""BLAS_WARMUP: default off, never raises, and never moves the RNG stream.

CPU-safe: the GEMM-running checks skip without a GPU."""

import unittest

import torch

from generative_recommenders.dlrm_v4.train import blas_warmup as bw


class BlasWarmupTest(unittest.TestCase):
    def test_default_off_is_noop(self) -> None:
        cpu = torch.random.get_rng_state()
        bw.run_blas_warmup(device=torch.device("cpu"))  # enabled defaults to 0
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))

    def test_cpu_device_skips_without_rng_change(self) -> None:
        cpu = torch.random.get_rng_state()
        ok, failed, _ = bw.warmup(torch.device("cpu"))
        self.assertEqual((ok, failed), (0, 0))
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))

    def test_tiny_m(self) -> None:
        # Must stay far below any production token count and a multiple of 16
        # for torch._scaled_mm.
        self.assertEqual(bw.WARMUP_M % 16, 0)
        self.assertLess(bw.WARMUP_M, 1024)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_gpu_warmup_rng_unchanged(self) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        cpu = torch.random.get_rng_state()
        cuda = torch.cuda.get_rng_state(device)
        ok, failed, _ = bw.warmup(device)
        self.assertGreater(ok, 0)
        self.assertEqual(failed, 0)
        self.assertTrue(torch.equal(cpu, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state(device)))


if __name__ == "__main__":
    unittest.main()
