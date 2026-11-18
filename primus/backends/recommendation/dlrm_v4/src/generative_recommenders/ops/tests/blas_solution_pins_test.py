# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Per-stack pin tables of ops/blas_solution_pins.py (CPU: the stack is mocked).

  * v26.3 validators + the gin request: the same ten TunableOp rows and the same
    synthesized cache file as before the per-stack change; no ScaledGemm row
  * ROCm 10.2 validators + the gin request: only the three ScaledGemm rows, no
    TunableOp line written; the v26.3 keys are skipped, not an error
  * a key of no table raises; validators of no table disable pinning (no raise)
  * scaled_mm without a matching row (or off-GPU) is torch._scaled_mm
"""

import os
import unittest
from typing import Dict, List
from unittest import mock

import torch
from generative_recommenders.ops import blas_solution_pins as pins

GIN = ("uvqk_fwd,uvqk_wgrad,out_fwd,out_wgrad,out_dgrad,targets_vk_wgrad,preproc_dgrad_n1024,"
       "preproc_dgrad_n512,preproc_l2_fwd,preproc_wgrad_n1024,uvqk_dgrad_fp8,targets_vk_dgrad_fp8,"
       "preproc_l2_dgrad_fp8")
OLD_GIN_ROWS = ["uvqk_fwd", "uvqk_wgrad", "out_fwd", "out_wgrad", "out_dgrad", "targets_vk_wgrad",
                "preproc_dgrad_n1024", "preproc_dgrad_n512", "preproc_l2_fwd", "preproc_wgrad_n1024"]


class _Tunable:
    def __init__(self, validators: Dict[str, str]) -> None:
        self.validators = validators
        self.files: List[str] = []

    def get_validators(self):
        return tuple(self.validators.items())

    def is_enabled(self) -> bool:
        return True

    def tuning_is_enabled(self) -> bool:
        return False

    def read_file(self, path: str) -> bool:
        with open(path) as f:
            self.files.append(f.read())
        return True


class PinStacksTest(unittest.TestCase):
    def _reset(self) -> None:
        pins._state = None
        pins._active = []
        pins._active_smm = {}
        pins._stack = None
        pins._seen.clear()
        pins._path = None
        pins._smm_broken.clear()

    def _run(self, validators: Dict[str, str], request: str) -> _Tunable:
        self._reset()
        tun = _Tunable(validators)
        with mock.patch.dict(os.environ, {pins.ENV_VAR: request}), \
                mock.patch.object(torch.cuda, "is_available", return_value=True), \
                mock.patch.object(torch.version, "hip", "7.17.26412"), \
                mock.patch("torch.cuda.tunable.get_validators", tun.get_validators), \
                mock.patch("torch.cuda.tunable.is_enabled", tun.is_enabled), \
                mock.patch("torch.cuda.tunable.tuning_is_enabled", tun.tuning_is_enabled), \
                mock.patch("torch.cuda.tunable.read_file", tun.read_file):
            pins.install_for_tokens(2365018)
            pins.install_for_tokens(2365018)  # once per count
        return tun

    def tearDown(self) -> None:
        self._reset()

    def test_v263_rows_unchanged(self) -> None:
        tun = self._run(pins.EXPECTED_ENV, GIN)
        self.assertTrue(pins._state)
        self.assertEqual(pins._stack.env, pins.EXPECTED_ENV)
        self.assertEqual([p[0] for p in pins._active], OLD_GIN_ROWS)
        self.assertEqual(pins._active_smm, {})
        self.assertEqual(len(tun.files), 1)
        expect = [f"Validator,{k},{v}" for k, v in pins.EXPECTED_ENV.items()]
        expect += [f"{sig},{tmpl.format(M=2365018)},{sol},0.0"
                   for name, sig, tmpl, sol in pins._PINS if name in OLD_GIN_ROWS]
        self.assertEqual(tun.files[0], "\n".join(expect) + "\n")

    def test_rocm102_scaled_mm_rows_only(self) -> None:
        tun = self._run(pins.EXPECTED_ENV_ROCM102, GIN)
        self.assertTrue(pins._state)
        self.assertEqual(pins._active, [])
        self.assertEqual(sorted(p.name for p in pins._active_smm.values()),
                         ["preproc_l2_dgrad_fp8", "targets_vk_dgrad_fp8", "uvqk_dgrad_fp8"])
        self.assertEqual(tun.files, [])  # nothing for TunableOp on this stack
        self.assertEqual(sorted((p.k, p.n) for p in pins._active_smm.values()),
                         [(512, 256), (1024, 512), (2048, 512)])
        for p in pins._active_smm.values():
            self.assertEqual((p.a_dtype, p.b_dtype, p.out_dtype, p.solution),
                             (torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16, 33058))

    def test_rocm102_only_v263_keys_requested(self) -> None:
        self._run(pins.EXPECTED_ENV_ROCM102, "uvqk_wgrad,out_wgrad")
        self.assertFalse(pins._state)

    def test_all_on_rocm102(self) -> None:
        self._run(pins.EXPECTED_ENV_ROCM102, "1")
        self.assertEqual(len(pins._active_smm), 3)

    def test_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._run(pins.EXPECTED_ENV_ROCM102, "uvqk_dgrad_fp8,no_such_pin")

    def test_unknown_stack_is_a_noop(self) -> None:
        other = dict(pins.EXPECTED_ENV_ROCM102, HIPBLASLT_VERSION="100600-0000000")
        tun = self._run(other, GIN)
        self.assertFalse(pins._state)
        self.assertEqual(tun.files, [])

    def test_scaled_mm_without_row_is_torch(self) -> None:
        self._reset()
        pins._state = False
        a = torch.randn(32, 64)
        b = torch.randn(48, 64).t()
        sentinel = torch.zeros(1)
        with mock.patch.object(torch, "_scaled_mm", return_value=sentinel) as smm:
            out = pins.scaled_mm(a, b, torch.ones(1), torch.ones(1), out_dtype=torch.bfloat16)
        self.assertIs(out, sentinel)
        smm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
