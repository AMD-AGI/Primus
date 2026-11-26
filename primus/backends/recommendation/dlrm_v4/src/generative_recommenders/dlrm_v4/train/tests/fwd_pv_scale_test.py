#!/usr/bin/env python3
# pyre-strict
"""FP8 P@V forward scale knob (X4C19 iter50): DLRM_HSTU_FP8_FWD_PV_SCALE parsing
and the compile kwargs fp8_fwd_pv_config hands to fwd_kernel_0.

CPU-only: nothing here compiles or launches a kernel. The kernel itself (pv_s:
rcp(fma(ex, c, c)), c = f32(8 / s)) is covered by the GPU checks of the
iteration that added it."""

import os
import unittest
from unittest import mock


def _mha():  # pyre-ignore[3]
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

    return mha


class FwdPvScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mha = _mha()
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        # The once-latch is process state: keep this test from leaking into it.
        self._latch = set(self.mha._fp8_fwd_pv_logged)
        self.addCleanup(self._restore_latch)

    def _restore_latch(self) -> None:
        self.mha._fp8_fwd_pv_logged.clear()
        self.mha._fp8_fwd_pv_logged.update(self._latch)

    def _scale(self, value):  # pyre-ignore[2,3]
        env = {"DLRM_HSTU_FP8_FWD_PV_SCALE": value} if value is not None else {}
        with mock.patch.dict(os.environ, env, clear=False):
            if value is None:
                os.environ.pop("DLRM_HSTU_FP8_FWD_PV_SCALE", None)
            return self.mha._fp8_fwd_pv_scale()[0]

    def test_scale_parsing(self) -> None:
        self.assertEqual(self._scale(None), 10.04)
        self.assertEqual(self._scale(""), 10.04)
        self.assertEqual(self._scale(" "), 10.04)
        self.assertEqual(self._scale("pow2"), 0.0)
        self.assertEqual(self._scale(" POW2 "), 0.0)
        self.assertEqual(self._scale("0"), 0.0)
        self.assertEqual(self._scale("0.0"), 0.0)
        self.assertEqual(self._scale("8"), 8.0)
        self.assertEqual(self._scale("10.04"), 10.04)
        # Canonicalized to 6 decimals: one accepted value = one kernel.
        self.assertEqual(self._scale("10.043046056197529"), 10.043046)
        self.assertEqual(self._scale("4"), 4.0)
        self.assertEqual(self._scale("32"), 32.0)
        for bad in ("3.99", "32.5", "-8", "nan", "inf", "abc", "10,04"):
            self.assertIsNone(self._scale(bad), bad)

    def _pipe_cfg(self) -> dict:  # pyre-ignore[24]
        mha = self.mha
        try:
            cfg = dict(mha._load_fwd_kernel().SHIPPED)
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable here: {error}")
        cfg.update(mha._FP8_FWD_CFG)
        cfg.update(mha._FP8_FWD_E2_CFG)
        cfg.update(mha._FP8_FWD_PIPE_CFG)
        cfg["mask_arith"] = mha._FP8_FWD_MASK_ARITH
        return cfg

    def _pv(self, env, cfg):  # pyre-ignore[2,3]
        keys = ("DLRM_HSTU_FP8_FWD_PV", "DLRM_HSTU_FP8_FWD_PV_SCALE")
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in keys:
                os.environ.pop(key, None)
            os.environ.update(env)
            return self.mha.fp8_fwd_pv_config(cfg)

    def test_config(self) -> None:
        cfg = self._pipe_cfg()
        self.assertEqual(self._pv({}, cfg), {})
        default = self._pv({"DLRM_HSTU_FP8_FWD_PV": "1"}, cfg)
        self.assertEqual(default.get("pv"), "k128")
        self.assertEqual(default.get("pv_s"), 10.04)
        # pow2: exactly the iteration-46 kwargs (no pv_s key at all).
        pow2 = self._pv({"DLRM_HSTU_FP8_FWD_PV": "1",
                         "DLRM_HSTU_FP8_FWD_PV_SCALE": "pow2"}, cfg)
        self.assertEqual(pow2, dict(self.mha._FP8_FWD_PV_CFG))
        self.assertNotIn("pv_s", pow2)
        # Fail closed: an invalid scale raises instead of running the control.
        with self.assertRaises(ValueError):
            self._pv({"DLRM_HSTU_FP8_FWD_PV": "1",
                      "DLRM_HSTU_FP8_FWD_PV_SCALE": "abc"}, cfg)
        # Not the FWD_PIPE config: declined (bf16-P forward), no raise.
        other = dict(cfg)
        other["BM"] = 64
        self.assertEqual(self._pv({"DLRM_HSTU_FP8_FWD_PV": "1",
                                   "DLRM_HSTU_FP8_FWD_PV_SCALE": "abc"}, other), {})


if __name__ == "__main__":
    unittest.main()
