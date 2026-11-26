#!/usr/bin/env python3
# pyre-strict
"""V-ring swizzle lab knob (X4C19 iter70): DLRM_HSTU_FP8_LAB_FWD_V8SWZ parsing,
the compile kwargs fp8_fwd_v8swz_config hands to fwd_kernel_0, and the default-off
identity of compile_hstu_fwd's v8_swz parameter.

CPU-only: nothing here compiles or launches a kernel. The kernel change
(hstu_fwd.py v8_swz=1: V ring rows 16..31 +128 B) is covered by the CPU AOT ISA
checks and the GPU checks of the iteration that ported it (x4c19 iter53/iter70)."""

import inspect
import os
import unittest
from unittest import mock

_KNOB = "DLRM_HSTU_FP8_LAB_FWD_V8SWZ"


def _mha():  # pyre-ignore[3]
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

    return mha


class FwdV8SwzKnobTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mha = _mha()
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        # The once-latch is process state: keep this test from leaking into it.
        self._latch = set(self.mha._fp8_fwd_v8swz_logged)
        self.addCleanup(self._restore_latch)

    def _restore_latch(self) -> None:
        self.mha._fp8_fwd_v8swz_logged.clear()
        self.mha._fp8_fwd_v8swz_logged.update(self._latch)

    def _swz(self, value, cfg):  # pyre-ignore[2,3]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            if value is not None:
                os.environ[_KNOB] = value
            return self.mha.fp8_fwd_v8swz_config(cfg)

    def test_off_values_change_nothing(self) -> None:
        pv_cfg = {"pv": "k128"}
        for value in (None, "", "0"):
            self.assertEqual(self._swz(value, pv_cfg), {}, value)
            self.assertEqual(self._swz(value, {}), {}, value)

    def test_armed_only_with_pv(self) -> None:
        self.assertEqual(self._swz("1", {"pv": "k128", "pv_k": 3}), {"v8_swz": 1})
        # Not the pv=k128 forward: declined, no raise, nothing added.
        self.assertEqual(self._swz("1", {}), {})
        self.assertEqual(self._swz("1", {"pv": ""}), {})

    def test_invalid_value_fails_closed(self) -> None:
        for bad in ("2", "true", "on", " 1", "1 ", "yes"):
            with self.assertRaises(ValueError, msg=bad):
                self._swz(bad, {"pv": "k128"})

    def test_default_config_has_no_v8swz(self) -> None:
        # With the knob unset, _fp8_fwd_config must not carry v8_swz, so the
        # shipped compile kwargs (and hence the kernel) are unchanged.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            try:
                cfg = self.mha._fp8_fwd_config()
            except Exception as error:  # pragma: no cover - environment specific
                self.skipTest(f"_fp8_fwd_config not evaluable here: {error}")
        self.assertNotIn("v8_swz", cfg)

    def test_kernel_parameter_default_off(self) -> None:
        try:
            kernel = self.mha._load_fwd_kernel()
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable here: {error}")
        # _load_fwd_kernel() returns None (it does not raise) when FlyDSL is
        # missing, e.g. in a CPU-only test container.
        if kernel is None:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable: {self.mha._fwd_kernel_error}")
        params = inspect.signature(kernel.compile_hstu_fwd).parameters
        self.assertIn("v8_swz", params)
        self.assertEqual(params["v8_swz"].default, 0)

    def test_warmup_keeps_the_latch(self) -> None:
        # Ship17 row (100) v2: the line F1 prints is the arming record. The
        # latch is not restored (not in _MHA_SETS; a _KEEP_GLOBALS entry so
        # VERIFY ignores it) and the line leaves F1's stdout capture
        # (_STDOUT_PASS), so it prints once, before RUN_START. The end-to-end
        # check is fwd_rider_knob_test.test_warmup_prints_each_line_once.
        try:
            from generative_recommenders.dlrm_v4.train import fp8_warmup
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup not importable here: {error}")
        self.assertNotIn("_fp8_fwd_v8swz_logged", fp8_warmup._MHA_SETS)
        self.assertIn((fp8_warmup._P_FLYDSL + ".hstu_flydsl_mha",
                       "_fp8_fwd_v8swz_logged"), fp8_warmup._KEEP_GLOBALS)
        self.assertIn("[hstu-fwd] " + _KNOB + "=", fp8_warmup._STDOUT_PASS)


if __name__ == "__main__":
    unittest.main()
