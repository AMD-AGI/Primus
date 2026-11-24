#!/usr/bin/env python3
# pyre-strict
"""Forward epilogue rider lab knob (X4C19 iter71): DLRM_HSTU_FP8_LAB_FWD_RIDER
parsing, the compile kwargs fp8_fwd_rider_config hands to fwd_kernel_0, its
composition with DLRM_HSTU_FP8_LAB_FWD_V8SWZ, compile_hstu_fwd's `rider`
parameter (default-off identity, fail-closed masks, LDS symbol suffix), and
the FP8 warm-up's handling of the two lab once-latches (ship17 row (100) v2:
kept, so each line prints once, in warm-up item F1, before RUN_START).

CPU-only: nothing here launches a kernel. compile_hstu_fwd is called with a
recording SmemAllocator stand-in, which only builds the kernel closures (no
MLIR is traced before a launch). The kernel change itself (hstu_fwd.py
rider=3) is covered by the CPU AOT ISA checks of x4c19 iter71 (rider)."""

import contextlib
import inspect
import io
import os
import unittest
from unittest import mock

_KNOB = "DLRM_HSTU_FP8_LAB_FWD_RIDER"
_V8SWZ = "DLRM_HSTU_FP8_LAB_FWD_V8SWZ"
# The production forward (ship15.1 gin): FWD_PIPE + MASK_ARITH + FWD_PV.
_PROD_ENV = {"DLRM_HSTU_FP8_FWD_PIPE": "1", "DLRM_HSTU_FP8_FWD_MASK_ARITH": "1",
             "DLRM_HSTU_FP8_FWD_PV": "1"}
_SHIPPED_SYM = ("hstu_fwd_smem_128_32_16_0_w4_k128_16_1_110_pf1_pd1_ws0_ma3"
                "_pv8k3_s1_vp16_x111_ps10040000")
_LATCH_SETS = ("_fp8_fwd_rider_logged", "_fp8_fwd_v8swz_logged",
               "_fp8_fwd_pv_logged", "_fp8_fwd_mask_arith_logged")
_LATCH_SCALARS = ("_fp8_fwd_pipe_logged",)


def _mha():  # pyre-ignore[3]
    from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

    return mha


class _Alloc:
    """SmemAllocator stand-in: records global_sym_name, keeps the offsets."""

    names = []  # pyre-ignore[4]

    def __init__(self, _base, arch=None, global_sym_name=""):  # pyre-ignore[2]
        self.ptr = 0
        _Alloc.names.append(global_sym_name)

    @staticmethod
    def _align(ptr, align):  # pyre-ignore[2,3]
        return (ptr + align - 1) // align * align


class FwdRiderKnobTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mha = _mha()
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        # Once-latches are process state: keep this test from leaking into them.
        saved = {n: set(getattr(self.mha, n)) for n in _LATCH_SETS}
        scalars = {n: getattr(self.mha, n) for n in _LATCH_SCALARS}

        def restore() -> None:
            for n, v in saved.items():
                getattr(self.mha, n).clear()
                getattr(self.mha, n).update(v)
            for n, v in scalars.items():
                setattr(self.mha, n, v)
        self.addCleanup(restore)

    def _env(self, values):  # pyre-ignore[2,3]
        """Context: the DLRM_* knobs of this test set to `values` exactly."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_")}
        env.update({k: v for k, v in values.items() if v is not None})
        return mock.patch.dict(os.environ, env, clear=True)

    def _rider(self, value, cfg):  # pyre-ignore[2,3]
        with self._env({_KNOB: value}):
            return self.mha.fp8_fwd_rider_config(cfg)

    def test_off_values_change_nothing(self) -> None:
        for value in (None, "", "0"):
            self.assertEqual(self._rider(value, {"pv": "k128"}), {}, value)
            self.assertEqual(self._rider(value, {}), {}, value)

    def test_values_map_to_masks(self) -> None:
        self.assertEqual(self._rider("1", {"pv": "k128", "pv_k": 3}),
                         {"rider": 3})
        # The lab value "2" (rider 7, R2c inline asm) is gone (row (100) v2).
        with self.assertRaises(ValueError):
            self._rider("2", {"pv": "k128", "pv_k": 3})

    def test_declined_without_pv(self) -> None:
        self.assertEqual(self._rider("1", {}), {})
        self.assertEqual(self._rider("1", {"pv": ""}), {})

    def test_invalid_value_fails_closed(self) -> None:
        for bad in ("2", "3", "4", "7", "-1", "01", "1.0", "true", "on", " 1",
                    "1 ", "yes", "R2a"):
            for cfg in ({"pv": "k128"}, {}):
                with self.assertRaises(ValueError, msg=(bad, cfg)):
                    self._rider(bad, cfg)

    def test_default_config_has_no_rider(self) -> None:
        for env in ({}, _PROD_ENV):
            with self._env(env):
                try:
                    cfg = self.mha._fp8_fwd_config()
                except Exception as error:  # pragma: no cover
                    self.skipTest(f"_fp8_fwd_config not evaluable: {error}")
            self.assertNotIn("rider", cfg)

    def test_composes_with_v8swz(self) -> None:
        with self._env(_PROD_ENV):
            try:
                base = self.mha._fp8_fwd_config()
            except Exception as error:  # pragma: no cover
                self.skipTest(f"_fp8_fwd_config not evaluable: {error}")
        self.assertEqual(base.get("pv"), "k128")
        cases = {
            ("1", None): {"rider": 3},
            ("1", "1"): {"rider": 3, "v8_swz": 1},
            (None, "1"): {"v8_swz": 1},
        }
        for (rd, vz), extra in cases.items():
            with self._env(dict(_PROD_ENV, **{_KNOB: rd, _V8SWZ: vz})):
                cfg = self.mha._fp8_fwd_config()
            want = dict(base)
            want.update(extra)
            self.assertEqual(cfg, want, (rd, vz))
        # Not the pv=k128 forward: both lab knobs decline, nothing added.
        with self._env({_KNOB: "1", _V8SWZ: "1",
                        "DLRM_HSTU_FP8_FWD_PIPE": "1"}):
            cfg = self.mha._fp8_fwd_config()
        self.assertNotIn("rider", cfg)
        self.assertNotIn("v8_swz", cfg)

    def test_kernel_parameter_default_off(self) -> None:
        kernel = self._kernel()
        params = inspect.signature(kernel.compile_hstu_fwd).parameters
        self.assertIn("rider", params)
        self.assertEqual(params["rider"].default, 0)

    def _warmup(self):  # pyre-ignore[3]
        try:
            from generative_recommenders.dlrm_v4.train import fp8_warmup
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup not importable here: {error}")
        return fp8_warmup

    def test_warmup_keeps_the_latch(self) -> None:
        # Row (100) v2: the F1 line is the arming record. The latch is not
        # restored (not in _MHA_SETS, a _KEEP_GLOBALS entry so VERIFY ignores
        # it) and the line leaves F1's stdout capture (_STDOUT_PASS).
        fw = self._warmup()
        self.assertNotIn("_fp8_fwd_rider_logged", fw._MHA_SETS)
        self.assertIn((fw._P_FLYDSL + ".hstu_flydsl_mha",
                       "_fp8_fwd_rider_logged"), fw._KEEP_GLOBALS)
        self.assertIn("[hstu-fwd] " + _KNOB + "=", fw._STDOUT_PASS)

    def test_warmup_prints_each_line_once(self) -> None:
        """An F1 stand-in run by fp8_warmup's own _Runner between
        _Guards.take() and restore() reaches both config functions (F1 does,
        through _fp8_fwd_config): each line prints once, during the item, the
        latch outlives the restore (the timed region prints nothing) and the
        VERIFY digest sees no difference. Armed and declined forms."""
        fw = self._warmup()

        class _Bufs:
            def reset(self) -> None:
                pass

        lines = {"armed": (f"[hstu-fwd] {_V8SWZ}=1 armed: v8_swz=1",
                           f"[hstu-fwd] {_KNOB}=1 armed: rider=3"),
                 "declined": (f"[hstu-fwd] {_V8SWZ}=1 declined: ",
                              f"[hstu-fwd] {_KNOB}=1 declined: ")}
        for form, cfg in (("armed", {"pv": "k128", "pv_k": 3}),
                          ("declined", {})):
            for name in ("_fp8_fwd_v8swz_logged", "_fp8_fwd_rider_logged"):
                getattr(self.mha, name).clear()  # a fresh process

            def calls(cfg=cfg) -> None:  # pyre-ignore[2]
                self.mha.fp8_fwd_v8swz_config(cfg)
                self.mha.fp8_fwd_rider_config(cfg)

            warm, timed = io.StringIO(), io.StringIO()
            with self._env({_KNOB: "1", _V8SWZ: "1"}):
                guards = fw._Guards()
                runner = fw._Runner(fw.torch.device("cpu"), _Bufs(), True,
                                    set(), False)
                with contextlib.redirect_stdout(warm):
                    guards.take(verify=True)
                    try:
                        with mock.patch.object(fw.torch.cuda, "synchronize",
                                               lambda *a, **k: None):
                            runner.run("F1", "flydsl_fwd_fp8", "test", calls)
                    finally:
                        guards.restore()
                guards.verify_globals()
                with contextlib.redirect_stdout(timed):
                    for _ in range(3):  # the real calls after RUN_START
                        calls()
            self.assertEqual((runner.ok, runner.failed), (1, 0), form)
            self.assertEqual(guards.changed, [], form)
            self.assertEqual(guards.globals_changed, [], form)
            for line in lines[form]:
                self.assertEqual(warm.getvalue().count(line), 1, line)
                self.assertEqual(timed.getvalue().count(line), 0, line)

    def _kernel(self):  # pyre-ignore[3]
        kernel = self.mha._load_fwd_kernel()
        if kernel is None:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable: {self.mha._fwd_kernel_error}")
        return kernel

    def _build(self, **kw):  # pyre-ignore[2,3]
        """compile_hstu_fwd (uncached) at the production kwargs + `kw`;
        returns the LDS global symbol name it chose."""
        kernel = self._kernel()
        with self._env(_PROD_ENV):
            cfg = self.mha._fp8_fwd_config()
        cfg.update(kw)
        _Alloc.names = []
        with mock.patch.object(kernel, "SmemAllocator", _Alloc), \
                mock.patch.object(kernel, "get_rocm_arch", lambda: "gfx950"):
            kernel.compile_hstu_fwd.__wrapped__(
                4, 128, 0.08838834764831843, rs=1536, **cfg)
        self.assertEqual(len(_Alloc.names), 1)
        return _Alloc.names[0]

    def test_symbol_suffix(self) -> None:
        self.assertEqual(self._build(), _SHIPPED_SYM)
        self.assertEqual(self._build(rider=0), _SHIPPED_SYM)
        self.assertEqual(self._build(rider=3), _SHIPPED_SYM + "_rd3")
        self.assertEqual(self._build(v8_swz=1, rider=3),
                         _SHIPPED_SYM + "_vz1_rd3")
        self.assertEqual(self._build(v8_swz=1), _SHIPPED_SYM + "_vz1")

    def test_kernel_rejects_bad_masks(self) -> None:
        # Only 0 and 3 (R2a + R2b): R2c (bit 4) is gone, and so are the lab
        # partial masks 1, 2 and 6 (row (100) v2).
        for bad in (1, 2, 4, 5, 6, 7, 8, 15, -1, True, 3.0, "3"):
            with self.assertRaises(AssertionError, msg=repr(bad)):
                self._build(rider=bad)
        self.assertTrue(self._build(rider=3).endswith("_rd3"))
        # rider is a pv="k128" knob: the bf16-P forward refuses it.
        kernel = self._kernel()
        with mock.patch.object(kernel, "SmemAllocator", _Alloc), \
                mock.patch.object(kernel, "get_rocm_arch", lambda: "gfx950"):
            with self.assertRaises(AssertionError):
                kernel.compile_hstu_fwd.__wrapped__(
                    4, 128, 0.08838834764831843, rider=3)


if __name__ == "__main__":
    unittest.main()
