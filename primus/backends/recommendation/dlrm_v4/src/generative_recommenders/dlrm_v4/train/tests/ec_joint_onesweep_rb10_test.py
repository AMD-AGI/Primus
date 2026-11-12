"""yambda_5b.gin item (102) EC_JOINT_ONESWEEP_RB10 (rccl iter106; ship17 row): CPU gates (no GPU, no extension build).

  DISPATCH  ops/onesweep_sort.ec_joint_segments calls the variant build's ec_joint_w only when that build is
            loaded, not latched off, and the joint keys need 19 or 20 bits (2^17 < U <= 2^19), with end_bit 19/20
            and the ws32w workspace. Every other case calls the default ec_joint exactly as before (same end_bit,
            the ws32 workspace), and a default build (no ec_joint_w) never takes the wide path.
  CHECKS    the first two wide calls also run the 9-bit build and compare raw bits; a mismatch returns the 9-bit
            result and latches the wide path off for the process; a raising wide call does the same; once the
            checks are used up the reference is no longer run; the armed line is logged once.
            A raise of the 9-bit reference itself propagates (the caller fails (c) closed) and does not latch item (102).
  BUILD     extension(): switch off -> one load() of the default module in the default directory with the default
            flags (lock and staging at the call-time _BUILD_DIR); switch on with (c) on -> the variant first (its
            own name, directory, lock and staging, -DDLRM_ONESWEEP_EC_RB10 in both flag lists); (c) off -> the
            default build; a variant that raises or lacks ec_joint_w leaves the default build in use (_FAILED
            untouched). prebuild() logs "onesweep_sort_ext_rb10=1" / "NOT armed ... rb10=0" only with the switch and
            (c) on.
            _stage_sources() / _locked() with no directory read osort._BUILD_DIR at call time, as before the item
            (ops/tests/onesweep_sort_test.py's build-lock tests rebind it).
  RADIX     an LSD radix sort emulation: digit splits [9, 9, 1] and [10, 9] (19-bit keys), [9, 9, 2] and [10, 10]
            (20-bit keys) give the same stable permutation as torch.argsort(stable=True): the bitwise claim
            (an LSD sort over the same bits is stable for any digit split), independent of rocPRIM.
  GIN       apply_env_bootstrap declares the knob once (Optional[bool] = None) and binds it with one env-wins
            _bind_bool call; yambda_5b.gin binds it exactly once, to True (default on: lab GO, job 74262).
The GPU raw-bit gate (variant build vs the 9-bit build and the item (41)(d) chain at U 1.31e5-5.24e5, both 19- and
20-bit edges) is the lab kit
named in the rccl iter106 findings; this file never touches a GPU."""

import ast
import logging
import os
import re
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

import torch

from generative_recommenders.ops import onesweep_sort as osort

KNOB = "DLRM_HSTU_FP8_EC_JOINT_ONESWEEP_RB10"
_TRAIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GIN = os.path.join(_TRAIN, "gin", "yambda_5b.gin")
_BOOT = os.path.join(_TRAIN, "_env_bootstrap.py")
DEFAULT_CUDA_FLAGS = ["-O3", "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
                      "--offload-arch=gfx950"]
_GLOBALS = ("_EXT", "_FAILED", "_RB10_FAILED", "_RB10_OFF", "_RB10_CHECKS", "_RB10_LOGGED", "_BUILD_DIR")


def _ref_joint(a, b, U, kout, vout, off_a, off_b):
    """CPU model of dlrm_onesweep_ec_joint's outputs: stable sort of [a ; b + U], member-relative positions with
    member b's order at the 16-byte-aligned offset ta + (ta & 1), offsets J[k] = #{keys < k}."""
    ta = a.numel()
    keys = torch.cat([a, b + U])
    order = torch.argsort(keys, stable=True)
    sk = keys[order]
    kout.copy_(sk.to(torch.int32))
    pos = torch.where(order < ta, order, order - ta)
    sh = ta & 1
    vout[:ta] = pos[:ta]
    vout[ta + sh:] = pos[ta:]
    j = torch.searchsorted(sk, torch.arange(2 * U + 1, dtype=torch.int64))
    off_a.copy_(j[: U + 1])
    off_b.copy_(j[U:] - ta)


class _FakeExt:
    """Both builds' Python surface; ec_joint / ec_joint_w write _ref_joint's outputs and record each call."""

    def __init__(self, wide: bool = True, wide_bug: str = "", ref_bug: str = "") -> None:
        self.calls: List[Dict[str, Any]] = []
        self.wide_bug = wide_bug
        self.ref_bug = ref_bug
        if wide:
            self.radix_bits32w = lambda: 10
            self.ws32w = lambda n, e: 1000 + e
            self.ec_joint_w = self._joint_w

    def radix_bits32(self) -> int:
        return 9

    def ws32(self, n: int, e: int) -> int:
        return 100 + e

    def gap_capacity(self, U: int) -> int:
        return 1

    def _record(self, name, end_bit, ws):
        self.calls.append({"fn": name, "end_bit": int(end_bit), "ws": ws.numel()})

    def ec_joint(self, a, b, U, end_bit, kout, vout, ktmp, vtmp, ws, off_a, off_b, gaps):
        self._record("ec_joint", end_bit, ws)
        if self.ref_bug == "raise":
            raise RuntimeError("onesweep_sort: HIP error 2 (9-bit build)")
        _ref_joint(a, b, U, kout, vout, off_a, off_b)

    def _joint_w(self, a, b, U, end_bit, kout, vout, ktmp, vtmp, ws, off_a, off_b, gaps):
        self._record("ec_joint_w", end_bit, ws)
        if self.wide_bug == "raise":
            raise RuntimeError("onesweep_sort: HIP error 1")
        _ref_joint(a, b, U, kout, vout, off_a, off_b)
        if self.wide_bug == "order":
            vout[0] += 1
        elif self.wide_bug == "offsets":
            off_b[-1] -= 1


def _ids(n: int, U: int, seed: int, lo: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, U, (n,), generator=g, dtype=torch.int64)


class _OsortState(unittest.TestCase):
    """Saves and restores the module globals and the knob, so other tests in the process see no change."""

    def setUp(self) -> None:
        self._saved = {k: getattr(osort, k) for k in _GLOBALS}
        osort._EXT = None
        osort._FAILED = None
        osort._RB10_FAILED = None
        osort._RB10_OFF = False
        osort._RB10_CHECKS = 2
        osort._RB10_LOGGED = False
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop(KNOB, None)
        self._dev = mock.patch.object(osort, "_on_current_device", lambda t: True)
        self._dev.start()

    def tearDown(self) -> None:
        self._dev.stop()
        self._env.stop()
        for k, v in self._saved.items():
            setattr(osort, k, v)


class DispatchTest(_OsortState):
    def _run(self, ext, U, ta=3001, tb=2048, seed=0):
        osort._EXT = ext
        a, b = _ids(ta, U, seed), _ids(tb, U, seed + 1)
        segs = osort.ec_joint_segments(a, b, U)
        self.assertIsNotNone(segs)
        ref_ext = _FakeExt(wide=False)
        sh = ta & 1
        kout = torch.empty(ta + tb, dtype=torch.int32)
        vout = torch.empty(ta + tb + sh, dtype=torch.int64)
        off_a = torch.empty(U + 1, dtype=torch.int64)
        off_b = torch.empty(U + 1, dtype=torch.int64)
        ref_ext.ec_joint(a, b, U, 0, kout, vout, kout, vout, torch.empty(1), off_a, off_b, None)
        (oa, fa), (ob, fb) = segs
        self.assertTrue(torch.equal(oa, vout[:ta]) and torch.equal(ob, vout[ta + sh:]))
        self.assertTrue(torch.equal(fa, off_a) and torch.equal(fb, off_b))
        self.assertEqual(oa.data_ptr() % 16, 0)
        self.assertEqual(ob.data_ptr() % 16, 0)
        return [(c["fn"], c["end_bit"], c["ws"]) for c in ext.calls]

    def test_18_bit_keys_stay_on_the_9_bit_build(self) -> None:
        U = 1 << 17  # 2U - 1 = 2^18 - 1: 18 bits, 2 passes already
        self.assertEqual(self._run(_FakeExt(), U), [("ec_joint", 18, 118)])

    def test_small_u_keeps_the_9_bit_minimum(self) -> None:
        self.assertEqual(self._run(_FakeExt(), 300), [("ec_joint", 10, 110)])

    def test_19_bit_keys_take_the_wide_build_with_a_reference(self) -> None:
        U = (1 << 17) + 1
        with self.assertLogs(osort.logger, level="WARNING") as cm:
            calls = self._run(_FakeExt(), U)
        self.assertEqual(calls, [("ec_joint_w", 19, 1019), ("ec_joint", 19, 119)])
        self.assertTrue(any("armed OK: ec joint onesweep rb10" in m for m in cm.output))

    def test_4n_u_takes_the_wide_build(self) -> None:
        with self.assertLogs(osort.logger, level="WARNING"):
            calls = self._run(_FakeExt(), 155280)  # the 4n trace's U (rank 0, step 54)
        self.assertEqual(calls[0], ("ec_joint_w", 19, 1019))

    def test_20_bit_keys_take_the_wide_build(self) -> None:
        U = 1 << 19  # 2U - 1 = 2^20 - 1: 20 bits
        with self.assertLogs(osort.logger, level="WARNING"):
            calls = self._run(_FakeExt(), U, ta=1000, tb=999)
        self.assertEqual(calls[0], ("ec_joint_w", 20, 1020))

    def test_21_bit_keys_stay_on_the_9_bit_build(self) -> None:
        U = (1 << 19) + 1
        self.assertEqual(self._run(_FakeExt(), U, ta=500, tb=501), [("ec_joint", 21, 121)])

    def test_default_build_never_takes_the_wide_path(self) -> None:
        self.assertEqual(self._run(_FakeExt(wide=False), 155280), [("ec_joint", 19, 119)])

    def test_checks_are_used_up_then_no_reference(self) -> None:
        ext = _FakeExt()
        with self.assertLogs(osort.logger, level="WARNING") as cm:
            for s in range(4):
                self._run(ext, 155280, seed=10 * s)
        fns = [c["fn"] for c in ext.calls]
        self.assertEqual(fns, ["ec_joint_w", "ec_joint", "ec_joint_w", "ec_joint", "ec_joint_w", "ec_joint_w"])
        self.assertEqual(sum("armed OK: ec joint onesweep rb10" in m for m in cm.output), 1)
        self.assertEqual(osort._RB10_CHECKS, 0)
        self.assertFalse(osort._RB10_OFF)


class FailClosedTest(_OsortState):
    def _call(self, ext, U=155280):
        osort._EXT = ext
        a, b = _ids(2000, U, 1), _ids(1500, U, 2)
        return osort.ec_joint_segments(a, b, U)

    def _assert_9_bit_result(self, segs, U=155280):
        ext = _FakeExt(wide=False)
        osort._EXT = ext
        osort._RB10_OFF = True
        a, b = _ids(2000, U, 1), _ids(1500, U, 2)
        ref = osort.ec_joint_segments(a, b, U)
        for (x0, x1), (y0, y1) in zip(segs, ref):
            self.assertTrue(torch.equal(x0, y0) and torch.equal(x1, y1))

    def test_order_mismatch_returns_the_9_bit_result_and_latches_off(self) -> None:
        ext = _FakeExt(wide_bug="order")
        with self.assertLogs(osort.logger, level="ERROR") as cm:
            segs = self._call(ext)
        self.assertTrue(any("ec joint onesweep rb10 SELF-CHECK FAILED" in m and "member 0 order" in m
                            for m in cm.output))
        self.assertTrue(osort._RB10_OFF)
        self._assert_9_bit_result(segs)
        ext.calls.clear()
        osort._EXT = ext
        osort.ec_joint_segments(_ids(10, 155280, 3), _ids(11, 155280, 4), 155280)
        self.assertEqual([c["fn"] for c in ext.calls], ["ec_joint"])

    def test_offsets_mismatch_is_caught(self) -> None:
        with self.assertLogs(osort.logger, level="ERROR") as cm:
            segs = self._call(_FakeExt(wide_bug="offsets"))
        self.assertTrue(any("member 1 offsets" in m for m in cm.output))
        self.assertTrue(osort._RB10_OFF)
        self._assert_9_bit_result(segs)

    def test_raising_wide_call_falls_back_and_latches_off(self) -> None:
        ext = _FakeExt(wide_bug="raise")
        with self.assertLogs(osort.logger, level="WARNING") as cm:
            segs = self._call(ext)
        self.assertTrue(any("ec joint onesweep rb10 FAILED CLOSED" in m for m in cm.output))
        self.assertTrue(osort._RB10_OFF)
        self.assertEqual([c["fn"] for c in ext.calls], ["ec_joint_w", "ec_joint"])
        self._assert_9_bit_result(segs)


class ReferenceRaiseTest(_OsortState):
    def test_a_raising_9_bit_reference_propagates_and_rb10_stays_on(self) -> None:
        osort._EXT = _FakeExt(ref_bug="raise")
        U = 155280
        with self.assertRaisesRegex(RuntimeError, "9-bit build"):
            osort.ec_joint_segments(_ids(2000, U, 1), _ids(1500, U, 2), U)
        self.assertFalse(osort._RB10_OFF)  # the caller (ec_offsets) fails (c) closed; item (102) is not blamed
        self.assertEqual(osort._RB10_CHECKS, 2)


class BuildSelectionTest(_OsortState):
    C_SWITCH = "DLRM_HSTU_FP8_EC_JOINT_ONESWEEP"

    def _extension(self, variant_module=None, variant_raises=False, call="extension"):
        loads: List[Dict[str, Any]] = []
        dirs: Dict[str, List[Any]] = {"locked": [], "staged": []}
        default_mod = SimpleNamespace(name="default")

        def fake_load(**kw):
            kw = dict(kw, _rocm_arch_env=os.environ.get("PYTORCH_ROCM_ARCH"))
            loads.append(kw)
            if kw["name"].endswith("_rb10"):
                if variant_raises:
                    raise RuntimeError("hipcc: error")
                return variant_module
            return default_mod

        def fake_locked(fn, build_dir=None):
            dirs["locked"].append(build_dir)
            return fn()

        def fake_stage(build_dir=None):
            dirs["staged"].append(build_dir)
            return [str(osort._BUILD_DIR if build_dir is None else build_dir)]

        with mock.patch("torch.utils.cpp_extension.load", fake_load), \
                mock.patch.object(osort, "_locked", fake_locked), \
                mock.patch.object(osort, "_stage_sources", fake_stage), \
                mock.patch.object(osort, "_arch", lambda: "gfx950"):
            ext = osort.prebuild() if call == "prebuild" else osort.extension()
        if call == "prebuild":
            ext = osort._EXT
        return ext, loads, default_mod, dirs

    def _variant(self):
        return SimpleNamespace(name="rb10", ec_joint_w=lambda *a: None, radix_bits32w=lambda: 10)

    def test_switch_off_builds_only_the_default_module(self) -> None:
        os.environ[self.C_SWITCH] = "1"
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx942;gfx950"  # as the training image sets it
        ext, loads, default_mod, dirs = self._extension()
        self.assertIs(ext, default_mod)
        self.assertEqual(len(loads), 1)
        (kw,) = loads
        self.assertEqual(kw.pop("_rocm_arch_env"), "gfx942;gfx950")  # the default build leaves the image's env alone
        self.assertEqual(kw, {  # the whole load() call, as before the item
            "name": "dlrm_hstu_onesweep_sort",
            "sources": ["/tmp/dlrm_hstu_onesweep_sort"],
            "extra_cflags": ["-O3"],
            "extra_cuda_cflags": DEFAULT_CUDA_FLAGS,
            "build_directory": "/tmp/dlrm_hstu_onesweep_sort",
            "verbose": False,
        })
        self.assertEqual(dirs, {"locked": [None], "staged": [None]})  # call-time _BUILD_DIR, as before the item

    def test_switch_on_builds_the_variant_first(self) -> None:
        os.environ[KNOB] = "1"
        os.environ[self.C_SWITCH] = "1"
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx942;gfx950"
        variant = self._variant()
        ext, loads, _, dirs = self._extension(variant_module=variant)
        self.assertIs(ext, variant)
        self.assertEqual(len(loads), 1)
        (kw,) = loads
        self.assertEqual(kw["name"], "dlrm_hstu_onesweep_sort_rb10")
        # torch adds $PYTORCH_ROCM_ARCH's archs before ours; gfx942 (64 KiB LDS) cannot hold the 10-bit kernels,
        # so the variant is built for the current device only (lab 73242), and the env is restored afterwards
        self.assertEqual(kw["_rocm_arch_env"], "gfx950")
        self.assertEqual(os.environ.get("PYTORCH_ROCM_ARCH"), "gfx942;gfx950")
        self.assertEqual(kw["build_directory"], "/tmp/dlrm_hstu_onesweep_sort_rb10")
        # the same flags as the default build plus the define (the "instruction-identical 9-bit code" premise)
        self.assertEqual(kw["extra_cflags"], ["-O3", "-DDLRM_ONESWEEP_EC_RB10"])
        self.assertEqual(kw["extra_cuda_cflags"], DEFAULT_CUDA_FLAGS[:3] + ["-DDLRM_ONESWEEP_EC_RB10"]
                         + DEFAULT_CUDA_FLAGS[3:])
        self.assertFalse(kw["verbose"])
        self.assertEqual(kw["sources"], ["/tmp/dlrm_hstu_onesweep_sort_rb10"])
        self.assertEqual(dirs, {"locked": [osort._RB10_BUILD_DIR], "staged": [osort._RB10_BUILD_DIR]})
        self.assertIsNone(osort._RB10_FAILED)

    def test_variant_needs_c_on(self) -> None:
        os.environ[KNOB] = "1"
        os.environ.pop(self.C_SWITCH, None)
        ext, loads, default_mod, _ = self._extension(variant_module=self._variant())
        self.assertIs(ext, default_mod)
        self.assertEqual([kw["name"] for kw in loads], ["dlrm_hstu_onesweep_sort"])

    def test_switch_value_0_is_off(self) -> None:
        os.environ[KNOB] = "0"
        os.environ[self.C_SWITCH] = "1"
        ext, loads, default_mod, _ = self._extension()
        self.assertIs(ext, default_mod)
        self.assertEqual([kw["name"] for kw in loads], ["dlrm_hstu_onesweep_sort"])

    def test_variant_build_failure_falls_back_to_the_default_build(self) -> None:
        os.environ[KNOB] = "1"
        os.environ[self.C_SWITCH] = "1"
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
        with self.assertLogs(osort.logger, level="WARNING") as cm:
            ext, loads, default_mod, dirs = self._extension(variant_raises=True)
        self.assertIs(ext, default_mod)
        self.assertEqual([kw["name"] for kw in loads], ["dlrm_hstu_onesweep_sort_rb10", "dlrm_hstu_onesweep_sort"])
        self.assertEqual(dirs["locked"], [osort._RB10_BUILD_DIR, None])
        self.assertEqual([kw["_rocm_arch_env"] for kw in loads], ["gfx950", None])  # override undone after a raise
        self.assertNotIn("PYTORCH_ROCM_ARCH", os.environ)
        self.assertTrue(any("ec joint onesweep rb10 build FAILED" in m for m in cm.output))
        self.assertIsNotNone(osort._RB10_FAILED)
        self.assertIsNone(osort._FAILED)

    def test_variant_reporting_9_bit_digits_is_a_failure(self) -> None:
        os.environ[KNOB] = "1"
        os.environ[self.C_SWITCH] = "1"
        stale = SimpleNamespace(name="rb9", ec_joint_w=lambda *a: None, radix_bits32w=lambda: 9)
        with self.assertLogs(osort.logger, level="WARNING"):
            ext, loads, default_mod, _ = self._extension(variant_module=stale)
        self.assertIs(ext, default_mod)
        self.assertEqual(len(loads), 2)

    def test_variant_without_the_wide_entry_point_is_a_failure(self) -> None:
        os.environ[KNOB] = "1"
        os.environ[self.C_SWITCH] = "1"
        with self.assertLogs(osort.logger, level="WARNING"):
            ext, loads, default_mod, _ = self._extension(variant_module=SimpleNamespace(name="stale"))
        self.assertIs(ext, default_mod)
        self.assertEqual(len(loads), 2)
        self.assertIsNone(osort._FAILED)

    def test_prebuild_arming_lines(self) -> None:
        os.environ[KNOB] = "1"
        os.environ[self.C_SWITCH] = "1"
        with self.assertLogs(osort.logger, level="INFO") as cm:
            self._extension(variant_module=self._variant(), call="prebuild")
        self.assertTrue(any("armed OK: prebuild before run_start: onesweep_sort_ext_rb10=1" in m for m in cm.output))
        self.assertTrue(any("armed OK: prebuild before run_start: onesweep_sort_ext=1" in m for m in cm.output))
        self.setUp_reset()
        with self.assertLogs(osort.logger, level="INFO") as cm:
            self._extension(variant_raises=True, call="prebuild")
        self.assertTrue(any("DLRMV4 NOT armed: prebuild before run_start: onesweep_sort_ext_rb10=0" in m
                            for m in cm.output))
        self.assertTrue(any("armed OK: prebuild before run_start: onesweep_sort_ext=1" in m for m in cm.output))
        self.setUp_reset()
        os.environ[KNOB] = "0"
        with self.assertLogs(osort.logger, level="INFO") as cm:
            self._extension(call="prebuild")
        self.assertFalse(any("onesweep_sort_ext_rb10" in m for m in cm.output))
        self.setUp_reset()
        os.environ[KNOB] = "1"
        os.environ.pop(self.C_SWITCH, None)
        os.environ["DLRM_HSTU_FP8_POSENC_TS_ONESWEEP"] = "1"  # (b) alone: the default build, no rb10 line
        with self.assertLogs(osort.logger, level="INFO") as cm:
            ext, loads, default_mod, _ = self._extension(variant_module=self._variant(), call="prebuild")
        self.assertIs(ext, default_mod)
        self.assertFalse(any("onesweep_sort_ext_rb10" in m for m in cm.output))

    def setUp_reset(self) -> None:
        osort._EXT = None
        osort._FAILED = None
        osort._RB10_FAILED = None


class WithArchHelperTest(unittest.TestCase):
    """_with_arch (item (102)'s variant build pins PYTORCH_ROCM_ARCH with it) restores the env afterwards."""

    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()

    def test_with_arch_helper(self) -> None:
        os.environ["PYTORCH_ROCM_ARCH"] = "a;b"
        self.assertEqual(osort._with_arch("gfx950", lambda: os.environ["PYTORCH_ROCM_ARCH"]), "gfx950")
        self.assertEqual(os.environ["PYTORCH_ROCM_ARCH"], "a;b")
        self.assertEqual(osort._with_arch(None, lambda: os.environ["PYTORCH_ROCM_ARCH"]), "a;b")


class BuildDirAtCallTimeTest(_OsortState):
    def test_default_build_dir_is_read_at_call_time(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            osort._BUILD_DIR = Path(td) / "rebound"
            staged = osort._stage_sources()
            self.assertTrue(all(p.startswith(str(osort._BUILD_DIR / "src")) for p in staged), staged)
            stale = osort._BUILD_DIR / "lock"
            stale.write_text("")
            self.assertEqual(osort._locked(lambda: "ran"), "ran")
            self.assertTrue((osort._BUILD_DIR / "build.flock").exists())
            self.assertFalse(stale.exists())  # the stale torch lock of THIS directory was removed
            rb10 = Path(td) / "rb10"
            self.assertEqual(osort._locked(lambda: "rb10", rb10), "rb10")
            self.assertTrue((rb10 / "build.flock").exists())


def _lsd_perm(keys: torch.Tensor, split: List[int]) -> torch.Tensor:
    perm = torch.arange(keys.numel())
    shift = 0
    for bits in split:
        digit = (keys[perm] >> shift) & ((1 << bits) - 1)
        perm = perm[torch.argsort(digit, stable=True)]
        shift += bits
    return perm


class RadixSplitTest(unittest.TestCase):
    def test_digit_split_does_not_change_the_stable_permutation(self) -> None:
        g = torch.Generator().manual_seed(7)
        for bits, splits in ((19, ([9, 9, 1], [10, 9])), (20, ([9, 9, 2], [10, 10]))):
            U = 1 << (bits - 1)
            a = torch.randint(0, U, (30000,), generator=g, dtype=torch.int64)
            b = torch.randint(0, U, (20001,), generator=g, dtype=torch.int64)
            a[:5000] = a[5000:10000]  # many duplicate ids: ties must keep token order
            keys = torch.cat([a, b + U])
            self.assertLess(int(keys.max()), 1 << bits)
            want = torch.argsort(keys, stable=True)
            for split in splits:
                self.assertEqual(sum(split), bits)
                self.assertTrue(torch.equal(_lsd_perm(keys, split), want), (bits, split))


class GinBindingTest(unittest.TestCase):
    def test_bootstrap_declares_and_binds_once(self) -> None:
        self._bootstrap_declares_and_binds_once(KNOB)

    def test_dropped_arch_pin_knob_is_absent(self) -> None:
        # the former item (103) ONESWEEP_ARCH_PIN was dropped for ship17 (item (102) pins its own variant build)
        self.assertFalse(hasattr(osort, "_ARCH_PIN_SWITCH"))
        for path in (_GIN, _BOOT):
            with open(path) as f:
                self.assertNotIn("ONESWEEP_ARCH_PIN", f.read(), path)

    def _bootstrap_declares_and_binds_once(self, KNOB: str) -> None:
        with open(_BOOT) as f:
            tree = ast.parse(f.read())
        (fn,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "apply_env_bootstrap"]
        args = fn.args.args + fn.args.kwonlyargs
        self.assertEqual([a.arg for a in args].count(KNOB), 1)
        arg = next(a for a in args if a.arg == KNOB)
        self.assertEqual(ast.unparse(arg.annotation), "Optional[bool]")
        defaults = dict(zip(reversed([a.arg for a in fn.args.args]), reversed(fn.args.defaults)))
        defaults.update({a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)})
        self.assertTrue(isinstance(defaults[KNOB], ast.Constant) and defaults[KNOB].value is None)
        calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and n.args
                 and isinstance(n.args[0], ast.Constant) and n.args[0].value == KNOB]
        self.assertEqual(len(calls), 1)
        self.assertEqual(getattr(calls[0].func, "id", None), "_bind_bool")
        self.assertEqual(len(calls[0].args), 2)  # bound to its own parameter, not to a neighbour's value
        self.assertTrue(isinstance(calls[0].args[1], ast.Name) and calls[0].args[1].id == KNOB)

    def test_gin_binds_it_once_to_true(self) -> None:
        with open(_GIN) as f:
            lines = f.read().splitlines()
        pat = re.compile(r"^\s*apply_env_bootstrap\.%s\s*=\s*(.*?)\s*$" % KNOB)
        self.assertEqual([m.group(1) for m in map(pat.match, lines) if m], ["True"])
        names = [m.group(1) for m in (re.match(r"^\s*apply_env_bootstrap\.(\w+)\s*=", ln) for ln in lines) if m]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])

    def test_switch_name_matches_the_module(self) -> None:
        self.assertEqual(osort._RB10_SWITCH, KNOB)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
