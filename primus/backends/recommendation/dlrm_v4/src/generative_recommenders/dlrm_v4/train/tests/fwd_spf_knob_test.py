#!/usr/bin/env python3
# pyre-strict
"""SPF fold fwd (ship17 item (105); X4C19 iter71 lab knob, code default OFF):
DLRM_HSTU_FP8_LAB_FWD_SPF parsing, the host snap of the attention qkv scale
(ops/fp8_qkv_spf.py), the forward's arming (hstu_flydsl_mha.fp8_fwd_spf_config:
one armed line per process and MODE), the F1 warm-up's handling of that line
(printed after the item, outside its stdout capture; the latch is kept, not in
fp8_warmup._MHA_SETS), the calibration hook (triton_fp8.calibrate_scale_buffer
spf_alpha at every qkv / targets-only VK publication) and compile_hstu_fwd's
spf parameter (default-off identity, LDS symbol suffix order).

The gin binding of item (105) is pinned by fwd_spf_gin_test.py (not here: this
file also runs where the knob is env-only).

CPU-only: nothing here launches a kernel. The kernel change (hstu_fwd.py
spf=True) is covered by the CPU AOT ISA checks of x4c19 iter71 / iter74. The
snap and source tests need only the standard library and run under Python 3.9;
the launcher / warm-up / calibration / kernel tests skip where their modules
(torch, triton, flydsl) are not importable."""

import ast
import contextlib
import io
import logging
import math
import os
import random
import struct
import unittest
from fractions import Fraction
from unittest import mock

_KNOB = "DLRM_HSTU_FP8_LAB_FWD_SPF"
_ALPHA = 1.0 / (128 ** 0.5)  # stu.py: attention_dim 128, attn_alpha unset
_GR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # .../generative_recommenders



def _spf():  # pyre-ignore[3]
    from generative_recommenders.ops import fp8_qkv_spf

    return fp8_qkv_spf


# Independent binary32 model (exact rationals, explicit round-to-nearest-even),
# so the snap's struct-based f32 arithmetic is not checked against itself.
def _rne32(x: Fraction) -> Fraction:
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    a = abs(x)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, -126)                       # subnormal quantum below 2**-126
    q = Fraction(2) ** (e - 23)            # spacing of binary32 at this binade
    n = a / q
    fl = n.numerator // n.denominator
    rem = n - fl
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and fl % 2 == 1):
        fl += 1
    return sign * fl * q


def _coeff_exact(s: float, alpha: float, log2e: float) -> Fraction:
    a32 = _rne32(Fraction(alpha))
    l32 = _rne32(Fraction(log2e))
    s2 = _rne32(Fraction(s) * Fraction(s))
    return _rne32(_rne32(s2 * a32) * l32)


def _nextup32(x: float) -> float:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    return struct.unpack("<f", struct.pack("<I", b + 1))[0]


class SpfSnapTest(unittest.TestCase):
    def setUp(self) -> None:
        self.spf = _spf()

    def test_knob_parsing(self) -> None:
        spf = self.spf
        for value in (None, "", "0"):
            env = {} if value is None else {_KNOB: value}
            self.assertEqual(spf.spf_mode(env), "off", value)
        self.assertEqual(spf.spf_mode({_KNOB: "snap"}), "snap")
        self.assertEqual(spf.spf_mode({_KNOB: "1"}), "fold")
        for bad in ("2", "true", "on", " 1", "1 ", "SNAP", "fold", "yes", "-1"):
            with self.assertRaises(ValueError, msg=bad):
                spf.spf_mode({_KNOB: bad})
        with mock.patch.dict(os.environ, {_KNOB: "snap"}):
            self.assertEqual(spf.spf_mode(), "snap")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            self.assertEqual(spf.spf_mode(), "off")

    def test_grid_is_exact_power_of_two_every_j(self) -> None:
        spf = self.spf
        prev = None
        for j in range(spf.J_MIN, spf.J_MAX + 1):
            s = spf.grid_scale(j, _ALPHA)
            self.assertIsNotNone(s, j)
            # exactly 2**j in the kernel's order, the backward's order and an
            # independent exact-rational binary32 model
            self.assertEqual(spf.score_coeff32(s, _ALPHA), 2.0 ** j, j)
            self.assertEqual(spf.score_coeff32_bwd(s, _ALPHA), 2.0 ** j, j)
            self.assertEqual(_coeff_exact(s, _ALPHA, spf.LOG2E),
                             Fraction(2) ** j, j)
            self.assertEqual(spf.grid_exponent(s, _ALPHA), j)
            if prev is not None:
                # the worst snap ratio is s_j / nextup(s_(j-1)) for f32 inputs
                self.assertLess(s / _nextup32(prev), math.sqrt(2.0), j)
            prev = s

    def test_snap_properties_random(self) -> None:
        spf = self.spf
        rng = random.Random(71)
        for _ in range(4000):
            s = spf.f32(2.0 ** rng.uniform(-30.0, 4.0))
            sn, j = spf.snap_scale(s, _ALPHA)
            self.assertGreaterEqual(sn, s)
            self.assertLess(sn / s, math.sqrt(2.0))
            self.assertEqual(spf.score_coeff32(sn, _ALPHA), 2.0 ** j)
            self.assertEqual(spf.snap_scale(sn, _ALPHA), (sn, j))
            below = spf.grid_scale(j - 1, _ALPHA)
            self.assertLess(below, s)          # smallest grid point >= s

    def test_snap_typical_value(self) -> None:
        # Config-derived (no data): uvqk init std sqrt(512 * 6/2560/3) = 0.632,
        # Gaussian amax ~6.9 std over 8 ranks x 1.9e6 tokens x 1536 -> 4.37;
        # s = f32(f32(4.37 * 1.5) * f32(1/448)) = 0.0146325 -> j = -15.
        spf = self.spf
        sn, j = spf.snap_scale(spf.f32(0.0146325), _ALPHA)
        self.assertEqual(j, -15)
        self.assertEqual(struct.unpack("<I", struct.pack("<f", sn))[0],
                         0x3C7D75E0)

    def test_arm_check(self) -> None:
        spf = self.spf
        s = spf.grid_scale(-15, _ALPHA)
        self.assertEqual(spf.arm_check(s, _ALPHA), -15)
        for bad in (0.0146325, spf.f32(0.0146325), 2.0 ** -6, 0.0, -s,
                    float("nan"), float("inf")):
            with self.assertRaises(RuntimeError, msg=repr(bad)):
                spf.arm_check(bad, _ALPHA)
        # outside [J_MIN, J_MAX]: on the grid but not armable
        far = spf.grid_scale(spf.J_MIN - 1, _ALPHA)
        with self.assertRaises(RuntimeError):
            spf.arm_check(far, _ALPHA)




class SpfLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        self.mha = mha
        self.spf = _spf()
        # The once-latch (mode -> line) is process state: keep it unleaked.
        self._latch = dict(mha._fp8_fwd_spf_logged)
        self.addCleanup(self._restore_latch)

    def _restore_latch(self) -> None:
        self.mha._fp8_fwd_spf_logged.clear()
        self.mha._fp8_fwd_spf_logged.update(self._latch)

    def _cfg(self, value, cfg, scale):  # pyre-ignore[2,3]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            if value is not None:
                os.environ[_KNOB] = value
            return self.mha.fp8_fwd_spf_config(cfg, scale, _ALPHA)

    def test_off_changes_nothing(self) -> None:
        for value in (None, "", "0"):
            # off: any scale, any config, nothing added, no check
            self.assertEqual(self._cfg(value, {"pv": "k128"}, 0.0146325), {})
            self.assertEqual(self._cfg(value, {}, 2.0 ** -6), {})

    def test_snap_and_fold(self) -> None:
        grid = self.spf.grid_scale(-15, _ALPHA)
        pv = {"pv": "k128", "pv_k": 3}
        self.assertEqual(self._cfg("snap", pv, grid), {})
        self.assertEqual(self._cfg("snap", {}, grid), {})
        self.assertEqual(self._cfg("1", pv, grid), {"spf": True})
        with self.assertRaises(RuntimeError):
            self._cfg("1", {}, grid)               # fold needs pv=k128
        for value in ("snap", "1"):
            with self.assertRaises(RuntimeError, msg=value):
                self._cfg(value, pv, 0.0146325)    # off the grid: fail closed
        with self.assertRaises(ValueError):
            self._cfg("2", pv, grid)

    def test_fail_closed_messages(self) -> None:
        # The texts item (105)'s gin block lists under "Must be ABSENT".
        grid = self.spf.grid_scale(-15, _ALPHA)
        pv = {"pv": "k128", "pv_k": 3}
        with self.assertRaisesRegex(RuntimeError, "is not on the SPF grid"):
            self._cfg("1", pv, 0.0146325)
        far = self.spf.grid_scale(self.spf.J_MAX + 1, _ALPHA)
        with self.assertRaisesRegex(RuntimeError, "outside the arming range"):
            self._cfg("1", pv, far)
        with self.assertRaisesRegex(
                RuntimeError, f"{_KNOB}=1 needs the pv=k128 forward"):
            self._cfg("1", {"pv": ""}, grid)
        with self.assertRaisesRegex(ValueError,
                                    "use unset, '0', 'snap' or '1'"):
            self._cfg("on", pv, grid)

    def test_armed_line_once_per_mode(self) -> None:
        # The latch is keyed by the MODE ("snap" / "fold"), not by the scale:
        # two calls at different grid scales print ONE armed line per mode.
        self.mha._fp8_fwd_spf_logged.clear()
        s15 = self.spf.grid_scale(-15, _ALPHA)
        s14 = self.spf.grid_scale(-14, _ALPHA)
        pv = {"pv": "k128", "pv_k": 3}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(self._cfg("1", pv, s15), {"spf": True})
            self.assertEqual(self._cfg("1", pv, s14), {"spf": True})
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(
            f"[hstu-fwd] {_KNOB}=1 armed: spf=True (QK MFMA E8M0 A-scale "
            "byte 112 = 2**-15; score multiply folded)"), lines[0])
        self.assertEqual(self.mha._fp8_fwd_spf_logged, {"fold": lines[0]})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(self._cfg("snap", pv, s14), {})
            self.assertEqual(self._cfg("snap", pv, s15), {})
            self.assertEqual(self._cfg("1", pv, s15), {"spf": True})
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(f"[hstu-fwd] {_KNOB}=snap armed: "
                                            "snapped control"), lines[0])
        self.assertIn("score scale = -2**-14 exactly", lines[0])
        self.assertEqual(sorted(self.mha._fp8_fwd_spf_logged), ["fold", "snap"])

    def test_default_config_has_no_spf(self) -> None:
        for value in (None, "snap", "1"):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop(_KNOB, None)
                if value is not None:
                    os.environ[_KNOB] = value
                try:
                    cfg = self.mha._fp8_fwd_config()
                except Exception as error:  # pragma: no cover
                    self.skipTest(f"_fp8_fwd_config not evaluable: {error}")
            self.assertNotIn("spf", cfg)

    def test_warmup_scale(self) -> None:
        s = 2.0 ** -6
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            self.assertEqual(self.mha.fp8_fwd_spf_warmup_scale(s, _ALPHA), s)
            for value in ("snap", "1"):
                os.environ[_KNOB] = value
                w = self.mha.fp8_fwd_spf_warmup_scale(s, _ALPHA)
                self.assertEqual(self.spf.arm_check(w, _ALPHA), -14)
                self.assertGreaterEqual(w, s)

    def test_kernel_parameter_default_off(self) -> None:
        try:
            kernel = self.mha._load_fwd_kernel()
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable here: {error}")
        # _load_fwd_kernel() returns None (it does not raise) when FlyDSL is
        # missing, e.g. in a CPU-only test container.
        if kernel is None:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable: {self.mha._fwd_kernel_error}")
        import inspect

        params = inspect.signature(kernel.compile_hstu_fwd).parameters
        self.assertIn("spf", params)
        self.assertIs(params["spf"].default, False)
        names = list(params)
        # merge order of the lab knobs: v8_swz, rider, spf
        self.assertLess(names.index("v8_swz"), names.index("rider"))
        self.assertLess(names.index("rider"), names.index("spf"))


class _Alloc:
    """SmemAllocator stand-in: records global_sym_name, keeps the offsets."""

    names = []  # pyre-ignore[4]

    def __init__(self, _base, arch=None, global_sym_name=""):  # pyre-ignore[2]
        self.ptr = 0
        _Alloc.names.append(global_sym_name)

    @staticmethod
    def _align(ptr, align):  # pyre-ignore[2,3]
        return (ptr + align - 1) // align * align


# The production forward (ship15.1 gin): FWD_PIPE + MASK_ARITH + FWD_PV.
_PROD_ENV = {"DLRM_HSTU_FP8_FWD_PIPE": "1", "DLRM_HSTU_FP8_FWD_MASK_ARITH": "1",
             "DLRM_HSTU_FP8_FWD_PV": "1"}
_SHIPPED_SYM = ("hstu_fwd_smem_128_32_16_0_w4_k128_16_1_110_pf1_pd1_ws0_ma3"
                "_pv8k3_s1_vp16_x111_ps10040000")


class SpfKernelSymbolTest(unittest.TestCase):
    """compile_hstu_fwd (uncached, closures only: no MLIR is traced before a
    launch) at the production kwargs: the spf suffix is present only when on,
    after the v8_swz and rider suffixes."""

    def setUp(self) -> None:
        try:
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_flydsl_mha not importable here: {error}")
        self.mha = mha
        kernel = mha._load_fwd_kernel()
        if kernel is None:  # pragma: no cover - environment specific
            self.skipTest(f"hstu_fwd not importable: {mha._fwd_kernel_error}")
        self.kernel = kernel
        saved = {n: set(getattr(mha, n)) for n in (
            "_fp8_fwd_pv_logged", "_fp8_fwd_mask_arith_logged",
            "_fp8_fwd_v8swz_logged", "_fp8_fwd_rider_logged")}
        pipe = mha._fp8_fwd_pipe_logged

        def restore() -> None:
            for n, v in saved.items():
                getattr(mha, n).clear()
                getattr(mha, n).update(v)
            mha._fp8_fwd_pipe_logged = pipe
        self.addCleanup(restore)

    def _build(self, **kw):  # pyre-ignore[2,3]
        env = {k: v for k, v in os.environ.items() if not k.startswith("DLRM_")}
        env.update(_PROD_ENV)
        with mock.patch.dict(os.environ, env, clear=True):
            cfg = self.mha._fp8_fwd_config()
        cfg.update(kw)
        _Alloc.names = []
        with mock.patch.object(self.kernel, "SmemAllocator", _Alloc), \
                mock.patch.object(self.kernel, "get_rocm_arch", lambda: "gfx950"):
            self.kernel.compile_hstu_fwd.__wrapped__(
                4, 128, 0.08838834764831843, rs=1536, **cfg)
        self.assertEqual(len(_Alloc.names), 1)
        return _Alloc.names[0]

    def test_symbol_suffix(self) -> None:
        self.assertEqual(self._build(), _SHIPPED_SYM)
        self.assertEqual(self._build(spf=False), _SHIPPED_SYM)
        self.assertEqual(self._build(spf=True), _SHIPPED_SYM + "_spf1")
        self.assertEqual(self._build(v8_swz=1, spf=True),
                         _SHIPPED_SYM + "_vz1_spf1")
        self.assertEqual(self._build(v8_swz=1, rider=3, spf=True),
                         _SHIPPED_SYM + "_vz1_rd3_spf1")
        self.assertEqual(self._build(v8_swz=1, rider=3),
                         _SHIPPED_SYM + "_vz1_rd3")

    def test_kernel_rejects_bad_values(self) -> None:
        for bad in (2, "1", None, 0.5):  # (1 == True passes: not bad)
            with self.assertRaises(AssertionError, msg=repr(bad)):
                self._build(spf=bad)
        # spf is a pv="k128" knob: the bf16-P forward refuses it.
        with mock.patch.object(self.kernel, "SmemAllocator", _Alloc), \
                mock.patch.object(self.kernel, "get_rocm_arch", lambda: "gfx950"):
            with self.assertRaises(AssertionError):
                self.kernel.compile_hstu_fwd.__wrapped__(
                    4, 128, 0.08838834764831843, spf=True)


def _read(rel: str) -> str:
    with open(os.path.join(_GR, rel)) as f:
        return f.read()


class SpfWarmupTest(unittest.TestCase):
    """F1 compiles the armed build before RUN_START; its armed line prints
    after the item, outside the item's stdout capture; the latch is kept."""

    def test_f1_source(self) -> None:
        src = _read("dlrm_v4/train/fp8_warmup.py")
        self.assertEqual(src.count("mha.fp8_fwd_spf_warmup_scale(s, alpha)"), 1)
        i_run = src.index('r.run("F1", "flydsl_fwd_fp8"')
        i_f2 = src.index("# ---- F2:", i_run)
        before, after = src[:i_run], src[i_run:i_f2]
        self.assertIn("spf_before = dict(mha._fp8_fwd_spf_logged)",
                      before[before.rindex("def f1()"):])
        self.assertEqual(after.count("_spf_armed_reprint(mha, spf_before)"), 1)
        self.assertEqual(src.count("_spf_armed_reprint("), 2)  # def + F1

    def test_f1_capture_then_reprint(self) -> None:
        # The real item runner drops the armed line F1's forward prints
        # (stdout captured); _spf_armed_reprint prints it after the item.
        try:
            import torch
            from generative_recommenders.dlrm_v4.train import fp8_warmup as fw
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup / hstu_flydsl_mha not importable: {error}")
        spf = _spf()
        saved = dict(mha._fp8_fwd_spf_logged)

        def restore() -> None:
            mha._fp8_fwd_spf_logged.clear()
            mha._fp8_fwd_spf_logged.update(saved)
        self.addCleanup(restore)
        mha._fp8_fwd_spf_logged.clear()

        class _Bufs:
            def reset(self) -> None:
                pass

        r = fw._Runner(torch.device("cpu"), _Bufs(), True, set(), False)
        s14 = spf.grid_scale(-14, _ALPHA)

        def f1() -> None:
            mha.fp8_fwd_spf_config({"pv": "k128", "pv_k": 3}, s14, _ALPHA)

        out = io.StringIO()
        with mock.patch.dict(os.environ, {_KNOB: "1"}), \
                mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None), \
                contextlib.redirect_stdout(out):
            before = dict(mha._fp8_fwd_spf_logged)
            r.run("F1", "flydsl_fwd_fp8", "fwd_spf_knob_test", f1)
            inside = out.getvalue()
            n = fw._spf_armed_reprint(mha, before)
            # a later warm-up pass (latch kept) prints nothing
            n2 = fw._spf_armed_reprint(mha, dict(mha._fp8_fwd_spf_logged))
        self.assertEqual((r.ok, r.failed), (1, 0))
        self.assertEqual(inside, "")               # dropped by the capture
        self.assertEqual((n, n2), (1, 0))
        lines = out.getvalue().splitlines()
        self.assertEqual(lines, [mha._fp8_fwd_spf_logged["fold"]])
        self.assertTrue(lines[0].startswith(
            f"[hstu-fwd] {_KNOB}=1 armed: spf=True (QK MFMA E8M0 A-scale "
            "byte 113 = 2**-14; score multiply folded)"), lines[0])

    def test_latch_is_kept_not_restored(self) -> None:
        tree = ast.parse(_read("dlrm_v4/train/fp8_warmup.py"))
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                    isinstance(node.targets[0], ast.Name):
                consts[node.targets[0].id] = ast.unparse(node.value)
        self.assertNotIn("_fp8_fwd_spf_logged", consts["_MHA_SETS"])
        self.assertNotIn("_fp8_fwd_spf_logged", consts["_MHA_DICTS"])
        self.assertIn("'_fp8_fwd_spf_logged')", consts["_KEEP_GLOBALS"])

    def test_guards_keep_the_latch(self) -> None:
        try:
            from generative_recommenders.dlrm_v4.train import fp8_warmup as fw
            from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"fp8_warmup / hstu_flydsl_mha not importable: {error}")
        saved = dict(mha._fp8_fwd_spf_logged)

        def restore() -> None:
            mha._fp8_fwd_spf_logged.clear()
            mha._fp8_fwd_spf_logged.update(saved)
        self.addCleanup(restore)
        mha._fp8_fwd_spf_logged.clear()
        guards = fw._Guards()
        guards.take(verify=True)
        try:
            # what F1's fp8_fwd_spf_config does on its first armed call
            mha._fp8_fwd_spf_logged["fold"] = "fwd_spf_knob_test sentinel"
        finally:
            guards.restore()
        guards.verify_globals()
        self.assertEqual(guards.globals_changed, [])
        self.assertEqual(guards.changed, [])
        self.assertEqual(mha._fp8_fwd_spf_logged,
                         {"fold": "fwd_spf_knob_test sentinel"})


def _qkv_calls(rel: str):  # pyre-ignore[3]
    """(line, first-arg source, spf_alpha source or None) of every
    calibrate_scale_buffer call in one source file."""
    out = []
    for node in ast.walk(ast.parse(_read(rel))):
        if isinstance(node, ast.Call) and getattr(
                node.func, "id", getattr(node.func, "attr", None)) == \
                "calibrate_scale_buffer":
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
            out.append((node.lineno, ast.unparse(node.args[0]),
                        kw.get("spf_alpha")))
    return out


class SpfCallSiteTest(unittest.TestCase):
    """Every calibrate_scale_buffer call that publishes an attention qkv (or
    targets-only VK) scale passes spf_alpha=attn_alpha; no other call does.
    An unsnapped qkv scale reaching the fold arm raises in-model."""

    def test_every_qkv_publication_passes_spf_alpha(self) -> None:
        found = {}
        for root, _dirs, files in os.walk(_GR):
            if os.sep + "tests" in root[len(_GR):] or "__pycache__" in root:
                continue
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                rel = os.path.relpath(os.path.join(root, fn), _GR)
                if "calibrate_scale_buffer(" not in _read(rel):
                    continue
                calls = _qkv_calls(rel)
                if calls:
                    found[rel] = calls
        tpa = "ops/triton/triton_hstu_preprocess_and_attention.py"
        qkv = [(rel, ln) for rel, calls in found.items()
               for ln, arg, sa in calls if "qkv" in arg]
        self.assertEqual(len(qkv), 3, found)
        for rel, calls in found.items():
            for ln, arg, sa in calls:
                if "qkv" in arg:
                    self.assertEqual((rel, sa), (tpa, "attn_alpha"), (rel, ln))
                else:
                    self.assertIsNone(sa, (rel, ln, arg))
        self.assertEqual(sorted(found), sorted([
            tpa, "ops/triton/triton_hstu_linear.py"]))


class SpfCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch
            from generative_recommenders.ops.triton import triton_fp8
        except Exception as error:  # pragma: no cover - environment specific
            self.skipTest(f"triton_fp8 not importable here: {error}")
        self.torch = torch
        self.f8 = triton_fp8
        self.spf = _spf()

    def _publish(self, value, spf_alpha, knob):  # pyre-ignore[2,3]
        torch, f8 = self.torch, self.f8
        state = torch.zeros((f8.fp8_scale_state_size(),), dtype=torch.float32)
        steps = f8.fp8_scale_state_size() - 2
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_KNOB, None)
            if knob is not None:
                os.environ[_KNOB] = knob
            out = None
            for i in range(steps):
                out = f8.calibrate_scale_buffer(
                    state, value * (1.0 + 0.25 * i), synchronize=False,
                    spf_alpha=spf_alpha)
                if i < steps - 1:
                    self.assertEqual(out, 0.0)
        return out, float(state[0])

    def test_publication(self) -> None:
        torch = self.torch
        g = torch.Generator().manual_seed(71)
        value = torch.randn((64, 96), generator=g).to(torch.bfloat16)
        base, base_state = self._publish(value, None, None)
        self.assertEqual(base, base_state)
        # knob off: spf_alpha changes nothing
        off, off_state = self._publish(value, _ALPHA, None)
        self.assertEqual((off, off_state), (base, base_state))
        # not a qkv buffer (spf_alpha None): never snapped
        self.assertEqual(self._publish(value, None, "snap"), (base, base_state))
        for knob in ("snap", "1"):
            got, got_state = self._publish(value, _ALPHA, knob)
            want, j = self.spf.snap_scale(base, _ALPHA)
            self.assertEqual(got, want, knob)
            self.assertEqual(got_state, want, knob)
            self.assertEqual(self.spf.arm_check(got, _ALPHA), j)
            self.assertGreaterEqual(got, base)
            self.assertLess(got / base, math.sqrt(2.0))
        with self.assertRaises(ValueError):
            self._publish(value, _ALPHA, "on")

    def test_publication_line(self) -> None:
        # One INFO line per publication (the arming proof of the real scale),
        # none while calibrating or with the knob off.
        torch = self.torch
        value = torch.full((8, 8), 3.0, dtype=torch.bfloat16)
        records = []

        class _H(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        h = _H(logging.DEBUG)
        lg = self.f8.logger
        old = lg.level
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
        try:
            self._publish(value, _ALPHA, None)
            self.assertEqual(records, [])
            got, _ = self._publish(value, _ALPHA, "1")
        finally:
            lg.removeHandler(h)
            lg.setLevel(old)
        spf_recs = [r for r in records if "SPF grid" in r.getMessage()]
        msgs = [r.getMessage() for r in spf_recs]
        self.assertEqual(len(msgs), 1, msgs)
        self.assertEqual(spf_recs[0].levelno, logging.INFO)
        self.assertTrue(msgs[0].startswith(
            f"[hstu-fp8] {_KNOB}: qkv scale snapped onto the SPF grid at "
            "publication: s_cal="), msgs[0])
        j = self.spf.arm_check(got, _ALPHA)
        self.assertIn(f" j={j} ", msgs[0])
        self.assertTrue(msgs[0].endswith(f"e8m0_byte={127 + j}"), msgs[0])


if __name__ == "__main__":
    unittest.main()
