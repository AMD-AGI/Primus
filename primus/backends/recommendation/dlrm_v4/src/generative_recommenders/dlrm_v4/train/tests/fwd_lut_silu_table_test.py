# pyre-strict
"""Row (120) LUT-SiLU fwd: the host table module ops/fp8_lut_silu.py (CPU, stdlib only).

Covers the API contract, the knob parser, the kernel's f32 score_scale order, the
scale contract, the e4m3fn encoder (all bytes, RNE ties, saturation), the table's
special cells, determinism, golden crc32s, and the per-cell error bound of the
table against exact math (compared with the shipped chain's pure rounding bound).
"""

import ast
import math
import os
import random
import re
import struct
import unittest
from unittest import mock

from generative_recommenders.ops import fp8_lut_silu as L

ALPHA = 1.0 / 128 ** 0.5
PV_S, PV_K = 10.04, 3
_OPS = os.path.dirname(os.path.abspath(L.__file__))


def _rne24(x: float) -> float:
    """Independent binary32 rounding of a normal-range double (no struct)."""
    if x == 0.0:
        return x
    m, e = math.frexp(x)                      # |m| in [0.5, 1)
    return math.ldexp(round(math.ldexp(m, 24)), e - 24)   # round(): exact RNE


def _brute_e4m3(v: float) -> int:
    """Nearest finite e4m3fn value (ties -> even code), saturating."""
    sign = 0x80 if math.copysign(1.0, v) < 0.0 else 0
    a = abs(v)
    if a >= 448.0:
        return sign | 0x7E
    best = min(range(0x7F), key=lambda c: (abs(L.e4m3_value(c) - a), c % 2))
    return sign | best


class ApiTest(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(L.KNOB, "DLRM_HSTU_FP8_LUT_SILU_FWD")
        self.assertEqual((L.BITS, L.SIZE, L.J_MIN, L.J_MAX), (13, 8192, -108, 103))
        self.assertEqual(L.LOG2E, 1.4426950408889634)
        for name in ("lut_mode", "f32", "bits32", "from_bits32", "mul32", "score_scale32",
                     "split_scale", "e4m3_byte", "e4m3_value", "p_exact", "build_table",
                     "table_crc32"):
            self.assertTrue(callable(getattr(L, name)), name)

    def test_log2e_matches_kernel_source(self) -> None:
        with open(os.path.join(_OPS, "flydsl", "hstu_fwd.py")) as f:
            src = f.read()
        self.assertTrue(re.search(r"^LOG2E = 1\.4426950408889634$", src, re.M))

    def test_stdlib_only_imports(self) -> None:
        with open(L.__file__) as f:
            tree = ast.parse(f.read())
        mods = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                mods.add((node.module or "").split(".")[0])
        self.assertLessEqual(mods, {"bisect", "decimal", "math", "os", "struct", "zlib",
                                    "fractions", "typing", "sys"})


class LutModeTest(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(L.lut_mode({}), "off")
        self.assertEqual(L.lut_mode({L.KNOB: ""}), "off")
        self.assertEqual(L.lut_mode({L.KNOB: "0"}), "off")
        self.assertEqual(L.lut_mode({L.KNOB: "1"}), "on")
        for bad in ("2", "true", "on", " 1", "1 ", "01", "yes", "-1", "13", "True"):
            with self.assertRaises(ValueError) as cm:
                L.lut_mode({L.KNOB: bad})
            self.assertIn(L.KNOB, str(cm.exception))
            self.assertIn(repr(bad), str(cm.exception))

    def test_reads_os_environ(self) -> None:
        with mock.patch.dict(os.environ, {L.KNOB: "1"}):
            self.assertEqual(L.lut_mode(), "on")
        with mock.patch.dict(os.environ, {L.KNOB: "x"}):
            with self.assertRaises(ValueError):
                L.lut_mode()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(L.lut_mode(), "off")


class ScoreScaleTest(unittest.TestCase):
    def test_literals(self) -> None:
        self.assertEqual(L.bits32(L.f32(-L.LOG2E)), 0xBFB8AA3B)
        self.assertEqual(L.bits32(L.f32(ALPHA)), 0x3DB504F3)

    def test_order_vs_independent(self) -> None:
        rng = random.Random(120)
        scales = [0.0295061, 0.0470843, 0.393415, 1.03795, 0.0119803, 0.015469998121261597]
        scales += [2.0 ** rng.uniform(-30.0, 6.0) for _ in range(3000)]
        for s in scales:
            s32 = _rne24(s)
            want = _rne24(_rne24(_rne24(s32 * s32) * _rne24(ALPHA)) * _rne24(-L.LOG2E))
            got = L.score_scale32(s, ALPHA)
            self.assertEqual(L.bits32(got), L.bits32(want), s)
            self.assertLess(got, 0.0)

    def test_matches_spf_if_importable(self) -> None:
        try:
            from generative_recommenders.ops import fp8_qkv_spf as spf
        except ImportError:
            self.skipTest("fp8_qkv_spf not in this tree")
        rng = random.Random(121)
        for _ in range(2000):
            s = L.f32(2.0 ** rng.uniform(-30.0, 6.0))
            self.assertEqual(L.score_scale32(s, ALPHA), -spf.score_coeff32(s, ALPHA))

    def test_qkv_scale_is_rounded_to_f32_first(self) -> None:
        s = 0.0295061                                 # not an f32
        self.assertEqual(L.score_scale32(s, ALPHA), L.score_scale32(L.f32(s), ALPHA))

    def test_non_finite_inputs_do_not_raise_until_split(self) -> None:
        for s in (0.0, math.inf, math.nan, 1e30, 1e-30):
            ss = L.score_scale32(s, ALPHA)
            with self.assertRaises(ValueError):
                L.split_scale(ss)


class SplitScaleTest(unittest.TestCase):
    def test_valid(self) -> None:
        for ss in (-(2.0 ** -15), -(2.0 ** L.J_MIN), -(2.0 ** L.J_MAX),
                   -L.from_bits32(0x7F7FFFFF & ((L.J_MAX + 127) << 23 | 0x7FFFFF)),
                   L.score_scale32(0.0295061, ALPHA)):
            j, r = L.split_scale(ss)
            self.assertTrue(L.J_MIN <= j <= L.J_MAX and 1.0 <= r < 2.0)
            self.assertEqual(L.f32(r), r)
            self.assertEqual(-math.ldexp(r, j), ss)
        self.assertEqual(L.split_scale(-(2.0 ** -15)), (-15, 1.0))

    def test_rejects(self) -> None:
        for bad in (2.0 ** -15, 0.0, -0.0, -(2.0 ** -130), -(2.0 ** -126) * 0.5, math.inf,
                    -math.inf, math.nan, -(2.0 ** (L.J_MIN - 1)), -(2.0 ** (L.J_MAX + 1)),
                    -0.1, -1e300):
            with self.assertRaises(ValueError, msg=repr(bad)):
                L.split_scale(bad)
            with self.assertRaises(ValueError, msg=repr(bad)):
                L.build_table(bad, PV_S, PV_K)


class E4m3Test(unittest.TestCase):
    def test_all_bytes_round_trip(self) -> None:
        for b in range(256):
            v = L.e4m3_value(b)
            if b & 0x7F == 0x7F:
                self.assertTrue(math.isnan(v))
            self.assertEqual(L.e4m3_byte(v), b, hex(b))
        self.assertEqual(L.e4m3_value(0x7E), 448.0)
        self.assertEqual(L.e4m3_value(0x08), 2.0 ** -6)
        self.assertEqual(L.e4m3_value(0x01), 2.0 ** -9)
        self.assertEqual(math.copysign(1.0, L.e4m3_value(0x80)), -1.0)

    def test_ties_and_saturation(self) -> None:
        cases = ((2.0 ** -10, 0x00), (3 * 2.0 ** -10, 0x02), (1.0625, 0x38), (1.1875, 0x3A),
                 (432.0, 0x7E), (448.0, 0x7E), (455.0, 0x7E), (464.0, 0x7E), (480.0, 0x7E),
                 (1e9, 0x7E), (math.inf, 0x7E), (-math.inf, 0xFE), (-464.0, 0xFE),
                 (0.0, 0x00), (-0.0, 0x80), (-(2.0 ** -11), 0x80), (7.5 * 2.0 ** -9, 0x08),
                 (math.nan, 0x7F), (-math.nan, 0xFF))
        for v, want in cases:
            self.assertEqual(L.e4m3_byte(v), want, repr(v))

    def test_vs_brute_force(self) -> None:
        rng = random.Random(122)
        vals = [rng.uniform(-500.0, 500.0) for _ in range(2000)]
        vals += [math.copysign(2.0 ** rng.uniform(-12.0, 9.0), rng.random() - 0.5)
                 for _ in range(3000)]
        for c in range(0x7E):                     # every tie and its neighbours
            t = 0.5 * (L.e4m3_value(c) + L.e4m3_value(c + 1))
            vals += [t, -t, math.nextafter(t, 0.0) if hasattr(math, "nextafter") else t * (1 - 2 ** -52),
                     t * (1 + 2.0 ** -52)]
        for v in vals:
            self.assertEqual(L.e4m3_byte(v), _brute_e4m3(v), repr(v))


class TableTest(unittest.TestCase):
    GOLDEN = ((0xB8E8D276, PV_S, "fb899e88"),     # iter73 gauss_heads s_cal 0.0295061394572258
              (0xBE0CAD02, PV_S, "d2d8fb64"),     # iter73 t3_heads s_cal 1.0379464626312256
              (0xB8000000, PV_S, "07d7e3c3"),     # j = -15, r = 1 (row (105)'s grid, where present)
              (0xB8000000, 0.0, "380fa97a"))      # pv_s = 0 -> S = 2**pv_k

    def test_golden_crc32_and_determinism(self) -> None:
        for bits, pv_s, crc in self.GOLDEN:
            ss = L.from_bits32(bits)
            t1 = L.build_table(ss, pv_s, PV_K)
            self.assertIsInstance(t1, bytes)
            self.assertEqual(len(t1), L.SIZE)
            self.assertEqual(t1, L.build_table(ss, pv_s, PV_K))
            self.assertEqual("%08x" % L.table_crc32(t1), crc, hex(bits))

    def test_scale_keys(self) -> None:
        self.assertEqual(L.score_scale32(0.0295061394572258, ALPHA), L.from_bits32(0xB8E8D276))
        self.assertEqual(L.score_scale32(1.0379464626312256, ALPHA), L.from_bits32(0xBE0CAD02))

    def test_special_cells(self) -> None:
        t = L.build_table(L.score_scale32(0.0295061, ALPHA), PV_S, PV_K)
        self.assertEqual(t[0xF80], 0xFE)          # +inf
        self.assertEqual(t[0x1F80], 0x00)         # -inf
        for k in range(0xF81, 0x1000):
            self.assertEqual(t[k], 0x7F)
            self.assertEqual(t[0x1000 + k], 0xFF)
        self.assertEqual(t[0], 0x80)              # +0 cell: x = -r*y_mid < 0 -> -0
        self.assertEqual(t[0x1000], 0x00)
        self.assertEqual(t[0xF7F], 0xFE)          # f16 max-finite cell (+ OVFL clamp)
        self.assertEqual(t[0x1F7F], 0x00)

    def test_entries_are_midpoint_rule(self) -> None:
        ss = L.from_bits32(0xBE0CAD02)
        _, r = L.split_scale(ss)
        t = L.build_table(ss, PV_S, PV_K)
        for i in range(0, L.SIZE, 7):
            if (i & 0xFFF) >= 0xF80:
                continue
            h = ((i & 0xFFF) << 3) | 4
            y = struct.unpack("<e", struct.pack("<H", h))[0] * (-1.0 if i >> 12 else 1.0)
            self.assertEqual(L.cell_y_mid(i), y)
            self.assertEqual(t[i], L.e4m3_byte(L.p_exact(-r * y, PV_S)), hex(i))

    def test_bad_pv(self) -> None:
        ss = -(2.0 ** -15)
        for pv_s, pv_k in ((-1.0, 3), (math.nan, 3), (math.inf, 3), (10.04, 3.5), (10.04, "3"),
                           (10.04, math.inf)):
            with self.assertRaises(ValueError):
                L.build_table(ss, pv_s, pv_k)

    def test_index_model_vs_cells(self) -> None:
        rng = random.Random(123)
        for _ in range(20000):
            b = rng.getrandbits(32)
            if (b >> 23) & 0xFF == 0xFF:
                continue
            y = L.from_bits32(b)
            lo, hi = L.cell_y_bounds(L.lut_index_of_y(y))
            self.assertTrue(lo <= y <= hi, (hex(b), lo, hi))
        self.assertEqual(L.lut_index_of_y(70000.0), 0xF7F)
        self.assertEqual(L.lut_index_of_y(-3.0e38), 0x1F7F)
        self.assertEqual(L.lut_index_of_y(math.inf), 0xF80)
        self.assertEqual(L.lut_index_of_y(-math.inf), 0x1F80)

    def test_p_exact(self) -> None:
        self.assertEqual(L.p_exact(math.inf, PV_S), 0.0)
        self.assertEqual(L.p_exact(-math.inf, PV_S), -math.inf)
        self.assertTrue(math.isnan(L.p_exact(math.nan, PV_S)))
        self.assertEqual(L.p_exact(1e6, PV_S), 0.0)
        self.assertAlmostEqual(L.p_exact(-1e6, PV_S), -1.004e7)
        self.assertAlmostEqual(L.p_exact(-1.0, 1.0), -2.0 / 3.0, places=15)
        self.assertAlmostEqual(L.X0, 1.2784645427610738 * L.LOG2E, places=12)
        p0 = L.p_exact(L.X0, 1.0)
        self.assertGreater(p0, L.p_exact(L.X0 - 1e-3, 1.0))
        self.assertGreater(p0, L.p_exact(L.X0 + 1e-3, 1.0))


class ErrorBoundTest(unittest.TestCase):
    """Bounds vs the exact S*x/(1+2^x) clamped to +-448 (the e4m3 range of both chains).

    Derivation [D]: the shipped chain modelled as e4m3_byte(exact P) errs by at most
    half an e4m3 ulp: 16 in P8 units (top binade [256, 448], ulp 32), 1/17 relative
    (value 8.5 * 2**k). The LUT byte is that rounding at the cell midpoint, and a
    13-bit RTZ cell spans 8 f16 ulps (relative width <= 2**-7), so the cell adds its
    own P variation. abs: near saturation P ~ S*x is linear; the worst cell is the
    first of an f16 binade, y in [32, 32.25), at the boundary 432:
    16 + 432 * 0.125/32.125 = 17.6809 (<= 16 + 448 * 2**-7 / 2 = 17.75). rel: in the
    decaying tail (P ~ S*x*2**-x, |d ln P/dx| = 0.616 at x = 12.93) the worst cell
    is y in (-8.0625, -8] at the lowest normal boundary 8.5 * 2**-9:
    (1 + 1/17) * 1.03148 - 1 = 0.092154.

    Ceilings from the r sweep REL_SWEEP (x4c19 iter76 T2; pv_s = 10.04, pv_k = 3; the
    table reads only r, so j is free): 2002 evenly spaced f32 r in [1, 2) at j = -14
    give max rel 0.090595, abs 17.6533; the continuous-r sup over all 7936 cells and
    every e4m3 boundary is rel 0.092154 (cell 0x1900, attained at f32 0xB8CE1F8F) and
    abs 17.680934 (cell 0x0A00, f32 0xB8AB711E reaches 17.680921). Asserted:
    sup + ~0.4% margin, rel <= 0.0925 (+0.00035) and abs <= 17.75 (+0.069).
    """

    REL_CEIL = 0.0925
    ABS_CEIL = 17.75
    # evidence scales, then the f32 scales that attain the rel / abs sup (REL_SWEEP)
    SCALES = (0xB8E8D276, 0xBE0CAD02, 0xB8000000, 0xB8CE1F8F, 0xB8AB711E)

    def test_bounds(self) -> None:
        for bits in self.SCALES:
            ss = L.from_bits32(bits)
            rep = L.table_error_report(ss, PV_S, PV_K)
            self.assertLessEqual(rep["lut_max_abs_p8"], self.ABS_CEIL)
            self.assertLessEqual(rep["lut_max_rel_ge_min_normal"], self.REL_CEIL)
            self.assertEqual(rep["ship_max_abs_p8"], 16.0)
            self.assertLessEqual(rep["ship_max_rel_ge_min_normal"], 1.0 / 17.0 + 1e-12)
            self.assertGreaterEqual(rep["lut_max_abs_p8"], rep["ship_max_abs_p8"])
            # 126 boundaries on the P < 0 branch + 2 x 72 below max P = S*P(X0) ~ 4.03
            self.assertEqual(rep["n_straddle_cells"], 270)
            self.assertLessEqual(rep["max_span_ulps"], 1)      # every moved byte is +-1 ulp
            self.assertEqual(rep["n_entry_outside_span"], 0)
            self.assertEqual(rep["n_finite_cells"], 2 * 0xF80)

    def test_sweep_sup_scales(self) -> None:
        # REL_SWEEP's measured sup (continuous-r value 0.092154 / 17.680934) and its cells
        rep = L.table_error_report(L.from_bits32(0xB8CE1F8F), PV_S, PV_K)
        self.assertAlmostEqual(rep["lut_max_rel_ge_min_normal"], 0.0921536, places=6)
        self.assertEqual(rep["argmax_cell"]["lut_rel"], "0x1900")
        rep = L.table_error_report(L.from_bits32(0xB8AB711E), PV_S, PV_K)
        self.assertAlmostEqual(rep["lut_max_abs_p8"], 17.680921, places=5)
        self.assertEqual(rep["argmax_cell"]["lut_abs"], "0x0a00")

    def test_per_cell_lut_vs_ship(self) -> None:
        ss = L.from_bits32(0xB8E8D276)
        _, r = L.split_scale(ss)
        t = L.build_table(ss, PV_S, PV_K)
        for i in range(L.SIZE):
            ce = L.cell_error(i, t[i], r, PV_S)
            if ce is None:
                continue
            lo, hi = ce["p"]
            width = min(hi, 448.0) - max(lo, -448.0)
            self.assertLessEqual(ce["lut_abs"], ce["ship_abs"] + max(width, 0.0) + 1e-9, hex(i))
            if not ce["straddle"]:
                self.assertEqual(t[i], L.e4m3_byte(lo), hex(i))


if __name__ == "__main__":
    unittest.main()
