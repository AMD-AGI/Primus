# pyre-strict
"""Host side of row (120) LUT-SiLU fwd (DLRM_HSTU_FP8_LUT_SILU_FWD): the P8 table.

The shipped FP8 forward (ops/flydsl/hstu_fwd.py, pv="k128", epi_fold) turns
every QK^T accumulator element `raw` into the e4m3 P8 byte

    x  = raw * score_scale,   score_scale = fl(fl(fl(s*s) * f32(alpha)) * f32(-LOG2E))
    P8 = e4m3_rne_sat(S * x / (1 + 2**x)),   S = pv_s if pv_s else 2**pv_k

(s = qkv_scale; S * x/(1+2^x) = -S*LOG2E * silu(z) with z = -x/LOG2E the
true score). With the knob on, score_scale = -2**j * r (r in [1, 2)) is split:
the QK MFMA takes the E8M0 A-scale byte 127 + j, so the accumulator is
y = 2**j * raw exactly; v_cvt_pkrtz_f16_f32 (round toward zero, MODE.FP16_OVFL
= 1: a finite overflow gives +-65504) makes an f16 h, and the top 13 bits of h
(sign | exp5 | mantissa[9:3]) index an 8 KB table T of e4m3 bytes.

Construction of T (a pure function of (bits of score_scale, pv_s, pv_k); it
is computed on the host from the formula only, never from data):
  * finite cell i: e4m3_rne_sat(S * x/(1 + 2**x)) at x = -r * y_mid(i), where
    y_mid(i) is the f16 with bits (i << 3) | 4 (the midpoint of the RTZ cell;
    with sign 1 the value is negated). x is exact in fp64 (24 x 11 bits). The
    function is evaluated in fp64; when the fp64 value lies within 2**-40
    (relative) of an e4m3 rounding boundary the decision is redone: exactly
    with Fraction when x is an integer with |x| <= 4096 (an exact tie goes to
    the even code), else with 80-digit Decimal (2**x = exp(x ln 2)), which
    raises ArithmeticError (fail closed) when the value is within 1e-70
    (relative) of the boundary, i.e. an unresolvable tie. For S >= 4 no exact
    tie reaches the Decimal path: 2**x is irrational for a non-integer x, and
    an integer |x| > 4096 is never re-decided (|P| > 448 saturates, or
    P < 2**-10 rounds to zero). So the bytes are the exact-math RNE bytes and
    do not depend on the host libm.
  * +inf cell (f16 0x7C00 prefix, sign 0): 0xFE (-448, y -> +inf);
    -inf cell (sign 1): 0x00 (+0, y -> -inf);
  * NaN cells (exp5 = 31, mantissa bits != 0): 0x7F | sign << 7.
  * e4m3 encoding: round to nearest even, e4m3 subnormals (2**-9 quantum),
    signed zero, |v| >= 448 after rounding saturates to 0x7E | sign.
Contract on the scale: score_scale must be a normal finite negative binary32
with exponent j in [J_MIN, J_MAX] (the range where 2**j * raw stays a normal
f32 for every e4m3 dot, |dot| in [2**-18, 128*448**2]); anything else raises
ValueError (fail closed).

Analysis helpers (tests / evidence): cell_y_bounds (the f32 y set of each RTZ
cell, incl. the f16 max-finite cell that also holds every finite |y| >= 65504),
cell_error (exact bound of |T[i] - sat(S*x/(1+2^x))| over the whole cell, and
the same bound for the shipped chain modelled as e4m3_byte(exact P)), and
table_error_report. P(x) = x/(1+2^x) rises on (-inf, X0] and falls on
[X0, inf): one interior extremum (a maximum of P, i.e. silu's minimum at
z = -1.27846), X0 = 1.27846 * LOG2E ~ 1.8444, the root of 1 + 2^x - x ln2 2^x.

Stdlib only, Python 3.9, no mutable state. `python3 fp8_lut_silu.py
--selftest` runs the exhaustive self-checks.
"""

import bisect
import decimal
import math
import os
import struct
import zlib
from fractions import Fraction
from typing import Dict, List, Mapping, Optional, Tuple

KNOB = "DLRM_HSTU_FP8_LUT_SILU_FWD"
BITS = 13
SIZE = 8192
# hstu_fwd.LOG2E (the kernel builds f32(-LOG2E) = 0xbfb8aa3b from this literal).
LOG2E = 1.4426950408889634
# 2**j * raw stays a normal f32 for every e4m3 dot, |dot| in [2**-18, 128*448**2]:
# 2**-108 * 2**-18 = 2**-126 and 2**103 * 128*448**2 < 2**128.
J_MIN = -108
J_MAX = 103

E4M3_MAX = 448.0
E4M3_MIN_NORMAL = 2.0 ** -6
_HALF = SIZE // 2                 # 4096 cells per sign
_INF_CELL = 0x7C00 >> 3           # 0xF80
_TOP_CELL = 0x7BF8 >> 3           # 0xF7F: f16 max finite 0x7BFF (+ RTZ/OVFL clamp)
_GUARD = 2.0 ** -40               # near-boundary band for the exact re-decision


# ------------------------------------------------- binary32 RNE helpers
def f32(x: float) -> float:
    """x rounded to binary32 (RNE, subnormals kept). A finite x past the f32
    range gives +-inf (IEEE round-to-nearest overflow) where struct.pack alone
    raises OverflowError."""
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


def bits32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def mul32(a: float, b: float) -> float:
    """binary32 product of two binary32 values (the fp64 product of two 24-bit
    significands is exact, so one rounding gives the f32 product)."""
    return f32(a * b)


F32_MAX = from_bits32(0x7F7FFFFF)


def score_scale32(qkv_scale: float, alpha: float) -> float:
    """fwd_kernel_0's f32 score_scale: s = f32(qkv_scale) (an f32 kernel arg),
    fl(fl(fl(s*s) * f32(alpha)) * f32(-LOG2E)), equal to the kernel's
    emit_body product (ops/flydsl/hstu_fwd.py). ship17 row (105), where
    present, computes the same product with the opposite sign (RNE is
    sign-symmetric)."""
    s = f32(qkv_scale)
    return mul32(mul32(mul32(s, s), f32(alpha)), f32(-LOG2E))


def split_scale(ss: float) -> Tuple[int, float]:
    """(j, r) with ss == -2**j * r exactly, r in [1, 2) an f32. Raises
    ValueError unless ss is a normal finite NEGATIVE binary32 with
    J_MIN <= j <= J_MAX."""
    v = float(ss)
    if not (math.isfinite(v) and v < 0.0):
        raise ValueError(f"{KNOB}: score_scale {ss!r} is not a finite negative number")
    try:
        b = bits32(v)
    except OverflowError:
        raise ValueError(f"{KNOB}: score_scale {ss!r} is outside binary32") from None
    if from_bits32(b) != v:
        raise ValueError(f"{KNOB}: score_scale {ss!r} is not a binary32 value")
    e = (b >> 23) & 0xFF
    if e == 0:
        raise ValueError(f"{KNOB}: score_scale {ss!r} is an f32 subnormal")
    j = e - 127
    if not J_MIN <= j <= J_MAX:
        raise ValueError(f"{KNOB}: score_scale {ss!r} has j={j} outside [{J_MIN}, {J_MAX}]")
    return j, from_bits32((127 << 23) | (b & 0x7FFFFF))


def lut_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """$DLRM_HSTU_FP8_LUT_SILU_FWD (read per call): unset, "" or "0" -> "off";
    "1" -> "on"; anything else raises ValueError (fail closed)."""
    raw = (os.environ if env is None else env).get(KNOB)
    if raw is None or raw in ("", "0"):
        return "off"
    if raw == "1":
        return "on"
    raise ValueError(f"{KNOB}={raw!r}: use unset, '0' or '1'")


# ------------------------------------------------------------------- e4m3fn
def _e4m3_mag_value(c: int) -> float:
    e, m = c >> 3, c & 7
    return m * 2.0 ** -9 if e == 0 else (8 + m) * 2.0 ** (e - 10)


def _e4m3_code(a: float) -> int:
    """Magnitude code 0..0x7E of RNE(a), a >= 0 (inf allowed), saturating."""
    if a >= E4M3_MAX:
        return 0x7E
    if a < E4M3_MIN_NORMAL:
        return int(round(a * 512.0))          # exact scaling; round() is RNE; 8 -> 0x08
    _, e = math.frexp(a)
    eb = e - 1                                # a in [2**eb, 2**(eb + 1)), eb in [-6, 8]
    n = int(round(math.ldexp(a, 3 - eb)))     # 8..16 (exact scaling, RNE)
    return min((eb + 7) * 8 + n - 8, 0x7E)    # n == 16 carries into the next binade


def e4m3_byte(v: float) -> int:
    """e4m3fn byte of v: RNE, subnormals, signed zero, |v| >= 448 after
    rounding (incl. +-inf) -> 0x7E | sign, NaN -> 0x7F | sign."""
    sign = 0x80 if math.copysign(1.0, v) < 0.0 else 0
    if v != v:
        return 0x7F | sign
    return sign | _e4m3_code(abs(v))


def e4m3_value(b: int) -> float:
    """Decode an e4m3fn byte (0x7F / 0xFF = NaN, 0x80 = -0.0)."""
    if not 0 <= b <= 0xFF:
        raise ValueError(f"e4m3 byte out of range: {b!r}")
    c = b & 0x7F
    v = math.nan if c == 0x7F else _e4m3_mag_value(c)
    return -v if b & 0x80 else v


# Positive RNE boundaries: midpoints of consecutive codes 0..0x7E (126, exact).
_TIES = tuple(0.5 * (_e4m3_mag_value(c) + _e4m3_mag_value(c + 1)) for c in range(0x7E))
_SIGNED_TIES = tuple(sorted([-t for t in _TIES] + list(_TIES)))


# ------------------------------------------------------------------ the math
def p_exact(x: float, S: float) -> float:
    """S * x / (1 + 2**x) in fp64, overflow-safe (2**x is never formed for
    x > 0). Limits: x = +inf -> +0.0, x = -inf -> -inf; NaN -> NaN."""
    if x != x:
        return math.nan
    if x > 0.0:
        if x == math.inf:
            return 0.0
        t = 2.0 ** (-x)                       # (0, 1]; underflows to 0 for huge x
        return S * (x * t) / (1.0 + t)
    return S * x / (1.0 + 2.0 ** x)


def _solve_x0() -> float:
    lo, hi = 1.5, 2.5                         # f(x) = 1 + 2^x - x ln2 2^x: f(lo) > 0 > f(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid in (lo, hi):
            break
        u = 2.0 ** mid
        if 1.0 + u - mid * math.log(2.0) * u > 0.0:
            lo = mid
        else:
            hi = mid
    return lo


X0 = _solve_x0()                              # ~1.8444; max of x/(1+2^x)


def _exact_above(x: float, S: float, mid: float) -> int:
    """sign(|S*x/(1+2^x)| - mid) for an e4m3 boundary mid. Exact (Fraction)
    when x is an integer with |x| <= 4096; 0 means an exact tie. Otherwise
    80-digit Decimal, which never returns 0: within 1e-70 (relative) of mid it
    raises ArithmeticError (fail closed, an unresolvable tie). For S >= 4 no
    exact tie takes that path: 2**x is irrational for a non-integer x, and
    p8_byte never re-decides an integer |x| > 4096 (|P| > 448 saturates, or
    P < 2**-10 rounds to zero)."""
    if x == math.floor(x) and abs(x) <= 4096:
        p = abs(Fraction(S) * Fraction(x) / (1 + Fraction(2) ** int(x)))
        m = Fraction(mid)
        return (p > m) - (p < m)
    with decimal.localcontext() as ctx:
        ctx.prec = 80
        dx = decimal.Decimal(x)
        u = (dx * decimal.Decimal(2).ln()).exp()
        p = abs(decimal.Decimal(S) * dx / (1 + u))
        m = decimal.Decimal(mid)
        if abs(p - m) <= m * decimal.Decimal("1e-70"):
            raise ArithmeticError(f"{KNOB}: unresolvable e4m3 tie at x={x!r} S={S!r}")
        return 1 if p > m else -1


def p8_byte(x: float, S: float) -> int:
    """e4m3_rne_sat(S * x/(1 + 2**x)) for finite x with the exact-math RNE
    decision (fp64 plus an exact re-decision within 2**-40 of a boundary)."""
    p = p_exact(x, S)
    sign = 0x80 if math.copysign(1.0, p) < 0.0 else 0
    a = abs(p)
    c_lo = _e4m3_code(a * (1.0 - _GUARD))
    c_hi = _e4m3_code(a * (1.0 + _GUARD))
    if c_lo == c_hi:
        return sign | c_lo
    mid = 0.5 * (_e4m3_mag_value(c_lo) + _e4m3_mag_value(c_hi))   # c_hi == c_lo + 1
    cmp = _exact_above(x, S, mid)
    if cmp == 0:
        return sign | (c_lo if c_lo % 2 == 0 else c_hi)          # tie -> even code
    return sign | (c_hi if cmp > 0 else c_lo)


def pv_gain(pv_s: float, pv_k: int) -> float:
    """S of the P8 scaling: pv_s when nonzero, else 2**pv_k."""
    try:
        s = float(pv_s)
        k = int(pv_k)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{KNOB}: bad pv_s/pv_k {pv_s!r}/{pv_k!r}") from None
    if k != pv_k or not -64 <= k <= 64:
        raise ValueError(f"{KNOB}: pv_k must be an integer in [-64, 64], got {pv_k!r}")
    if not (math.isfinite(s) and s >= 0.0):
        raise ValueError(f"{KNOB}: pv_s must be finite and >= 0, got {pv_s!r}")
    return s if s != 0.0 else math.ldexp(1.0, k)


def _f16_pos_value(h: int) -> float:
    e, m = h >> 10, h & 0x3FF
    return m * 2.0 ** -24 if e == 0 else (1024 + m) * 2.0 ** (e - 25)


def cell_y_mid(i: int) -> float:
    """f16 value with bits (i << 3) | 4 (sign from bit 12)."""
    v = _f16_pos_value(((i & 0xFFF) << 3) | 4)
    return -v if i >> 12 else v


def table_entry(i: int, r: float, S: float) -> int:
    """T[i] for scale residual r and gain S (the table rule in this module's
    docstring)."""
    if not 0 <= i < SIZE:
        raise ValueError(f"cell index out of range: {i!r}")
    sign = i >> 12
    k = i & 0xFFF
    if k >= _INF_CELL:
        if k == _INF_CELL:
            return 0x00 if sign else 0xFE
        return 0x7F | (sign << 7)
    return p8_byte(-r * cell_y_mid(i), S)


def build_table(score_scale: float, pv_s: float, pv_k: int) -> bytes:
    """The 8192-byte e4m3 table for the kernel's f32 score_scale (the table
    rule in this module's docstring)."""
    _, r = split_scale(score_scale)
    S = pv_gain(pv_s, pv_k)
    return bytes(table_entry(i, r, S) for i in range(SIZE))


def table_crc32(tbl: bytes) -> int:
    return zlib.crc32(bytes(tbl)) & 0xFFFFFFFF


def lut_index_of_y(y: float) -> int:
    """Cell of an f32 y under v_cvt_pkrtz_f16_f32 with MODE.FP16_OVFL = 1
    (reference model; quiet-NaN payload: f16 mantissa bit 9 forced)."""
    b = bits32(y)
    sign = b >> 31
    e = (b >> 23) & 0xFF
    m = b & 0x7FFFFF
    if e == 255:
        h = 0x7C00 if m == 0 else (0x7E00 | (m >> 13))
    elif e >= 143:                            # |y| >= 65536: finite overflow clamps
        h = 0x7BFF
    elif e >= 113:
        h = ((e - 112) << 10) | (m >> 13)
    elif e == 0:
        h = 0
    else:
        h = ((m | 0x800000) >> (126 - e)) if 126 - e < 32 else 0
    return (sign << 12) | (h >> 3)


# ------------------------------------------------------------------ analysis
def cell_y_bounds(i: int) -> Optional[Tuple[float, float]]:
    """Closure [lo, hi] (signed) of the f32 y the RTZ index maps to cell i:
    [v(h0), v(h0 + 8)) for sign 0, mirrored (-v(h0 + 8), -v(h0)] for sign 1;
    the f16 max-finite cell holds [65280, F32_MAX] (finite |y| >= 65504 is
    clamped to 0x7BFF); the +-0 cell holds f32 zero/subnormals up to 2**-21.
    None for the inf / NaN cells."""
    k = i & 0xFFF
    if k >= _INF_CELL:
        return None
    lo = _f16_pos_value(k << 3)
    hi = F32_MAX if k == _TOP_CELL else _f16_pos_value((k << 3) + 8)
    return (-hi, -lo) if i >> 12 else (lo, hi)


def cell_f32_count(i: int) -> int:
    """Number of f32 bit patterns in cell i (finite cells; incl. +-0)."""
    k = i & 0xFFF
    if k >= _INF_CELL:
        return 0
    lo = bits32(_f16_pos_value(k << 3))
    if k == _TOP_CELL:
        return 0x7F7FFFFF + 1 - lo
    return bits32(_f16_pos_value((k << 3) + 8)) - lo


def _sat(p: float) -> float:
    return max(-E4M3_MAX, min(E4M3_MAX, p))


def cell_error(i: int, byte: int, r: float, S: float) -> Optional[Dict[str, object]]:
    """Exact (fp64-evaluated) bounds over the WHOLE cell i (closure) of
      lut_abs  = max |T - sat(P(x))|,   ship_abs = max |e4m3(P(x)) - sat(P(x))|
    (P = S*x/(1+2^x), x = -r*y, sat = clamp to +-448: the e4m3 output range of
    both chains), the relative forms when |sat P| >= 2**-6 on the whole cell,
    and whether e4m3_byte(P) is constant on the cell (straddle = not). None
    for the inf/NaN cells. P is unimodal (max at X0), so its range over the
    cell is spanned by the endpoints and X0; the shipped rounding error peaks
    at the endpoints or at a boundary (tie) inside the range."""
    yb = cell_y_bounds(i)
    if yb is None:
        return None
    xa, xb = sorted((-r * yb[0], -r * yb[1]))
    pa, pb = p_exact(xa, S), p_exact(xb, S)
    lo, hi = min(pa, pb), max(pa, pb)
    if xa < X0 < xb:
        hi = max(hi, p_exact(X0, S))
    t = e4m3_value(byte)
    slo, shi = _sat(lo), _sat(hi)
    lut_abs = max(abs(t - slo), abs(t - shi))
    k0 = bisect.bisect_right(_SIGNED_TIES, lo)
    k1 = bisect.bisect_left(_SIGNED_TIES, hi)
    cands = [lo, hi] + list(_SIGNED_TIES[k0:k1])
    ship_err = [abs(e4m3_value(e4m3_byte(p)) - _sat(p)) for p in cands]
    rel_ok = slo >= E4M3_MIN_NORMAL or shi <= -E4M3_MIN_NORMAL
    b_lo, b_hi = e4m3_byte(lo), e4m3_byte(hi)
    ordl = (b_lo & 0x7F) * (-1 if b_lo & 0x80 else 1)
    ordh = (b_hi & 0x7F) * (-1 if b_hi & 0x80 else 1)
    ordt = (byte & 0x7F) * (-1 if byte & 0x80 else 1)
    return {
        "x": (xa, xb), "p": (lo, hi), "t": t,
        "lut_abs": lut_abs,
        "lut_rel": max(abs(t / slo - 1.0), abs(t / shi - 1.0)) if rel_ok else None,
        "ship_abs": max(ship_err),
        "ship_rel": (max(e / abs(_sat(p)) for e, p in zip(ship_err, cands))
                     if rel_ok else None),
        "straddle": b_lo != b_hi,
        "span_ulps": abs(ordh - ordl),
        "t_in_span": min(ordl, ordh) <= ordt <= max(ordl, ordh),
        "saturates": hi > E4M3_MAX or lo < -E4M3_MAX,
    }


def table_error_report(score_scale: float, pv_s: float, pv_k: int,
                       tbl: Optional[bytes] = None) -> Dict[str, object]:
    """Aggregate cell_error over every finite cell of the table."""
    j, r = split_scale(score_scale)
    S = pv_gain(pv_s, pv_k)
    if tbl is None:
        tbl = build_table(score_scale, pv_s, pv_k)
    agg = {"lut_abs": 0.0, "ship_abs": 0.0, "lut_rel": 0.0, "ship_rel": 0.0}
    arg = {}
    n_cells = n_strad = n_sat = 0
    f32_all = f32_strad = 0
    max_span = 0
    t_out = 0
    strad_cells: List[int] = []
    for i in range(SIZE):
        ce = cell_error(i, tbl[i], r, S)
        if ce is None:
            continue
        n_cells += 1
        cnt = cell_f32_count(i)
        f32_all += cnt
        for key in agg:
            v = ce[key]
            if v is not None and v > agg[key]:
                agg[key] = v
                arg[key] = i
        n_sat += bool(ce["saturates"])
        if ce["straddle"]:
            n_strad += 1
            f32_strad += cnt
            strad_cells.append(i)
        max_span = max(max_span, int(ce["span_ulps"]))
        t_out += not ce["t_in_span"]
    unit = S * LOG2E
    return {
        "score_scale": score_scale, "score_scale_bits": "0x%08x" % bits32(score_scale),
        "j": j, "r": r, "S": S, "crc32": "%08x" % table_crc32(tbl),
        "n_finite_cells": n_cells, "n_saturating_cells": n_sat,
        "lut_max_abs_p8": agg["lut_abs"], "lut_max_abs_silu": agg["lut_abs"] / unit,
        "lut_max_rel_ge_min_normal": agg["lut_rel"],
        "ship_max_abs_p8": agg["ship_abs"], "ship_max_abs_silu": agg["ship_abs"] / unit,
        "ship_max_rel_ge_min_normal": agg["ship_rel"],
        "argmax_cell": {k: "0x%04x" % v for k, v in arg.items()},
        "n_straddle_cells": n_strad, "frac_straddle_cells": n_strad / n_cells,
        "frac_straddle_f32": f32_strad / f32_all,
        "max_span_ulps": max_span, "n_entry_outside_span": t_out,
        "straddle_cells": ["0x%04x" % c for c in strad_cells],
    }


# ---------------------------------------------------------------- selftest
def _selftest(verbose: bool = True) -> int:  # noqa: C901
    fails: List[str] = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    for v in (None, "", "0"):
        check(lut_mode({} if v is None else {KNOB: v}) == "off", f"mode off {v!r}")
    check(lut_mode({KNOB: "1"}) == "on", "mode on")
    for bad in ("2", "true", "on", " 1", "1 ", "01", "yes", "-1", "13"):
        try:
            lut_mode({KNOB: bad})
            fails.append(f"no raise for {bad!r}")
        except ValueError as e:
            check(KNOB in str(e) and repr(bad) in str(e), f"message {bad!r}")
    check(bits32(f32(-LOG2E)) == 0xBFB8AA3B, "f32(-LOG2E) literal")
    check(bits32(f32(1.0 / 128 ** 0.5)) == 0x3DB504F3, "f32(alpha) literal")
    # e4m3: round trip of all 256 bytes, RNE ties, saturation edges
    for b in range(256):
        v = e4m3_value(b)
        if b & 0x7F == 0x7F:
            check(v != v and e4m3_byte(v) == b, f"nan byte {b:#04x}")
        else:
            check(e4m3_byte(v) == b, f"round trip {b:#04x}")
    for v, want in ((2.0 ** -10, 0x00), (-(2.0 ** -10), 0x80), (3 * 2.0 ** -10, 0x02),
                    (1.0625, 0x38), (1.1875, 0x3A), (15.5 * 2.0 ** -9, 0x10),
                    (432.0, 0x7E), (431.99, 0x7D), (448.0, 0x7E), (464.0, 0x7E),
                    (480.0, 0x7E), (-464.0, 0xFE), (math.inf, 0x7E), (-math.inf, 0xFE),
                    (0.0, 0x00), (-0.0, 0x80), (2.0 ** -11, 0x00), (-(2.0 ** -11), 0x80),
                    (1.5 * 2.0 ** -9, 0x02), (7.5 * 2.0 ** -9, 0x08)):
        check(e4m3_byte(v) == want, f"e4m3_byte({v!r}) = {e4m3_byte(v):#04x} != {want:#04x}")
    # every boundary and its neighbours
    for c in range(0x7E):
        t = _TIES[c]
        check(e4m3_byte(t) == (c if c % 2 == 0 else c + 1), f"tie {c}")
        check(e4m3_byte(t * (1 - 2.0 ** -50)) == c, f"below tie {c}")
        check(e4m3_byte(t * (1 + 2.0 ** -50)) == c + 1, f"above tie {c}")
    # tables
    alpha = 1.0 / 128 ** 0.5
    for s in (0.0295061, 0.0470843, 0.393415, 1.03795, 0.0119803, 0.015469998121261597):
        ss = score_scale32(s, alpha)
        j, r = split_scale(ss)
        check(-math.ldexp(r, j) == ss and 1.0 <= r < 2.0, f"split {s}")
        tbl = build_table(ss, 10.04, 3)
        check(len(tbl) == SIZE and tbl == build_table(ss, 10.04, 3), f"len/determinism {s}")
        check(tbl[_INF_CELL] == 0xFE and tbl[_HALF + _INF_CELL] == 0x00, f"inf cells {s}")
        check(all(tbl[k] == 0x7F and tbl[_HALF + k] == 0xFF
                  for k in range(_INF_CELL + 1, _HALF)), f"nan cells {s}")
        check(tbl[0] == 0x80 and tbl[_HALF] == 0x00, f"zero cells {s}")
        check(tbl[_TOP_CELL] == 0xFE and tbl[_HALF + _TOP_CELL] == 0x00, f"top cells {s}")
        # monotone in y: sign 0 non-increasing; sign 1 unimodal (peak at X0)
        vp = [e4m3_value(tbl[k]) for k in range(_INF_CELL)]
        check(all(vp[k + 1] <= vp[k] for k in range(len(vp) - 1)), f"monotone +y {s}")
        vn = [e4m3_value(tbl[_HALF + k]) for k in range(_INF_CELL)]
        kp = max(range(len(vn)), key=lambda k: vn[k])
        check(all(vn[k + 1] >= vn[k] for k in range(kp)) and
              all(vn[k + 1] <= vn[k] for k in range(kp, len(vn) - 1)), f"unimodal -y {s}")
        for k in range(SIZE):
            yb = cell_y_bounds(k)
            if yb is None:
                continue
            for y in (yb[0], yb[1], cell_y_mid(k)):
                if abs(y) < F32_MAX and y != 0.0 and f32(y) == y and y not in (
                        _f16_pos_value((k & 0xFFF) * 8 + 8), -_f16_pos_value((k & 0xFFF) * 8 + 8)):
                    check(lut_index_of_y(y) == k, f"index of y={y!r} cell {k:#06x}")
        rep = table_error_report(ss, 10.04, 3, tbl)
        check(rep["max_span_ulps"] <= 1 and rep["n_entry_outside_span"] == 0, f"span {s}")
    check(abs(X0 - 1.2784645427610738 * LOG2E) < 1e-9, f"X0 {X0!r}")
    for bad in (0.0, -0.0, 1.0, -math.inf, math.nan, -(2.0 ** -130), -(2.0 ** -109),
                -(2.0 ** 104), -0.1):
        try:
            split_scale(bad)
            fails.append(f"split_scale accepted {bad!r}")
        except ValueError:
            pass
    if verbose:
        ss = score_scale32(0.0295061, alpha)
        print(f"fp8_lut_silu selftest {'PASS' if not fails else 'FAIL'}: entries={SIZE} "
              f"X0={X0!r} prod-like ss={ss!r} (0x{bits32(ss):08x}) crc32="
              f"{table_crc32(build_table(ss, 10.04, 3)):08x}"
              + ("" if not fails else f" ({len(fails)}): " + "; ".join(fails[:10])))
    return 0 if not fails else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
