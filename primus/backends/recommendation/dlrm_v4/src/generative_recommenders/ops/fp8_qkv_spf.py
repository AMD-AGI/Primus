# pyre-strict
"""SPF host side: the e4m3 q/k/v scale snap behind DLRM_HSTU_FP8_LAB_FWD_SPF.

SPF is the score-scale power-of-two fold (X4C19 iter70/71; LAB knob, default
OFF, deliberately NOT bound in gin). fwd_kernel_0 (ops/flydsl/hstu_fwd.py)
multiplies every QK^T accumulator element by the uniform f32

    score_scale = fl(fl(fl(s * s) * A) * -L)       A = f32(alpha), L = f32(log2 e)

(s = qkv_scale, the one per-tensor dequant scale of the e4m3 q/k/v; the order
and the two literals 0x3db504f3 / 0xbfb8aa3b are the ones the shipped ISA
issues). The calibrated s = amax * margin / 448 is never a point where that
product is a power of two, so the multiply cannot ride the MFMA's E8M0 scale
field. SPF snaps s UP to the f32 grid

    G = { s : score_coeff32(s, alpha) == 2**j exactly, j an integer }

(the smallest grid point >= the calibrated scale: no new saturation, at most
0.5 binade of e4m3 range given up). Then score_scale = -2**j EXACTLY, and

  * "snap"  (the SNAPPED CONTROL): only the published scale changes; the kernel
    is the shipped one and multiplies by exactly -2**j;
  * "1"     (snap + fold): the QK MFMA takes the runtime E8M0 A-scale byte
    127 + j (v_mfma_scale_f32_16x16x128_f8f6f4) and the per-element multiply
    is deleted; exp2 and the silu multiply read -acc through VOP3 neg source
    modifiers. Bitwise equal to "snap" under the conditions in CONTRACT.md
    (the scaled MFMA is exact power-of-two scaling of the unscaled one; no f32
    subnormal accumulator, guaranteed by J_MIN/J_MAX below).

For alpha = 1/sqrt(128) (stu.py: attention_dim 128, attn_alpha unset) every j
in [J_MIN, J_MAX] has exactly ONE f32 preimage s_j, with mantissa 0x33393e
(even j) or 0x7d75e0 (odd j), and s_{j+2} = 2 * s_j (selftest below).

Stdlib only, Python 3.9 (node-side): imported by ops/triton/triton_fp8.py
(calibration snap) and ops/flydsl/hstu_flydsl_mha.py (arming). It holds no
mutable state. `python3 fp8_qkv_spf.py --selftest` runs the exhaustive checks.
"""

import math
import os
import struct
from typing import Mapping, Optional, Tuple

KNOB = "DLRM_HSTU_FP8_LAB_FWD_SPF"

# hstu_fwd.LOG2E (the kernel builds f32(-LOG2E) = 0xbfb8aa3b from this literal).
LOG2E = 1.4426950408889634

# Arming range of j. Every e4m3 value is an integer multiple of 2**-9, so a
# nonzero QK^T dot of e4m3 operands (exact products, any summation) has
# magnitude >= 2**-18, and |dot| <= 128 * 448**2 < 2**24.62. The folded
# accumulator 2**j * dot is then a NORMAL f32 (no subnormal, no overflow) for
# -108 <= j <= 103, which is the range where power-of-two scaling commutes
# with every rounding on the path (and the E8M0 byte 127 + j is in 19..230).
J_MIN = -108
J_MAX = 103

_F32_MAX_BITS = 0x7F800000
_SCAN_ULPS = 16


def f32(x: float) -> float:
    """x rounded to binary32 (round-to-nearest-even); OverflowError past f32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits32(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits32(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def mul32(a: float, b: float) -> float:
    """IEEE binary32 product of two binary32 values (v_mul_f32, RNE, denormals
    on: fwd_kernel_0 runs float_denorm_mode_32 = 3). The double product of two
    24-bit significands is exact, so one rounding to f32 is the f32 product."""
    return f32(a * b)


def score_coeff32(s: float, alpha: float) -> float:
    """|score_scale| exactly as fwd_kernel_0 computes it: fl(fl(fl(s*s)*A)*L).
    (The kernel multiplies by -L; RNE is sign-symmetric, so its score_scale is
    exactly -score_coeff32.)"""
    return mul32(mul32(mul32(s, s), f32(alpha)), f32(LOG2E))


def score_coeff32_bwd(s: float, alpha: float) -> float:
    """|c_t| in bwd_dkdv's order: fl(fl(A * fl(s*s)) * L) (score_alpha_scale =
    f32(alpha) * score_input_scale, then * -LOG2E). Equal to score_coeff32 for
    every s (IEEE multiplication is commutative); kept for the contract check."""
    return mul32(mul32(f32(alpha), mul32(s, s)), f32(LOG2E))


def pow2_exponent(c: float) -> Optional[int]:
    """j when c == 2**j exactly (c a positive finite double), else None."""
    if not (c > 0.0 and math.isfinite(c)):
        return None
    m, e = math.frexp(c)
    return e - 1 if m == 0.5 else None


def _is_f32(s: float) -> bool:
    try:
        return f32(s) == s
    except OverflowError:
        return False


def grid_exponent(s: float, alpha: float) -> Optional[int]:
    """j when s is a positive finite f32 on the SPF grid (score_coeff32(s,
    alpha) == 2**j exactly), else None. No range check (see arm_check)."""
    s = float(s)
    if not (s > 0.0 and math.isfinite(s) and _is_f32(s)):
        return None
    try:
        c = score_coeff32(s, alpha)
    except OverflowError:
        return None
    return pow2_exponent(c)


def grid_scale(j: int, alpha: float) -> Optional[float]:
    """The smallest f32 s with score_coeff32(s, alpha) == 2**j, or None.

    score_coeff32 is monotone non-decreasing in s > 0, so the preimage of 2**j
    is a run of consecutive f32 values; it sits within a few ulps of the real
    root sqrt(2**j / (A*L)), and the scan covers +-_SCAN_ULPS around it."""
    target = math.ldexp(1.0, j)
    denom = f32(alpha) * f32(LOG2E)
    try:
        b0 = bits32(f32(math.sqrt(target / denom)))
    except (OverflowError, ValueError, ZeroDivisionError):
        return None
    best = None
    for d in range(-_SCAN_ULPS, _SCAN_ULPS + 1):
        b = b0 + d
        if not 0 < b < _F32_MAX_BITS:
            continue
        s = from_bits32(b)
        try:
            c = score_coeff32(s, alpha)
        except OverflowError:
            continue
        if c == target and (best is None or s < best):
            best = s
    return best


def snap_scale(s_cal: float, alpha: float) -> Tuple[float, int]:
    """(s_snap, j): the smallest grid point >= s_cal (an f32 > 0), on the grid
    of exponent j. s_cal already on the grid is returned unchanged.

    By monotonicity, s_j >= s  <=>  score_coeff32(s) <= 2**j, so the first
    candidate is j = ceil(log2(score_coeff32(s_cal))); a j without an f32
    preimage (none in [J_MIN, J_MAX] for alpha = 1/sqrt(128)) is skipped."""
    s = float(s_cal)
    if not (s > 0.0 and math.isfinite(s) and _is_f32(s)):
        raise ValueError(f"SPF snap needs a positive finite f32 scale, got {s_cal!r}")
    c = score_coeff32(s, alpha)
    if not c > 0.0:
        raise ValueError(f"SPF snap: score coefficient of {s!r} underflows to 0")
    m, e = math.frexp(c)
    j = e - 1 if m == 0.5 else e
    for jj in range(j, j + 4):
        sj = grid_scale(jj, alpha)
        if sj is not None and sj >= s:
            return sj, jj
    raise ValueError(f"SPF snap: no grid point at or above {s!r} (j from {j})")


def spf_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """$DLRM_HSTU_FP8_LAB_FWD_SPF (read per call): unset or "0" -> "off",
    "snap" -> "snap" (snapped control), "1" -> "fold". Anything else raises
    ValueError (fail closed: a typo must not silently run another arm)."""
    raw = (os.environ if environ is None else environ).get(KNOB, "")
    if raw in ("", "0"):
        return "off"
    if raw == "snap":
        return "snap"
    if raw == "1":
        return "fold"
    raise ValueError(f"{KNOB}={raw!r}: use unset, '0', 'snap' or '1'")


def arm_check(s: float, alpha: float) -> int:
    """j for a scale the forward may run under snap/fold; raises RuntimeError
    when s is off the grid (e.g. a scale published by an unsnapped run and
    restored from its checkpoint) or j is outside [J_MIN, J_MAX]."""
    j = grid_exponent(s, alpha)
    if j is None:
        raise RuntimeError(
            f"{KNOB}: qkv_scale={float(s)!r} is not on the SPF grid (its f32 "
            f"score coefficient is not a power of two). The scale is snapped "
            f"when calibration publishes it; a scale restored from a run "
            f"without the knob is not. Start the arm from scratch (or from a "
            f"checkpoint of the same arm), or unset the knob.")
    if not J_MIN <= j <= J_MAX:
        raise RuntimeError(
            f"{KNOB}: qkv_scale={float(s)!r} gives j={j}, outside the arming "
            f"range [{J_MIN}, {J_MAX}] (normal-f32 accumulator condition)")
    return j


def describe(s_cal: float, s_snap: float, j: int) -> str:
    return (f"s_cal={s_cal!r} -> s_snap={s_snap!r} (0x{bits32(s_snap):08x}) "
            f"j={j} ratio={s_snap / s_cal:.9f} e8m0_byte={127 + j}")


# ---------------------------------------------------------------- selftest
def _selftest(verbose: bool = True) -> int:  # noqa: C901
    import random

    fails = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    # 1. knob parsing
    for v in (None, "", "0"):
        env = {} if v is None else {KNOB: v}
        check(spf_mode(env) == "off", f"mode off for {v!r}")
    check(spf_mode({KNOB: "snap"}) == "snap", "mode snap")
    check(spf_mode({KNOB: "1"}) == "fold", "mode fold")
    for bad in ("2", "true", "on", " 1", "1 ", "SNAP", "Snap", "yes", "fold", "-1"):
        try:
            spf_mode({KNOB: bad})
            fails.append(f"no raise for {bad!r}")
        except ValueError:
            pass

    # 2. exhaustive grid structure for the production alpha
    alpha = 1.0 / (128 ** 0.5)
    sq2 = math.sqrt(2.0)
    max_ratio = 0.0
    grid = {}
    for j in range(J_MIN - 1, J_MAX + 2):
        sj = grid_scale(j, alpha)
        check(sj is not None, f"no grid point for j={j}")
        if sj is None:
            continue
        grid[j] = sj
        # uniqueness: neighbours are off the grid
        b = bits32(sj)
        check(score_coeff32(from_bits32(b - 1), alpha) != math.ldexp(1.0, j),
              f"j={j}: two preimages (below)")
        check(score_coeff32(from_bits32(b + 1), alpha) != math.ldexp(1.0, j),
              f"j={j}: two preimages (above)")
        check(score_coeff32(sj, alpha) == math.ldexp(1.0, j), f"fwd chain j={j}")
        check(score_coeff32_bwd(sj, alpha) == math.ldexp(1.0, j), f"bwd chain j={j}")
        check(grid_exponent(sj, alpha) == j, f"grid_exponent j={j}")
        check((b & 0x7FFFFF) in (0x33393E, 0x7D75E0), f"mantissa j={j}: {b:08x}")
    for j in range(J_MIN, J_MAX + 2):
        if j in grid and j - 1 in grid:
            r = grid[j] / from_bits32(bits32(grid[j - 1]) + 1)
            max_ratio = max(max_ratio, r)
            check(r < sq2, f"ratio bound j={j}: {r!r}")
        if j in grid and j + 2 in grid:
            check(grid[j + 2] == 2.0 * grid[j], f"s(j+2) == 2 s(j) at j={j}")

    # 3. snap properties on seeded random f32 scales (log-uniform 2^-40..2^8)
    rng = random.Random(20261013)
    n = 0
    worst = 0.0
    for _ in range(20000):
        s = f32(math.ldexp(1.0, 0) * 2.0 ** rng.uniform(-40.0, 8.0))
        sn, j = snap_scale(s, alpha)
        n += 1
        check(sn >= s, f"snap below input {s!r}")
        check(sn / s < sq2, f"snap ratio {s!r} -> {sn!r}")
        check(grid_exponent(sn, alpha) == j, f"snap off grid {s!r}")
        check(snap_scale(sn, alpha) == (sn, j), f"snap not idempotent {s!r}")
        prev = grid.get(j - 1) or grid_scale(j - 1, alpha)
        check(prev is None or prev < s, f"snap not minimal {s!r}")
        worst = max(worst, sn / s)
    # 4. boundaries
    for j in (-20, -15, -16, 0, 7):
        sj = grid[j]
        check(snap_scale(sj, alpha) == (sj, j), f"on-grid unchanged j={j}")
        up = from_bits32(bits32(sj) + 1)
        check(snap_scale(up, alpha) == (grid[j + 1], j + 1), f"nextup j={j}")
        dn = from_bits32(bits32(sj) - 1)
        check(snap_scale(dn, alpha) == (sj, j), f"nextdown j={j}")
    # 5. arming range
    check(arm_check(grid[-15], alpha) == -15, "arm_check on grid")
    for bad_s in (0.0146325, f32(0.0146325), 2.0 ** -6):
        try:
            arm_check(bad_s, alpha)
            fails.append(f"arm_check accepted {bad_s!r}")
        except RuntimeError:
            pass
    # 6. other alphas: snap invariants only (existence is per alpha)
    for a2 in (1.0 / 8.0, 0.1, 1.0 / (64 ** 0.5), 1.0):
        for _ in range(500):
            s = f32(2.0 ** rng.uniform(-30.0, 4.0))
            sn, j = snap_scale(s, a2)
            check(sn >= s and grid_exponent(sn, a2) == j, f"alpha={a2} s={s!r}")
    if verbose:
        print(f"[fp8_qkv_spf selftest] grid j in [{J_MIN - 1}, {J_MAX + 1}]: "
              f"{len(grid)} points; max s_j/nextup(s_(j-1)) = {max_ratio!r} "
              f"(sqrt2 = {sq2!r}); random snaps {n}, worst ratio {worst!r}; "
              f"s_-15 = {grid[-15]!r} (0x{bits32(grid[-15]):08x})")
        print("[fp8_qkv_spf selftest] " + ("PASS" if not fails else
              f"FAIL ({len(fails)}): " + "; ".join(fails[:10])))
    return 0 if not fails else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
