# pyre-strict
"""SPF backward fold, host side: DLRM_HSTU_FP8_LAB_BWD_SPF (R15 iter75 tscv2).

LAB knob, read per call, code default OFF, deliberately NOT bound in gin. It is
R15's share of the SPF contract v1.1 (x4c19 probes/iter71/spf/CONTRACT.md,
sections 1, 2, 5, 6): the backward half of the score-scale power-of-two fold,
i.e. bwd_dkdv's VALU token "tsc" (R15 iter74 f2, ops/flydsl/hstu_bwd_dkdv.py),
layered on x4c19's forward knob DLRM_HSTU_FP8_LAB_FWD_SPF (ops/fp8_qkv_spf.py),
whose calibration snap is what puts qkv_scale on the SPF grid.

  unset / "0"  off: bwd_dkdv compiles exactly as without this module. A "tsc"
               token in DLRM_HSTU_FP8_BWD_VALU (any case or padding), or an
               exact "tsc" in config["valu"] from any other path, is stripped
               with one log line per process; any other spelling there fails
               compile_hstu_bwd_dkdv's valu assert. A stray token never arms.
  "1"          the fold. Needs DLRM_HSTU_FP8_LAB_FWD_SPF in {"snap", "1"}
               (ValueError at parse otherwise). On every backward call whose
               q AND k are e4m3, before compile and launch:
                 * arm_check_bwd(qkv_scale, alpha): fp8_qkv_spf.arm_check (on
                   the SPF grid, j in [J_MIN, J_MAX]), then j in [BWD_J_MIN,
                   BWD_J_MAX]; RuntimeError otherwise. This is contract
                   section 6: a scale restored from a checkpoint written
                   without the knob is off the grid and the arm refuses it;
                 * body_reason(): the vf4 + msat + msplit=1 + k128 body the
                   token is built on (the kernel's own assert); ValueError
                   otherwise, so the arm never runs unfolded under its label;
                 * "tsc" joins the VALU tokens (config["valu"], the compile
                   key; j is a runtime value, one build serves every j) and
                   one ARMED line per process names j and the E8M0 byte.
               bwd_dkdv's call() guard (tsc_exponent, the same range) stays
               as the second check. Calls with bf16 q or k (the calibration
               steps) are not armed and do not raise.
  other        ValueError.

Accepted range [BWD_J_MIN, BWD_J_MAX] = [-40, -7] (E8M0 bytes 87..120):
  * upper -7: tsc needs vf4, and vf4 needs vf3p's guard 0 < s <= 0.25. The
    grid has s_-7 = 0.24752 <= 0.25 < s_-6 = 0.35005 (s_(j+2) = 2 s_j), so a
    j >= -6 never reaches the tsc body. Refusing it in the arm check names the
    real reason instead of a later "vf4 declined" body error.
  * lower -40: the extent of R15 iter74's exactness evidence (c_t == -2**j for
    all 39 j in [-40, -2]; the tsc chain bitwise equal to vf4 in an ISA-level
    emulation sampled at every j in [-40, -2] and exhaustive over all 2**32
    score values at j = -40, -15, -14, -7). x4c19's a'/b' check covers
    [-24, -6], a subset. The plausible in-model j is -15 (s about 0.015);
    j = -40 is s = 2.7e-6.
The forward accepts [J_MIN, J_MAX] = [-108, 103]. A forward-armable scale
outside [-40, -7] is refused by this arm check (RuntimeError).
hstu_bwd_dkdv.TSC_K_MIN/TSC_K_MAX are literals and MUST equal BWD_J_MIN and
BWD_J_MAX. The launcher refuses to arm when they differ, and a unit test checks
them.

Stdlib only, Python 3.9 (node-side); holds no mutable state.
`python3 fp8_qkv_spf_bwd.py --selftest` runs the checks.
"""

import math
import os
import struct
import sys
from typing import Any, Iterable, Mapping, Optional

if __package__ in (None, ""):  # run as a script: put the tree root on sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))

from generative_recommenders.ops import fp8_qkv_spf as fwd  # noqa: E402

BWD_KNOB = "DLRM_HSTU_FP8_LAB_BWD_SPF"
TOKEN = "tsc"
BWD_J_MIN = -40
BWD_J_MAX = -7
# hstu_flydsl_mha._VF3P_QKV_SCALE_MAX: vf3p (and so vf4, and so tsc) needs
# 0 < qkv_scale <= this bound.
VF3P_QKV_SCALE_MAX = 0.25


def bwd_spf_mode(environ: Optional[Mapping[str, str]] = None) -> str:
    """$DLRM_HSTU_FP8_LAB_BWD_SPF (read per call): unset or "0" -> "off",
    "1" -> "fold". "1" needs DLRM_HSTU_FP8_LAB_FWD_SPF in {"snap", "1"}: the
    forward's calibration snap is what makes the scale foldable. Any other
    value, or "1" without that forward knob, raises ValueError (an invalid
    forward value raises fp8_qkv_spf's ValueError)."""
    env = os.environ if environ is None else environ
    raw = env.get(BWD_KNOB, "")
    if raw in ("", "0"):
        return "off"
    if raw != "1":
        raise ValueError(f"{BWD_KNOB}={raw!r}: use unset, '0' or '1'")
    fmode = fwd.spf_mode(env)
    if fmode not in ("snap", "fold"):
        raise ValueError(
            f"{BWD_KNOB}=1 needs {fwd.KNOB} in {{'snap', '1'}} (the forward "
            f"calibration snap puts qkv_scale on the SPF grid); it is "
            f"{env.get(fwd.KNOB, '<unset>')!r}")
    return "fold"


def arm_check_bwd(s: float, alpha: float) -> int:
    """j for a scale the backward fold may run with: fp8_qkv_spf.arm_check
    (RuntimeError when s is off the SPF grid or j is outside the forward's
    [J_MIN, J_MAX]; re-raised naming this knob and the backward range, the
    original chained), then RuntimeError unless BWD_J_MIN <= j <= BWD_J_MAX."""
    try:
        j = fwd.arm_check(s, alpha)
    except RuntimeError as e:
        raise RuntimeError(
            f"{BWD_KNOB}=1: {e} (backward range [{BWD_J_MIN}, {BWD_J_MAX}]; "
            f"unset {BWD_KNOB} for this arm)") from e
    if not BWD_J_MIN <= j <= BWD_J_MAX:
        raise RuntimeError(
            f"{BWD_KNOB}: qkv_scale={float(s)!r} is on the SPF grid with "
            f"j={j}, outside the backward fold's range [{BWD_J_MIN}, "
            f"{BWD_J_MAX}] (j <= -7: vf3p/vf4 guard qkv_scale <= "
            f"{VF3P_QKV_SCALE_MAX}; j >= -40: the verified extent). Unset the "
            f"knob for this arm.")
    return j


def body_reason(config: Mapping[str, Any], fp8_mfma: str,
                tokens: Iterable[str]) -> Optional[str]:
    """None when compile_hstu_bwd_dkdv's kwargs (`config` plus fp8_mfma and
    the VALU `tokens`) are the body the tsc token is built on, else why not.
    The same four conditions as the kernel's assert: vf4 and msat in the VALU
    tokens, msplit == 1 and fp8_mfma == "k128"."""
    vs = set(tokens)
    missing = sorted({"vf4", "msat"} - vs)
    if missing:
        return "VALU token(s) %s not in the effective set (valu=%s)" % (
            ",".join(missing), ",".join(sorted(vs)) or "<none>")
    if int(config.get("msplit", 0) or 0) != 1:
        return "needs msplit=1 (msplit=%r)" % (config.get("msplit"),)
    if fp8_mfma != "k128":
        return "needs the k128 e4m3 QK MFMA (fp8_mfma=%r)" % (fp8_mfma,)
    return None


def warmup_scale(s: float, alpha: float,
                 environ: Optional[Mapping[str, str]] = None) -> float:
    """The runtime scale fp8_warmup's FP8 backward items pass: `s` unchanged
    with the knob off, else snapped onto the SPF grid (fp8_qkv_spf.snap_scale:
    the smallest grid point >= s, by f32 table compare), so the warmup passes
    the per-call grid check and compiles the arm's own build."""
    if bwd_spf_mode(environ) == "off":
        return s
    return fwd.snap_scale(float(s), float(alpha))[0]


def armed_line(j: int, s: float, fwd_value: str, valu: str) -> str:
    """The once-per-process ARMED line (the marker of the fold arm)."""
    b = struct.unpack("<I", struct.pack("<f", float(s)))[0]
    return (f"{BWD_KNOB}=1 ARMED: bwd_dkdv VALU token {TOKEN} (S = Q K^T "
            f"MFMA E8M0 A-scale byte {127 + j} = 2**{j}, B byte 127; t = "
            f"qk*c_t folded) j={j} e8m0_byte={127 + j}; qkv_scale="
            f"{float(s)!r} (0x{b:08x}) on the SPF grid, j in [{BWD_J_MIN}, "
            f"{BWD_J_MAX}]; {fwd.KNOB}={fwd_value}; valu={valu} (declared "
            f"precision class: the snapped control's)")


def strip_line(source: str, value: str, knob_value: str) -> str:
    """The once-per-process line for a stripped stray tsc token."""
    return (f"{BWD_KNOB}: VALU token {TOKEN} in {source}={value!r} stripped "
            f"(only {BWD_KNOB}=1 adds it, after its per-call grid check; "
            f"{BWD_KNOB}={knob_value or '<unset>'})")


# ---------------------------------------------------------------- selftest
def _f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def _selftest(verbose: bool = True) -> int:  # noqa: C901
    import random

    fails = []

    def check(ok: bool, what: str) -> None:
        if not ok:
            fails.append(what)

    def raises(exc, fn, *a) -> bool:  # pyre-ignore[2]
        try:
            fn(*a)
        except exc:
            return True
        except Exception:  # noqa: BLE001 - wrong type counts as a failure
            return False
        return False

    # 1. knob parsing, every value class
    fk = fwd.KNOB
    for v in (None, "", "0"):
        for fv in (None, "", "0", "snap", "1"):
            env = {} if v is None else {BWD_KNOB: v}
            if fv is not None:
                env[fk] = fv
            check(bwd_spf_mode(env) == "off", f"off for {v!r} (fwd {fv!r})")
    for fv in ("snap", "1"):
        check(bwd_spf_mode({BWD_KNOB: "1", fk: fv}) == "fold", f"fold fwd {fv}")
    for fv in (None, "", "0"):
        env = {BWD_KNOB: "1"} if fv is None else {BWD_KNOB: "1", fk: fv}
        check(raises(ValueError, bwd_spf_mode, env), f"1 without fwd {fv!r}")
    check(raises(ValueError, bwd_spf_mode, {BWD_KNOB: "1", fk: "on"}),
          "1 with an invalid fwd value")
    for bad in ("2", "true", "on", " 1", "1 ", "snap", "fold", "yes", "-1",
                "tsc", "01", "1.0", "00", " "):
        for fv in (None, "snap", "1"):
            env = {BWD_KNOB: bad} if fv is None else {BWD_KNOB: bad, fk: fv}
            check(raises(ValueError, bwd_spf_mode, env), f"no raise {bad!r}")

    # 2. range constants and the vf3p bound
    alpha = 1.0 / (128 ** 0.5)
    check(fwd.J_MIN <= BWD_J_MIN <= BWD_J_MAX <= fwd.J_MAX, "range nesting")
    s7, s6 = fwd.grid_scale(-7, alpha), fwd.grid_scale(-6, alpha)
    check(s7 is not None and s6 is not None and s7 <= VF3P_QKV_SCALE_MAX < s6,
          f"vf3p bound: s_-7={s7!r} <= 0.25 < s_-6={s6!r}")
    check(BWD_J_MAX == -7, "BWD_J_MAX is the largest j under the vf3p guard")

    # 3. arm_check_bwd over every grid j and its off-grid neighbours
    n_ok = 0
    for j in range(fwd.J_MIN - 2, fwd.J_MAX + 3):
        sj = fwd.grid_scale(j, alpha)
        if sj is None:
            continue
        inside = BWD_J_MIN <= j <= BWD_J_MAX
        if inside:
            check(arm_check_bwd(sj, alpha) == j, f"arm j={j}")
            n_ok += 1
        else:
            check(raises(RuntimeError, arm_check_bwd, sj, alpha),
                  f"j={j} outside the range armed")
        for d in (-2, -1, 1, 2):
            check(raises(RuntimeError, arm_check_bwd,
                         _from_bits(_bits(sj) + d), alpha),
                  f"off-grid neighbour {d} of j={j} armed")
    check(n_ok == BWD_J_MAX - BWD_J_MIN + 1, f"armable j count {n_ok}")
    for bad in (0.0146325, _f32(0.0146325), 0.0173, 2.0 ** -6, 0.0, -0.01547,
                float("nan"), float("inf")):
        check(raises(RuntimeError, arm_check_bwd, bad, alpha), f"armed {bad!r}")
    rng = random.Random(20261014)
    for _ in range(5000):
        s = _f32(2.0 ** rng.uniform(-24.0, -1.0))
        if fwd.grid_exponent(s, alpha) is None:
            check(raises(RuntimeError, arm_check_bwd, s, alpha), f"armed {s!r}")

    # 4. the fold's host-side constants are exact pow2 rescalings, normal,
    #    for every armable j (the kernel's own f32 chain, hstu_bwd_dkdv vf4 +
    #    tsc: c_t, cx, N' = 1/f32(1/N), ncxn = -(cx*N'), sas, 2^j, 2^-j)
    tiny = 2.0 ** -126  # smallest normal f32
    worst_cop = 0
    for j in range(BWD_J_MIN, BWD_J_MAX + 1):
        s = fwd.grid_scale(j, alpha)
        a32, l32 = _f32(alpha), _f32(fwd.LOG2E)
        sas = _f32(a32 * _f32(s * s))
        ct = _f32(sas * -l32)
        check(ct == -(2.0 ** j), f"c_t != -2**{j}")
        byte = (_bits(ct) >> 23) & 0xFF
        check(byte == 127 + j and 1 <= byte <= 253, f"byte j={j}")
        p2k, p2mk = _from_bits(byte << 23), _from_bits((254 - byte) << 23)
        check(p2k == 2.0 ** j and p2mk == 2.0 ** -j, f"pow2 j={j}")
        cx = _f32(a32 * s)
        for n in (2, 128, 2000, 2121, 2992, 3000, 3058, 4096, 65536):
            nf = _f32(1.0 / _f32(1.0 / n))
            ncxn = -_f32(cx * nf)
            ncxn2 = _f32(ncxn * p2mk)
            check(ncxn2 == ncxn * 2.0 ** -j and abs(ncxn2) >= tiny,
                  f"ncxn'' j={j} N={n}")
            sas2 = _f32(sas * p2mk)
            check(sas2 == sas * 2.0 ** -j and sas2 >= tiny, f"sas'' j={j}")
            # cast operand 2^j / (mu2_ssc * sc), sc = 2^round(log2 cx),
            # mu2_ssc = 2^(ceil(log2 N) + K), K <= 9 (launcher assert)
            ka = round(math.log2(cx))
            c = int(math.ceil(math.log2(max(n, 2))))
            for k_ssc in (-2, 3, 9):
                e = j - (c + k_ssc) - ka
                worst_cop = min(worst_cop, e)
                check(-126 <= e <= 127, f"cast operand 2^{e} j={j} N={n}")
        # tau = 2^j * qk: e4m3 dot products are 0 or 2^-18 <= |qk| < 2^24.62
        check(j - 18 >= -126 and j + 24.62 < 128, f"tau normal j={j}")
    if verbose:
        print(f"[fp8_qkv_spf_bwd selftest] armable j in [{BWD_J_MIN}, "
              f"{BWD_J_MAX}] ({n_ok} values, E8M0 bytes {127 + BWD_J_MIN}.."
              f"{127 + BWD_J_MAX}); s_-7 = {s7!r}, s_-6 = {s6!r}; smallest "
              f"cast-operand exponent {worst_cop}")
        print("[fp8_qkv_spf_bwd selftest] " + ("PASS" if not fails else
              f"FAIL ({len(fails)}): " + "; ".join(fails[:10])))
    return 0 if not fails else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv[1:]:
        sys.exit(_selftest())
    print(__doc__)
