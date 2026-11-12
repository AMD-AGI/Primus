# pyre-strict
"""IX-FP8F: the Indexed-X L1 forward (preprocessor pair 0, item + cat members) on FP8 e4m3 operands, TRAIN ONLY,
with a CONSISTENT backward (rccl R-C3 FP8F, ship14 precision layer P4; productionized from the rccl iteration-86
lab kit probes/iter86/fp8f/ixf8_kern.py, lab GO job 69324: f8 - ix = -0.77..-0.89 ms per module step early incl.
prep, _ecix_fwd8 127 VGPR / 0 spills).

DEFAULT OFF. Gate: DLRM_HSTU_FP8_EC_INDEXED_L1_FP8=1 (1/true/on/yes), read once per process; unset / "0" = off (kill
switch). It only acts while Indexed-X itself is armed (triton_ec_indexed_l1 = K: DLRM_HSTU_FP8_EC_INDEXED_L1=1,
ROCm gfx950). Backward mode: DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD = consistent (default) | ste; any other value
raises ValueError at first use. The caller (ops/ec_indexed_l1.fwd, train=True only) owns the decision and the
fallback: Unsupported -> the bf16 Indexed-X forward for that call; any other exception -> latch_off() -> bf16
Indexed-X for the rest of the run. Eval never comes here (FP8 forward is train-only).

RECIPE (per call of fwd8; emb_c = the EC's bf16 unique-row table U [U, 512], shared by both members):
  U prep  _ixf8_uprep, ONE launch, reads the bf16 emb_c:
            amax_r = max |x| over the FINITE entries of row r
            E_r    = ceil(log2(amax_r / 448)) EXACTLY from the fp32 bits of amax_r (448 = 1.75 * 2^8:
                     E = ex - 8 + (mantissa > 0.75)); a row whose amax is 0 or a bf16 subnormal (all-zero,
                     all-non-finite: every code is +-0 at any E) has E = 0 (actckpt's zero-row convention);
                     clamped to [E_MIN, E_MAX] = [-50, 50] (|E_art - E_alb| <= 100 keeps the cat rescale exact and
                     finite: |acc| <= 512 * 448^2 < 2^27, 2^27 * 2^100 < 2^128)
            U8     = e4m3fn RNE(x * 2^-E_r) (c10 fp8e4m3fn_from_fp32_value bit for bit; |q| <= 448 by construction;
                     NaN / inf elements -> NaN code 0x7F | sign)
            SX_r   = 2^E_r, SXI_r = 2^-E_r (exact fp32); NaN (loud: every h row gathering it is NaN, and its Ud row is
                     NaN) when the row holds a non-finite element or amax_r > 448 * 2^E_MAX
            Ud     = bf16(decode(U8) * SX_r) (consistent mode only): EXACT (<= 4 significant bits, 2^-59..2^59)
  W prep  _ixf8_w1q, ONE launch, 16 programs: bwd_fp8.quantize_weight's arithmetic AS IT RUNS ON THE GPU on the bf16
            w_c (per-tensor CURRENT amax with NaN -> scale NaN, clamp_min 1e-30, SW = amax * fl32(1/448) (ATen divides
            a CUDA tensor by a CPU scalar as a multiply by its fp32 reciprocal), inv = RN(1 / SW), q = clamp(w * inv,
            +-448), e4m3 RNE). Every program computes the whole-tensor amax redundantly (the max is exact and
            order-free), program 0 stores SW, each program casts its own 16 rows.
            Wd = bf16_rne(fp32(decode(W8)) * SW) (consistent mode only).
  forward _ecix_fwd8 (rccl's lab kernel, ROWSCALE=1 path, tile f128x256x64w8s2k32 = the lab's f128x256x128 re-blocked
            to BK 64 for Triton 3.9 (TILE below), v_mfma_f32_32x32x64_f8f6f4):
            h[m, n] = bf16_rne( acc * (SX[last] * SW) + b1[n] ), acc = fp32 sum over the member's halves of
            U8[idx] . W8[n, half]; cat: after the artist half acc *= SX[art] * SXI[alb] (exact pow2), then the album
            half accumulates into the same acc. Device-pointer scales (no host sync).
  returns (h, table, w_bwd): consistent -> (h, Ud, Wd): the head saves Ud (the unchanged K wgrad gathers it) and Wd
            (dX = dh @ Wd and E-X2b's rec.w1), i.e. the gradient of the quantized forward; ste -> (h, emb_c, w_c).
  The two members of one step pass the SAME emb_c object: its U prep runs once and is reused by the other member
  (identity + version-counter + stream checked; dropped after both members used it). That is not a computation
  reduction: it is one cast of one tensor, like the EC's own emb_c cast.

RECIPE LABEL: the U scale is per-row POWER-OF-TWO, actckpt's lab arm 'd' -- NOT arm 'dE' (a per-row fp32 scale
amax/448). DECLARED ENGINEERING CHOICE (user decision 2026-10-12), not derived from training-data analysis: a pow2
scale keeps the dequantized table exact (Ud = bf16(U8 * 2^e) is exact for e in [-50, 50]), so the consistent-mode
backward is the exact gradient of the quantized forward, and the cat mid-K rescale (SX[art] * SXI[alb]) is exact.
Synthetic-only re-derivation (rccl iter99, probes/iter99/fp8f_synth; 480 cases: init-law uniform, Gaussian with
log-uniform row std, Student-t nu=3 + 1% x100 outlier rows and tiny/zero rows, uniform and Zipf ids, item and cat):
the h1 SNR median of 'd' is within 0.6 dB of 'dE' on every family. Under the bounded-cost rule, R1 fails (a
per-tensor scale is never >= 6 dB worse than per-row; it matched dE on these families) and R2 fails only on worst
cases (h1 up to 2.1 dB and consistent wgrad up to 4.4 dB below dE with x100 outlier rows; consistent wgrad 1.8 dB
below dE on the cat tiny/zero-row worst case). dE and per-tensor remain ship16 candidates.

MLPerf closed division: no computation reduction. Every token's output row is computed from its own gathered row(s);
the scales are per unique row (U) and per tensor (W1); nothing is summed or reused across tokens, samples or steps.

SPECIALIZATION (as K): M / n_rows are do_not_specialize, IDX do_not_specialize_on_alignment; every other pointer must
be 16-byte aligned and its storage within 2 GiB ('S' class; load-bearing for the fwd8 register budget). fwd8 checks
every input before the first launch and raises Unsupported otherwise; the outputs it allocates are in the class.
prebuild() compiles every kernel at that class for both members and both backward modes and checks them exactly.

Test hook: ALLOW_CPU_INTERPRETER = True plus TRITON_INTERPRET=1 lets the wrappers take CPU tensors (host self-test).
"""

import logging
import os
import weakref
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from generative_recommenders.ops.triton import triton_ec_indexed_l1 as K

try:  # CPU-only test environments may lack Triton; only the kernels need it.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

logger: logging.Logger = logging.getLogger(__name__)

ENV_VAR: str = "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8"
BWD_ENV_VAR: str = "DLRM_HSTU_FP8_EC_INDEXED_L1_FP8_BWD"
BWD_MODES: Tuple[str, ...] = ("consistent", "ste")
TAG: str = "[ec-indexed-l1-fp8]"
ARMED_PREFIX: str = "armed OK: " + TAG
HID: int = K.HID  # 256
WIDTH: int = K.WIDTH  # 512
E4M3_MAX: float = 448.0
INV448: float = 1.0 / 448.0  # rounded to fp32 where used (Triton fp32 constant / torch fp32 tensor)
E_MIN, E_MAX = -50, 50
AMAX_FLOOR: float = 1.0e-30  # quantize_weight's clamp_min
# TILE: the lab GO tile f128x256x128w8s2k32 (iter82 t1b) re-blocked along K to BK 64 for Triton 3.9 (ROCm 10.2 port,
# ec-kernels lane). Same instruction (one K-64 v_mfma_f32_32x32x64_f8f6f4 step per 64 K), same ascending K order, so
# the same bits. Why: at BK 128 Triton 3.9's LLVM schedules every global prefetch of the step ahead of its MFMAs (3.6
# issued the W8 loads after them): 158 / 164 VGPRs > FWD8_MAX_VGPR. At BK 64 the step's operands are half as large,
# but the pipelined loop then has 7 iterations, which LLVM unrolls by 2 with an exit between the copies (two register
# sets for the 64-VGPR accumulator: 170 / 174 VGPRs); the K loop's trip count is therefore opaque to LLVM (_ecix_fwd8:
# D // BK + (M < 0), always D // BK since M >= 0): no unroll, 116 / 126 VGPRs, 0 spills, occupancy 4, LDS 24 KiB.
FWD8_BM, FWD8_BN, FWD8_BK = 128, 256, 64
FWD8_TILE: str = "f128x256x64w8s2k32"
FWD8_KW: Dict[str, int] = dict(num_warps=8, num_stages=2, waves_per_eu=0, matrix_instr_nonkdim=32)
FWD8_MAX_VGPR: int = 128  # compile guard (occupancy 4 needs <= 128); lab: 127, 0 spills
UPREP_ROWS: int = 4  # rows per _ixf8_uprep program
UPREP_KW: Dict[str, int] = dict(num_warps=4)
W1Q_BLOCK_R: int = 16  # rows cast per _ixf8_w1q program: grid 256 // 16 = 16 programs
W1Q_AMAX_ELEMS: int = 8192  # amax tile [AMAX_R, C], AMAX_R = 8192 // C (16 item, 8 cat)
W1Q_C: Tuple[int, ...] = (WIDTH, 2 * WIDTH)
W1Q_KW: Dict[str, int] = dict(num_warps=8)
MAX_PTR_RANGE: int = K.MAX_PTR_RANGE
MAX_ROWS: int = K.MAX_ROWS  # h [M, 256] bf16 in the 'S' class
MAX_TABLE_ROWS: int = K.MAX_TABLE_ROWS  # emb_c / Ud [U, 512] bf16 in the 'S' class (U8 is half of that)
KERNEL_NAMES: Tuple[str, ...] = ("_ixf8_uprep", "_ixf8_w1q", "_ecix_fwd8")
assert WIDTH % FWD8_BK == 0 and HID % FWD8_BN == 0 and HID % W1Q_BLOCK_R == 0

ALLOW_CPU_INTERPRETER: bool = False  # test hook only (needs TRITON_INTERPRET=1 as well)

_KNOB: Optional[bool] = None
_BWD: Optional[str] = None
_BROKEN: Optional[str] = None
_IGNORED_SAID: bool = False
_ARMED: Set[str] = set()
# name -> Triton CompiledKernel of the last launch (tests / diagnostics: n_regs, n_spills, asm).
LAST_KERNELS: Dict[str, Any] = {}
STATS: Dict[str, int] = {"uprep_launch": 0, "uprep_reuse": 0, "uprep_dropped_on_free": 0, "w1q_launch": 0,
                         "fwd8_launch": 0}


class Unsupported(ValueError):
    """The inputs are outside the class the FP8F kernels were validated and prebuilt for. Raised BEFORE any launch;
    the caller runs the bf16 Indexed-X forward for that call."""


def _say(msg: str) -> None:
    K._say(msg)  # arm_print_all_ranks: every rank, " [rank N]"


def _interpreting() -> bool:
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


# ------------------------------------------------------------------------------------------------------------ gate
def knob_on() -> bool:
    """DLRM_HSTU_FP8_EC_INDEXED_L1_FP8 in (1, true, on, yes), read once per process."""
    global _KNOB
    if _KNOB is None:
        _KNOB = os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "on", "yes")
    return bool(_KNOB)


def bwd_mode() -> str:
    """'consistent' (default; also for unset / empty) or 'ste'. Any other value raises ValueError (every call until
    fixed; the caller latches FP8F off, loudly). Read once per process."""
    global _BWD
    if _BWD is None:
        raw = os.environ.get(BWD_ENV_VAR, "")
        v = raw.strip().lower() or "consistent"
        if v not in BWD_MODES:
            raise ValueError(f"{BWD_ENV_VAR}={raw!r}: expected one of {BWD_MODES}")
        _BWD = v
    return _BWD


def fp8f_enabled() -> bool:
    """knob on, Indexed-X armed (K.enabled()) and not latched off. Knob off: returns before touching anything else.
    Knob on while Indexed-X was never armed: one IGNORED line (every rank) and False."""
    global _IGNORED_SAID
    if not knob_on():
        return False
    if _BROKEN is not None:
        return False
    if not K.enabled():
        if K._BROKEN is None and not _IGNORED_SAID:  # never armed (a mid-run IX latch printed its own line)
            _IGNORED_SAID = True
            _say(f"{TAG} {ENV_VAR}=1 IGNORED (NOT armed): Indexed-X ({K.ENV_VAR}) is not armed on this process; "
                 "the shipped path runs")
        return False
    return True


def latch_off(site: str, exc: BaseException) -> None:
    """FP8F off for the rest of the process (Indexed-X itself stays on: the caller runs the bf16 forward)."""
    global _BROKEN, _UCACHE
    _UCACHE = None
    if _BROKEN is None:
        _BROKEN = f"{site}: {type(exc).__name__}: {str(exc)[:300]}"
        _say(f"{TAG} FAILED CLOSED at {site}: {type(exc).__name__}: {str(exc)[:300]} -> bf16 Indexed-X for the rest "
             "of the run")


def reset_for_testing() -> None:
    global _KNOB, _BWD, _BROKEN, _IGNORED_SAID, _UCACHE
    _KNOB = _BWD = _BROKEN = None
    _IGNORED_SAID = False
    _UCACHE = None
    _ARMED.clear()
    LAST_KERNELS.clear()
    for k in STATS:
        STATS[k] = 0


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ------------------------------------------------------------------------------------------------------------ kernels
_KERNELS_ERROR: Optional[str] = "Triton is not importable" if triton is None else None
if triton is not None:
    try:

        @triton.jit
        def _e4m3_rne(q):
            """fp32 -> e4m3fn code (uint32 in [0, 255]): c10 fp8e4m3fn_from_fp32_value bit for bit. Integer RNE on
            the normal range, the 2^14 denorm-add below 2^-6, NaN / inf / |q| > 464 -> 0x7F | sign."""
            qb = q.to(tl.uint32, bitcast=True)
            sgn = (qb >> 24) & 0x80
            a = qb & 0x7FFFFFFF
            nrm = (a - 0x3C000000 + 0x7FFFF + ((a >> 20) & 1)) >> 20
            sub = (a.to(tl.float32, bitcast=True) + 16384.0).to(tl.uint32, bitcast=True) - 0x46800000
            code = tl.where(a < 0x3C800000, sub, tl.minimum(nrm, 0x7F))
            return code | sgn

        @triton.jit
        def _e4m3_decode(code):
            """e4m3fn code (uint32 in [0, 255]) -> its exact fp32 value; 0x7F / 0xFF -> NaN. Signed zeros kept."""
            s = (code & 0x80) << 24
            ef = (code >> 3) & 0xF
            mf = code & 0x7
            nrm = ((ef + 120) << 23) | (mf << 20)
            sub = (mf.to(tl.float32) * 0.001953125).to(tl.uint32, bitcast=True)  # m * 2^-9, exact
            bits = tl.where(ef == 0, sub, nrm) | s
            bits = tl.where((code & 0x7F) == 0x7F, 0x7FC00000 | s, bits)
            return bits.to(tl.float32, bitcast=True)

        @triton.jit
        def _bf16_rne_bits(v):
            """fp32 -> bf16 bits (uint32 in [0, 65535]) by integer round-to-nearest-even (c10 round_to_nearest_even;
            NaN -> 0x7FC0). Independent of the backend's own fp32 -> bf16 conversion (the interpreter truncates)."""
            b = v.to(tl.uint32, bitcast=True)
            r = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
            return tl.where((b & 0x7FFFFFFF) > 0x7F800000, 0x7FC0, r)

        @triton.jit(do_not_specialize=["n_rows"])
        def _ixf8_uprep(X, U8, SX, SXI, UD, n_rows,
                        D: tl.constexpr, ROWS: tl.constexpr, E_MIN: tl.constexpr, E_MAX: tl.constexpr,
                        WRITE_UD: tl.constexpr):
            """bf16 U (int16 view) [n_rows, D] -> U8 (uint8 view) + SX + SXI fp32 [n_rows] (+ Ud bf16 (int16 view) when
            WRITE_UD) in one pass; ROWS rows per program (the module docstring has the recipe)."""
            pid = tl.program_id(0)
            r = pid * ROWS + tl.arange(0, ROWS)
            rmask = r < n_rows
            offs = r.to(tl.int64)[:, None] * D + tl.arange(0, D)[None, :]
            m2 = rmask[:, None]
            xbits = (tl.load(X + offs, mask=m2, other=0).to(tl.uint32) & 0xFFFF) << 16
            xb = xbits.to(tl.float32, bitcast=True)  # exact bf16 value
            ax = tl.abs(xb)
            fin = ax < float("inf")  # False for inf and NaN
            amax = tl.max(tl.where(fin, ax, 0.0), axis=1)
            bad = tl.max(tl.where(fin, 0, 1), axis=1) > 0
            ab = amax.to(tl.uint32, bitcast=True)
            ef = (ab >> 23).to(tl.int32)
            e = ef - 135 + ((ab & 0x7FFFFF) > 0x600000).to(tl.int32)  # ceil(log2(amax / 448)), exact
            # zero or bf16-subnormal amax: E = 0 (every code is +-0 at any E; flush-to-zero-proof: a max that flushes
            # a subnormal amax to 0 gives the same E)
            e = tl.where(ef == 0, 0, e)
            e = tl.maximum(e, E_MIN)  # tiny rows
            bad = bad | (e > E_MAX)
            e = tl.minimum(e, E_MAX)
            p2 = ((127 + e).to(tl.uint32) << 23).to(tl.float32, bitcast=True)  # 2^e, exact
            p2i = ((127 - e).to(tl.uint32) << 23).to(tl.float32, bitcast=True)  # 2^-e, exact
            sx = tl.where(bad, float("nan"), p2)
            tl.store(SX + r, sx, mask=rmask)
            tl.store(SXI + r, tl.where(bad, float("nan"), p2i), mask=rmask)
            code = _e4m3_rne(xb * p2i[:, None])
            tl.store(U8 + offs, code.to(tl.uint8), mask=m2)
            if WRITE_UD:
                ud = _e4m3_decode(code) * sx[:, None]  # exact (or NaN on a bad row)
                ub = tl.where(ud != ud, 0x7FC0, ud.to(tl.uint32, bitcast=True) >> 16)
                tl.store(UD + offs, ub.to(tl.int16), mask=m2)

        @triton.jit
        def _ixf8_w1q(W, W8, SW, WD, R: tl.constexpr, C: tl.constexpr, BLOCK_R: tl.constexpr,
                      AMAX_R: tl.constexpr, INV448: tl.constexpr, WRITE_WD: tl.constexpr):
            """W1_8 at SW = amax * fl32(1/448) (quantize_weight on the GPU), inv = RN(1 / SW), clamp +-448, e4m3 RNE;
            + Wd = bf16_rne(decode(W8) * SW) when WRITE_WD. W [R, C] bf16 contiguous; grid (R // BLOCK_R,). R and C
            are constexprs (one compile per W1 class), so the amax loop bound is static."""
            pid = tl.program_id(0)
            cols = tl.arange(0, C)
            vmax = tl.zeros((AMAX_R, C), dtype=tl.float32)
            vnan = tl.zeros((AMAX_R, C), dtype=tl.int32)
            for r0 in range(0, R, AMAX_R):
                o = (r0 + tl.arange(0, AMAX_R))[:, None] * C + cols[None, :]
                x = tl.load(W + o).to(tl.float32)
                vmax = tl.maximum(vmax, tl.where(x == x, tl.abs(x), 0.0))
                vnan = tl.maximum(vnan, tl.where(x == x, 0, 1))
            amax = tl.maximum(tl.max(tl.max(vmax, axis=1), axis=0), 1.0e-30)
            scale = amax * INV448
            scale = tl.where(tl.max(tl.max(vnan, axis=1), axis=0) > 0, float("nan"), scale)
            inv = tl.math.div_rn(1.0, scale)
            tl.store(SW, scale, mask=pid == 0)
            o = (pid * BLOCK_R + tl.arange(0, BLOCK_R))[:, None] * C + cols[None, :]
            x = tl.load(W + o).to(tl.float32)
            q = tl.minimum(tl.maximum(x * inv, -448.0), 448.0)
            code = _e4m3_rne(q)
            tl.store(W8 + o, code.to(tl.uint8))
            if WRITE_WD:
                tl.store(WD + o, _bf16_rne_bits(_e4m3_decode(code) * scale).to(tl.int16))

        @triton.jit(do_not_specialize=["M"], do_not_specialize_on_alignment=["IDX"])
        def _ecix_fwd8(U8, SX, SXI, IDX, W8, SW, BIAS, C, M,
                       NH: tl.constexpr, D: tl.constexpr, NOUT: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
            """C[m, :] = bf16_rne(acc * (SX[last] * SW) + BIAS); A row m = U8[IDX[m*NH + h]] for K half h. The
            _ecix_fwd loop structure (h = 0..NH-1, k = 0..D-1 step BK, one fp32 accumulator); between the halves acc
            is rescaled by the exact pow2 SX[prev] * SXI[cur]. Rows >= M of the last block read the block's first
            row (valid) and are not stored. rccl iter86 ixf8_kern._ecix_fwd8 with ROWSCALE=1 (the lab GO binary)."""
            KT: tl.constexpr = NH * D
            NPN: tl.constexpr = NOUT // BN
            pid = tl.program_id(0)
            pid_m = pid // NPN
            pid_n = pid % NPN
            row0 = pid_m * BM
            offs_r = tl.arange(0, BM)
            rmask = offs_r < M - row0
            offs_rc = tl.where(rmask, offs_r, 0)
            offs_k = tl.arange(0, BK)
            offs_n = pid_n * BN + tl.arange(0, BN)
            # K steps per half: D // BK, written so LLVM cannot see a constant trip count (M >= 0 always; the
            # wrappers never launch M <= 0), which keeps it from unrolling the pipelined loop with a mid-loop exit
            # (module comment at FWD8_BK). Same values, same order.
            nk = D // BK + (M < 0).to(tl.int32)
            acc = tl.zeros((BM, BN), dtype=tl.float32)
            for h in tl.static_range(NH):
                ih = tl.load(IDX + (row0 + offs_rc) * NH + h)
                if h > 0:  # mid-K: re-express the previous halves' sum in this half's row-scale units
                    ip = tl.load(IDX + (row0 + offs_rc) * NH + (h - 1))
                    rs = tl.load(SX + ip) * tl.load(SXI + ih)
                    acc = acc * rs[:, None]
                a_ptrs = U8 + (ih.to(tl.int32) * D)[:, None] + offs_k[None, :]
                b_ptrs = W8 + h * D + offs_n[None, :] * KT + offs_k[:, None]
                for k in range(0, nk):
                    a = tl.load(a_ptrs)
                    b = tl.load(b_ptrs)
                    acc = tl.dot(a, b, acc)
                    a_ptrs += BK
                    b_ptrs += BK
            il = tl.load(IDX + (row0 + offs_rc) * NH + (NH - 1))
            sc = tl.load(SX + il) * tl.load(SW)
            acc = acc * sc[:, None]
            bias = tl.load(BIAS + offs_n).to(tl.float32)
            acc = acc + bias[None, :]
            c_ptrs = C + (row0 + offs_r).to(tl.int64)[:, None] * NOUT + offs_n[None, :]
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=rmask[:, None])

    except Exception as exc:  # pragma: no cover
        _KERNELS_ERROR = f"{type(exc).__name__}: {str(exc)[:200]}"


# ------------------------------------------------------------------------------------------------------------ checks
def _require(ok: bool, msg: str) -> None:
    if not ok:
        raise Unsupported(msg)


def _ptr_class_ok(name: str, t: torch.Tensor, align: bool = True) -> None:
    """The one production specialization class: 16-byte aligned ('D') unless align=False, storage within 2 GiB."""
    if t.numel() == 0:
        return
    if align:
        _require(t.data_ptr() % 16 == 0, f"{name} data_ptr not 16-byte aligned (new Triton 'D' class)")
    _require(t.untyped_storage().size() <= MAX_PTR_RANGE,
             f"{name} storage {t.untyped_storage().size()} B > 2 GiB (new Triton 'S' / buffer-ops class)")


def _check_device(dev: torch.device) -> None:
    _require(_KERNELS_ERROR is None, f"FP8F Triton kernels unavailable ({_KERNELS_ERROR})")
    if dev.type == "cpu" and ALLOW_CPU_INTERPRETER and _interpreting():
        return  # host self-test under the Triton interpreter
    try:
        K._check_device(dev)  # Triton, HIP, GPU, current device, gfx950 (cached per device)
    except K.IndexedL1Unsupported as exc:
        raise Unsupported(str(exc)) from None


def _check_table(emb_c: torch.Tensor) -> int:
    """emb_c: the bf16 unique-row table [U >= 1, 512] contiguous, in the 'S' class. -> U."""
    _require(isinstance(emb_c, torch.Tensor), "emb_c must be a tensor")
    _check_device(emb_c.device)
    _require(emb_c.dtype == torch.bfloat16, f"emb_c dtype {emb_c.dtype} != bfloat16")
    _require(emb_c.dim() == 2 and emb_c.shape[1] == WIDTH and emb_c.shape[0] >= 1,
             f"emb_c shape {tuple(emb_c.shape)} != (U >= 1, {WIDTH})")
    _require(K._strides_ok(emb_c, (WIDTH, 1)), f"emb_c strides {tuple(emb_c.stride())} != ({WIDTH}, 1)")
    n = int(emb_c.shape[0])
    _require(n <= MAX_TABLE_ROWS, f"U={n} > {MAX_TABLE_ROWS}: emb_c / Ud would leave the 2 GiB 'S' class")
    _ptr_class_ok("emb_c", emb_c)
    return n


def _check_weight(w: torch.Tensor, dev: torch.device) -> Tuple[int, int]:
    """w: the bf16 W1 autocast cast [256, C] contiguous, C in (512, 1024), in the 'S' class. -> (R, C)."""
    _require(isinstance(w, torch.Tensor) and w.device == dev, f"w_c must be a tensor on {dev}")
    _require(w.dtype == torch.bfloat16, f"w_c dtype {w.dtype} != bfloat16 (pass the autocast cast)")
    _require(w.dim() == 2 and int(w.shape[0]) == HID and int(w.shape[1]) in W1Q_C,
             f"w_c shape {tuple(w.shape)} not ({HID}, C in {W1Q_C})")
    r, c = int(w.shape[0]), int(w.shape[1])
    _require(K._strides_ok(w, (c, 1)), f"w_c strides {tuple(w.stride())} != ({c}, 1)")
    _ptr_class_ok("w_c", w)
    return r, c


def check_fwd8(emb_c: torch.Tensor, idx: torch.Tensor, nh: int, w_c: torch.Tensor, b_c: torch.Tensor) -> int:
    """Raise Unsupported unless (emb_c, idx, nh, w_c, b_c) is the validated FP8F class. -> M."""
    _require(isinstance(nh, int) and not isinstance(nh, bool) and nh in (1, 2), f"nh={nh!r} (1 item, 2 cat)")
    _check_table(emb_c)
    dev = emb_c.device
    _require(isinstance(idx, torch.Tensor) and idx.device == dev, f"idx must be a tensor on {dev}")
    _require(idx.dtype == torch.int64, f"idx dtype {idx.dtype} != int64")
    _require(idx.dim() == 1 and (idx.numel() <= 1 or idx.stride(0) == 1), "idx must be 1-D contiguous")
    _require(idx.numel() % nh == 0, f"idx numel {idx.numel()} not a multiple of nh={nh}")
    m = idx.numel() // nh
    _require(m <= MAX_ROWS, f"M={m} > {MAX_ROWS}: h [M, {HID}] bf16 would leave the 2 GiB 'S' class")
    _require(m * nh < 2**31, f"M*nh={m * nh} overflows int32 token arithmetic")
    _ptr_class_ok("idx", idx, align=False)  # IDX is do_not_specialize_on_alignment
    _, c = _check_weight(w_c, dev)
    _require(c == nh * WIDTH, f"w_c K={c} != nh * {WIDTH} = {nh * WIDTH}")
    _require(isinstance(b_c, torch.Tensor) and b_c.device == dev and b_c.dtype == torch.bfloat16,
             f"b_c must be a bf16 tensor on {dev}")
    _require(tuple(b_c.shape) == (HID,) and K._strides_ok(b_c, (1,)), f"b_c shape {tuple(b_c.shape)} != ({HID},)")
    _ptr_class_ok("b_c", b_c)
    return m


# ------------------------------------------------------------------------------------------------------------ launches
def _launch_uprep(emb_c: torch.Tensor, write_ud: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                 Optional[torch.Tensor]]:
    n, dev = int(emb_c.shape[0]), emb_c.device
    u8 = torch.empty((n, WIDTH), dtype=torch.float8_e4m3fn, device=dev)
    sx = torch.empty((n,), dtype=torch.float32, device=dev)
    sxi = torch.empty((n,), dtype=torch.float32, device=dev)
    ud = torch.empty((n, WIDTH), dtype=torch.bfloat16, device=dev) if write_ud else None
    if n:
        STATS["uprep_launch"] += 1
        LAST_KERNELS["uprep_ud" if write_ud else "uprep"] = _ixf8_uprep[(_cdiv(n, UPREP_ROWS),)](
            emb_c.view(torch.int16), u8.view(torch.uint8), sx, sxi, (ud.view(torch.int16) if ud is not None else sx),
            n, D=WIDTH, ROWS=UPREP_ROWS, E_MIN=E_MIN, E_MAX=E_MAX, WRITE_UD=bool(write_ud), **UPREP_KW)
    return u8, sx, sxi, ud


def _launch_w1q(w: torch.Tensor, write_wd: bool) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    r, c, dev = int(w.shape[0]), int(w.shape[1]), w.device
    w8 = torch.empty((r, c), dtype=torch.float8_e4m3fn, device=dev)
    sw = torch.empty((1,), dtype=torch.float32, device=dev)
    wd = torch.empty((r, c), dtype=torch.bfloat16, device=dev) if write_wd else None
    STATS["w1q_launch"] += 1
    LAST_KERNELS["w1q_%d%s" % (c, "_wd" if write_wd else "")] = _ixf8_w1q[(r // W1Q_BLOCK_R,)](
        w, w8.view(torch.uint8), sw, (wd.view(torch.int16) if wd is not None else sw),
        R=r, C=c, BLOCK_R=W1Q_BLOCK_R, AMAX_R=W1Q_AMAX_ELEMS // c, INV448=INV448, WRITE_WD=bool(write_wd),
        **W1Q_KW)
    return w8, sw, wd


def _launch_fwd8(u8: torch.Tensor, sx: torch.Tensor, sxi: torch.Tensor, idx: torch.Tensor, nh: int,
                 w8: torch.Tensor, sw: torch.Tensor, b1: torch.Tensor, m: int) -> torch.Tensor:
    out = torch.empty((m, HID), dtype=torch.bfloat16, device=u8.device)
    if m == 0:
        return out
    STATS["fwd8_launch"] += 1
    LAST_KERNELS["fwd8_%d" % nh] = _ecix_fwd8[(_cdiv(m, FWD8_BM) * (HID // FWD8_BN),)](
        u8, sx, sxi, idx, w8, sw, b1, out, m,
        NH=nh, D=WIDTH, NOUT=HID, BM=FWD8_BM, BN=FWD8_BN, BK=FWD8_BK, **FWD8_KW)
    return out


# ------------------------------------------------------------------------------------------------------------ wrappers
def uprep(emb_c: torch.Tensor, write_ud: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                Optional[torch.Tensor]]:
    """bf16 U [U, 512] -> (U8 e4m3 [U, 512], SX fp32 [U], SXI fp32 [U], Ud bf16 [U, 512] or None), one launch.
    Raises Unsupported before launching."""
    _check_table(emb_c)
    return _launch_uprep(emb_c, write_ud)


def wprep(w_c: torch.Tensor, write_wd: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """bf16 W1 [256, C] -> (W8 e4m3 [256, C], SW fp32 [1], Wd bf16 [256, C] or None), one 16-program launch.
    Raises Unsupported before launching."""
    _require(isinstance(w_c, torch.Tensor), "w_c must be a tensor")
    _check_device(w_c.device)
    _check_weight(w_c, w_c.device)
    return _launch_w1q(w_c, write_wd)


def ecix_fwd8(u8: torch.Tensor, sx: torch.Tensor, sxi: torch.Tensor, idx: torch.Tensor, nh: int, w8: torch.Tensor,
              sw: torch.Tensor, b1: torch.Tensor) -> torch.Tensor:
    """h [M, 256] bf16 from given FP8 operands (tests / prebuild; fwd8 is the production entry). Raises Unsupported
    before launching."""
    _require(isinstance(nh, int) and not isinstance(nh, bool) and nh in (1, 2), f"nh={nh!r}")
    _require(isinstance(u8, torch.Tensor) and u8.dtype == torch.float8_e4m3fn and u8.dim() == 2
             and u8.shape[1] == WIDTH and u8.shape[0] >= 1 and K._strides_ok(u8, (WIDTH, 1)),
             "u8 must be e4m3 [U >= 1, 512] contiguous")
    dev = u8.device
    _check_device(dev)
    n = int(u8.shape[0])
    _require(n <= 2 * MAX_TABLE_ROWS, f"U={n} outside the 'S' class")
    _ptr_class_ok("u8", u8)
    for name, t, numel in (("sx", sx, n), ("sxi", sxi, n), ("sw", sw, 1)):
        _require(isinstance(t, torch.Tensor) and t.device == dev and t.dtype == torch.float32 and t.dim() == 1
                 and t.numel() == numel and t.is_contiguous(), f"{name} must be fp32 [{numel}] contiguous on {dev}")
        _ptr_class_ok(name, t)
    _require(isinstance(idx, torch.Tensor) and idx.device == dev and idx.dtype == torch.int64 and idx.dim() == 1
             and (idx.numel() <= 1 or idx.stride(0) == 1) and idx.numel() % nh == 0,
             "idx must be int64 1-D contiguous with numel % nh == 0")
    m = idx.numel() // nh
    _require(m <= MAX_ROWS and m * nh < 2**31, f"M={m} outside the 'S' class")
    _ptr_class_ok("idx", idx, align=False)
    kt = nh * WIDTH
    _require(isinstance(w8, torch.Tensor) and w8.device == dev and w8.dtype == torch.float8_e4m3fn
             and tuple(w8.shape) == (HID, kt) and K._strides_ok(w8, (kt, 1)), f"w8 must be e4m3 ({HID}, {kt})")
    _ptr_class_ok("w8", w8)
    _require(isinstance(b1, torch.Tensor) and b1.device == dev and b1.dtype == torch.bfloat16
             and tuple(b1.shape) == (HID,) and K._strides_ok(b1, (1,)), f"b1 must be bf16 ({HID},)")
    _ptr_class_ok("b1", b1)
    return _launch_fwd8(u8, sx, sxi, idx, nh, w8, sw, b1, m)


def _drop_ucache(ref: Any) -> None:
    """Item (84) U-cache lifetime. weakref callback: the emb_c of the cached U prep died; drop the entry so its U8 / SX / SXI (/ Ud) are freed.
    Without it an entry used by one member only (asymmetric publish, a per-member Unsupported, an IX or IX-FP8F latch
    between the members) kept ~66-120 MB per rank alive until the next fwd8 call, or for the rest of the run after a
    latch. Ud is still held by the autograd node that saved it, so the backward is unaffected."""
    global _UCACHE
    c = _UCACHE
    if c is not None and c.ref is ref:
        _UCACHE = None
        STATS["uprep_dropped_on_free"] += 1


class _UPrepEntry:
    """The U prep of one emb_c object, shared by the two members of one step (no reference cycle: weakref; the
    weakref's callback drops the entry when emb_c dies, so the prep never outlives the step's table)."""

    __slots__ = ("ref", "version", "ptr", "stream", "write_ud", "out", "used")

    def __init__(self, emb_c: torch.Tensor, stream: Any, write_ud: bool, out: Tuple[Any, ...], nh: int) -> None:
        self.ref = weakref.ref(emb_c, _drop_ucache)
        self.version = emb_c._version
        self.ptr = emb_c.data_ptr()
        self.stream = stream
        self.write_ud = write_ud
        self.out = out
        self.used = {nh}

    def hit(self, emb_c: torch.Tensor, stream: Any, write_ud: bool, nh: int) -> bool:
        return (self.ref() is emb_c and self.version == emb_c._version and self.ptr == emb_c.data_ptr()
                and self.stream == stream and self.write_ud == write_ud and nh not in self.used)


_UCACHE: Optional[_UPrepEntry] = None


def _stream_key(dev: torch.device) -> Any:
    if dev.type != "cuda":
        return None
    s = torch.cuda.current_stream(dev)
    return (s.device_index, s.stream_id)


def _uprep_for(emb_c: torch.Tensor, write_ud: bool, nh: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                                      Optional[torch.Tensor]]:
    """The U prep of emb_c, reused when the other member already ran it on this very tensor (same object, same
    version counter, same stream) in this step; dropped once both members used it."""
    global _UCACHE
    stream = _stream_key(emb_c.device)
    c = _UCACHE
    if c is not None and c.hit(emb_c, stream, write_ud, nh):
        c.used.add(nh)
        if {1, 2} <= c.used:
            _UCACHE = None
        STATS["uprep_reuse"] += 1
        return c.out
    out = _launch_uprep(emb_c, write_ud)
    _UCACHE = _UPrepEntry(emb_c, stream, write_ud, out, nh)
    return out


def _armed(nh: int, m: int, u: int, mode: str) -> None:
    kind = "item" if nh == 1 else "cat"
    if kind in _ARMED:
        return
    _ARMED.add(kind)
    kt = nh * WIDTH
    bwd_txt = ("the IX wgrad reads Ud = bf16(U8 * 2^E), dX and E-X2b read Wd = bf16(W1_8 * SW)"
               if mode == "consistent" else "STE: the IX wgrad and dX read the bf16 U and W1")
    cat_txt = ", exact pow2 artist->album mid-K rescale" if nh == 2 else ""
    _say(f"{ARMED_PREFIX} fwd ({kind}): train-only L1 forward on FP8 operands -- e4m3 U8 per-row pow2 (d) scale "
         f"2^E, E = ceil(log2(amax/448)) in [{E_MIN}, {E_MAX}] (zero row E = 0, non-finite row -> NaN h) of the bf16 "
         f"U [U={u}, {WIDTH}] (_ixf8_uprep); W1_8 per-tensor amax/448 e4m3 [{HID}, {kt}] (_ixf8_w1q); _ecix_fwd8 "
         f"{FWD8_TILE} M={m}, fp32 accumulate{cat_txt}; bwd={mode} ({bwd_txt}); eval keeps the bf16 Indexed-X "
         f"forward; kill switch {ENV_VAR}=0")


def fwd8(emb_c: torch.Tensor, idx: torch.Tensor, nh: int, w_c: torch.Tensor,
         b_c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The IX-FP8F train forward of one member -> (h, table, w_bwd).
    h: bf16 [M, 256] (same shape / dtype / layout as K.ecix_fwd); table: Ud (consistent) or emb_c (ste); w_bwd: Wd
    (consistent) or w_c (ste). emb_c: bf16 [U, 512] (the EC's table, both members); idx: int64 [M] (item, nh=1) or
    [2M] interleaved artist/album (cat, nh=2); w_c: bf16 [256, 512 * nh]; b_c: bf16 [256] (the autocast casts).
    Raises Unsupported BEFORE any launch outside the validated class, ValueError on a bad BWD env value; any other
    exception is a compile / launch fault (the caller latches off). Prints the armed line once per member."""
    mode = bwd_mode()
    m = check_fwd8(emb_c, idx, nh, w_c, b_c)
    cons = mode == "consistent"
    u8, sx, sxi, ud = _uprep_for(emb_c, cons, nh)
    w8, sw, wd = _launch_w1q(w_c, cons)
    h = _launch_fwd8(u8, sx, sxi, idx, nh, w8, sw, b_c, m)
    _armed(nh, m, int(emb_c.shape[0]), mode)
    if cons:
        assert ud is not None and wd is not None
        return h, ud, wd
    return h, emb_c, w_c


# ----------------------------------------------------------------------------------------------------------- references
def exponents(amax: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 row amax -> (E int32, over bool): E = ceil(log2(amax / 448)) from the fp32 bits, 0 for a zero or
    subnormal amax, clamped to [E_MIN, E_MAX]; over = E > E_MAX before the clamp."""
    a = amax.float().contiguous()
    ab = a.view(torch.int32)
    ef = (ab >> 23) & 0xFF
    e = ef - 135 + ((ab & 0x7FFFFF) > 0x600000).to(torch.int32)
    e = torch.where(ef == 0, torch.zeros_like(e), e).clamp_min(E_MIN)
    over = e > E_MAX
    return e.clamp_max(E_MAX), over


def pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e as fp32, exact (e int in [-126, 127])."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    """e4m3fn codes (uint8 or float8_e4m3fn) -> fp32 values (0x7F / 0xFF -> NaN)."""
    return codes.contiguous().view(torch.float8_e4m3fn).float()


def e4m3fn_codes(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3fn codes (uint8) by c10's non-saturating rule, whatever the torch version: RNE; NaN, +-inf and
    |v| > 464 (past the last rounding step to 448) -> 0x7F | sign, like the kernel's _e4m3_rne. torch 2.16's CPU
    conversion saturates those to +-448 (0x7E | sign) instead; torch 2.12's gives NaN. Every other code is the same
    in both (all bf16 values x 2^-10..2^8 compared, ROCm 10.2 port), so only the bad rows' codes needed this."""
    v = v.float().contiguous()
    codes = v.to(torch.float8_e4m3fn).view(torch.uint8)
    sign = ((v.view(torch.int32) >> 24) & 0x80).to(torch.uint8)
    return torch.where(~(v.abs() <= 464.0), sign | 0x7F, codes)


def reference_uprep(table: torch.Tensor) -> Dict[str, torch.Tensor]:
    """torch reference of _ixf8_uprep on the bf16 table (any device): codes uint8 [U, 512], e int32 [U] (clamped),
    bad bool [U], sx / sxi fp32 [U] (NaN on bad rows), ud bf16 [U, 512]."""
    xb = table.float()
    fin = torch.isfinite(xb)
    amax = torch.where(fin, xb.abs(), torch.zeros_like(xb)).amax(1)
    e, over = exponents(amax)
    bad = (~fin).any(1) | over
    nan = torch.full_like(amax, float("nan"))
    sx = torch.where(bad, nan, pow2(e))
    sxi = torch.where(bad, nan, pow2(-e))
    codes = e4m3fn_codes(xb * pow2(-e)[:, None])
    ud = (decode_e4m3(codes) * sx[:, None]).to(torch.bfloat16)
    return dict(codes=codes, e=e, bad=bad, sx=sx, sxi=sxi, ud=ud)


def reference_wprep(w: torch.Tensor) -> Dict[str, torch.Tensor]:
    """torch replica of bwd_fp8.quantize_weight AS IT RUNS ON THE GPU (amax / 448 is ATen's multiply by fl32(1/448)
    for a CPU-scalar divisor on CUDA; on a CPU tensor quantize_weight divides exactly instead), written out so any
    device gives the GPU bits: w8 float8_e4m3fn, sw fp32 [1], wd bf16 = bf16_rne(fp32(w8) * sw)."""
    wf = w.float()
    amax = w.detach().abs().amax().float().clamp_min(AMAX_FLOOR)
    sw = (amax * torch.tensor(INV448, dtype=torch.float32, device=amax.device)).reshape(1)
    inv = torch.div(torch.ones_like(sw), sw)
    w8 = (wf * inv).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    wd = (w8.float() * sw).to(torch.bfloat16)
    return dict(w8=w8, sw=sw, wd=wd)


def reference_h8(u8: torch.Tensor, sx: torch.Tensor, idx: torch.Tensor, nh: int, w8: torch.Tensor, sw: torch.Tensor,
                 b1: torch.Tensor, chunk: int = 65536) -> torch.Tensor:
    """fp64 h of the FP8 operands (own codes): sum_h (U8[idx_h] * SX[idx_h]) @ (W8_h * SW)^T + b1, every product
    exact in fp64, fp64 sums (exact on the prebuild classes). Token chunks of `chunk` rows bound the fp64
    temporaries (production M on a GPU: ~0.25 GB per chunk and half)."""
    m = idx.numel() // nh
    wv = decode_e4m3(w8).double() * sw.double()
    sxd = sx.double()
    u8b = u8.view(torch.uint8)
    out = b1.double().expand(m, HID).clone()
    for r0 in range(0, m, max(1, chunk)):
        r1 = min(m, r0 + max(1, chunk))
        for h in range(nh):
            ih = idx[r0 * nh + h:r1 * nh:nh]
            a = decode_e4m3(u8b.index_select(0, ih)).double() * sxd.index_select(0, ih)[:, None]
            out[r0:r1] += a @ wv[:, h * WIDTH:(h + 1) * WIDTH].t()
    return out


def round_bf16(v: torch.Tensor, mode: str = "rne") -> torch.Tensor:
    """values exactly representable in fp32 -> bf16 by RNE (GPU kernels) or RTZ (the Triton interpreter's fp32 ->
    bf16 cast truncates)."""
    f = v.float()
    if mode == "rne":
        return f.to(torch.bfloat16)
    if mode == "rtz":
        return (f.contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)
    raise ValueError(mode)


def reference_fwd8(emb_c: torch.Tensor, idx: torch.Tensor, nh: int, w_c: torch.Tensor, b_c: torch.Tensor,
                   mode: Optional[str] = None, rounding: str = "rne") -> Tuple[torch.Tensor, torch.Tensor,
                                                                               torch.Tensor]:
    """Drop-in torch reference of fwd8 (CPU tests): the same codes and scales (reference_uprep / reference_wprep), h
    = bf16(fp64 exact own-codes product + b1) (the kernel is within 1 bf16 ulp of it: fp32 accumulation order),
    table / w_bwd per the backward mode (default: bwd_mode())."""
    mode = bwd_mode() if mode is None else mode
    ru, rw = reference_uprep(emb_c), reference_wprep(w_c)
    h = round_bf16(reference_h8(ru["codes"], ru["sx"], idx, nh, rw["w8"], rw["sw"], b_c), rounding)
    if mode == "consistent":
        return h, ru["ud"], rw["wd"]
    return h, emb_c, w_c


# ------------------------------------------------------------------------------------------------------------ compare
def _nan_class_bits_mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    """Raw-bit mismatches of two same-shape float tensors, NaN == NaN (payloads / sign not compared)."""
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    if a.shape != b.shape:
        return max(a.numel(), b.numel())
    ia = a.view(torch.int16 if a.element_size() == 2 else torch.int32)
    ib = b.view(torch.int16 if b.element_size() == 2 else torch.int32)
    na, nb = torch.isnan(a.float()), torch.isnan(b.float())
    return int(((ia != ib) & ~(na & nb)).sum())


def canon_codes(c: torch.Tensor) -> torch.Tensor:
    """e4m3fn codes as uint8 on the CPU with both NaN codes (0x7F, 0xFF) mapped to 0x7F."""
    c = c.detach().cpu().contiguous().view(torch.uint8)
    return torch.where((c & 0x7F) == 0x7F, torch.full_like(c, 0x7F), c)


def uprep_compare(u8: torch.Tensor, sx: torch.Tensor, sxi: torch.Tensor, ud: Optional[torch.Tensor],
                  ref: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Kernel U prep vs reference_uprep, raw bits (NaN codes / NaN floats compared by class). All 0 = exact."""
    out = dict(codes=int((canon_codes(u8) != canon_codes(ref["codes"])).sum()),
               sx=_nan_class_bits_mismatch(sx, ref["sx"]), sxi=_nan_class_bits_mismatch(sxi, ref["sxi"]))
    if ud is not None:
        out["ud"] = _nan_class_bits_mismatch(ud, ref["ud"])
    return out


def wprep_compare(w8: torch.Tensor, sw: torch.Tensor, wd: Optional[torch.Tensor],
                  ref: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Kernel W prep vs reference_wprep. codes: byte for byte at every element whose replica q is not NaN; nanq_bad:
    elements whose replica q IS NaN (only with a NaN / inf W: SW NaN / inf, every h of the member non-finite) must
    hold {+-448, NaN} (the clamp's NaN rule is the backend's: NaN-dropping v_med3 on gfx950, NaN-propagating numpy in
    the interpreter); scale: bits (NaN by class); wd: bits at the non-NaN-q elements (NaN by class), non-finite at
    the NaN-q elements. All four 0 = exact."""
    k = w8.detach().cpu().contiguous().view(torch.uint8).reshape(-1).to(torch.int32)
    r = ref["w8"].detach().cpu().contiguous().view(torch.uint8).reshape(-1).to(torch.int32)
    nanq = (r & 0x7F) == 0x7F
    k_cls = ((k & 0x7F) == 0x7F) | ((k & 0x7F) == 0x7E)
    out: Dict[str, Any] = dict(codes=int(((k != r) & ~nanq).sum()), nanq_bad=int((nanq & ~k_cls).sum()),
                               scale=_nan_class_bits_mismatch(sw.reshape(-1), ref["sw"].reshape(-1)),
                               nanq=int(nanq.sum()))
    if wd is not None:
        a = wd.detach().cpu().contiguous().reshape(-1)
        b = ref["wd"].detach().cpu().contiguous().reshape(-1)
        ia, ib = a.view(torch.int16), b.view(torch.int16)
        na, nb = torch.isnan(a.float()), torch.isnan(b.float())
        out["wd"] = int((((ia != ib) & ~(na & nb)) & ~nanq).sum()) + int((nanq & torch.isfinite(a.float())).sum())
    return out


def wprep_ok(cmp: Dict[str, Any]) -> bool:
    return cmp["codes"] == 0 and cmp["nanq_bad"] == 0 and cmp["scale"] == 0 and cmp.get("wd", 0) == 0


# ------------------------------------------------------------------------------------------------------------ prebuild
def _tern(r: int, c: int, a: int, b: int, lo: int = -1, hi: int = 1) -> torch.Tensor:
    i = torch.arange(r, dtype=torch.int64)[:, None]
    j = torch.arange(c, dtype=torch.int64)[None, :]
    return ((i * a + j * b) % (hi - lo + 1) + lo).to(torch.float64)


def _prebuild_idx(nh: int, m: int, n_rows: int) -> torch.Tensor:
    """K.prebuild's index pattern: a hot row over many blocks, the last row U-1, an artist / album split (cat)."""
    t = torch.arange(m, dtype=torch.int64)
    if nh == 1:
        idx = torch.where(t < 3000, torch.zeros_like(t), 1 + (t * 7) % (n_rows - 1))
    else:
        half = n_rows // 2
        art = torch.where(t < 2500, torch.full_like(t, 1), (t * 5) % half)
        alb = half + (t * 11) % (n_rows - half)
        idx = torch.stack([art, alb], dim=1).reshape(-1)
    idx[min(5, idx.numel() - 1)] = n_rows - 1
    return idx


def prebuild_inputs(nh: int, m: int, n_rows: int) -> Dict[str, torch.Tensor]:
    """CPU fwd8 inputs whose fp32 arithmetic is EXACT (e4m3-representable integers x exact pow2 scales): U8 codes
    ((3r + 7c) mod 5) - 2, row exponents ((7r + 3) mod 5) - 2 (mid-K rescale 2^-4..2^4), W8 in {-1, 0, 1},
    SW = 2^-3, b1 in {-1, 0, 1}; every partial sum fits 24 bits, so h is the exact value rounded once."""
    e = (torch.arange(n_rows, dtype=torch.int32) * 7 + 3) % 5 - 2
    return dict(u8=_tern(n_rows, WIDTH, 3, 7, -2, 2).float().to(torch.float8_e4m3fn), sx=pow2(e), sxi=pow2(-e),
                idx=_prebuild_idx(nh, m, n_rows),
                w8=_tern(HID, nh * WIDTH, 7, 2 + nh).float().to(torch.float8_e4m3fn),
                sw=torch.tensor([0.125], dtype=torch.float32), b1=_tern(1, HID, 3, 1).view(HID).float().to(
                    torch.bfloat16))


def extreme_inputs(nh: int, m: int = 300) -> Dict[str, torch.Tensor]:
    """CPU fwd8 inputs of the scale extremes: rows 0 zero (E 0), 1 E_MAX, 2 E_MIN, 3 a NaN-scaled (bad) row, 4..11
    E in [-2, 2]; codes in {0, 1, 2} and W8[n, :] = +-1 per output column (no cancellation, so a dominated half is
    absorbed the same way in any summation order), SW = 2^-3, b1 in {-1, 0, 1}. idx pairs every extreme with every
    other (|E_art - E_alb| up to 100) and touches the bad row from both halves; M = 300 has a masked tail block."""
    n_rows = 12
    codes = (_tern(n_rows, WIDTH, 5, 3, 0, 2) * (torch.arange(n_rows) != 0).double()[:, None]).float()
    e = torch.tensor([0, E_MAX, E_MIN, 0] + [(-2 + (r % 5)) for r in range(4, n_rows)], dtype=torch.int32)
    sx, sxi = pow2(e), pow2(-e)
    sx[3] = float("nan")
    sxi[3] = float("nan")
    t = torch.arange(m, dtype=torch.int64)
    if nh == 1:
        idx = t % n_rows
    else:
        art = t % n_rows
        alb = (t // n_rows + 2 * t) % n_rows
        idx = torch.stack([art, alb], dim=1).reshape(-1)
    sgn = torch.where(torch.arange(HID) % 2 == 0, 1.0, -1.0)[:, None]
    return dict(u8=codes.to(torch.float8_e4m3fn), sx=sx, sxi=sxi, idx=idx,
                w8=(sgn * torch.ones(HID, nh * WIDTH)).to(torch.float8_e4m3fn),
                sw=torch.tensor([0.125], dtype=torch.float32),
                b1=_tern(1, HID, 3, 1).view(HID).float().to(torch.bfloat16))


def emulate_fwd8(p: Dict[str, torch.Tensor], nh: int, rounding: str = "rne") -> torch.Tensor:
    """The kernel's own fp32 operation sequence in torch (exact integer dots; acc * pow2 rescale; + next dot; *
    (SX * SW); + b1; one bf16 rounding). Equal to the kernel bit for bit on inputs whose dots are exact integers and
    whose dominated halves cannot cancel (prebuild_inputs, extreme_inputs)."""
    idx, sx, sxi = p["idx"], p["sx"], p["sxi"]
    w = decode_e4m3(p["w8"]).double()
    acc = None
    for h in range(nh):
        ih = idx[h::nh]
        dot = (decode_e4m3(p["u8"].view(torch.uint8).index_select(0, ih)).double()
               @ w[:, h * WIDTH:(h + 1) * WIDTH].t()).float()  # exact integers (< 2^24)
        if acc is None:
            acc = dot
        else:
            ip = idx[h - 1::nh]
            acc = acc * (sx.index_select(0, ip) * sxi.index_select(0, ih))[:, None] + dot
    il = idx[nh - 1::nh]
    acc = acc * (sx.index_select(0, il) * p["sw"])[:, None] + p["b1"].float()[None, :]
    return round_bf16(acc, rounding)


def uprep_edge_table(n_rows: int = 96, seed: int = 3) -> torch.Tensor:
    """bf16 [n_rows >= 32, 512]: lognormal rows plus the edge rows: 0 zeros, 1 signed zeros, 2 tiny (E clamped to
    E_MIN), 3 bf16 subnormals, 4 amax 448 (E 0), 5 amax 450, 6 powers of two, 7 in-range huge (E ~ 49), 8 over-range
    (bad), 9 NaN, 10 +inf, 11 -inf, 12 negative NaN, 13 e4m3 rounding ties (normal and subnormal), 14 all NaN,
    15 amax 448 * 2^E_MAX (E = E_MAX), 16 amax 450 * 2^E_MAX (bad), 17 amax 448 * 2^E_MIN (E = E_MIN), 18 amax
    448 * 2^(E_MIN - 1) (clamped), 19 mixed signs with exact zeros."""
    assert n_rows >= 32
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_rows, WIDTH, generator=g) * torch.exp(2.0 * torch.randn(n_rows, 1, generator=g)) * 0.05
    sgn = torch.where(torch.rand(WIDTH, generator=g) < 0.5, -1.0, 1.0)
    x[0] = 0.0
    x[1] = sgn * 0.0
    x[2] = torch.randn(WIDTH, generator=g) * 1e-20
    x[3] = sgn * torch.ldexp(1.0 + torch.rand(WIDTH, generator=g), torch.randint(-133, -127, (WIDTH,), generator=g))
    x[4] = torch.randn(WIDTH, generator=g).clamp(-1, 1) * 100.0
    x[4, 7] = 448.0
    x[5] = x[4]
    x[5, 9] = -450.0
    x[6] = sgn * torch.ldexp(torch.ones(WIDTH), torch.randint(-12, 3, (WIDTH,), generator=g))
    x[7] = torch.randn(WIDTH, generator=g) * 2.0 ** 55
    x[8] = torch.randn(WIDTH, generator=g) * 1e30
    x[9] = torch.randn(WIDTH, generator=g)
    x[9, 11] = float("nan")
    x[10] = torch.randn(WIDTH, generator=g)
    x[10, 13] = float("inf")
    x[11] = torch.randn(WIDTH, generator=g)
    x[11, 17] = float("-inf")
    x[12] = torch.randn(WIDTH, generator=g)
    x[13] = 0.0
    x[13, 0] = 448.0
    k = torch.arange(1, WIDTH)
    ties = torch.cat([torch.tensor([1.0625, 1.1875, 1.3125, 1.4375, 1.5625, 1.6875, 1.8125, 1.9375])
                      * 2.0 ** float(j) for j in range(-6, 8)]
                     + [torch.tensor([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]) * 2.0 ** -9])
    x[13, 1:1 + ties.numel()] = ties * torch.where(k[:ties.numel()] % 2 == 0, -1.0, 1.0)
    x[14] = float("nan")
    x[15] = torch.randn(WIDTH, generator=g).clamp(-1, 1) * 2.0 ** 58
    x[15, 3] = 448.0 * 2.0 ** E_MAX
    x[16] = x[15]
    x[16, 5] = 450.0 * 2.0 ** E_MAX
    x[17] = torch.randn(WIDTH, generator=g).clamp(-1, 1) * 2.0 ** -45
    x[17, 3] = 448.0 * 2.0 ** E_MIN
    x[18] = torch.randn(WIDTH, generator=g).clamp(-1, 1) * 2.0 ** -46
    x[18, 3] = 448.0 * 2.0 ** (E_MIN - 1)
    x[19] = torch.randn(WIDTH, generator=g)
    x[19, ::5] = 0.0
    x[19, 1::5] = -0.0
    xb = x.to(torch.bfloat16)
    nb = xb.view(torch.int16)
    nb[12, 23] = -64  # 0xFFC0: a negative quiet NaN
    return xb.contiguous()


def uprep_encoder_table() -> torch.Tensor:
    """bf16 rows holding EVERY bf16 bit pattern: the finite ones with |v| <= 448 (511 per row, each row also holding
    448, so E = 0 and the codes are the plain e4m3 RNE of the bf16 values: an exhaustive encoder / decoder check) and,
    in rows of their own, every non-finite pattern (bad rows)."""
    p = torch.arange(65536, dtype=torch.int32)
    v = (p << 16).view(torch.float32)
    rows = []
    for keep in (v.abs() <= 448.0, ~torch.isfinite(v)):
        vals = v[keep]
        per = WIDTH - 1
        n = -(-vals.numel() // per)
        body = torch.zeros(n * per, dtype=torch.float32)
        body[:vals.numel()] = vals
        rows.append(torch.cat([torch.full((n, 1), 448.0), body.view(n, per)], dim=1))
    return torch.cat(rows).to(torch.bfloat16).contiguous()  # exact: every value is a bf16 value


W1Q_EDGE: Tuple[str, ...] = ("random", "zeros", "signed_zeros", "subnormal", "subnormal_mixed", "amax_448",
                             "amax_above_448", "huge", "nan", "inf", "neg_inf", "nan_inf")


def w1q_edge_weights(c: int, seed: int = 29) -> Dict[str, torch.Tensor]:
    """bf16 [256, c] W1 test weights (rccl iter86 w1q_edge_weights, bf16 class; no global RNG)."""
    g0 = torch.Generator().manual_seed(c)
    out = {"random": torch.randn(HID, c, generator=g0) * 0.03}
    g = torch.Generator().manual_seed(seed + c)
    base = torch.randn(HID, c, generator=g) * 0.03
    sgn = torch.where(torch.rand(HID, c, generator=g) < 0.5, -1.0, 1.0)
    out["zeros"] = torch.zeros(HID, c)
    out["signed_zeros"] = sgn * 0.0
    sub = sgn * torch.ldexp(1.0 + torch.rand(HID, c, generator=g), torch.randint(-149, -127, (HID, c), generator=g))
    out["subnormal"] = sub
    mix = base.clone()
    pick = torch.rand(HID, c, generator=g) < 0.15
    mix[pick] = sub[pick]
    out["subnormal_mixed"] = mix
    a = torch.randn(HID, c, generator=g).clamp(-1, 1) * 100.0
    a[3, 7] = 448.0
    out["amax_448"] = a
    b = a.clone()
    b[200, 9] = -450.0  # the next bf16 above 448
    out["amax_above_448"] = b
    out["huge"] = sgn * 3.0e38 * (0.5 + 0.5 * torch.rand(HID, c, generator=g))
    for name, sets in (("nan", ((5, 11, float("nan")),)), ("inf", ((6, 13, float("inf")),)),
                       ("neg_inf", ((7, 17, float("-inf")),)),
                       ("nan_inf", ((5, 11, float("nan")), (6, 13, float("inf"))))):
        t = base.clone()
        for i, j, v in sets:
            t[i, j] = v
        out[name] = t
    return {k: out[k].to(torch.bfloat16).contiguous() for k in W1Q_EDGE}


def _guard(name: str, ck: Any) -> Optional[str]:
    """Compile guard of the forward (the lab GO binary: 127 VGPR, 0 spills, occupancy 4; Triton 3.9 at BK 64: 116 / 126
    VGPR, 0 spills). None if fine or unreadable
    (interpreter)."""
    n_regs, n_spills = getattr(ck, "n_regs", None), getattr(ck, "n_spills", None)
    if n_regs is None or n_spills is None:
        return None
    if int(n_spills) != 0 or int(n_regs) > FWD8_MAX_VGPR:
        return f"{name}: n_regs {n_regs} / n_spills {n_spills} (need <= {FWD8_MAX_VGPR} / 0)"
    return None


def prebuild(device: Any, m: int = 20100, n_rows: int = 96) -> str:
    """Before RUN_START: compile every FP8F kernel at the production specialization class (fresh 16-byte-aligned
    tensors within 2 GiB, n_rows / M unspecialized) -- _ixf8_uprep x (Ud on, off), _ixf8_w1q x (item C 512, cat C 1024)
    x (Wd on, off), _ecix_fwd8 x (item, cat) -- and check them EXACTLY: the U prep bitwise vs reference_uprep on the
    edge rows and on every bf16 bit pattern; W1_8 / SW / Wd vs reference_wprep (the GPU quantize_weight arithmetic) on
    12 edge weights per class; _ecix_fwd8 bitwise vs the exact reference at K.prebuild's M / U (masked tail block)
    and on the scale-extreme / NaN-row class; plus the forward compile guard (<= 128 VGPR, 0 spills). No global RNG
    (seeded CPU generators). Validates the BWD env (ValueError). Raises RuntimeError on any mismatch (the caller
    latches FP8F off). Returns the full marker line "[ec-indexed-l1-fp8] pre-run_start prebuild OK (...)" for the
    caller to print as is."""
    mode = bwd_mode()
    dev = torch.device(device)
    if dev.type == "cuda" and dev.index is None:
        dev = torch.device("cuda", torch.cuda.current_device())
    _check_device(dev)
    rounding = "rtz" if _interpreting() else "rne"
    bad: List[str] = []
    tables = (("edge", uprep_edge_table(n_rows)), ("encoder", uprep_encoder_table()))
    n_tab = sum(int(t.shape[0]) for _, t in tables)
    for write_ud in (True, False):
        for tag, x in tables:
            ref = reference_uprep(x)
            got = uprep_compare(*_launch_uprep(x.to(dev), write_ud), ref)
            if any(got.values()):
                bad.append(f"uprep {tag} Ud={write_ud}: {got}")
    n_w = 0
    for c in W1Q_C:
        for name, w in w1q_edge_weights(c).items():
            ref = reference_wprep(w)
            for write_wd in (True, False):
                cmp = wprep_compare(*_launch_w1q(w.to(dev), write_wd), ref)
                n_w += 1
                if not wprep_ok(cmp):
                    bad.append(f"w1q C={c} {name} Wd={write_wd}: {cmp}")
    guards: List[str] = []
    for nh in (1, 2):
        for tag, p in (("exact", prebuild_inputs(nh, m, n_rows)), ("extreme", extreme_inputs(nh))):
            q = {k: v.to(dev) for k, v in p.items()}
            h = _launch_fwd8(q["u8"], q["sx"], q["sxi"], q["idx"], nh, q["w8"], q["sw"], q["b1"],
                             q["idx"].numel() // nh)
            ref = emulate_fwd8(p, nh, rounding)
            nb = _nan_class_bits_mismatch(h, ref)
            if nb:
                bad.append(f"fwd8 {tag} nh={nh}: {nb} of {h.numel()} differ")
            if tag == "extreme":  # loud rows: exactly the tokens that touch the NaN-scaled row
                touch = torch.zeros(p["idx"].numel() // nh, dtype=torch.bool)
                for hh in range(nh):
                    touch |= p["idx"][hh::nh] == 3
                hn = torch.isnan(h.detach().cpu().float())
                if not (bool(hn[touch].all()) and not bool(hn[~touch].any())):
                    bad.append(f"fwd8 extreme nh={nh}: NaN rows != the tokens touching the bad row")
        why = _guard(f"_ecix_fwd8 nh={nh}", LAST_KERNELS.get("fwd8_%d" % nh))
        if why is not None:
            bad.append("compile guard " + why)
        ck = LAST_KERNELS.get("fwd8_%d" % nh)
        guards.append(f"{'item' if nh == 1 else 'cat'} {getattr(ck, 'n_regs', 'n/a')} VGPR / "
                      f"{getattr(ck, 'n_spills', 'n/a')} spills")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    if bad:
        raise RuntimeError(f"{TAG} prebuild differs from the exact reference: " + "; ".join(bad))
    return (f"{TAG} pre-run_start prebuild OK (_ixf8_uprep x Ud on/off, _ixf8_w1q x item+cat x Wd on/off, _ecix_fwd8 "
            f"x item+cat; exact checks: U prep bitwise on {n_tab} rows (edge rows, every bf16 pattern), W1_8 / SW / Wd "
            f"vs the GPU quantize_weight replica on {n_w} weight cases, _ecix_fwd8 bitwise M={m} U={n_rows} + the "
            f"scale-extreme / NaN-row class; _ecix_fwd8 {'; '.join(guards)}; bwd={mode}; device={dev})")
