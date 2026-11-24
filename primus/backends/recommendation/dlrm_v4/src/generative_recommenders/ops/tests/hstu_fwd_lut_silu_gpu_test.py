#!/usr/bin/env python3
"""Row (120) LUT-SiLU forward: GPU bitwise unit test (gfx950 only).

Ladder-chain (v2) variant: the tree carries ship17 item (105) SPF fold fwd,
bound ON in its yambda_5b.gin (DLRM_HSTU_FP8_LAB_FWD_SPF=1), so the production
forward compiles with spf=True and every call's qkv_scale must be on the SPF
grid (score_scale = -2**j exactly, r = 1; an off-grid scale raises).

Runs the REAL launcher hstu_flydsl_mha.flydsl_hstu_attention_fwd_fp8 under the
production FlyDSL e4m3 forward env with DLRM_HSTU_FP8_LUT_SILU_FWD = "1" (LUT),
"0" (kill switch) and unset (code default), and checks every P8 byte, in two
passes:
  main  DLRM_HSTU_FP8_LAB_FWD_SPF = "0" (LUT alone): 13 off-grid scales, r in
        [1.2, 1.8), plus one repeated scale (table cache hit);
  spf   DLRM_HSTU_FP8_LAB_FWD_SPF = "1" (the production chain): 4 SPF-grid
        scales (fp8_qkv_spf.grid_scale), r = 1 tables, the LUT build compiled
        with spf=True, and knob 0 / code default compiled with item (105)'s
        kwargs (spf=True, no lut).

Inputs make every byte observable: q and k rows are nonzero only in dims 0 and
1 (random e4m3, both signs, zeros and exact cancellations), drawn so each dot
raw = q0*k0 + q1*k1 is exact both in f32 and in the gfx950 f8f6f4 MFMA. That
MFMA truncates each product toward zero at 2**(A - 13), A = the largest
operand-exponent sum ea + eb of the dot's products (fitted to job 77833 of
x4c19 iter76: its one mismatch, raw 127.4921875 read as 127.5, and the 0
mismatches of its other scales; the scaled and unscaled MFMAs agree bitwise,
fwdlab 74317). Every pair in one (sequence, head) has its q and k exponents
inside windows of 4, so the products' exponent sums differ by <= 6 and each
product is a multiple of 2**(A - 12): nothing is truncated, and make_inputs
checks it. V is the identity on the first keys of
each sequence (v[n, h, d] = 1.0 iff d == n, for n < min(L, 128)); V rows
n >= 128 are zero. So out[m, h, d] = bf16(f32(dec(P8[m, d]) * out_scale)) for
every unmasked (m, d) (the one nonzero product is exact) and 0 for d > m
(causal). Each output element is decoded back to its e4m3 byte by an
unambiguous nearest match (+0 and -0 are one class: they are
indistinguishable after the accumulation).

Both kernel paths that produce P8 are checked. N = 256 gives two m-blocks
(hstu_fwd.py BM = 128): a workgroup at m0 = 0 runs only the masked tail group
(keys 0..127, d <= m); a workgroup at m0 = 128 (the sequences of 200 and 256
rows) runs n_groups = m0 // 128 = 1 unmasked HEAD group (keys 0..127) before
its masked tail group (keys 128..255). Rows m >= 128 therefore read every
d < 128 from the head group (head elements); their tail group runs real
masked work on nonzero q/k that adds 0 through the zero V rows. Rows m < 128
are tail elements (d <= m; d > m must be zero). A second LUT launch per main
and spf scale (the "tail2" pass) makes that m0 = 128 masked tail group
observable: same q/k, V the identity on the keys n in [128, min(L, 256))
mapped to output dim n - 128 (v[n, h, n - 128] = 1.0), zero on the keys
< 128. Then out[m, d] = P8[m, 128 + d] for 128 + d <= m on the rows
m >= 128 (expected build_table()[idx(2**j * q[m] . k[128 + d])]), 0 for
128 + d > m (the causal mask inside the diagonal block), and every row
m < 128 is 0.

Variant cells (knob "1", SPF "0" as the main pass, same inputs): item (100)
RB1 off (V8SWZ and RIDER "0": lut=13 without v8_swz / rider, its own build)
and item (51)'s power-of-two P8 scale (DLRM_HSTU_FP8_FWD_PV_SCALE=pow2:
pv_s absent, S = 2**pv_k, its own build and tables) are checked bitwise
like the main sweep at 3 scales each; item (51) off (DLRM_HSTU_FP8_FWD_PV=0)
and the mask-arith kill (MASK_ARITH=0) must make the launcher decline item
(120) (one decline line, no lut kwarg, no table) and run the same kernel as
knob "0" (bitwise equal output) at 1 scale each.

Index probe (informational, never fails the test): on the v1.1 wide-spread
inputs (make_inputs_wide, the pattern of job 77833) and identity-coded
tables (coded_table) the LUT kernel reports its own 13-bit table index per
element; the counts of elements whose index differs from each dot model
(dot_models: exact, tz13, rne13, nmax13) are printed on one
"LUT_SILU_GPU_PROBE ..." line before the PASS line.

LUT expectation (ops/fp8_lut_silu.py build_table; the hstu_fwd.py `lut`
kwarg comment): y = 2**j * raw (exact: the QK MFMA E8M0 A-scale and the
truncation-free dots above), h = f16 RTZ
of y with |y| >= 65504 clamped to 65504 (v_cvt_pkrtz, MODE.FP16_OVFL),
idx = h >> 3, byte = fp8_lut_silu.build_table(score_scale, pv_s, pv_k)[idx]:
asserted exact for EVERY element, head and tail. The f16-subnormal cells are
reached; at pv_s=10.04 they all hold +-0, so this checks they map to a zero
byte, not RTZ-vs-flush. The shipped kernel is checked against
e4m3(pv_s * x / (1 + 2**x)), x = f32(raw * score_scale), at >= 99.9% (it
uses hardware exp2/rcp; this proves the decode, not the kernel).

Run (one gfx950 GPU, training container):
    python3 -B -m pytest -q -s -p no:cacheprovider <this file>
    python3 -B <this file>
On success it prints exactly one line "LUT_SILU_GPU_UNIT PASS ...":
    LUT_SILU_GPU_UNIT PASS calls=40 elems=4314440 head_elems=1331200
    tail_elems=2983240 cells=7933 lut_mism=0 ship_agree=<x> lut_builds=1
    lut_jit_entries=1 tail2_calls=13 tail2_elems=565968 tail2_mism=0
    spf_calls=12 spf_elems=1327520 spf_head_elems=409600
    spf_tail_elems=917920 spf_cells=7930 spf_lut_mism=0 spf_ship_agree=<y>
    spf_lut_builds=1 spf_lut_jit_entries=1 spf_tail2_calls=4
    spf_tail2_elems=174144 spf_tail2_mism=0 var_calls=10 var_elems=1991280
    var_mism=0 var_builds=4
(one line; x, y >= 0.999; the inputs are deterministic, seed 120), preceded
by one informational "LUT_SILU_GPU_PROBE ..." line (see _probe).
"""

import contextlib
import importlib
import importlib.util
import io
import math
import os
import random
import re
import struct
import unittest
from typing import Dict, List, Optional, Sequence, Tuple

import torch

KNOB = "DLRM_HSTU_FP8_LUT_SILU_FWD"
SPF_KNOB = "DLRM_HSTU_FP8_LAB_FWD_SPF"
# Production env of the FlyDSL e4m3 forward (ladder2 L18 yambda_5b.gin: (26)(a)
# FWD_PIPE, (48)(d) MASK_ARITH, (51) FWD_PV, (100) RB1 V8SWZ + RIDER, (105) SPF
# fold fwd, current-stream launches); DLRM_HSTU_FP8_FWD_PV_SCALE is not bound
# (unset = 10.04). The main pass overrides SPF to "0" (see the docstring).
PROD_ENV: Dict[str, str] = {
    "HSTU_ATTN_FWD": "flydsl",
    "DLRM_HSTU_FP8_FWD": "1",
    "DLRM_HSTU_FP8_FWD_PIPE": "1",
    "DLRM_HSTU_FP8_FWD_MASK_ARITH": "1",
    "DLRM_HSTU_FP8_FWD_PV": "1",
    "DLRM_HSTU_FP8_LAB_FWD_V8SWZ": "1",
    "DLRM_HSTU_FP8_LAB_FWD_RIDER": "1",
    SPF_KNOB: "1",
    "DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM": "1",
}
UNSET_ENV: Tuple[str, ...] = (
    KNOB, "DLRM_HSTU_FP8_FWD_PV_SCALE", "DLRM_HSTU_FP8_FWD_FLYDSL")
# The production compile kwargs that env gives (the launcher's
# _fp8_fwd_config(), hstu_fwd.py compile_hstu_fwd; rs comes from the q/k/v
# layout; the knob adds lut=13, hstu_flydsl_mha.fp8_fwd_lut_config), with
# item (105) on: the launcher adds spf=True per call, after _fp8_fwd_config()
# and SPF's grid check of the call's runtime scale.
PROD_CFG_OFF: Dict[str, object] = {
    "BM": 128, "BN": 32, "sched": "iglp0", "waves": 4, "fp8_mfma": "k128",
    "fp8_v": True, "epi_fold": True, "mask_split": True, "kv_prefetch": True,
    "p_direct": True, "mask_arith": 3, "pv": "k128", "pv_k": 3,
    "pv_split": True, "pf_late": True, "pv_tg": True, "pv_s": 10.04,
    "v8_swz": 1, "rider": 3, "spf": True,
}
PROD_CFG_ON: Dict[str, object] = dict(PROD_CFG_OFF, lut=13)


def without_spf(cfg: Dict[str, object]) -> Dict[str, object]:
    """The kwargs without item (105): _fp8_fwd_config() (SPF is added at the
    call site) and the compile key of the main pass (SPF "0")."""
    return {k: v for k, v in cfg.items() if k != "spf"}

def cfg_without(cfg: Dict[str, object], *keys: str) -> Dict[str, object]:
    return {k: v for k, v in cfg.items() if k not in keys}


_PV_KEYS: Tuple[str, ...] = ("pv", "pv_k", "pv_split", "pf_late", "pv_tg",
                             "pv_s")
# Variant cells (env on top of PROD_ENV with SPF "0", as the main pass;
# knob "1") and the compile kwargs
# the launcher must give them (hstu_flydsl_mha._fp8_fwd_config):
# (a) item (100) RB1 off: lut=13 without v8_swz / rider (its own build);
# (b) item (51) power-of-two P8 scale: "pow2" (= "0") leaves pv_s out of
#     the kwargs, so S = 2**pv_k = 8 (its own build and tables);
# (c) item (51) off: no pv=k128, the knob declines (bf16-P kernel);
# (d) the mask-arith kill: pv=k128 without mask_arith=3, the knob declines.
VAR_RB1_ENV: Dict[str, str] = {"DLRM_HSTU_FP8_LAB_FWD_V8SWZ": "0",
                               "DLRM_HSTU_FP8_LAB_FWD_RIDER": "0"}
VAR_RB1_CFG: Dict[str, object] = cfg_without(PROD_CFG_ON, "v8_swz", "rider", "spf")
VAR_POW2_ENV: Dict[str, str] = {"DLRM_HSTU_FP8_FWD_PV_SCALE": "pow2"}
VAR_POW2_CFG: Dict[str, object] = cfg_without(PROD_CFG_ON, "pv_s", "spf")
VAR_PV0_ENV: Dict[str, str] = {"DLRM_HSTU_FP8_FWD_PV": "0"}
VAR_PV0_CFG: Dict[str, object] = cfg_without(PROD_CFG_OFF, "v8_swz", "rider", "spf",
                                             *_PV_KEYS)
VAR_MA0_ENV: Dict[str, str] = {"DLRM_HSTU_FP8_FWD_MASK_ARITH": "0"}
VAR_MA0_CFG: Dict[str, object] = cfg_without(PROD_CFG_OFF, "mask_arith", "spf")
VAR_J: Tuple[int, ...] = (-14, -3, 1)     # cells (a), (b): main-sweep scales
VAR_KILL_J = -3                           # cells (c), (d)
DECLINE_TXT = "[hstu-fwd] DLRM_HSTU_FP8_LUT_SILU_FWD=1 declined:"
# The launcher's once-latches a variant cell may touch (restored after it).
LATCHES: Tuple[str, ...] = (
    "_fp8_fwd_pv_logged", "_fp8_fwd_v8swz_logged", "_fp8_fwd_rider_logged",
    "_fp8_fwd_lut_logged", "_fp8_fwd_mask_arith_logged",
    "_fp8_fwd_spf_logged")

H = 4
D = 128
# The last two sequences (> 128 rows) make the m0 = 128 workgroups that run
# the head group (see the module docstring); N_MAX = 256 -> m_blocks = 2.
SEQ_LENS: Tuple[int, ...] = (128, 128, 128, 128, 1, 37, 64, 100, 256, 200)
N_MAX = 256
BM = 128                      # hstu_fwd.py m-block rows (head group width)
ROW_STRIDE = 3 * H * D        # production reads q/k/v in place at rs 1536
ALPHA = 1.0 / math.sqrt(128.0)
SEED = 120
# score_scale exponents j (|score_scale| = 2**j * r): |y| = 2**j * |raw|
# reaches the f16-subnormal cells (at pv_s=10.04 they all hold +-0, so this
# checks they map to a zero byte, not RTZ-vs-flush) and the 65504 clamp cell
# and beyond (|raw| in [2**-18, 2**19]).
J_TARGETS: Tuple[int, ...] = (-15, -14, -13, -11, -9, -7, -5, -3, -2, -1, 0, 1, 2)
# The spf pass: SPF-grid scales (score_scale = -2**j exactly), from the
# scales that reach the f16-subnormal cells (-15: the AOT/ladder scale
# 0.015469998121261597; -14: F1's snapped warm-up scale) to the 65504 clamp
# (2).
SPF_J_TARGETS: Tuple[int, ...] = (-15, -14, -6, 2)
SHIP_AGREE_MIN = 0.999
# |o - c| <= DEC_TOL * |c| (bf16 RNE error < 2**-8 relative; the f32
# epilogue adds <= 2**-24).
DEC_TOL = 2.0 ** -8
DEC_MARGIN = 4.0              # second-best distance >= DEC_MARGIN * best
CLAMP_CELL = 0x7BFF >> 3      # f16 max-finite cell (every finite |y| >= 65504)
# Index probe (informational): scales (main-sweep qkv_scale of each j; -13
# is job 77833's 0.035837557166814804), coded-table shifts, the element of
# job 77833's one mismatch (seq 8 = tokens 714.., m=136, head 0, key 96).
PROBE_J: Tuple[int, ...] = (-15, -13, 2)
PROBE_SHIFTS: Tuple[int, ...] = (0, 6, 12)
CODE_BASE = 0x08              # e4m3 2**-6: codes 0x08..0x47 (positive normal)
PROBE_ELEM: Tuple[int, int, int] = (850, 0, 96)
PROBE_ELEM_J = -13
PROBE_ELEM_IDX = 1151         # job 77833 read table[1151] there (exact: 1150)
DOT_MODELS: Tuple[str, ...] = ("exact", "tz13", "rne13", "nmax13")
TABLE_LINE = re.compile(
    r"\[hstu-fwd\] DLRM_HSTU_FP8_LUT_SILU_FWD table built: "
    r"score_scale=(\S+) \(0x([0-9a-f]{8})\) j=(-?\d+) r=(\S+) S=(\S+) "
    r"bytes=8192 crc32=([0-9a-f]{8})")


# --------------------------------------------------------------------------
# Pure helpers (CPU; importable without a GPU, flydsl or the launcher).
# --------------------------------------------------------------------------
def lut_module():
    """generative_recommenders.ops.fp8_lut_silu (the item (120) table module:
    score_scale32, split_scale, build_table), imported lazily."""
    return importlib.import_module("generative_recommenders.ops.fp8_lut_silu")


def spf_module():
    """generative_recommenders.ops.fp8_qkv_spf (item (105)), imported lazily."""
    return importlib.import_module("generative_recommenders.ops.fp8_qkv_spf")


def spf_scale_for_j(j: int, alpha: float = ALPHA) -> float:
    """The SPF grid point of exponent j (fp8_qkv_spf.grid_scale: the smallest
    f32 s whose score coefficient is exactly 2**j), so score_scale = -2**j."""
    s = spf_module().grid_scale(j, alpha)
    if s is None:
        raise AssertionError("no SPF grid point for j=%d" % j)
    return s


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def mul32(a: float, b: float) -> float:
    return f32(f32(a) * f32(b))   # f32*f32 is exact in f64: one RNE


def e4m3_value(b: int) -> float:
    """e4m3fn decode (0x7F / 0xFF = NaN)."""
    if (b & 0x7F) == 0x7F:
        return float("nan")
    e, m = (b >> 3) & 0xF, b & 7
    v = math.ldexp(m, -9) if e == 0 else math.ldexp(8 + m, e - 10)
    return -v if b & 0x80 else v


_MAGS = torch.tensor([e4m3_value(b) for b in range(0x7F)], dtype=torch.float64)
_POW2 = torch.tensor([math.ldexp(1.0, k) for k in range(-64, 65)],
                     dtype=torch.float64)


def merge_zero(codes: torch.Tensor) -> torch.Tensor:
    """-0 (0x80) and +0 (0x00) are one class after the P@V accumulation."""
    return torch.where(codes == 0x80, torch.zeros_like(codes), codes)


def out_scale32(qkv_scale: float, n_max: int, pv_s: float, pv_k: int) -> float:
    """The kernel epilogue's f32 out_scale (hstu_fwd.py fp8_v + epi_fold + PV8):
    ((qkv_scale * (inv_max * f32(-1/LOG2E))) * f32(1/pv_s or 2**-pv_k))."""
    log2e = 1.4426950408889634
    fold = mul32(f32(1.0 / float(n_max)), f32(-1.0 / log2e))
    os_ = mul32(f32(qkv_scale), fold)
    return mul32(os_, f32(1.0 / pv_s if pv_s else 2.0 ** (-pv_k)))


def qkv_scale_for_j(j: int, k: int, alpha: float = ALPHA) -> float:
    """An f32 qkv_scale whose score_scale has exponent j (r away from 1, 2)."""
    r_t = 1.2 + 0.6 * ((k * 0.6180339887498949) % 1.0)
    return f32(math.sqrt(math.ldexp(r_t, j) / (alpha * 1.4426950408889634)))


def _e4m3_exp(b: int) -> int:
    """Exponent of a finite nonzero e4m3 byte as the MFMA sees it (the
    subnormals share the first binade's -6): value = M * 2**(e - 3), M < 16."""
    ef = (b >> 3) & 0xF
    return -6 if ef == 0 else ef - 7


def _byte_window_class(b: int) -> int:
    """ulp exponent of a finite nonzero e4m3 byte: value = M * 2**u, M < 16."""
    ef = (b >> 3) & 0xF
    return -9 if ef <= 1 else ef - 10


_FINITE_NZ: Tuple[int, ...] = tuple(
    b for b in range(256) if (b & 0x7F) not in (0x00, 0x7F))


def make_inputs(seed: int = SEED, seq_lens: Sequence[int] = SEQ_LENS,
                heads: int = H) -> Dict[str, object]:
    """q/k bytes of dims 0, 1 (uint8 [T, heads, 2]) and the sequence offsets.

    Per (sequence, head) the q bytes come from one ulp-exponent window of 4
    and the k bytes from another (window starts -9..2, so the classes reach
    the e4m3 top binade and its subnormals), so for every (m, n) the two
    products' exponent sums ea + eb differ by <= 6 (asserted below). Then
    raw = q0*k0 + q1*k1 is exact in f32 (exact_raw() re-checks it) and in
    the gfx950 f8f6f4 MFMA, which truncates each product at 2**(A - 13),
    A = max(ea + eb) (see the module docstring): every product is a
    multiple of 2**(A - 12). Zeros, all-zero q rows and exact cancellations
    (q1 == q0 with k1 == -k0) give raw = 0."""
    rng = random.Random(seed)
    tot = sum(seq_lens)
    q2 = torch.zeros(tot, heads, 2, dtype=torch.uint8)
    k2 = torch.zeros(tot, heads, 2, dtype=torch.uint8)
    offsets = [0]
    for L in seq_lens:
        offsets.append(offsets[-1] + L)
    for si, L in enumerate(seq_lens):
        s0 = offsets[si]
        for h in range(heads):
            a = rng.randint(-9, 2)
            b = rng.randint(-9, 2)
            qset = [x for x in _FINITE_NZ if a <= _byte_window_class(x) <= a + 3]
            kset = [x for x in _FINITE_NZ if b <= _byte_window_class(x) <= b + 3]
            for n in range(L):
                kv = [0 if rng.random() < 0.12 else rng.choice(kset)
                      for _ in range(2)]
                if rng.random() < 0.08 and kv[0]:
                    kv[1] = kv[0] ^ 0x80
                qv = [0 if rng.random() < 0.12 else rng.choice(qset)
                      for _ in range(2)]
                u = rng.random()
                if u < 0.05:
                    qv = [0, 0]
                elif u < 0.13 and qv[0]:
                    qv[1] = qv[0]
                q2[s0 + n, h, 0], q2[s0 + n, h, 1] = qv[0], qv[1]
                k2[s0 + n, h, 0], k2[s0 + n, h, 1] = kv[0], kv[1]
    # Truncation-free dots: within each (sequence, head) the exponents of the
    # nonzero q bytes span <= 3 and so do those of the nonzero k bytes.
    for si, L in enumerate(seq_lens):
        s0 = offsets[si]
        for h in range(heads):
            for t2, nr in ((q2, L), (k2, L)):
                e = [_e4m3_exp(int(x)) for x in t2[s0:s0 + nr, h].flatten()
                     if int(x) & 0x7F]
                if e and max(e) - min(e) > 3:
                    raise AssertionError("input generator: exponent window > 4")
    return {"q2": q2, "k2": k2, "offsets": offsets,
            "seq_lens": tuple(seq_lens), "heads": heads}


def make_inputs_wide(seed: int = SEED, seq_lens: Sequence[int] = SEQ_LENS,
                     heads: int = H) -> Dict[str, object]:
    """The v1.1 generator, kept for the index probe only (never gating):
    ulp-exponent windows of 8 (starts -9..-2), so the two products'
    exponent sums of one dot may differ by up to 14 and the gfx950 QK MFMA
    may round them (job 77833: token 850, head 0, key 96 at j = -13). The
    dots stay exact in f32 (exact_raw() checks it). Same seed, same draws
    as the v1.1 test: the same bytes."""
    rng = random.Random(seed)
    tot = sum(seq_lens)
    q2 = torch.zeros(tot, heads, 2, dtype=torch.uint8)
    k2 = torch.zeros(tot, heads, 2, dtype=torch.uint8)
    offsets = [0]
    for L in seq_lens:
        offsets.append(offsets[-1] + L)
    for si, L in enumerate(seq_lens):
        s0 = offsets[si]
        for h in range(heads):
            a = rng.randint(-9, -2)
            b = rng.randint(-9, -2)
            qset = [x for x in _FINITE_NZ if a <= _byte_window_class(x) <= a + 7]
            kset = [x for x in _FINITE_NZ if b <= _byte_window_class(x) <= b + 7]
            for n in range(L):
                kv = [0 if rng.random() < 0.12 else rng.choice(kset)
                      for _ in range(2)]
                if rng.random() < 0.08 and kv[0]:
                    kv[1] = kv[0] ^ 0x80
                qv = [0 if rng.random() < 0.12 else rng.choice(qset)
                      for _ in range(2)]
                u = rng.random()
                if u < 0.05:
                    qv = [0, 0]
                elif u < 0.13 and qv[0]:
                    qv[1] = qv[0]
                q2[s0 + n, h, 0], q2[s0 + n, h, 1] = qv[0], qv[1]
                k2[s0 + n, h, 0], k2[s0 + n, h, 1] = kv[0], kv[1]
    return {"q2": q2, "k2": k2, "offsets": offsets,
            "seq_lens": tuple(seq_lens), "heads": heads}


def decode_bytes_f64(b: torch.Tensor) -> torch.Tensor:
    table = torch.tensor([0.0 if (x & 0x7F) == 0x7F else e4m3_value(x)
                          for x in range(256)], dtype=torch.float64)
    return table[b.to(torch.int64)]


def exact_raw(inp: Dict[str, object], key_off: int = 0) -> torch.Tensor:
    """f64 [T, heads, D]: raw[s0+m, h, n] = q[s0+m, h] . k[s0+key_off+n, h]
    for the keys key_off + n < min(L, key_off + D) the identity V of
    qkv_buffer(inp, key_off) exposes (0 elsewhere). Raises unless every
    value is exact in f32."""
    q = decode_bytes_f64(inp["q2"])
    k = decode_bytes_f64(inp["k2"])
    offs, lens = inp["offsets"], inp["seq_lens"]
    raw = torch.zeros(q.shape[0], q.shape[1], D, dtype=torch.float64)
    for si, L in enumerate(lens):
        s0, nk = offs[si], min(L - key_off, D)
        if nk <= 0:
            continue
        qs = q[s0:s0 + L]                                    # [L, H, 2]
        ks = k[s0 + key_off:s0 + key_off + nk]               # [nk, H, 2]
        r = (qs[:, None, :, 0] * ks[None, :, :, 0]
             + qs[:, None, :, 1] * ks[None, :, :, 1])        # [m, n, H]
        raw[s0:s0 + L, :, :nk] = r.permute(0, 2, 1)
    raw = raw + 0.0                                          # -0 -> +0
    if not bool(torch.equal(raw.to(torch.float32).to(torch.float64), raw)):
        raise AssertionError("input generator: a q.k dot is not exact in f32")
    return raw


def valid_mask(seq_lens: Sequence[int], heads: int = H,
               key_off: int = 0) -> torch.Tensor:
    """bool [T, heads, D]: True where key_off + d <= m (the causal, unmasked
    P of key key_off + d; key_off = 0: d <= m, every d < D on the rows
    m >= BM; key_off = BM, the tail2 pass: rows m >= BM, d <= m - BM)."""
    rows = []
    for L in seq_lens:
        m = torch.arange(L)[:, None]
        rows.append(torch.arange(D)[None, :] + key_off <= m)
    return torch.cat(rows, 0)[:, None, :].expand(-1, heads, -1).contiguous()


def head_mask(seq_lens: Sequence[int], heads: int = H) -> torch.Tensor:
    """bool [T, heads, D]: True on the rows m >= BM of each sequence, whose
    out[m, d] (d < D = BM) is a byte of the m0 = 128 workgroup's unmasked
    head group (hstu_fwd.py n_groups = m0 // 128); False on the tail-group
    rows m < BM."""
    rows = torch.cat([torch.arange(L) >= BM for L in seq_lens], 0)
    return rows[:, None, None].expand(-1, heads, D).contiguous()


def jit_entries(fn) -> Optional[int]:  # pyre-ignore[2]
    """FlyDSL JIT artifacts held by one compile_hstu_fwd callable: the size
    of its launcher's JitFunction._mem_cache (flydsl 0.1.1
    compiler/jit_function.py; one entry per distinct argument-type key,
    compiled or loaded from the disk cache; scales are runtime Float32
    values, not part of the key). None when the attribute is absent."""
    cache = getattr(getattr(fn, "launch", None), "_mem_cache", None)
    return len(cache) if isinstance(cache, dict) else None


def qkv_buffer(inp: Dict[str, object], key_off: int = 0) -> torch.Tensor:
    """uint8 [T, 3*heads*D] = v | q | k column slices (row stride 1536, the
    production in-place layout). V is the identity on the keys
    key_off <= n < min(L, key_off + D) of each sequence, mapped to output
    dim n - key_off (0x38 = 1.0); every other V row is zero. key_off = 0:
    a key >= 128 (the tail group of an m0 = 128 workgroup) adds 0 to every
    output; key_off = BM (the tail2 pass): only that tail group shows."""
    q2, k2 = inp["q2"], inp["k2"]
    tot, heads = q2.shape[0], q2.shape[1]
    hd = heads * D
    buf = torch.zeros(tot, 3 * hd, dtype=torch.uint8)
    v = buf[:, 0:hd].view(tot, heads, D)
    q = buf[:, hd:2 * hd].view(tot, heads, D)
    k = buf[:, 2 * hd:3 * hd].view(tot, heads, D)
    q[:, :, 0:2] = q2
    k[:, :, 0:2] = k2
    offs = inp["offsets"]
    for si, L in enumerate(inp["seq_lens"]):
        if L > key_off:
            n = torch.arange(key_off, min(L, key_off + D))
            v[offs[si] + n, :, n - key_off] = 0x38
    return buf


def _exp_t(b: torch.Tensor) -> torch.Tensor:
    """int64 _e4m3_exp of every byte of b (zeros and subnormals: -6)."""
    ef = (b.to(torch.int64) >> 3) & 0xF
    return torch.where(ef == 0, torch.full_like(ef, -6), ef - 7)


def dot_models(inp: Dict[str, object]) -> Dict[str, torch.Tensor]:
    """f64 [T, heads, D]: the dot q[s0+m, h] . k[s0+n, h] (keys n < min(L,
    D), as exact_raw) under four models of the gfx950 f8f6f4 QK MFMA, for
    the index probe. With p_c = q_c * k_c (c = 0, 1; exact in f64), ea / eb
    the operand exponents (_e4m3_exp: e4m3 subnormals have exponent -6) and
    the max taken over the dot's NONZERO products only (a dot whose
    products are all zero is 0 in every model):
      exact   p_0 + p_1.
      tz13    A = max(ea + eb); each p_c truncated toward zero to a
              multiple of 2**(A - 13); then summed exactly (WBND's fit of
              job 77833).
      rne13   as tz13 with round-half-to-even to a multiple of 2**(A - 13).
      nmax13  E = max floor(log2 |p_c|) (the normalized leading-bit position
              of the largest product, E = ea + eb or ea + eb + 1); each p_c
              rounded half-to-even to a multiple of 2**(E - 13); summed
              exactly.
    All three rounded models reproduce job 77833 (j = -15, -14: 0, j = -13:
    1 mismatch, at token 850 head 0 key 96, raw 127.4921875 -> 127.5);
    rounding the exact SUM to 13 bits below its own leading bit does not
    (it keeps 127.4921875)."""
    q = decode_bytes_f64(inp["q2"])
    k = decode_bytes_f64(inp["k2"])
    eq, ek = _exp_t(inp["q2"]), _exp_t(inp["k2"])
    offs, lens = inp["offsets"], inp["seq_lens"]
    neg = -100000.0
    shape = (q.shape[0], q.shape[1], D)
    p = [torch.zeros(shape, dtype=torch.float64) for _ in range(2)]
    es = [torch.full(shape, neg, dtype=torch.float64) for _ in range(2)]
    for si, L in enumerate(lens):
        s0, nk = offs[si], min(L, D)
        for c in (0, 1):
            pc = q[s0:s0 + L, None, :, c] * k[None, s0:s0 + nk, :, c]
            ec = (eq[s0:s0 + L, None, :, c]
                  + ek[None, s0:s0 + nk, :, c]).to(torch.float64)
            p[c][s0:s0 + L, :, :nk] = pc.permute(0, 2, 1)
            es[c][s0:s0 + L, :, :nk] = torch.where(
                pc != 0, ec, torch.full_like(ec, neg)).permute(0, 2, 1)
    a_sum = torch.maximum(es[0], es[1])

    def nexp(x: torch.Tensor) -> torch.Tensor:
        _, e = torch.frexp(x.abs())
        return torch.where(x != 0, (e - 1).to(torch.float64),
                           torch.full_like(x, neg))

    e_max = torch.maximum(nexp(p[0]), nexp(p[1]))

    def at(x: torch.Tensor, anchor: torch.Tensor, rne: bool) -> torch.Tensor:
        g = torch.pow(torch.full_like(x, 2.0), (anchor - 13).clamp(min=-400))
        return (torch.round(x / g) if rne else torch.trunc(x / g)) * g

    return {"exact": p[0] + p[1] + 0.0,
            "tz13": at(p[0], a_sum, False) + at(p[1], a_sum, False) + 0.0,
            "rne13": at(p[0], a_sum, True) + at(p[1], a_sum, True) + 0.0,
            "nmax13": at(p[0], e_max, True) + at(p[1], e_max, True) + 0.0}


def coded_table(shift: int) -> bytes:
    """Identity-coded table for the index probe: entry i is the positive
    normal e4m3 byte CODE_BASE + ((i >> shift) & 0x3F), so three launches
    (PROBE_SHIFTS) recover the 13-bit index of every element."""
    return bytes(CODE_BASE + ((i >> shift) & 0x3F) for i in range(8192))


def f16_rtz_bits(y: torch.Tensor) -> torch.Tensor:
    """int64 f16 bits of RTZ(y), |y| >= 65504 clamped to 65504 (v_cvt_pkrtz
    with MODE.FP16_OVFL; RTZ never rounds up). y: f64, finite."""
    a = y.abs()
    _, e = torch.frexp(a)                       # a = m * 2**e, m in [0.5, 1)
    ue = torch.clamp(e.to(torch.int64) - 1, min=-14) - 10   # f16 ulp exponent
    ulp = _POW2[torch.clamp(ue, -64, 64) + 64]
    t = torch.floor(a / ulp) * ulp              # exact: power-of-two scaling
    t = torch.where(a >= 65504.0, torch.full_like(t, 65504.0), t)
    bits = t.to(torch.float16).view(torch.int16).to(torch.int64) & 0x7FFF
    return bits | (torch.signbit(y).to(torch.int64) << 15)


def lut_index(y: torch.Tensor) -> torch.Tensor:
    """13-bit table index = f16 sign | exp5 | top 7 mantissa bits."""
    return f16_rtz_bits(y) >> 3


def lut_expected(raw: torch.Tensor, j: int, table: bytes
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(codes, idx): the expected P8 byte class of every element of the LUT
    kernel, y = 2**j * raw exactly."""
    if len(table) != 8192:
        raise AssertionError("build_table returned %d bytes" % len(table))
    y = raw * math.ldexp(1.0, j)
    idx = lut_index(y)
    tbl = torch.tensor(list(table), dtype=torch.int64)
    return merge_zero(tbl[idx]), idx


def e4m3_rne_sat_bytes(p: torch.Tensor) -> torch.Tensor:
    """e4m3fn bytes of f64 p: RNE, saturating to +-448, subnormals, signed
    zero, NaN -> 0x7F | sign."""
    a = p.abs()
    hi = torch.searchsorted(_MAGS, a.contiguous())
    hi_c = torch.clamp(hi, max=0x7E)
    lo_c = torch.clamp(hi - 1, min=0)
    d_hi = _MAGS[hi_c] - a
    d_lo = a - _MAGS[lo_c]
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi_c % 2 == 0))
    mag = torch.where(pick_hi, hi_c, lo_c)
    mag = torch.where(a >= 448.0, torch.full_like(mag, 0x7E), mag)
    mag = torch.where(torch.isnan(p), torch.full_like(mag, 0x7F), mag)
    return mag | (torch.signbit(p).to(torch.int64) << 7)


def ship_expected(raw: torch.Tensor, ss: float, S: float) -> torch.Tensor:
    """Expected P8 byte class of the shipped kernel (exact math model):
    x = f32(raw * score_scale), P8 = e4m3_sat(S * x / (1 + 2**x))."""
    x = (raw * ss).to(torch.float32).to(torch.float64)   # one RNE: mul32
    p = S * x / (1.0 + torch.exp2(x))
    return merge_zero(e4m3_rne_sat_bytes(p))


def _candidates(os_: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sorted (value, byte) of dec(b) * os for every finite byte, 0x80 merged
    into the 0x00 class."""
    pairs = sorted((e4m3_value(b) * os_, b) for b in range(256)
                   if (b & 0x7F) != 0x7F and b != 0x80)
    return (torch.tensor([p[0] for p in pairs], dtype=torch.float64),
            torch.tensor([p[1] for p in pairs], dtype=torch.int64))


def decode(o: torch.Tensor, os_: float
           ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Nearest-match decode of f64 outputs o to byte classes.

    Returns (codes, ok, worst best/second distance ratio). ok is False where
    the match is not unambiguous: the best candidate is farther than DEC_TOL
    relative, the second best is closer than DEC_MARGIN times the best, or a
    nonzero value would decode to the zero class (or o is not finite)."""
    vals, bytes_ = _candidates(os_)
    n = vals.numel()
    flat = o.reshape(-1).to(torch.float64).contiguous()
    pos = torch.searchsorted(vals, flat)
    inf = torch.full_like(flat, float("inf"))
    best_d, sec_d = inf.clone(), inf.clone()
    best_i = torch.zeros_like(pos)
    for off in (-2, -1, 0, 1):
        ii = pos + off
        ok_i = (ii >= 0) & (ii < n)
        iic = torch.clamp(ii, 0, n - 1)
        d = torch.where(ok_i, (flat - vals[iic]).abs(), inf)
        d = torch.where(torch.isnan(d), inf, d)
        better = d < best_d
        sec_d = torch.where(better, best_d, torch.minimum(sec_d, d))
        best_i = torch.where(better, iic, best_i)
        best_d = torch.where(better, d, best_d)
    c1 = vals[best_i]
    ok = ((best_d <= DEC_TOL * c1.abs()) & (sec_d >= DEC_MARGIN * best_d)
          & torch.isfinite(flat) & torch.isfinite(best_d))
    ratio = torch.where(sec_d > 0, best_d / sec_d, torch.zeros_like(sec_d))
    worst = float(ratio[ok].max()) if bool(ok.any()) else float("nan")
    return bytes_[best_i].reshape(o.shape), ok.reshape(o.shape), worst


def check_output(label: str, o: torch.Tensor, os_: float,
                 valid: torch.Tensor, want: Optional[torch.Tensor],
                 idx: Optional[torch.Tensor] = None,
                 head: Optional[torch.Tensor] = None) -> Dict[str, object]:
    """Decode o (f64 [T, heads, D]) and compare with `want` byte classes on
    the valid (d <= m) elements. Raises AssertionError on an ambiguous decode
    or a nonzero masked element; returns the codes and the mismatch count
    (want None: decode only), split into head-group (head mask) and
    tail-group elements when `head` is given."""
    masked_bad = (~valid) & (o != 0)
    if bool(masked_bad.any()):
        w = masked_bad.nonzero()[:5].tolist()
        raise AssertionError(
            "%s: %d masked (d > m) elements are nonzero, first [t, h, d] %s "
            "values %s" % (label, int(masked_bad.sum()), w,
                           [float(o[t, h, d]) for t, h, d in w]))
    codes, ok, worst = decode(o, os_)
    amb = valid & ~ok
    if bool(amb.any()):
        w = amb.nonzero()[:5].tolist()
        raise AssertionError(
            "%s: %d outputs do not decode unambiguously to an e4m3 byte "
            "(os=%r), first [t, h, d] %s values %s" % (
                label, int(amb.sum()), os_, w,
                [float(o[t, h, d]) for t, h, d in w]))
    hd = head if head is not None else torch.zeros_like(valid)
    res = {"codes": codes, "worst_ratio": worst,
           "n_valid": int(valid.sum()), "n_head": int((valid & hd).sum())}
    if want is not None:
        bad = valid & (codes != want)
        res["mism"] = int(bad.sum())
        res["head_mism"] = int((bad & hd).sum())
        res["first_mism"] = [
            {"thd": t_h_d, "got": int(codes[tuple(t_h_d)]),
             "want": int(want[tuple(t_h_d)]),
             "idx": (int(idx[tuple(t_h_d)]) if idx is not None else None),
             "group": "head" if bool(hd[tuple(t_h_d)]) else "tail",
             "value": float(o[tuple(t_h_d)])}
            for t_h_d in bad.nonzero()[:8].tolist()]
    return res


def parse_table_lines(lines: Sequence[str]) -> List[Dict[str, object]]:
    out = []
    for ln in lines:
        m = TABLE_LINE.search(ln)
        if m:
            out.append({"bits": int(m.group(2), 16), "j": int(m.group(3)),
                        "r": float(m.group(4)), "S": float(m.group(5)),
                        "crc": int(m.group(6), 16), "line": ln})
    return out


# --------------------------------------------------------------------------
# GPU test
# --------------------------------------------------------------------------
def _skip_reason() -> Optional[str]:
    if not torch.cuda.is_available():
        return "no GPU visible to torch"
    if importlib.util.find_spec("flydsl") is None:
        return "flydsl is not importable"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
               or "")
    if not arch.startswith("gfx950"):
        return "device arch %r is not gfx950" % arch
    return None


class HstuFwdLutSiluGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        why = _skip_reason()
        if why:
            raise unittest.SkipTest("LUT-SiLU GPU unit test skipped: " + why)
        keys = set(PROD_ENV) | set(UNSET_ENV) | {"FLYDSL_GPU_ARCH"}
        saved = {k: os.environ.get(k) for k in keys}

        def _restore_env() -> None:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        cls.addClassCleanup(_restore_env)
        os.environ.setdefault("FLYDSL_GPU_ARCH", "gfx950")
        for k in UNSET_ENV:
            os.environ.pop(k, None)
        os.environ.update(PROD_ENV)
        from generative_recommenders.ops.flydsl import hstu_flydsl_mha as mha

        cls.mha = mha
        cls.lut = lut_module()
        cls.kern = mha._load_fwd_kernel()
        if cls.kern is None:
            raise AssertionError("hstu_fwd import failed: %r"
                                 % (mha._fwd_kernel_error,))
        for name in ("_fwd_lut_tables", "_fwd_lut_table",
                     "fp8_fwd_lut_config"):
            if not hasattr(mha, name):
                raise AssertionError(
                    "launcher has no %s: the item (120) launcher code "
                    "(hstu_flydsl_mha.fp8_fwd_lut_config, _fwd_lut_table) "
                    "is missing" % name)
        # A fresh table cache, so every scale must build (and print) once.
        saved_tables = dict(mha._fwd_lut_tables)
        mha._fwd_lut_tables.clear()

        def _restore_tables() -> None:
            mha._fwd_lut_tables.clear()
            mha._fwd_lut_tables.update(saved_tables)

        cls.addClassCleanup(_restore_tables)
        cls.fresh_lut = not getattr(mha, "_fp8_fwd_lut_logged", False)
        # Count compiles: wrap the module attribute the launcher calls.
        orig = cls.kern.compile_hstu_fwd
        cls.records = []

        def counting(*a, **kw):
            info = getattr(orig, "cache_info", None)
            m0 = info().misses if info else None
            fn = orig(*a, **kw)
            m1 = info().misses if info else None
            cls.records.append({"kw": dict(kw), "fn": fn, "misses": (
                None if m0 is None else m1 - m0)})
            return fn

        cls.kern.compile_hstu_fwd = counting

        def _restore_compile() -> None:
            cls.kern.compile_hstu_fwd = orig

        cls.addClassCleanup(_restore_compile)
        cls.lines = []

    def _knob(self, val: Optional[str]) -> None:
        if val is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = val

    def _config(self, val: Optional[str]) -> dict:
        self._knob(val)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cfg = self.mha._fp8_fwd_config()
        self.lines.extend(buf.getvalue().splitlines())
        return cfg

    def _launch(self, val: Optional[str], qkv, offs, s: float, spf: str,
                record: bool = True):
        self._knob(val)
        os.environ[SPF_KNOB] = spf
        q, k, v = qkv
        n0 = len(self.records)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = self.mha.flydsl_hstu_attention_fwd_fp8(
                q, k, v, offs, N_MAX, ALPHA, s)
            torch.cuda.synchronize()
        lines = buf.getvalue().splitlines()
        if record:
            self.lines.extend(lines)
        recs = self.records[n0:]
        self.assertEqual(len(recs), 1, "knob=%r spf=%r: %d compile_hstu_fwd "
                         "calls" % (val, spf, len(recs)))
        self.assertEqual(tuple(out.shape), (q.shape[0], H, D))
        self.assertEqual(out.dtype, torch.bfloat16)
        recs[0]["jit"] = jit_entries(recs[0]["fn"])   # after this launch
        return out.detach().cpu(), recs[0], lines

    def _fail(self, msg: str) -> None:
        tail = "\n".join(self.lines[-30:])
        raise AssertionError(msg + "\n--- launcher stdout (tail) ---\n" + tail)

    @staticmethod
    def _qkv(buf: torch.Tensor):
        """(q, k, v) e4m3 [T, H, D] views of a qkv_buffer() on the device."""
        f8 = buf.view(torch.float8_e4m3fn)
        tot, hd = buf.shape[0], H * D
        return (f8[:, hd:2 * hd].view(tot, H, D),
                f8[:, 2 * hd:3 * hd].view(tot, H, D),
                f8[:, 0:hd].view(tot, H, D))

    @contextlib.contextmanager
    def _env(self, env: Dict[str, str], redecline: bool = False):
        """One variant cell: `env` on top of the production env; restores
        it, the knob and the launcher's once-latches (LATCHES) afterwards.
        redecline: forget the knob's decline latch first, so the cell's own
        decline line prints (once)."""
        saved = {k: os.environ.get(k) for k in set(env) | {KNOB}}
        latches = []
        for name in LATCHES:
            obj = getattr(self.mha, name, None)
            if isinstance(obj, (set, dict)):
                latches.append((obj, obj.copy()))
        try:
            if redecline:
                self.mha._fp8_fwd_lut_logged.discard("declined")
            os.environ.update(env)
            yield
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            for obj, snap in latches:
                obj.clear()
                obj.update(snap)

    def _variant_cells(self, qkv, offs, raw, valid, head, scales,
                       expect_tables, main_fns) -> Dict[str, int]:
        """Item (100) off and item (51)'s power-of-two scale (bitwise LUT
        checks), item (51) off and the mask-arith kill (the launcher
        declines item (120)), all with the knob "1"; see the VAR_*
        constants. Returns the PASS-line counts."""
        lut = self.lut
        n = {"calls": 0, "elems": 0}
        fns: Dict[str, set] = {}
        misses: Dict[str, List[Optional[int]]] = {}

        def note(cell: str, rec) -> None:
            fns.setdefault(cell, set()).add(id(rec["fn"]))
            misses.setdefault(cell, []).append(rec["misses"])
            n["calls"] += 1

        def kw_of(rec) -> Dict[str, object]:
            kw = dict(rec["kw"])
            if kw.pop("rs", None) != ROW_STRIDE:
                self._fail("variant: rs=%r, want %d" % (rec["kw"].get("rs"),
                                                        ROW_STRIDE))
            return kw

        for cell, env, want_cfg in (("rb1_off", VAR_RB1_ENV, VAR_RB1_CFG),
                                    ("pow2", VAR_POW2_ENV, VAR_POW2_CFG)):
            with self._env(env):
                cfg = self._config("1")
                if cfg != want_cfg:
                    self._fail("%s (%r): knob 1 cfg %r != %r"
                               % (cell, env, cfg, want_cfg))
                pv_s, pv_k = float(cfg.get("pv_s", 0.0)), int(cfg["pv_k"])
                S = pv_s if pv_s else float(2 ** pv_k)
                if cell == "pow2" and (pv_s != 0.0 or S == PROD_CFG_ON["pv_s"]):
                    self._fail("pow2: pv_s=%r S=%r (want 0.0 and 2**pv_k)"
                               % (pv_s, S))
                for j in VAR_J:
                    s = scales[J_TARGETS.index(j)]
                    ss = lut.score_scale32(s, ALPHA)
                    table = lut.build_table(ss, pv_s, pv_k)
                    o, rec, lines = self._launch("1", qkv, offs, s, "0")
                    note(cell, rec)
                    if kw_of(rec) != want_cfg or getattr(rec["fn"], "lut",
                                                         None) != 13:
                        self._fail("%s j=%d: compiled kwargs %r call.lut=%r "
                                   "(want %r, 13)" % (
                                       cell, j, rec["kw"],
                                       getattr(rec["fn"], "lut", None),
                                       want_cfg))
                    built = parse_table_lines(lines)
                    if cell == "pow2":
                        # The launcher builds this cfg's table (new S).
                        expect_tables[(lut.bits32(ss), S)] = (
                            j, lut.table_crc32(table))
                        if len(built) != 1 or built[0]["S"] != S:
                            self._fail("pow2 j=%d: table lines %r (want one "
                                       "with S=%g)" % (
                                           j, [b["line"] for b in built], S))
                    elif built:
                        self._fail("%s j=%d: unexpected table lines %r"
                                   % (cell, j, [b["line"] for b in built]))
                    want, idx = lut_expected(raw, j, table)
                    res = check_output(
                        "%s LUT j=%d" % (cell, j), o.to(torch.float64),
                        out_scale32(s, N_MAX, pv_s, pv_k), valid, want, idx,
                        head)
                    if res["mism"]:
                        self._fail("%s LUT j=%d qkv_scale=%r: %d of %d P8 "
                                   "bytes (%d of %d head-group) != "
                                   "build_table(pv_s=%r, pv_k=%d)[idx(y)]; "
                                   "first %r" % (
                                       cell, j, s, res["mism"], res["n_valid"],
                                       res["head_mism"], res["n_head"], pv_s,
                                       pv_k, res["first_mism"]))
                    n["elems"] += res["n_valid"]
        for cell, env, want_cfg in (("pv0", VAR_PV0_ENV, VAR_PV0_CFG),
                                    ("ma0", VAR_MA0_ENV, VAR_MA0_CFG)):
            with self._env(env, redecline=True):
                n0 = len(self.lines)
                cfg = self._config("1")
                if cfg != want_cfg or "lut" in cfg:
                    self._fail("%s (%r): knob 1 cfg %r != %r (the launcher "
                               "must decline item (120))"
                               % (cell, env, cfg, want_cfg))
                s = scales[J_TARGETS.index(VAR_KILL_J)]
                o1, rec1, _ = self._launch("1", qkv, offs, s, "0")
                o0, rec0, _ = self._launch("0", qkv, offs, s, "0")
                note(cell, rec1)
                note(cell, rec0)
                cell_lines = self.lines[n0:]
            for rec in (rec1, rec0):
                if (kw_of(rec) != want_cfg
                        or getattr(rec["fn"], "lut", 0) != 0):
                    self._fail("%s: compiled kwargs %r call.lut=%r (want %r, "
                               "no lut)" % (cell, rec["kw"],
                                            getattr(rec["fn"], "lut", 0),
                                            want_cfg))
            declines = [ln for ln in cell_lines if DECLINE_TXT in ln]
            if len(declines) != 1 or parse_table_lines(cell_lines):
                self._fail("%s: %d '%s' lines (want 1) and %d table lines "
                           "(want 0)" % (cell, len(declines), DECLINE_TXT,
                                         len(parse_table_lines(cell_lines))))
            if rec1["fn"] is not rec0["fn"] or not torch.equal(
                    o1.view(torch.int16), o0.view(torch.int16)):
                self._fail("%s: knob 1 (declined) != knob 0 (want the same "
                           "callable and bitwise-equal output)" % cell)
        # One build per cell, distinct from each other and the main pass.
        ids = [next(iter(v)) for v in fns.values() if len(v) == 1]
        if (len(ids) != len(fns) or len(set(ids)) != len(ids)
                or set(ids) & set(main_fns)):
            self._fail("variant compile_hstu_fwd callables %r (want one per "
                       "cell, all distinct and new)"
                       % {c: len(v) for c, v in fns.items()})
        for cell, ms in misses.items():
            if ms[0] is not None and (ms[0] not in (0, 1) or any(ms[1:])):
                self._fail("%s: compile_hstu_fwd misses per call %r (want "
                           "first in (0, 1), then 0)" % (cell, ms))
        n["builds"] = len(fns)
        return n

    def _probe(self, offs, scales) -> str:
        """Index probe (informational: never fails the test). On the wide
        inputs (make_inputs_wide), identity V and identity-coded tables
        (coded_table, injected through the module-level build_table the
        launcher calls, with the table cache cleared before each launch)
        the LUT kernel writes its own 13-bit index per valid element. Counts
        the elements whose index differs from lut_index(2**j * dot) of each
        dot_models() model, per PROBE_J scale. Its table lines stay out of
        self.lines; build_table and the table cache are restored. The line
        never carries exception text (the GPU kit's readout rejects some
        words in any output line): an unrunnable probe prints
        "LUT_SILU_GPU_PROBE status=unavailable reason=<stage code>"."""
        lut, mha = self.lut, self.mha
        orig_build = lut.build_table
        saved = dict(mha._fwd_lut_tables)
        stage = "inputs"
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
            inp = make_inputs_wide()
            stage = "models"
            models = dot_models(inp)
            if not torch.equal(models["exact"], exact_raw(inp)):
                raise AssertionError("dot_models exact != exact_raw")
            valid = valid_mask(inp["seq_lens"])
            qkv = self._qkv(qkv_buffer(inp).to(dev))
            pv_s = float(PROD_CFG_ON["pv_s"])
            pv_k = int(PROD_CFG_ON["pv_k"])
            dis: Dict[str, List[int]] = {m: [] for m in DOT_MODELS}
            n_tables, elem = 0, "na"
            for j in PROBE_J:
                s = scales[J_TARGETS.index(j)]
                os_ = out_scale32(s, N_MAX, pv_s, pv_k)
                meas = torch.zeros(valid.shape, dtype=torch.int64)
                for shift in PROBE_SHIFTS:
                    stage = "launch_j%d_s%d" % (j, shift)
                    coded = coded_table(shift)
                    lut.build_table = (lambda ss, a, b, _t=coded: _t)
                    mha._fwd_lut_tables.clear()
                    o, _, lines = self._launch("1", qkv, offs, s, "0",
                                               record=False)
                    n_tables += len(parse_table_lines(lines))
                    stage = "decode_j%d_s%d" % (j, shift)
                    res = check_output("probe j=%d shift=%d" % (j, shift),
                                       o.to(torch.float64), os_, valid, None)
                    c = res["codes"] - CODE_BASE
                    if bool((valid & ((c < 0) | (c > 0x3F))).any()):
                        stage = "code_range_j%d_s%d" % (j, shift)
                        raise AssertionError(stage)
                    meas = meas | (torch.where(valid, c, torch.zeros_like(c))
                                   << shift)
                stage = "compare_j%d" % j
                for m in DOT_MODELS:
                    want = lut_index(models[m] * math.ldexp(1.0, j))
                    dis[m].append(int((valid & (meas != want)).sum()))
                if j == PROBE_ELEM_J:
                    elem = "%d" % int(meas[PROBE_ELEM])
            zero = [m for m in DOT_MODELS if not any(dis[m])]
            return ("LUT_SILU_GPU_PROBE status=ran js=%s elems=%d tables=%d "
                    "%s zero=%s t850_h0_k96_j-13_idx=%s is%d=%s" % (
                        ",".join("%d" % j for j in PROBE_J),
                        int(valid.sum()), n_tables,
                        " ".join("%s=%s" % (m, ",".join("%d" % x
                                                        for x in dis[m]))
                                 for m in DOT_MODELS),
                        ",".join(zero) or "none", elem, PROBE_ELEM_IDX,
                        "yes" if elem == "%d" % PROBE_ELEM_IDX else "no"))
        except Exception:  # noqa: B902 (informational probe)
            return "LUT_SILU_GPU_PROBE status=unavailable reason=%s" % stage
        finally:
            lut.build_table = orig_build
            mha._fwd_lut_tables.clear()
            mha._fwd_lut_tables.update(saved)

    def _check_kw(self, label: str, rec, want: Dict[str, object]) -> None:
        """The compiled kwargs of one launch are the production ones."""
        kw = dict(rec["kw"])
        if kw.pop("rs", None) != ROW_STRIDE:
            self._fail("%s: rs=%r, want %d (in-place layout)"
                       % (label, rec["kw"].get("rs"), ROW_STRIDE))
        if kw != want:
            self._fail("%s: compiled kwargs %r != production %r"
                       % (label, kw, want))

    def _jit_check(self, label: str, jit: List[Optional[int]]) -> str:
        """FlyDSL JIT entries of one LUT callable after each of its launches
        (jit_entries): one constant value >= 1, i.e. no scale added a JIT
        artifact. Returns it for the PASS line ("na": not exposed)."""
        known = [n for n in jit if n is not None]
        if known and (len(known) != len(jit) or min(known) < 1
                      or len(set(known)) != 1):
            self._fail("%s: FlyDSL JIT entries of the LUT callable after "
                       "each LUT launch %r (want one constant value >= 1: a "
                       "new scale must not JIT)" % (label, jit))
        return "%d" % known[0] if known else "na"

    def test_lut_silu_fwd_bitwise(self) -> None:  # noqa: C901
        lut = self.lut
        dev = torch.device("cuda", torch.cuda.current_device())
        cfg_on = self._config("1")
        cfg_off = self._config("0")
        cfg_unset = self._config(None)
        # _fp8_fwd_config() has no spf: item (105) is added per call.
        if cfg_on != without_spf(PROD_CFG_ON):
            self._fail("knob 1 cfg %r != production %r"
                       % (cfg_on, without_spf(PROD_CFG_ON)))
        if (cfg_off != without_spf(PROD_CFG_OFF)
                or cfg_unset != without_spf(PROD_CFG_OFF)):
            self._fail("knob 0/unset cfg %r / %r != production %r"
                       % (cfg_off, cfg_unset, without_spf(PROD_CFG_OFF)))
        # The lut path's only causal mask is mask_arith=3's P8 byte mask:
        # with DLRM_HSTU_FP8_FWD_MASK_ARITH=0 (pv=k128 still armed) the
        # launcher declines the knob (hstu_flydsl_mha.fp8_fwd_lut_config).
        ma_key = "DLRM_HSTU_FP8_FWD_MASK_ARITH"
        os.environ[ma_key] = "0"
        try:
            cfg_ma0 = self._config("1")
        finally:
            os.environ[ma_key] = PROD_ENV[ma_key]
        if "lut" in cfg_ma0 or cfg_ma0.get("pv") != "k128":
            self._fail("%s=0 with the knob on: cfg %r (want pv=k128 and no "
                       "lut: the launcher must decline item (120))"
                       % (ma_key, cfg_ma0))
        pv_s, pv_k = float(cfg_on["pv_s"]), int(cfg_on["pv_k"])
        S = pv_s if pv_s else float(2 ** pv_k)

        inp = make_inputs()
        raw = exact_raw(inp)
        valid = valid_mask(inp["seq_lens"])
        head = head_mask(inp["seq_lens"])
        qkv = self._qkv(qkv_buffer(inp).to(dev))
        # The tail2 pass: same q/k, V the identity on the keys 128..255.
        raw2 = exact_raw(inp, BM)
        valid2 = valid_mask(inp["seq_lens"], key_off=BM)
        qkv2 = self._qkv(qkv_buffer(inp, BM).to(dev))
        offs = torch.tensor(inp["offsets"], dtype=torch.int64, device=dev)

        scales = [qkv_scale_for_j(j, i) for i, j in enumerate(J_TARGETS)]
        calls = elems = head_elems = head_nz = 0
        tail2_calls = tail2_elems = tail2_nz = 0
        ship_n = ship_ok = 0
        cells = set()
        lut_fns, off_fns = set(), set()
        lut_misses: List[Optional[int]] = []
        lut_jit: List[Optional[int]] = []
        # (bits32(score_scale), S) -> (j, crc32) of every table the launcher
        # must build (and print) exactly once.
        expect_tables = {}
        first_lut = None
        for i, s in enumerate(scales):
            ss = lut.score_scale32(s, ALPHA)
            j, r = lut.split_scale(ss)
            if j != J_TARGETS[i]:
                self._fail("scale %r: j=%d, wanted %d" % (s, j, J_TARGETS[i]))
            table = lut.build_table(ss, pv_s, pv_k)
            expect_tables[(lut.bits32(ss), S)] = (j, lut.table_crc32(table))
            os_ = out_scale32(s, N_MAX, pv_s, pv_k)
            want_lut, idx = lut_expected(raw, j, table)
            want_ship = ship_expected(raw, ss, S)

            o_l, rec_l, _ = self._launch("1", qkv, offs, s, "0")
            o_0, rec_0, _ = self._launch("0", qkv, offs, s, "0")
            o_u, rec_u, _ = self._launch(None, qkv, offs, s, "0")
            calls += 3
            # The tail2 pass (m0 = 128 masked tail group): same LUT build,
            # same (cached) table, no compile.
            o_t, rec_t, _ = self._launch("1", qkv2, offs, s, "0")
            tail2_calls += 1
            if rec_t["fn"] is not rec_l["fn"]:
                self._fail("tail2 j=%d: a different compile_hstu_fwd callable "
                           "than the main LUT launch" % j)
            want2, idx2 = lut_expected(raw2, j, table)
            res_t = check_output("LUT tail2 j=%d" % j, o_t.to(torch.float64),
                                 os_, valid2, want2, idx2)
            if res_t["mism"]:
                self._fail("LUT tail2 (m0 = 128 masked tail group, keys "
                           "128..255) j=%d qkv_scale=%r: %d of %d P8 bytes != "
                           "build_table()[idx(y)]; first %r"
                           % (j, s, res_t["mism"], res_t["n_valid"],
                              res_t["first_mism"]))
            tail2_elems += res_t["n_valid"]
            tail2_nz += int((valid2 & (want2 != 0)).sum())
            self._check_kw("LUT j=%d" % j, rec_l, without_spf(PROD_CFG_ON))
            self._check_kw("knob 0 j=%d" % j, rec_0, without_spf(PROD_CFG_OFF))
            self._check_kw("unset j=%d" % j, rec_u, without_spf(PROD_CFG_OFF))
            if rec_l["kw"].get("lut") != 13 or getattr(rec_l["fn"], "lut",
                                                        None) != 13:
                self._fail("knob 1 compiled kw lut=%r call.lut=%r (want 13)"
                           % (rec_l["kw"].get("lut"),
                              getattr(rec_l["fn"], "lut", None)))
            for rec in (rec_0, rec_u):
                if rec["kw"].get("lut", 0) != 0 or getattr(rec["fn"], "lut",
                                                           0) != 0:
                    self._fail("knob 0/unset compiled a lut kernel: kw=%r"
                               % (rec["kw"],))
            if rec_l["kw"].get("rs") != ROW_STRIDE:
                self._fail("rs=%r, want %d (in-place layout)"
                           % (rec_l["kw"].get("rs"), ROW_STRIDE))
            lut_fns.add(id(rec_l["fn"]))
            off_fns.add(id(rec_0["fn"]))
            off_fns.add(id(rec_u["fn"]))
            lut_misses.append(rec_l["misses"])
            lut_jit.append(rec_l["jit"])
            lut_misses.append(rec_t["misses"])        # tail2: same callable
            lut_jit.append(rec_t["jit"])
            # (4) knob 0 == code default (same callable, bitwise-equal
            # output).
            if rec_0["fn"] is not rec_u["fn"]:
                self._fail("knob 0 != code default: knob 0 and unset got "
                           "different compile_hstu_fwd callables")
            if not torch.equal(o_0.view(torch.int16), o_u.view(torch.int16)):
                self._fail("scale %r: knob 0 output != code default (unset) "
                           "output (same callable, want bitwise-equal)" % s)

            o_lf = o_l.to(torch.float64)
            res_l = check_output("LUT j=%d" % j, o_lf, os_, valid, want_lut,
                                 idx, head)
            if res_l["mism"]:
                self._fail("LUT j=%d qkv_scale=%r score_scale=%r: %d of %d "
                           "P8 bytes (%d of %d head-group) != "
                           "build_table()[idx(y)]; first %r"
                           % (j, s, ss, res_l["mism"], res_l["n_valid"],
                              res_l["head_mism"], res_l["n_head"],
                              res_l["first_mism"]))
            res_s = check_output("ship j=%d" % j, o_0.to(torch.float64), os_,
                                 valid, want_ship)
            elems += res_l["n_valid"]
            head_elems += res_l["n_head"]
            head_nz += int((valid & head & (want_lut != 0)).sum())
            ship_n += res_s["n_valid"]
            ship_ok += res_s["n_valid"] - res_s["mism"]
            cells.update(torch.unique(idx[valid]).tolist())
            if first_lut is None:
                first_lut = (s, o_l)
        tail_elems = elems - head_elems
        # The head group (rows m >= 128 of the m0 = 128 workgroups) ran and
        # its bytes were observable (nonzero), and so did the tail groups,
        # the m0 = 128 one included (tail2).
        if (head_elems <= 0 or head_nz <= 0 or tail_elems <= 0
                or tail2_elems <= 0 or tail2_nz <= 0):
            self._fail("head/tail coverage: head_elems=%d (nonzero %d) "
                       "tail_elems=%d tail2_elems=%d (nonzero %d) (want all "
                       "> 0)" % (head_elems, head_nz, tail_elems, tail2_elems,
                                 tail2_nz))

        # Cache hit on a repeated scale: same bytes, no new table, no compile.
        s0, o_first = first_lut
        o_rep, rec_rep, _ = self._launch("1", qkv, offs, s0, "0")
        calls += 1
        if not torch.equal(o_rep.view(torch.int16), o_first.view(torch.int16)):
            self._fail("LUT output for a repeated scale differs (cache)")
        lut_fns.add(id(rec_rep["fn"]))
        lut_misses.append(rec_rep["misses"])
        lut_jit.append(rec_rep["jit"])

        # (2) the decode is right: the ship kernel matches its exact model.
        agree = ship_ok / float(ship_n)
        if agree < SHIP_AGREE_MIN:
            self._fail("ship kernel agrees with e4m3(exact P) on %.6f < %g of "
                       "%d elements: the decode or the out_scale model is "
                       "wrong" % (agree, SHIP_AGREE_MIN, ship_n))
        # (5) one compile_hstu_fwd build serves every scale: one distinct
        # LUT callable (lut_builds) and one off callable, no lru miss after
        # the first LUT call, and (where flydsl exposes it) no FlyDSL JIT
        # artifact added to that callable after its first launch.
        if len(lut_fns) != 1 or len(off_fns) != 1 or lut_fns & off_fns:
            self._fail("compile_hstu_fwd callables: lut %d, off %d (want 1 "
                       "and 1)" % (len(lut_fns), len(off_fns)))
        if lut_misses[0] is not None:
            want_first = (1,) if self.fresh_lut else (0, 1)
            if lut_misses[0] not in want_first or any(lut_misses[1:]):
                self._fail("compile_hstu_fwd misses per LUT call %r (want "
                           "first in %r, then 0: a new scale must not "
                           "build)" % (lut_misses, want_first))
        jit_txt = self._jit_check("LUT", lut_jit)
        # The spf pass (item (105) on, the production chain): SPF-grid
        # scales, score_scale = -2**j exactly, r = 1 tables built from the
        # same (snapped) qkv_scale the kernel receives; LUT compiled with
        # spf=True; knob 0 / unset compiled with item (105)'s kwargs.
        spf = spf_module()
        spf_calls = spf_elems = spf_ship_n = spf_ship_ok = 0
        spf_head_elems = spf_head_nz = 0
        spf_t2_calls = spf_t2_elems = spf_t2_nz = 0
        spf_cells = set()
        spf_lut_fns, spf_off_fns = set(), set()
        spf_misses: List[Optional[int]] = []
        spf_jit: List[Optional[int]] = []
        spf_bits = set()
        for j in SPF_J_TARGETS:
            s = spf_scale_for_j(j)
            ss = lut.score_scale32(s, ALPHA)
            if spf.arm_check(s, ALPHA) != j or ss != -math.ldexp(1.0, j):
                self._fail("SPF grid scale %r: score_scale %r is not -2**%d"
                           % (s, ss, j))
            if lut.split_scale(ss) != (j, 1.0):
                self._fail("SPF grid scale %r: split %r, want (%d, 1.0)"
                           % (s, lut.split_scale(ss), j))
            table = lut.build_table(ss, pv_s, pv_k)
            expect_tables[(lut.bits32(ss), S)] = (j, lut.table_crc32(table))
            spf_bits.add((lut.bits32(ss), S))
            os_ = out_scale32(s, N_MAX, pv_s, pv_k)
            want_lut, idx = lut_expected(raw, j, table)
            want_ship = ship_expected(raw, ss, S)

            o_l, rec_l, _ = self._launch("1", qkv, offs, s, "1")
            o_0, rec_0, _ = self._launch("0", qkv, offs, s, "1")
            o_u, rec_u, _ = self._launch(None, qkv, offs, s, "1")
            spf_calls += 3
            # The spf tail2 pass: the SPF+LUT build on the tail2 buffer.
            o_t, rec_t, _ = self._launch("1", qkv2, offs, s, "1")
            spf_t2_calls += 1
            if rec_t["fn"] is not rec_l["fn"]:
                self._fail("SPF tail2 j=%d: a different compile_hstu_fwd "
                           "callable than the SPF+LUT launch" % j)
            want2, idx2 = lut_expected(raw2, j, table)
            res_t = check_output("SPF+LUT tail2 j=%d" % j,
                                 o_t.to(torch.float64), os_, valid2, want2,
                                 idx2)
            if res_t["mism"]:
                self._fail("SPF+LUT tail2 (m0 = 128 masked tail group, keys "
                           "128..255) j=%d qkv_scale=%r: %d of %d P8 bytes != "
                           "build_table()[idx(y)] (r = 1); first %r"
                           % (j, s, res_t["mism"], res_t["n_valid"],
                              res_t["first_mism"]))
            spf_t2_elems += res_t["n_valid"]
            spf_t2_nz += int((valid2 & (want2 != 0)).sum())
            self._check_kw("SPF+LUT j=%d" % j, rec_l, PROD_CFG_ON)
            self._check_kw("SPF knob 0 j=%d" % j, rec_0, PROD_CFG_OFF)
            self._check_kw("SPF unset j=%d" % j, rec_u, PROD_CFG_OFF)
            if getattr(rec_l["fn"], "lut", None) != 13:
                self._fail("SPF+LUT call.lut=%r (want 13)"
                           % (getattr(rec_l["fn"], "lut", None),))
            # (4) knob 0 == code default (same callable, bitwise-equal
            # output), and no lut kernel.
            if (getattr(rec_0["fn"], "lut", 0) != 0
                    or rec_0["fn"] is not rec_u["fn"]):
                self._fail("SPF knob 0 != code default: a lut kernel or two "
                           "different compile_hstu_fwd callables")
            if not torch.equal(o_0.view(torch.int16), o_u.view(torch.int16)):
                self._fail("SPF scale %r: knob 0 output != code default "
                           "(unset) output (same callable, want "
                           "bitwise-equal)" % s)
            spf_lut_fns.add(id(rec_l["fn"]))
            spf_off_fns.add(id(rec_0["fn"]))
            spf_misses.append(rec_l["misses"])
            spf_jit.append(rec_l["jit"])
            spf_misses.append(rec_t["misses"])        # tail2: same callable
            spf_jit.append(rec_t["jit"])
            res_l = check_output("SPF+LUT j=%d" % j, o_l.to(torch.float64),
                                 os_, valid, want_lut, idx, head)
            if res_l["mism"]:
                self._fail("SPF+LUT j=%d qkv_scale=%r score_scale=%r: %d of %d "
                           "P8 bytes (%d of %d head-group) != "
                           "build_table()[idx(y)] (r = 1); first %r"
                           % (j, s, ss, res_l["mism"], res_l["n_valid"],
                              res_l["head_mism"], res_l["n_head"],
                              res_l["first_mism"]))
            res_s = check_output("SPF ship j=%d" % j, o_0.to(torch.float64),
                                 os_, valid, want_ship)
            spf_elems += res_l["n_valid"]
            spf_head_elems += res_l["n_head"]
            spf_head_nz += int((valid & head & (want_lut != 0)).sum())
            spf_ship_n += res_s["n_valid"]
            spf_ship_ok += res_s["n_valid"] - res_s["mism"]
            spf_cells.update(torch.unique(idx[valid]).tolist())
        spf_tail_elems = spf_elems - spf_head_elems
        if (spf_head_elems <= 0 or spf_head_nz <= 0 or spf_tail_elems <= 0
                or spf_t2_elems <= 0 or spf_t2_nz <= 0):
            self._fail("spf pass head/tail coverage: head_elems=%d (nonzero "
                       "%d) tail_elems=%d tail2_elems=%d (nonzero %d) (want "
                       "all > 0)" % (spf_head_elems, spf_head_nz,
                                     spf_tail_elems, spf_t2_elems, spf_t2_nz))
        spf_agree = spf_ship_ok / float(spf_ship_n)
        if spf_agree < SHIP_AGREE_MIN:
            self._fail("item (105)'s kernel agrees with e4m3(exact P) on %.6f "
                       "< %g of %d elements" % (spf_agree, SHIP_AGREE_MIN,
                                                spf_ship_n))
        # (5) for the spf pass: one compile_hstu_fwd build (spf_lut_builds)
        # serves every SPF-grid scale.
        if (len(spf_lut_fns) != 1 or len(spf_off_fns) != 1
                or (spf_lut_fns | spf_off_fns) & (lut_fns | off_fns)
                or spf_lut_fns & spf_off_fns):
            self._fail("spf pass compile_hstu_fwd callables: lut %d, off %d "
                       "(want 1 and 1, distinct from the main pass)"
                       % (len(spf_lut_fns), len(spf_off_fns)))
        if spf_misses[0] is not None:
            want_first = (1,) if self.fresh_lut else (0, 1)
            if spf_misses[0] not in want_first or any(spf_misses[1:]):
                self._fail("compile_hstu_fwd misses per SPF+LUT call %r (want "
                           "first in %r, then 0)" % (spf_misses, want_first))
        spf_jit_txt = self._jit_check("SPF+LUT", spf_jit)

        # Variant cells (SPF "0"; item (100) off, item (51) pow2 / off,
        # mask-arith kill): new builds only there, the pow2 cell's tables
        # added to expect_tables.
        var = self._variant_cells(
            qkv, offs, raw, valid, head, scales, expect_tables,
            lut_fns | off_fns | spf_lut_fns | spf_off_fns)
        # (6) one 'table built' line per distinct (scale, S), matching the
        # host.
        got = parse_table_lines(self.lines)
        if len(got) != len(expect_tables):
            self._fail("%d 'table built' lines for %d scales"
                       % (len(got), len(expect_tables)))
        for g in got:
            e = expect_tables.get((g["bits"], g["S"]))
            if e is None or e != (g["j"], g["crc"]):
                self._fail("table line %r does not match the host table %r"
                           % (g["line"], e))
        if len({(g["bits"], g["S"]) for g in got}) != len(expect_tables):
            self._fail("duplicate 'table built' lines: %r"
                       % [g["line"] for g in got])
        spf_got = [g for g in got if (g["bits"], g["S"]) in spf_bits]
        if len(spf_got) != len(spf_bits) or any(g["r"] != 1.0
                                                for g in spf_got):
            self._fail("SPF-grid table lines (want %d, all r = 1): %r"
                       % (len(spf_bits), [g["line"] for g in spf_got]))
        # Coverage: the f16-subnormal cells are reached (at pv_s=10.04 they
        # all hold +-0, so this checks they map to a zero byte, not
        # RTZ-vs-flush), the clamp cell, both signs.
        for label, cs in (("main", cells), ("spf", spf_cells)):
            need = {"subnormal": any((c & 0xF80) == 0 and c & 0x7F for c in cs),
                    "clamp": (CLAMP_CELL in cs) or ((CLAMP_CELL | 0x1000) in cs),
                    "neg": any(c & 0x1000 for c in cs),
                    "pos": any(not c & 0x1000 and c for c in cs)}
            if not all(need.values()):
                self._fail("%s pass input coverage too narrow (f16-subnormal "
                           "cells reached, clamp cell, both signs): %r"
                           % (label, need))
        # Every gating check passed. The index probe is informational.
        print(self._probe(offs, scales), flush=True)
        print("LUT_SILU_GPU_UNIT PASS calls=%d elems=%d head_elems=%d "
              "tail_elems=%d cells=%d lut_mism=0 ship_agree=%.6f "
              "lut_builds=%d lut_jit_entries=%s tail2_calls=%d "
              "tail2_elems=%d tail2_mism=0 spf_calls=%d spf_elems=%d "
              "spf_head_elems=%d spf_tail_elems=%d spf_cells=%d "
              "spf_lut_mism=0 spf_ship_agree=%.6f spf_lut_builds=%d "
              "spf_lut_jit_entries=%s spf_tail2_calls=%d spf_tail2_elems=%d "
              "spf_tail2_mism=0 var_calls=%d var_elems=%d var_mism=0 "
              "var_builds=%d"
              % (calls, elems, head_elems, tail_elems, len(cells), agree,
                 len(lut_fns), jit_txt, tail2_calls, tail2_elems, spf_calls,
                 spf_elems, spf_head_elems, spf_tail_elems, len(spf_cells),
                 spf_agree, len(spf_lut_fns), spf_jit_txt, spf_t2_calls,
                 spf_t2_elems, var["calls"], var["elems"], var["builds"]),
              flush=True)


if __name__ == "__main__":
    unittest.main()
