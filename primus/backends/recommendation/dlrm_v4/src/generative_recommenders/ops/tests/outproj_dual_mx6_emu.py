# pyre-unsafe
"""CPU models for the outproj_dual_mx6 tests (ops/tests/outproj_dual_mx6_test.py): a lane-level emulator of
ops/hip/outproj_dual_mx6.cu driven by the .cu TEXT, plus in-tree ports of the reference models it is checked against.

r14 iteration 66, producer track, DESIGN v1 (FROZEN) sec. 13 (T12, T13, T18). CPU only, numpy only; nothing here imports a
probe / session directory or the module under test (ops/hip/outproj_dual_mx6.py).

PORTS (logic copied; provenance in each section header):
  U2  probes/iter66/producer/spec/emu_triton_dual.py   the shipped Triton dual cast bwd_fp8._quantize_amax_dual_kernel
      (exact fp32 multiply, v_med3_f32, the e4m3 / e5m2 converters, the per-(inv, t) payload tables, the 'maxnum'
      PARTIAL model, publish() bits): dual_cast(), payload_luts(), tile_partials(), publish_inv_bits().
  U1  probes/iter66/producer/spec/g6_spec_helper.py    the REF-equal integer G6 rules (r16 REF mx6_l11_ref.py c49f2659,
      encode_g6 with neg_zero=True): e8m0_byte_np(), e2m3_code_np(), scale_operand_bits(), pack24_dwords(),
      encode_g6_int(); the W side quant_w_int() and the SPEC_G6 sec. 5 inverse-map scatter w6_scatter_np().
  App. B  DESIGN.md Appendix B emu_odm6.py: the lane-level emulator (buffer range checks, permlane16_swap, DPP quads,
      row_ror / xor reductions, the LDS staging + copy-out, LDS bank counts, write-once coverage) and its 10 mutants.

THE .cu IS THE SOURCE OF EVERY INDEX MAP (DESIGN sec. 13 T13). CuModel(text) preprocesses the .cu for a variant
(-DOUTPROJ_MX6_* macros), strips comments, finds the device functions and extracts, statement by statement, the C
expressions of every lane / band / load / staging / swap / G6 store / E8M0 / copy-out / PARTIAL / INV_USED map and the
schedule (the PF block's load_pass / pass_body / __syncthreads / copy_out sequence). A small C-subset evaluator
(integer types int / uint32_t / long long with C conversions, wrap-around, shifts, ?:, casts, calls) evaluates them on
numpy lane vectors. An edit of the .cu that breaks a map therefore changes what the emulator does, and the comparisons
fail; a construct the extractor does not recognise raises CuError (fail closed). Instruction semantics (the FP6 / fp8
converters, permlane16_swap, DPP, buffer range checks) are modelled; the helper functions that wrap them are pinned by
their normalised text (helper_text_drift()).

Every check that the kernel text must satisfy raises Fail (an emulated rule violation: a buffer access that straddles
num_records, a misaligned access, a staging write outside the image or not exactly once per tile, an LDS hazard
without a barrier, a copy-out store outside its descriptor, a scale operand that is not SPEC_G6's, an unconsumed
prefetch that read memory).
"""

import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CU_PATH = os.path.join(os.path.dirname(HERE), "hip", "outproj_dual_mx6.cu")
VECTORS_PATH = os.path.join(HERE, "outproj_dual_mx6_vectors.npz")

N_COLS = 512
BLOCK_M = 256
BLOCK_N = 128
E5M2_MAX = 57344.0
E4M3_MAX = 448.0
F32_QNAN = 0x7FC00000
F32_NEG_E5M2_MAX = 0xC7600000
F32_POS_E5M2_MAX = 0x47600000
F32_NEG_E4M3_MAX = 0xC3E00000
F32_POS_E4M3_MAX = 0x43E00000


class Fail(Exception):
    """An emulated kernel broke a rule (range, alignment, coverage, hazard, spec scale operand)."""


class CuError(Exception):
    """The .cu text lacks (or changed) a construct this emulator models: fail closed."""


# =====================================================================================================================
# U2 port: emu_triton_dual.py (r14 iter66 U2; the shipped ship15.1 b23945a Triton dual cast, Triton 3.6.0 gfx950 ISA)
# =====================================================================================================================
def _u32(b) -> np.ndarray:
    return np.asarray(b, dtype=np.uint32)


def f32_bits(v) -> np.ndarray:
    """uint32 bit patterns of float32 values (a Python float rounds RNE to fp32)."""
    return np.asarray(v, dtype=np.float32).view(np.uint32)


def is_nan32(b) -> np.ndarray:
    b = _u32(b)
    return ((b & 0x7F800000) == 0x7F800000) & ((b & 0x007FFFFF) != 0)


def is_snan32(b) -> np.ndarray:
    return is_nan32(b) & ((_u32(b) & 0x00400000) == 0)


def quiet32(b) -> np.ndarray:
    return _u32(b) | np.uint32(0x00400000)


def _order_key(b) -> np.ndarray:
    b = _u32(b).astype(np.int64)
    mag = b & 0x7FFFFFFF
    return np.where(b >> 31 == 1, -mag - 1, mag)


def _bitlen53(p: np.ndarray) -> np.ndarray:
    return np.frexp(p.astype(np.float64))[1].astype(np.int64)


def _u2_rne_shift(p: np.ndarray, k) -> np.ndarray:
    """round-half-even of p * 2^-k for int64 p >= 0 (k <= 0: exact left shift)."""
    p = p.astype(np.int64)
    k = np.asarray(k, dtype=np.int64)
    kr = np.clip(k, 1, 62)
    q = p >> kr
    r = p - (q << kr)
    half = np.int64(1) << (kr - 1)
    up = (r > half) | ((r == half) & ((q & 1) == 1))
    right = q + up.astype(np.int64)
    left = p << np.clip(-k, 0, 62)
    return np.where(k > 0, right, left)


def _decode32(b):
    b = _u32(b)
    sign = (b >> 31).astype(np.int64)
    e = ((b >> 23) & 0xFF).astype(np.int64)
    m = (b & 0x7FFFFF).astype(np.int64)
    sig = np.where(e == 0, m, m | (1 << 23))
    exp = np.where(e == 0, -149, e - 150)
    return sign, sig, exp


def f32_mul(a, b, ftz_out: bool = False) -> np.ndarray:
    """v_mul_f32, IEEE mode, RNE, denormals kept (denorm mode 3); ftz_out=True: a subnormal result flushes to signed
    zero (MODE.FP_DENORM f32 = 1, the rung-L3 DN1 setreg). NaN: quiet(first NaN operand); Inf * 0 -> 0x7FC00000."""
    a, b = np.broadcast_arrays(_u32(a), _u32(b))
    sa, ga, ea = _decode32(a)
    sb, gb, eb = _decode32(b)
    sign = (sa ^ sb).astype(np.uint32)
    a_nan, b_nan = is_nan32(a), is_nan32(b)
    a_inf = (a & 0x7FFFFFFF) == 0x7F800000
    b_inf = (b & 0x7FFFFFFF) == 0x7F800000
    a_zero = (a & 0x7FFFFFFF) == 0
    b_zero = (b & 0x7FFFFFFF) == 0
    p = ga * gb
    e = ea + eb
    pz = np.where(p > 0, p, 1)
    ue = e + _bitlen53(pz) - 1
    qe = np.where(ue >= -126, ue - 23, -149)
    n = _u2_rne_shift(pz, qe - e)
    mag = ((qe + 149) << 23) + n
    mag = np.where(mag >= 0x7F800000, 0x7F800000, mag)
    mag = np.where(p > 0, mag, 0).astype(np.uint32)
    if ftz_out:
        mag = np.where(mag < 0x00800000, np.uint32(0), mag).astype(np.uint32)
    out = mag | (sign << 31)
    out = np.where(a_inf | b_inf, np.uint32(0x7F800000) | (sign << 31), out)
    out = np.where((a_inf & b_zero) | (b_inf & a_zero), np.uint32(F32_QNAN), out)
    out = np.where(b_nan, quiet32(b), out)
    out = np.where(a_nan, quiet32(a), out)
    return out.astype(np.uint32)


def v_max_f32(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(_u32(a), _u32(b))
    r = np.where(_order_key(a) >= _order_key(b), a, b)
    r = np.where(is_nan32(b), a, r)
    r = np.where(is_nan32(a), b, r)
    r = np.where(is_snan32(b), quiet32(b), r)
    r = np.where(is_snan32(a), quiet32(a), r)
    return r.astype(np.uint32)


def v_min_f32(a, b) -> np.ndarray:
    a, b = np.broadcast_arrays(_u32(a), _u32(b))
    r = np.where(_order_key(a) <= _order_key(b), a, b)
    r = np.where(is_nan32(b), a, r)
    r = np.where(is_nan32(a), b, r)
    r = np.where(is_snan32(b), quiet32(b), r)
    r = np.where(is_snan32(a), quiet32(a), r)
    return r.astype(np.uint32)


def v_med3_f32(a, b, c) -> np.ndarray:
    """V_MED3_F32 (ISA pseudocode): any NaN -> v_min3_f32(a, b, c); else the median."""
    a, b, c = np.broadcast_arrays(_u32(a), _u32(b), _u32(c))
    anynan = is_nan32(a) | is_nan32(b) | is_nan32(c)
    mx = v_max_f32(v_max_f32(a, b), c)
    r = np.where(mx == a, v_max_f32(b, c), np.where(mx == b, v_max_f32(a, c), v_max_f32(a, b)))
    mn3 = v_min_f32(v_min_f32(a, b), c)
    return np.where(anynan, mn3, r).astype(np.uint32)


def cvt_f32_to_fp8(b, fmt: str, fp16_ovfl: bool = True) -> np.ndarray:
    """v_cvt_scalef32_pk_{bf8,fp8}_f32 at scale 1.0, per element: OCP e5m2 / e4m3fn, RNE, subnormals kept, sign kept
    on zero; finite overflow saturates with MODE.FP16_OVFL = 1; NaN -> 0x7F | sign (never reached after med3)."""
    if fmt == "e5m2":
        mbits, bias, max_code = 2, 15, 0x7B
    elif fmt == "e4m3fn":
        mbits, bias, max_code = 3, 7, 0x7E
    else:
        raise ValueError(fmt)
    b = _u32(b)
    sign, sig, exp = _decode32(b)
    emin = 1 - bias
    sigz = np.where(sig > 0, sig, 1)
    ue = exp + _bitlen53(sigz) - 1
    qe = np.maximum(ue, emin) - mbits
    n = _u2_rne_shift(sigz, qe - exp)
    code = ((qe + mbits + bias - 1) << mbits) + n
    code = np.where(sig > 0, code, 0)
    over = code > max_code
    code = np.where(over, max_code if fp16_ovfl else (0x7C if fmt == "e5m2" else 0x7F), code)
    inf = (b & 0x7FFFFFFF) == 0x7F800000
    code = np.where(inf, 0x7C if fmt == "e5m2" else (max_code if fp16_ovfl else 0x7F), code)
    code = np.where(is_nan32(b), 0x7F, code)
    return (code | (sign << 7)).astype(np.uint8)


def inv_bits_of(inv) -> np.uint32:
    """inv as fp32 bits: an int / np.uint32 is a bit pattern; a float is rounded to fp32."""
    if isinstance(inv, np.uint32) or (isinstance(inv, (int, np.integer)) and not isinstance(inv, bool)):
        return np.uint32(int(inv) & 0xFFFFFFFF)
    return np.uint32(f32_bits(np.float32(inv)))


def t_mul_bits(t_shift: int) -> np.uint32:
    return np.uint32(f32_bits(np.float32(2.0 ** -int(t_shift))))


_ALL16 = np.arange(1 << 16, dtype=np.uint32)


def payload_luts(inv, t_shift: int, ftz_out: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(e5m2 table, e4m3fn table), uint8[65536] indexed by the bf16 pattern: the dual cast's DST / DST_T byte at
    (inv, t) (U2). ftz_out: the same chain with subnormal products flushed (the DN1 MODE)."""
    ib = inv_bits_of(inv)
    x32 = _ALL16 << 16
    s = f32_mul(ib, x32, ftz_out)
    g8 = cvt_f32_to_fp8(v_med3_f32(s, F32_NEG_E5M2_MAX, F32_POS_E5M2_MAX), "e5m2")
    st = f32_mul(t_mul_bits(t_shift), s, ftz_out)
    e4 = cvt_f32_to_fp8(v_med3_f32(st, F32_NEG_E4M3_MAX, F32_POS_E4M3_MAX), "e4m3fn")
    return g8, e4


def n_partials(m: int) -> int:
    return -(-int(m) // BLOCK_M) * (N_COLS // BLOCK_N)


def tile_partials(x_u16: np.ndarray) -> np.ndarray:
    """PARTIAL (U2 'maxnum' model, GPU-validated class) as uint32 fp32 bits [4 * ceil(M/256)], index pid_m * 4 +
    pid_n: max over the 256 x 128 tile of |x| (rows >= M read +0.0), every NaN ignored; 0x7FC00000 iff all NaN."""
    x = np.asarray(x_u16, dtype=np.uint16)
    m, n = x.shape
    if n != N_COLS:
        raise ValueError("x must be [M, 512]")
    g0 = -(-m // BLOCK_M)
    a = np.zeros((g0 * BLOCK_M, N_COLS), dtype=np.uint16)
    a[:m] = x & 0x7FFF
    inc = a <= 0x7F80
    v = np.where(inc, a, 0).reshape(g0, BLOCK_M, N_COLS // BLOCK_N, BLOCK_N)
    any_inc = inc.reshape(g0, BLOCK_M, N_COLS // BLOCK_N, BLOCK_N).any(axis=(1, 3))
    mx = v.max(axis=(1, 3)).astype(np.uint32) << 16
    return np.where(any_inc, mx, np.uint32(F32_QNAN)).reshape(-1).astype(np.uint32)


def m_pad_of(m: int, t_tiled: bool = True) -> int:
    pad = 128 if t_tiled else 16
    return -(-int(m) // pad) * pad


def dual_cast(x_u16: np.ndarray, inv, t_shift: int, t_tiled: bool = True) -> Dict[str, Any]:
    """One quantize_row_major_dual(x, inv, t_dtype=e4m3fn, t_inv_mul=2^-t, t_limit=448, return_partial=True,
    t_tiled) launch (U2): g8 [M, 512], dout_t [M_PAD/128, 512, 128] (tiled; pad tokens = cast(+0.0)), partial bits."""
    x = np.asarray(x_u16, dtype=np.uint16)
    m, n = x.shape
    lut5, lut4 = payload_luts(inv, t_shift)
    mp = m_pad_of(m, t_tiled)
    pad_byte = lut4[0]
    if t_tiled:
        dt = np.full((mp // 128, N_COLS, 128), pad_byte, dtype=np.uint8)
        full = m // 128
        if full:
            dt[:full] = lut4[x[:full * 128]].reshape(full, 128, N_COLS).transpose(0, 2, 1)
        rem = m - full * 128
        if rem:
            dt[full, :, :rem] = lut4[x[full * 128:]].T
    else:
        dt = np.full((N_COLS, mp), pad_byte, dtype=np.uint8)
        dt[:, :m] = lut4[x].T
    pb = tile_partials(x)
    return dict(g8=lut5[x], dout_t=dt, m_pad=mp, pad_byte=int(pad_byte), partial_bits=pb,
                inv_bits=int(inv_bits_of(inv)), t_shift=int(t_shift))


def publish_inv_bits(pending: float, margin: float = 8.0, grad_max: float = E5M2_MAX) -> np.uint32:
    """_SiteScale.publish(): inv = fl32(1 / fl32(fl32(max(pending, 1e-30)) * fl32(margin / grad_max)))."""
    with np.errstate(all="ignore"):
        p = np.float32(pending)
        p = p if np.isnan(p) else np.maximum(p, np.float32(1.0e-30))
        scale = np.float32(p * np.float32(margin / grad_max))
        inv = np.float32(np.float32(1.0) / scale)
    return np.uint32(inv.view(np.uint32))


# =====================================================================================================================
# U1 port: g6_spec_helper.py (r14 iter66 U1; REF = r16 mx6_l11_ref.py c49f2659, encode_g6 / quant_w, neg_zero=True)
# =====================================================================================================================
KD, BLK, NBLK = 512, 32, 16
REC, HEAD_OFF, TAIL_OFF, SC_OFF = 400, 0, 256, 384
NOUT, WNB = 1536, 48
BYTE_NONFINITE = 255
_BL9 = np.array([int(v).bit_length() for v in range(1 << 9)], dtype=np.int64)


def e8m0_byte_np(a16) -> np.ndarray:
    """E8M0 byte of the integer block max a16 = max(h & 0x7FFF) (NaN-dominant): 255 if a16 >= 0x7F80, 127 if
    a16 == 0, else max(0, (a16 >> 7) - 2 + ((a16 & 0x7F) > 0x70))."""
    a = np.asarray(a16).astype(np.int64)
    b = np.maximum((a >> 7) - 2 + ((a & 0x7F) > 0x70).astype(np.int64), 0)
    b = np.where(a == 0, 127, b)
    return np.where(a >= 0x7F80, BYTE_NONFINITE, b)


def _bitlen(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.int64)
    if s.size and int(s.max()) < (1 << 9):
        return _BL9[s]
    out = np.zeros(s.shape, dtype=np.int64)
    v = s.copy()
    while np.any(v > 0):
        out += (v > 0)
        v >>= 1
    return out


def _u1_rne_shift(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """RNE(s * 2^u) for integers s >= 0 (s < 2^25), any integer u (ties to even)."""
    s = s.astype(np.int64)
    u = u.astype(np.int64)
    out_pos = np.left_shift(s, np.clip(u, 0, 30))
    sh = np.clip(-u, 1, 62)
    q = np.right_shift(s, sh)
    r = s - np.left_shift(q, sh)
    half = np.left_shift(np.int64(1), sh - 1)
    rnd = (r > half) | ((r == half) & ((q & 1) == 1))
    return np.where(u >= 0, out_pos, q + rnd.astype(np.int64))


def _e2m3_code_from_sig(s, sig, x, e) -> np.ndarray:
    """E2M3 code of (-1)^s * sig * 2^x / 2^e (|y| <= 7.5): nearest grid index (ties to even), sign bit = s."""
    sig = np.asarray(sig, dtype=np.int64)
    t = np.asarray(x, dtype=np.int64) - np.asarray(e, dtype=np.int64) + 3
    lg = np.where(sig > 0, _bitlen(sig) - 1, 0)
    p = lg + t
    if np.any((sig > 0) & (p > 5)):
        raise ValueError("|x / 2^e| > 7.5: not a finite-block element at this E8M0 byte")
    j = np.clip(p - 3, 0, 2)
    c = np.array([0, 8, 16], dtype=np.int64)[j] + _u1_rne_shift(sig, t - j)
    c = np.where(sig == 0, 0, c)
    return c | (np.asarray(s, dtype=np.int64) << 5)


def e2m3_code_np(h, byte) -> np.ndarray:
    """6-bit code of finite bf16 pattern h in a finite block whose E8M0 byte is `byte` (0..253)."""
    h = np.asarray(h).astype(np.int64) & 0xFFFF
    byte = np.asarray(byte).astype(np.int64)
    ex = (h >> 7) & 0xFF
    if np.any(ex == 0xFF):
        raise ValueError("non-finite element: its block's byte is 255 and its codes are 0")
    sig = np.where(ex == 0, h & 0x7F, 128 + (h & 0x7F))
    x = np.where(ex == 0, -133, ex - 134)
    return _e2m3_code_from_sig(h >> 15, sig, x, byte - 127)


def scale_operand_bits(byte) -> np.ndarray:
    """SPEC_G6 sec. 4: fp32 bits of the pk32 convert's scale operand: max(byte << 23, 0x00400000); 1.0 for 255."""
    b = np.asarray(byte).astype(np.int64)
    out = np.maximum(b << 23, 0x00400000)
    return np.where(b == BYTE_NONFINITE, 0x3F800000, out).astype(np.uint32)


def pack24_np(codes) -> np.ndarray:
    """int [..., 32] codes -> uint8 [..., 24]: code f at bits [6f, 6f + 6) of the little-endian 192-bit group."""
    c = np.asarray(codes).astype(np.int64)
    sh = c.shape[:-1]
    c = c.reshape(*sh, 8, 4)
    v = c[..., 0] | (c[..., 1] << 6) | (c[..., 2] << 12) | (c[..., 3] << 18)
    b = np.stack([v & 255, (v >> 8) & 255, (v >> 16) & 255], -1)
    return b.reshape(*sh, 24).astype(np.uint8)


def pack24_dwords(codes) -> np.ndarray:
    """int [..., 32] -> uint32 [..., 6]: the convert's result dwords D0..D5 (head D0..D3, tail D4..D5)."""
    sh = np.asarray(codes).shape[:-1]
    return np.ascontiguousarray(pack24_np(codes)).view(np.uint32).reshape(*sh, 6)


def _assemble_np(codes: np.ndarray, byte: np.ndarray) -> np.ndarray:
    r = codes.shape[0]
    grp = pack24_np(codes)
    rec = np.empty((r, REC), dtype=np.uint8)
    rec[:, HEAD_OFF:TAIL_OFF] = grp[:, :, 0:16].reshape(r, 256)
    rec[:, TAIL_OFF:SC_OFF] = grp[:, :, 16:24].reshape(r, 128)
    rec[:, SC_OFF:REC] = byte.reshape(r, 4, 4).transpose(0, 2, 1).reshape(r, 16).astype(np.uint8)
    return rec


def encode_g6_int(bits, chunk: int = 65536) -> np.ndarray:
    """bf16 bit patterns [M, 512] -> uint8 [M, 400], byte-equal to REF encode_g6 (neg_zero=True), strict."""
    h_all = np.asarray(bits)
    if h_all.ndim != 2 or h_all.shape[1] != KD:
        raise ValueError("bits must be [M, 512]")
    m = h_all.shape[0]
    out = np.empty((m, REC), dtype=np.uint8)
    for r0 in range(0, m, chunk):
        h = h_all[r0:r0 + chunk].astype(np.int64) & 0xFFFF
        r = h.shape[0]
        hb = h.reshape(r, NBLK, BLK)
        byte = e8m0_byte_np((hb & 0x7FFF).max(-1))
        fin = byte != BYTE_NONFINITE
        codes = e2m3_code_np(np.where(fin[..., None], hb, 0), np.where(fin, byte, 127)[..., None])
        codes = np.where(fin[..., None], codes, 0)
        out[r0:r0 + r] = _assemble_np(codes, byte)
    return out


def e8m0_byte_f32_np(a32) -> np.ndarray:
    a = np.asarray(a32).astype(np.int64)
    b = np.maximum((a >> 23) - 2 + ((a & 0x7FFFFF) > 0x700000).astype(np.int64), 0)
    b = np.where(a == 0, 127, b)
    return np.where(a >= 0x7F800000, BYTE_NONFINITE, b)


def quant_w_int(w) -> Tuple[np.ndarray, np.ndarray]:
    """REF quant_w (neg_zero=True), integer form: fp32 [1536, 512] -> (codes [1536, 512], e [48, 16]; 128 = NaN)."""
    u = np.ascontiguousarray(np.asarray(w, dtype=np.float32)).view(np.uint32).astype(np.int64)
    t = u.reshape(WNB, BLK, NBLK, BLK)
    byte = e8m0_byte_f32_np((t & 0x7FFFFFFF).max(axis=(1, 3)))
    fin = byte != BYTE_NONFINITE
    f4 = fin[:, None, :, None]
    ex = (t >> 23) & 0xFF
    mant = t & 0x7FFFFF
    sig = np.where(f4, np.where(ex == 0, mant, (1 << 23) | mant), 0)
    x = np.where(f4, np.where(ex == 0, -149, ex - 150), 0)
    ee = np.where(fin, byte, 127)[:, None, :, None] - 127
    codes = _e2m3_code_from_sig(np.where(f4, t >> 31, 0), sig, x, ee)
    codes = np.where(f4, codes, 0)
    return codes.reshape(NOUT, KD), np.where(fin, byte - 127, 128)


def w6_scatter_np(codes, e) -> Tuple[np.ndarray, np.ndarray]:
    """W6 / W6S via SPEC_G6 sec. 5's explicit inverse map (U1 w6_scatter_np): t = n >> 8, w = (n >> 5) & 7,
    j = (n >> 2) & 1, R = 4((n >> 3) & 3) + (n & 3), k = kb >> 2, q = kb & 3; head (4t + k) 24576 + 3072w + 1024j +
    64R + 16(q ^ ((n >> 3) & 3)); tail ... + 2048 + 512j + 32R + 8(q ^ 2((n >> 4) & 1)); W6S[16 nb + 4q + k]."""
    c = np.asarray(codes).astype(np.int64).reshape(NOUT, NBLK, BLK)
    grp = pack24_np(c)
    w6 = np.full(589824, 0xEE, dtype=np.uint8)
    n = np.arange(NOUT)[:, None]
    kb = np.arange(NBLK)[None, :]
    t, w, j = n >> 8, (n >> 5) & 7, (n >> 2) & 1
    r = 4 * ((n >> 3) & 3) + (n & 3)
    k, q = kb >> 2, kb & 3
    base = (4 * t + k) * 24576 + 3072 * w
    ho = base + 1024 * j + 64 * r + 16 * (q ^ ((n >> 3) & 3))
    to = base + 2048 + 512 * j + 32 * r + 8 * (q ^ (2 * ((n >> 4) & 1)))
    for i in range(16):
        w6[(ho + i).reshape(-1)] = grp[:, :, i].reshape(-1)
    for i in range(8):
        w6[(to + i).reshape(-1)] = grp[:, :, 16 + i].reshape(-1)
    eb = np.minimum(np.asarray(e).astype(np.int64) + 127, 255)
    w6s = np.empty(768, dtype=np.uint8)
    nb_, kb_ = np.meshgrid(np.arange(WNB), np.arange(NBLK), indexing="ij")
    w6s[16 * nb_ + 4 * (kb_ & 3) + (kb_ >> 2)] = eb[nb_, kb_]
    return w6, w6s


def rne_bf16_bits(x) -> np.ndarray:
    """fp32 values -> bf16 bit patterns, RNE (torch's rule); NaN -> 0x7FC0 with its sign."""
    u = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) & 0xFFFF
    nan = is_nan32(u.astype(np.uint32))
    r = np.where(nan, 0x7FC0 | ((u >> 16) & 0x8000), r)
    return r.astype(np.uint16)


# =====================================================================================================================
# C subset: comments, preprocessor, expressions (C integer semantics on numpy lane vectors), function bodies
# =====================================================================================================================
def strip_comments(src: str) -> str:
    """Remove // and /* */ comments outside string / char literals (newlines kept)."""
    out: List[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            if j < 0:
                raise CuError("unterminated /* comment")
            out.append("\n" * src.count("\n", i, j))
            i = j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


class CV:
    """A C value: int64 numpy data (any shape, or a scalar) and its C type: i32, u32, i64, u8, u16 or f32 (bits)."""

    __slots__ = ("v", "t")

    def __init__(self, v: Any, t: str) -> None:
        self.v = v
        self.t = t

    def arr(self) -> np.ndarray:
        return np.asarray(self.v, dtype=np.int64)

    def scalar(self) -> int:
        a = np.asarray(self.v)
        if a.ndim and a.size != 1:
            if not (a == a.reshape(-1)[0]).all():
                raise CuError("a lane-varying value where a uniform scalar is required")
            return int(a.reshape(-1)[0])
        return int(a)


def wrap(v: Any, t: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    if t == "u32":
        return v & 0xFFFFFFFF
    if t == "i32":
        return ((v + 0x80000000) & 0xFFFFFFFF) - 0x80000000
    if t == "u8":
        return v & 0xFF
    if t == "u16":
        return v & 0xFFFF
    if t == "i64":
        return v
    raise CuError(f"no integer arithmetic on type {t}")


def _promote(t: str) -> str:
    return "i32" if t in ("u8", "u16", "i32") else t


def _common(a: str, b: str) -> str:
    a, b = _promote(a), _promote(b)
    if "f32" in (a, b):
        raise CuError("float operand in an integer expression")
    if "i64" in (a, b):
        return "i64"
    if "u32" in (a, b):
        return "u32"
    return "i32"


def cv_u32(v: Any) -> CV:
    return CV(wrap(v, "u32"), "u32")


def cv_i32(v: Any) -> CV:
    return CV(wrap(v, "i32"), "i32")


_TOKEN_RE = re.compile(
    r"(?P<ws>\s+)"
    r"|(?P<flt>(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?[fF]?)"
    r"|(?P<num>(?:0[xX][0-9a-fA-F]+|\d+))(?P<suf>[uUlL]*)"
    r"|(?P<id>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)"
    r"|(?P<op><<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>?:(),\[\]{}])"
)
_BIN_PREC = {"||": 1, "&&": 2, "|": 3, "^": 4, "&": 5, "==": 6, "!=": 6, "<": 7, "<=": 7, ">": 7, ">=": 7,
             "<<": 8, ">>": 8, "+": 9, "-": 9, "*": 10, "/": 10, "%": 10}
_CAST_TYPES = {
    ("int",): "i32", ("int32_t",): "i32", ("unsigned",): "u32", ("unsigned", "int"): "u32", ("uint32_t",): "u32",
    ("long", "long"): "i64", ("int64_t",): "i64", ("long",): "i64", ("size_t",): "i64", ("uint64_t",): "i64",
    ("bool",): "i32", ("uint8_t",): "u8", ("uint16_t",): "u16",
}


def _tokenize(text: str) -> List[Tuple[str, Any, str]]:
    toks: List[Tuple[str, Any, str]] = []
    i = 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if not m:
            raise CuError(f"cannot tokenize {text[i:i + 20]!r} in {text!r}")
        i = m.end()
        if m.group("ws"):
            continue
        if m.group("flt"):
            toks.append(("flt", float(m.group("flt").rstrip("fF")), ""))
        elif m.group("num"):
            toks.append(("num", int(m.group("num"), 0), m.group("suf").lower()))
        elif m.group("id"):
            toks.append(("id", m.group("id"), ""))
        else:
            toks.append(("op", m.group("op"), ""))
    return toks


class _Parser:
    def __init__(self, text: str) -> None:
        self.text = text
        self.toks = _tokenize(text)
        self.i = 0

    def peek(self, k: int = 0) -> Optional[Tuple[str, Any, str]]:
        j = self.i + k
        return self.toks[j] if j < len(self.toks) else None

    def take(self, op: Optional[str] = None) -> Tuple[str, Any, str]:
        tok = self.peek()
        if tok is None or (op is not None and tok != ("op", op, "")):
            raise CuError(f"expected {op!r} at token {self.i} of {self.text!r}")
        self.i += 1
        return tok

    def parse(self) -> Any:
        node = self.ternary()
        if self.i != len(self.toks):
            raise CuError(f"trailing tokens in {self.text!r}")
        return node

    def ternary(self) -> Any:
        c = self.binary(1)
        if self.peek() == ("op", "?", ""):
            self.take("?")
            a = self.ternary()
            self.take(":")
            b = self.ternary()
            return ("tern", c, a, b)
        return c

    def binary(self, minp: int) -> Any:
        lhs = self.unary()
        while True:
            tok = self.peek()
            if tok is None or tok[0] != "op" or tok[1] not in _BIN_PREC or _BIN_PREC[tok[1]] < minp:
                return lhs
            self.i += 1
            rhs = self.binary(_BIN_PREC[tok[1]] + 1)
            lhs = ("bin", tok[1], lhs, rhs)

    def _cast_type(self) -> Optional[Tuple[str, int]]:
        if self.peek() != ("op", "(", ""):
            return None
        names: List[str] = []
        j = 1
        while True:
            tok = self.peek(j)
            if tok is None:
                return None
            if tok == ("op", ")", ""):
                break
            if tok[0] != "id" or tok[1] == "const":
                if tok[0] == "id" and tok[1] == "const":
                    j += 1
                    continue
                return None
            names.append(tok[1])
            j += 1
        ty = _CAST_TYPES.get(tuple(names))
        return (ty, j + 1) if ty else None

    def unary(self) -> Any:
        cast = self._cast_type()
        if cast is not None:
            self.i += cast[1]
            return ("cast", cast[0], self.unary())
        tok = self.peek()
        if tok is not None and tok[0] == "op" and tok[1] in ("-", "~", "!", "+"):
            self.i += 1
            return ("un", tok[1], self.unary())
        return self.postfix()

    def postfix(self) -> Any:
        node = self.primary()
        while True:
            tok = self.peek()
            if tok == ("op", "(", ""):
                self.i += 1
                args: List[Any] = []
                if self.peek() != ("op", ")", ""):
                    while True:
                        args.append(self.ternary())
                        if self.peek() == ("op", ",", ""):
                            self.i += 1
                            continue
                        break
                self.take(")")
                node = ("call", node, args)
            elif tok == ("op", "[", ""):
                self.i += 1
                idx = self.ternary()
                self.take("]")
                node = ("index", node, idx)
            else:
                return node

    def primary(self) -> Any:
        tok = self.take()
        if tok[0] == "num":
            v, suf = tok[1], tok[2]
            if "ll" in suf or "l" in suf:
                ty = "i64"
            elif "u" in suf:
                ty = "u32" if v < (1 << 32) else "i64"
            else:
                ty = "i32" if v < (1 << 31) else ("u32" if v < (1 << 32) else "i64")
            return ("num", v, ty)
        if tok[0] == "flt":
            return ("flt", tok[1])
        if tok[0] == "id":
            return ("id", tok[1])
        if tok == ("op", "(", ""):
            e = self.ternary()
            self.take(")")
            return e
        raise CuError(f"unexpected token {tok!r} in {self.text!r}")


_COMPILED: Dict[str, Any] = {}


def compile_expr(text: str) -> Any:
    node = _COMPILED.get(text)
    if node is None:
        node = _Parser(text).parse()
        _COMPILED[text] = node
    return node


def _cmp(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    return a >= b


def _eval(node: Any, env: Dict[str, Any]) -> CV:
    kind = node[0]
    if kind == "num":
        return CV(np.int64(node[1]), node[2])
    if kind == "flt":
        return CV(np.int64(int(f32_bits(np.float32(node[1])))), "f32")
    if kind == "id":
        if node[1] not in env:
            raise CuError(f"unknown identifier {node[1]!r}")
        v = env[node[1]]
        return v if isinstance(v, CV) else CV(np.int64(v), "i32")
    if kind == "cast":
        x = _eval(node[2], env)
        if x.t == "f32":
            raise CuError("cast of a float value")
        return CV(wrap(x.v, node[1]), node[1])
    if kind == "un":
        x = _eval(node[2], env)
        t = _promote(x.t)
        if node[1] == "-":
            return CV(wrap(-wrap(x.v, t), t), t)
        if node[1] == "~":
            return CV(wrap(~wrap(x.v, t), t), t)
        if node[1] == "!":
            return CV((np.asarray(x.v) == 0).astype(np.int64), "i32")
        return CV(wrap(x.v, t), t)
    if kind == "tern":
        c = np.asarray(_eval(node[1], env).v) != 0
        a, b = _eval(node[2], env), _eval(node[3], env)
        t = _common(a.t, b.t)
        return CV(np.where(c, wrap(a.v, t), wrap(b.v, t)), t)
    if kind == "call":
        if node[1][0] != "id":
            raise CuError("call of a non-identifier")
        fn = env.get(node[1][1])
        if not callable(fn):
            raise CuError(f"unknown function {node[1][1]!r}")
        return fn(*[_eval(a, env) for a in node[2]])
    if kind == "index":
        if node[1][0] != "id" or node[1][1] not in env:
            raise CuError("index of an unknown array")
        return env[node[1][1]][_eval(node[2], env)]
    op, x, y = node[1], _eval(node[2], env), _eval(node[3], env)
    if op in ("&&", "||"):
        a, b = np.asarray(x.v) != 0, np.asarray(y.v) != 0
        return CV((a & b if op == "&&" else a | b).astype(np.int64), "i32")
    if op in ("<<", ">>"):
        t = _promote(x.t)
        a, n = wrap(x.v, t), np.asarray(y.v, dtype=np.int64)
        if np.any((n < 0) | (n >= (64 if t == "i64" else 32))):
            raise CuError(f"shift count out of range in {op}")
        return CV(wrap((a << n) if op == "<<" else (a >> n), t), t)
    t = _common(x.t, y.t)
    a, b = wrap(x.v, t), wrap(y.v, t)
    if op in ("==", "!=", "<", "<=", ">", ">="):
        return CV(_cmp(op, a, b).astype(np.int64), "i32")
    if op == "+":
        r = a + b
    elif op == "-":
        r = a - b
    elif op == "*":
        r = a * b
    elif op in ("/", "%"):
        if np.any(b == 0):
            raise CuError("division by zero")
        q = np.abs(a) // np.abs(b)
        q = np.where((a < 0) ^ (b < 0), -q, q)
        r = q if op == "/" else a - b * q
    elif op == "&":
        r = a & b
    elif op == "|":
        r = a | b
    else:
        r = a ^ b
    return CV(wrap(r, t), t)


def ev(text: str, env: Dict[str, Any]) -> CV:
    """Evaluate C expression `text` (C integer semantics) in `env` (name -> CV / callable / indexable)."""
    return _eval(compile_expr(text), env)


def _match(text: str, i: int, open_c: str, close_c: str) -> int:
    """Index of the bracket closing text[i] == open_c."""
    if text[i] != open_c:
        raise CuError(f"expected {open_c!r}")
    depth = 0
    for j in range(i, len(text)):
        if text[j] == open_c:
            depth += 1
        elif text[j] == close_c:
            depth -= 1
            if depth == 0:
                return j
    raise CuError(f"unbalanced {open_c!r}")


def split_top(text: str, sep: str = ",") -> List[str]:
    """Split at top-level `sep` (outside (), [], {})."""
    out, depth, cur = [], 0, []
    for c in text:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if c == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    out.append("".join(cur).strip())
    return out


def preprocess(src: str, defines: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, str]]:
    """The .cu after comment stripping and #if / #ifdef / #ifndef / #else / #endif / #define / #error for `defines`
    (macro name -> value, as -D would set them). Returns (text, macros)."""
    macros: Dict[str, str] = {k: str(v) for k, v in (defines or {}).items()}
    out: List[str] = []
    stack: List[Tuple[bool, bool]] = []
    active = True
    for line in strip_comments(src).split("\n"):
        s = line.strip()
        if not s.startswith("#"):
            out.append(line if active else "")
            continue
        out.append("")
        kw, _, rest = s[1:].strip().partition(" ")
        rest = rest.strip()
        if kw in ("if", "ifdef", "ifndef"):
            if not active:
                cond = False
            elif kw == "ifdef":
                cond = rest in macros
            elif kw == "ifndef":
                cond = rest not in macros
            else:
                cond = pp_value(rest, macros) != 0
            stack.append((active, cond))
            active = active and cond
        elif kw == "else":
            parent, taken = stack[-1]
            active = parent and not taken
            stack[-1] = (parent, True)
        elif kw == "endif":
            active = stack.pop()[0]
        elif kw in ("elif",):
            raise CuError("#elif is not modelled")
        elif not active:
            continue
        elif kw == "define":
            name, _, val = rest.partition(" ")
            if "(" in name:
                raise CuError("function-like macros are not modelled")
            macros[name] = val.strip()
        elif kw == "error":
            raise CuError(f"#error {rest}")
        elif kw in ("include", "pragma"):
            continue
        else:
            raise CuError(f"preprocessor directive #{kw} is not modelled")
    if stack:
        raise CuError("unterminated #if")
    return "\n".join(out), macros


def pp_value(text: str, macros: Dict[str, str], depth: int = 0) -> int:
    """Integer value of a preprocessor expression / macro body (unknown identifiers are 0, as in C)."""
    if depth > 16:
        raise CuError("macro recursion")

    class _Env(dict):
        def __missing__(self, key: str) -> CV:
            if key in macros:
                return cv_i32(pp_value(macros[key], macros, depth + 1))
            return cv_i32(0)

        def __contains__(self, key: object) -> bool:
            return True

        def get(self, key: str, default: Any = None) -> Any:
            return None

    text = re.sub(r"defined\s*\(\s*(\w+)\s*\)", lambda m: "1" if m.group(1) in macros else "0", text)
    return ev(text, _Env()).scalar()


def find_function(text: str, name: str) -> Tuple[str, str]:
    """(parameter list, body) of the unique definition `name(...) {...}` in preprocessed text."""
    found: List[Tuple[str, str]] = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
        j = _match(text, m.end() - 1, "(", ")")
        k = j + 1
        while k < len(text) and text[k].isspace():
            k += 1
        if k < len(text) and text[k] == "{":
            e = _match(text, k, "{", "}")
            found.append((text[m.end():j], text[k + 1:e]))
    if len(found) != 1:
        raise CuError(f"{len(found)} definitions of {name}")
    return found[0]


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def assignments(text: str, lhs: str) -> List[str]:
    """Every `lhs = <expr>` (a declaration or a plain assignment) in `text`, in order; <expr> ends at a top-level ','
    or ';'. Compound (|=, +=) and comparisons are not matched."""
    pat = re.compile(r"(?<![\w.\]])" + re.escape(lhs) + r"\s*=(?!=)")
    out: List[str] = []
    for m in pat.finditer(text):
        prev = text[max(0, m.start() - 1):m.start()]
        if prev and prev in "|&^+-*/%<>!":
            continue
        depth, j = 0, m.end()
        while j < len(text):
            c = text[j]
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if depth == 0:
                    break
                depth -= 1
            elif c in ",;" and depth == 0:
                break
            j += 1
        out.append(norm_ws(text[m.end():j]))
    return out


def assign_positions(text: str, lhs: str) -> List[Tuple[int, str]]:
    """(position, expr) of every `lhs = <expr>` (see assignments())."""
    pat = re.compile(r"(?<![\w.\]])" + re.escape(lhs) + r"\s*=(?!=)")
    out = []
    for m in pat.finditer(text):
        expr = assignments(text[m.start():], lhs)
        if expr:
            out.append((m.start(), expr[0]))
    return out


def one(text: str, lhs: str, what: str = "") -> str:
    got = assignments(text, lhs)
    if len(got) != 1:
        raise CuError(f"{what or lhs}: {len(got)} assignments of {lhs!r} (need exactly 1)")
    return got[0]


def if_statements(text: str, stmt_re: str) -> List[Tuple[str, re.Match, int]]:
    """Every `if (<cond>) <stmt>` whose statement matches stmt_re: (cond, match, position)."""
    out = []
    for m in re.finditer(r"\bif\s*\(", text):
        j = _match(text, m.end() - 1, "(", ")")
        sm = re.compile(r"\s*" + stmt_re).match(text, j + 1)
        if sm:
            out.append((norm_ws(text[m.end():j]), sm, m.start()))
    return out


def loop_bounds(text: str, var: str) -> List[Tuple[str, str, str]]:
    """Every `for (int var = A; var < B; STEP)` header: (A, B, STEP) with STEP '++' or '<<= 1'."""
    pat = re.compile(r"for\s*\(\s*int\s+" + var + r"\s*=\s*([^;]+);\s*" + var + r"\s*<\s*([^;]+);\s*"
                     r"(\+\+\s*" + var + r"|" + var + r"\s*<<=\s*1)\s*\)")
    return [(norm_ws(m.group(1)), norm_ws(m.group(2)), "<<= 1" if "<<=" in m.group(3) else "++")
            for m in pat.finditer(text)]


def loop_range(text: str, var: str, what: str) -> List[int]:
    got = loop_bounds(text, var)
    if len(got) != 1:
        raise CuError(f"{what}: {len(got)} loops over {var}")
    a, b, step = got[0]
    lo, hi = pp_value(a, {}), pp_value(b, {})
    if step == "++":
        return list(range(lo, hi))
    out, v = [], lo
    while v < hi:
        out.append(v)
        v <<= 1
    return out


def loop_blocks(text: str, var: str) -> List[Tuple[List[int], str]]:
    """Every `for (int var = A; var < B; STEP) <body>` with its iteration values and body text ({...} or one stmt)."""
    pat = re.compile(r"for\s*\(\s*int\s+" + var + r"\s*=\s*([^;]+);\s*" + var + r"\s*<\s*([^;]+);\s*"
                     r"(\+\+\s*" + var + r"|" + var + r"\s*<<=\s*1)\s*\)")
    out = []
    for m in pat.finditer(text):
        lo, hi = pp_value(m.group(1), {}), pp_value(m.group(2), {})
        vals, v = [], lo
        while v < hi:
            vals.append(v)
            v = v + 1 if m.group(3).startswith("++") else v << 1
        k = m.end()
        while text[k].isspace():
            k += 1
        if text[k] == "{":
            body = text[k + 1:_match(text, k, "{", "}")]
        else:
            body = text[k:text.index(";", k) + 1]
        out.append((vals, body))
    return out


def the_loop(text: str, var: str, marker: str, what: str) -> Tuple[List[int], str]:
    got = [(v, b) for v, b in loop_blocks(text, var) if marker in b]
    if len(got) != 1:
        raise CuError(f"{what}: {len(got)} loops over {var} containing {marker!r}")
    return got[0]


def straight_line(body: str) -> List[Tuple[str, str]]:
    """Declarations / assignments of a straight-line block as (name, expr); `asm volatile` statements are skipped."""
    out = []
    for st in split_top(body, ";"):
        st = norm_ws(st)
        if not st or st.startswith("asm volatile"):
            continue
        m = re.fullmatch(r"(?:const\s+)?(?:uint32_t|int|bool)?\s*([A-Za-z_]\w*)\s*=(?!=)\s*(.+)", st)
        if not m:
            raise CuError(f"straight-line statement not modelled: {st!r}")
        out.append((m.group(1), m.group(2)))
    return out


# Helper functions whose bodies are instruction wrappers: modelled, and pinned by their normalised text (an edit of one
# of them is not silently emulated: CuModel raises CuError).
HELPER_PINS: Dict[str, str] = {
    "make_rsrc": "const uint64_t a = reinterpret_cast<uint64_t>(p); i32x4 r; r.x = (int)(uint32_t)a; r.y = (int)"
                 "(uint32_t)(a >> 32); r.z = (int)nrec; r.w = 0x00110000; return r;",
    "pkmax": "return __builtin_bit_cast(uint32_t, __builtin_elementwise_max(__builtin_bit_cast(us2, a), "
             "__builtin_bit_cast(us2, b)));",
    "pkadd": "return __builtin_bit_cast(uint32_t, __builtin_bit_cast(us2, a) + __builtin_bit_cast(us2, b));",
    "pkasr15": "return __builtin_bit_cast(uint32_t, __builtin_bit_cast(s2, a) >> (s2){15, 15});",
    "lo_f": "return __uint_as_float(d << 16);",
    "hi_f": "return __uint_as_float(d & 0xffff0000u);",
    "med448": "return __builtin_amdgcn_fmed3f(v, -448.0f, 448.0f);",
    "e4x4": "v2s z; z = __builtin_nondeterministic_value(z); const v2s lo = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(z, "
            "a, b, 1.0f, false); const v2s hv = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(lo, c, d, 1.0f, true); "
            "return __builtin_bit_cast(uint32_t, hv);",
    "cvt_pk32": 'v6u r; asm volatile("s_nop 1\\n\\tv_cvt_scalef32_pk32_fp6_bf16 %0, %1, %2\\n\\ts_nop 4" : "=&v"(r) : '
                '"v"(x), "v"(__uint_as_float(sbits))); return r;',
    "cvt_2xpk16": 'v6u r; const float one = 1.0f; asm volatile("s_nop 1\\n\\tv_cvt_scalef32_2xpk16_fp6_f32 %0, %1, %2, '
                  '%3\\n\\ts_nop 4" : "=&v"(r) : "v"(e), "v"(o), "v"(one)); return r;',
    "umax": "return a > b ? a : b;",
    "w_code": "const float v = y * inv2e; float a = fabsf(v); a = a > 7.5f ? 7.5f : a; uint32_t c; if (a < 2.0f) c = "
              "(uint32_t)rintf(a * 8.0f); else if (a < 4.0f) c = (uint32_t)rintf(a * 4.0f) + 8u; else c = (uint32_t)"
              "rintf(a * 2.0f) + 16u; if (!(a == a)) c = 0u; return c | ((__float_as_uint(v) >> 26) & 0x20u);",
}

VARIANT_DEFINES: Dict[str, Dict[str, int]] = {
    "pk32": {},                                                         # the production default (PF = 2)
    "f32dn1": {"OUTPROJ_MX6_G6_F32": 1, "OUTPROJ_MX6_DN1": 1},          # rung L3 (PF defaults to 1)
    "pf1": {"OUTPROJ_MX6_PF": 1},
    "reverse": {"OUTPROJ_MX6_REVERSE": 1},
}


class CuModel:
    """Every map, condition and schedule step of outproj_dual_mx6.cu for one variant, extracted from the text."""

    def __init__(self, src: str, defines: Optional[Dict[str, Any]] = None) -> None:
        self.src = src
        self.text, self.macros = preprocess(src, defines)
        mv = lambda n: pp_value(n, self.macros)  # noqa: E731
        self.g6_f32, self.dn1_macro = mv("OUTPROJ_MX6_G6_F32"), mv("OUTPROJ_MX6_DN1")
        self.pf, self.reverse = mv("OUTPROJ_MX6_PF"), mv("OUTPROJ_MX6_REVERSE")
        T = self.text
        self.pins: Dict[str, bool] = {}
        for name, want in HELPER_PINS.items():
            body = norm_ws(find_function(T, name)[1])
            if body != norm_ws(want):
                raise CuError(f"helper {name} changed (modelled helper, pinned text): {body!r}")
            self.pins[name] = True
        # namespace constants
        self.consts: Dict[str, int] = {}
        for name in ("NT", "BAND", "LDS_T", "LDS_BYTES", "W6CH", "W6SL"):
            self.consts[name] = ev(one(T, name), {k: cv_i32(v) for k, v in self.consts.items()}).scalar()
        self.abi = mv("ODM6_ABI_VERSION")
        self._kernel()
        self._load_pass()
        self._pass_body()
        self._copy_out()
        self._dpp()
        self._nanfree()
        self._g6_codes()
        self._launcher()
        self._w6()

    # ------------------------------------------------------------------------------------------------ kernel body
    def _kernel(self) -> None:
        params, K = find_function(self.text, "outproj_dual_mx6_kernel")
        self.k_params = [norm_ws(p).split()[-1] for p in split_top(params)]
        n = norm_ws(K)
        self.dn1_setreg = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 1" in n
        self.fp16_ovfl_setreg = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1" in n
        if not self.fp16_ovfl_setreg:
            raise CuError("the FP16_OVFL setreg (Triton's MODE) is missing from the producer prologue")
        self.lds_decl = re.search(r"__shared__ __attribute__\(\(aligned\((\d+)\)\)\) uint8_t lds\[(\w+)\];", n)
        if not self.lds_decl or self.lds_decl.group(2) != "LDS_BYTES":
            raise CuError("the producer's static LDS declaration changed")
        self.band_expr = one(K, "band")
        self.band0_expr = one(K, "band0")
        if norm_ws(one(K, "inv")) != "INV[0]":
            raise CuError("INV is not read as INV[0]")
        iu = if_statements(K, r"inv_used\[(\w+)\]\s*=\s*__float_as_uint\(inv\);")
        if len(iu) != 1:
            raise CuError("INV_USED store")
        self.iu_cond, self.iu_idx = iu[0][0], iu[0][1].group(1)
        self.rvl, self.rpl = one(K, "rvl"), one(K, "rpl")
        self.rows_valid, self.rows_pad = one(K, "rows_valid"), one(K, "rows_pad")
        self.rsrc: Dict[str, Tuple[str, str, str]] = {}
        for m in re.finditer(r"const i32x4 (\w+) = make_rsrc\((\w+) \+ ([^,]+), (.+?)\);", n):
            self.rsrc[m.group(1)] = (m.group(2), norm_ws(m.group(3)), norm_ws(m.group(4)))
        if sorted(self.rsrc) != ["rdt", "rg6", "rs"]:
            raise CuError(f"descriptors {sorted(self.rsrc)}")
        lane_names = ("L.tid", "L.wv", "L.lane", "L.h", "L.hb", "x", "L.k", "L.q", "L.b", "L.ld_off", "L.wl_base",
                      "L.sw", "g0", "g1")
        pos = []
        for name in lane_names:
            got = assign_positions(K, name)
            if len(got) != 1:
                raise CuError(f"lane map: {len(got)} assignments of {name!r}")
            pos.append((got[0][0], name, got[0][1]))
        self.lane = [(nm, ex) for _, nm, ex in sorted(pos)]
        self.acc_init = one(K, "acc")
        self.t1_expr = one(K, "t1")
        start = K.index("const bool t1")
        start = K.index(";", start) + 1
        stop = K.index("uint32_t a = acc;")
        self.schedule = self._parse_schedule(K[start:stop])
        tail = K[stop:]
        self.red_updates = assignments(tail, "a")[1:]
        if assignments(tail, "a")[0] != "acc":
            raise CuError("the PARTIAL reduction does not start from acc")
        m = re.search(r"uint32_t\* red = reinterpret_cast<uint32_t\*>\(lds \+ (.+?)\);", norm_ws(tail))
        if not m:
            raise CuError("PARTIAL scratch pointer")
        self.red_base = m.group(1)
        w = if_statements(tail, r"red\[([^\]]+)\]\s*=\s*a;")
        if len(w) != 1:
            raise CuError("PARTIAL scratch write")
        self.red_w_cond, self.red_w_idx = w[0][0], norm_ws(w[0][1].group(1))
        rd = if_statements(tail, r"\{")
        if len(rd) != 1:
            raise CuError("PARTIAL final block")
        self.red_r_cond, bpos = rd[0][0], rd[0][2]
        sync_pos = tail.find("__syncthreads();", w[0][2])
        if not (w[0][2] < sync_pos < bpos):
            raise CuError("no barrier between the PARTIAL scratch writes and reads")
        blk = norm_ws(tail[bpos:])
        m = re.search(r"uint32_t m = red\[([^\]]+)\];", blk)
        lp = re.search(r"for \(int w = (\d+); w < (\d+); \+\+w\) m = umax\(m, red\[([^\]]+)\]\);", blk)
        out = re.search(r"partial\[(.+?)\] = __uint_as_float\((.+)\);", blk)
        if not (m and lp and out):
            raise CuError("PARTIAL final reduction")
        self.red_r0, self.red_ws = m.group(1), list(range(int(lp.group(1)), int(lp.group(2))))
        self.red_rw, self.p_idx, self.p_val = lp.group(3), out.group(1), out.group(2)

    def _parse_schedule(self, region: str) -> List[Tuple[Any, ...]]:
        ops: List[Tuple[Any, ...]] = []
        i = 0
        while True:
            while i < len(region) and region[i].isspace():
                i += 1
            if i >= len(region):
                return ops
            m = re.compile(r"if\s*\(").match(region, i)
            if m:
                j = _match(region, m.end() - 1, "(", ")")
                k = j + 1
                while region[k].isspace():
                    k += 1
                e = _match(region, k, "{", "}")
                ops.append(("if", norm_ws(region[m.end():j]), self._parse_schedule(region[k + 1:e])))
                i = e + 1
                continue
            m = re.compile(r"v16u\s+([\w\s,]+);").match(region, i)
            if m:
                ops.append(("decl", [s.strip() for s in m.group(1).split(",")]))
                i = m.end()
                continue
            m = re.compile(r"(\w+)\s*\(").match(region, i)
            if not m:
                raise CuError(f"schedule statement not modelled: {region[i:i + 40]!r}")
            j = _match(region, m.end() - 1, "(", ")")
            if region[j + 1:j + 2] != ";":
                raise CuError("schedule call without ';'")
            args = split_top(region[m.end():j]) if region[m.end():j].strip() else []
            ops.append(("call", m.group(1), [norm_ws(a) for a in args]))
            i = j + 2

    # ------------------------------------------------------------------------------------------------ load_pass
    def _load_pass(self) -> None:
        params, B = find_function(self.text, "load_pass")
        self.lp_params = [norm_ws(p).split()[-1] for p in split_top(params)]
        self.lp_r0, self.lp_vo = one(B, "r0"), one(B, "vo")
        self.lp_iters, body = the_loop(B, "i", "odm6_buf_ld4", "load_pass")
        self.lp_loads: Dict[str, str] = {}
        for m in re.finditer(r"(\w+)\s*=\s*odm6_buf_ld4\(", body):
            j = _match(body, m.end() - 1, "(", ")")
            args = split_top(body[m.end():j])
            if len(args) != 4 or args[3] != "0" or args[2] != "0":
                raise CuError(f"buffer load arguments {args}")
            self.lp_loads[m.group(1)] = (args[0], norm_ws(args[1]))
        m = re.search(r"v16u&\s*R\s*=\s*(.+?)\s*\?\s*(\w+)\s*:\s*(\w+)\s*;", body)
        if not m:
            raise CuError("load_pass register selection")
        self.lp_rsel = (norm_ws(m.group(1)), m.group(2), m.group(3))
        self.lp_base = one(body, "base")
        self.lp_fill = [(norm_ws(a), b, c) for a, b, c in
                        re.findall(r"R\[([^\]]+)\]\s*=\s*\(uint32_t\)(\w+)\.([xyzw])\s*;", body)]
        if len(self.lp_fill) != 8:
            raise CuError("load_pass register fill")

    # ------------------------------------------------------------------------------------------------ pass_body
    def _pass_body(self) -> None:
        params, B = find_function(self.text, "pass_body")
        self.pb_params = [norm_ws(p).split()[-1] for p in split_top(params)]
        n = norm_ws(B)
        m = re.search(r"uint8_t\* const wbase = lds \+ ([^;]+);", n)
        if not m:
            raise CuError("staging base")
        self.pb_wbase = m.group(1)
        self.pb_jj, jb = the_loop(B, "jj", "e4x4", "dout_t loop")
        jn = norm_ws(jb)
        self.pb_dw = one(jb, "dw")
        self.pb_pick = []
        for k in range(4):
            m = re.search(r"\bd%d = (\w+)\[([^\]]+)\]" % k, jn)
            if not m:
                raise CuError(f"dout_t pick d{k}")
            self.pb_pick.append((m.group(1), m.group(2)))
        m = re.search(r"if \((.+?)\) \{ x0 = hi_f\(d0\); x1 = hi_f\(d1\); x2 = hi_f\(d2\); x3 = hi_f\(d3\); \} "
                      r"else \{ x0 = lo_f\(d0\); x1 = lo_f\(d1\); x2 = lo_f\(d2\); x3 = lo_f\(d3\); \}", jn)
        if not m:
            raise CuError("dout_t half select")
        self.pb_half = m.group(1)
        for chain in ("const float s0 = inv * x0, s1 = inv * x1, s2_ = inv * x2, s3 = inv * x3;",
                      "const float t0 = tmul * s0, t1 = tmul * s1, t2 = tmul * s2_, t3 = tmul * s3;"):
            if chain not in jn:
                raise CuError(f"dout_t arithmetic changed (Triton's op order is pinned): missing {chain!r}")
        m = re.search(r"\*reinterpret_cast<uint32_t\*>\(wbase \+ (.+?)\) = e4x4\(med448\((t\d)\), med448\((t\d)\), "
                      r"med448\((t\d)\), med448\((t\d)\)\);", jn)
        if not m:
            raise CuError("staging store")
        self.pb_st_off = m.group(1)
        self.pb_byte_src = [int(m.group(g)[1]) for g in (2, 3, 4, 5)]
        self.pb_swap_iters, sb = the_loop(B, "d", "permlane16_swap", "swap loop")
        sn = norm_ws(sb)
        self.pb_swaps = []
        for m in re.finditer(r"const auto (\w+) = __builtin_amdgcn_permlane16_swap\((\w+)\[([^\]]+)\], (\w+)\[([^\]]+)\], "
                             r"(\w+), (\w+)\);", sn):
            if (m.group(6), m.group(7)) != ("false", "false"):
                raise CuError("permlane16_swap with fi / bound_ctrl set is not modelled")
            res = m.group(1)
            outs = {k: (r, norm_ws(ix)) for r, ix, rr, k in
                    re.findall(r"(\w+)\[([^\]]+)\] = (\w+)\[([01])\];", sn) if rr == res}
            if sorted(outs) != ["0", "1"]:
                raise CuError("permlane16_swap results")
            self.pb_swaps.append(((m.group(2), norm_ws(m.group(3))), (m.group(4), norm_ws(m.group(5))),
                                  outs["0"], outs["1"]))
        if len(self.pb_swaps) != 2:
            raise CuError("permlane16_swap statements")
        self.pb_m_iters, mb = the_loop(B, "d", "pkmax(m0", "E8M0 max loop")
        self.pb_m_upd = dict(straight_line(mb))
        self.pb_m_init = {s: assignments(B, f"m{s}")[0] for s in (0, 1)}
        self.pb_slot = {}
        for s in (0, 1):
            b_all = assignments(B, f"b{s}")
            if len(b_all) != 2:
                raise CuError(f"b{s}")
            self.pb_slot[s] = dict(a=one(B, f"a{s}"), b=b_all[0], b_fix=b_all[1], bs=one(B, f"bs{s}"),
                                   e=one(B, f"e{s}"), v=assignments(B, f"v{s}")[0])
        m = re.search(r"const bool nf = __builtin_amdgcn_ballot_w64\((.+?)\) != 0ull;", n)
        if not m:
            raise CuError("ballot")
        self.pb_nf = m.group(1)
        nf_blocks = []
        for mm in re.finditer(r"if \(nf\) \{", B):
            k = mm.end() - 1
            nf_blocks.append((k, _match(B, k, "{", "}")))
        inside = lambda p: any(a < p < b for a, b in nf_blocks)  # noqa: E731
        self.pb_nan = [(c, mm.group(1), mm.group(2), inside(p)) for c, mm, p in
                       if_statements(B, r"(v[01])\s*=\s*nanfree_max1\((\w+)\);")]
        self.pb_codes = re.findall(r"v6u (c[01]) = g6_codes\((\w+), (\w+)\);", n)
        if [c for c, _, _ in self.pb_codes] != ["c0", "c1"]:
            raise CuError("g6_codes calls")
        self.pb_zero = []
        for c, mm, p in if_statements(B, r"(c[01])\s*=\s*\(v6u\)\{([^}]*)\};"):
            if [x.strip() for x in mm.group(2).split(",")] != ["0u"] * 6:
                raise CuError("byte-255 zeroing literal")
            self.pb_zero.append((c, mm.group(1), inside(p)))
        self.pb_acc = one(B, "acc")
        self.pb_rr, self.pb_ro = one(B, "rr"), one(B, "ro")
        self.pb_stores = []
        for m in re.finditer(r"odm6_buf_st([124])\(", B):
            j = _match(B, m.end() - 1, "(", ")")
            args = split_top(B[m.end():j])
            if len(args) != 5 or args[1] != "rg6" or args[3] != "0" or args[4] != "0":
                raise CuError(f"G6 store arguments {args}")
            if m.group(1) in ("4", "2"):
                dm = re.fullmatch(r"\(i32x[42]\)\{(.*)\}", norm_ws(args[0]))
                elems = re.findall(r"\(int\)(\w+)\[(\d+)\]", dm.group(1)) if dm else []
                if not elems or len(elems) != int(m.group(1)) or len({v for v, _ in elems}) != 1:
                    raise CuError(f"G6 store data {args[0]!r}")
                self.pb_stores.append(("vec", elems[0][0], [int(i) for _, i in elems], norm_ws(args[2]), None))
            else:
                pre = B[:m.start()]
                cm = list(re.finditer(r"\bif\s*\(", pre))
                if not cm:
                    raise CuError("E8M0 store condition")
                cj = _match(B, cm[-1].end() - 1, "(", ")")
                if B[cj + 1:m.start()].strip():
                    raise CuError("E8M0 store is not the statement of its if")
                self.pb_stores.append(("dword", norm_ws(args[0]), None, norm_ws(args[2]),
                                       norm_ws(B[cm[-1].end():cj])))
        if len(self.pb_stores) != 5:
            raise CuError(f"{len(self.pb_stores)} G6 stores (need 4 code stores + 1 E8M0 store)")
        self.pb_u_init = {s: one(B, f"u{s}") for s in (0, 1)}
        self.pb_u_dpp = re.findall(r"(u[01]) \|= (\w+)\(\1\);", n)

    # ------------------------------------------------------------------------------------------------ copy_out
    def _copy_out(self) -> None:
        params, B = find_function(self.text, "copy_out")
        self.co_params = [norm_ws(p).split()[-1] for p in split_top(params)]
        n = norm_ws(B)
        m = re.search(r"const int (cc = [^;]*);", n)
        if not m:
            raise CuError("copy-out lane decomposition")
        self.co_decl = [tuple(x.split(" = ", 1)) for x in split_top(m.group(1))]
        self.co_lane_off = one(B, "lane_off")
        self.co_iters, lb = the_loop(B, "it", "odm6_buf_st4", "copy-out loop")
        self.co_b, self.co_grp = one(lb, "b"), one(lb, "grp")
        ln = norm_ws(lb)
        m = re.search(r"v4u v = \*reinterpret_cast<const v4u\*>\(lds \+ (.+?)\);", ln)
        if not m:
            raise CuError("copy-out LDS read")
        self.co_rd = m.group(1)
        m = re.search(r"if \((.+?)\) v = \(v4u\)\{v\.([xyzw]), v\.([xyzw]), v\.([xyzw]), v\.([xyzw])\};", ln)
        self.co_swap = (m.group(1), ["xyzw".index(m.group(g)) for g in (2, 3, 4, 5)]) if m else None
        m = re.search(r"odm6_buf_st4\(", lb)
        j = _match(lb, m.end() - 1, "(", ")")
        args = split_top(lb[m.end():j])
        if norm_ws(args[0]) != "__builtin_bit_cast(i32x4, v)" or args[1] != "rdt" or args[3:] != ["0", "0"]:
            raise CuError(f"copy-out store arguments {args}")
        self.co_st = norm_ws(args[2])

    # ------------------------------------------------------------------------------------------------ helpers
    def _dpp(self) -> None:
        self.dpp: Dict[str, Tuple[int, int, int, bool]] = {}
        for name in re.findall(r"uint32_t (dpp_\w+)\(uint32_t v\)", self.text):
            body = norm_ws(find_function(self.text, name)[1])
            m = re.fullmatch(r"return \(uint32_t\)__builtin_amdgcn_mov_dpp\(\(int\)v, (\w+), (\w+), (\w+), (true|false)"
                             r"\);", body)
            if not m:
                raise CuError(f"{name} body")
            self.dpp[name] = (int(m.group(1), 0), int(m.group(2), 0), int(m.group(3), 0), m.group(4) == "true")

    def _nanfree(self) -> None:
        params, B = find_function(self.text, "nanfree_max1")
        self.nf_param = norm_ws(params).split()[-1]
        self.nf_init = assignments(B, "m")[0]
        self.nf_iters, lb = the_loop(B, "d", "pkmax", "nanfree loop")
        self.nf_body = straight_line(lb)
        m = re.search(r"return (.+?);", norm_ws(B))
        self.nf_ret = m.group(1)

    def _g6_codes(self) -> None:
        params, B = find_function(self.text, "g6_codes")
        ps = [norm_ws(p).split()[-1] for p in split_top(params)]
        self.gc_reg, self.gc_bs = ps[0].lstrip("&"), ps[1]
        n = norm_ws(B)
        if self.g6_f32:
            m = re.fullmatch(r"const float f = __uint_as_float\((.+?)\); v16f e, o; for \(int i = 0; i < 16; \+\+i\) "
                             r"\{ e\[i\] = lo_f\(R\[i\]\) \* f; o\[i\] = hi_f\(R\[i\]\) \* f; \} return cvt_2xpk16\(e, o\);", n)
            if not m:
                raise CuError("g6_codes (G6_F32 = 1) body")
            self.gc_scale = m.group(1)
        else:
            m = re.fullmatch(r"return cvt_pk32\(R, (.+)\);", n)
            if not m:
                raise CuError("g6_codes (pk32) body")
            self.gc_scale = m.group(1)

    def _launcher(self) -> None:
        params, B = find_function(self.text, "outproj_dual_mx6_launch")
        self.launch_params = [norm_ws(p) for p in split_top(params)]
        n = norm_ws(B)
        m = re.search(r"hipLaunchKernelGGL\(odm6::outproj_dual_mx6_kernel, dim3\((.+?)\), dim3\(odm6::(\w+)\), 0,", n)
        if not m:
            raise CuError("producer launch geometry")
        self.grid_expr, self.block_name = m.group(1), m.group(2)
        self.launch_checks = [norm_ws(c) for c, _, _ in
                              if_statements(B, r"return \(int\)hipErrorInvalidValue;")]
        _, W = find_function(self.text, "outproj_w6_shadow_launch")
        self.w6_launch_checks = [norm_ws(c) for c, _, _ in if_statements(W, r"return \(int\)hipErrorInvalidValue;")]
        wn = norm_ws(W)
        dims = re.findall(r"hipLaunchKernelGGL\(odm6::outproj_w6_shadow_kernel<(true|false)>, dim3\((\d+)\), "
                          r"dim3\((\d+)\),", wn)
        if len(dims) != 2 or len({(g, b) for _, g, b in dims}) != 1:
            raise CuError("W6 launch geometry")
        self.w6_grid, self.w6_block = int(dims[0][1]), int(dims[0][2])
        body = find_function(self.text, "outproj_dual_mx6_impl")[1]
        m = re.fullmatch(r'\s*return\s+((?:"(?:[^"\\]|\\.)*"\s*)+)OUTPROJ_MX6_SRC_SHA256\s*;\s*', body)
        if not m:
            raise CuError("impl literal")
        self.impl_literal = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))
        self.variant_expr = re.search(r"return (.+);", norm_ws(find_function(self.text, "outproj_dual_mx6_variant")[1])).group(1)
        self.partial_count_expr = re.search(r"return (.+);", norm_ws(find_function(self.text, "outproj_dual_mx6_partial_count")[1])).group(1)

    def _w6(self) -> None:
        params, B = find_function(self.text, "outproj_w6_shadow_kernel")
        n = norm_ws(B)
        self.w_decl = [(nm, one(B, nm, "W6 lanes")) for nm in ("tid", "lane", "gw", "kb", "n")]
        m16 = re.search(r"reinterpret_cast<const uint16_t\*>\(wv\) \+ (.+?)\);", n)
        m32 = re.search(r"reinterpret_cast<const float\*>\(wv\) \+ (.+?)\);", n)
        if not (m16 and m32):
            raise CuError("W6 source pointers")
        self.w_off16, self.w_off32 = m16.group(1), m32.group(1)
        it16, b16 = the_loop(B, "i", "lo_f", "W6 bf16 loads")
        it32, b32 = the_loop(B, "i", "float4", "W6 fp32 loads")
        self.w_it16, self.w_it32 = it16, it32
        self.w_k16 = the_loop(b16, "k", "lo_f", "W6 bf16 widen")[0]
        bn = norm_ws(b16)
        lo = re.search(r"v\[([^\]]+)\] = lo_f\(d\[k\]\); v\[([^\]]+)\] = hi_f\(d\[k\]\);", bn)
        if not lo or "const uint32_t d[4] = {t.x, t.y, t.z, t.w};" not in bn or "const v4u t = src[i];" not in bn:
            raise CuError("W6 bf16 widening")
        self.w_lo16, self.w_hi16 = lo.group(1), lo.group(2)
        f32 = re.findall(r"v\[([^\]]+)\] = t\.([xyzw]);", norm_ws(b32))
        if len(f32) != 4 or "const float4 t = src[i];" not in norm_ws(b32):
            raise CuError("W6 fp32 loads")
        self.w_f32 = [(ix, "xyzw".index(c)) for ix, c in f32]
        self.w_bm_iters, bmb = the_loop(B, "x", "bm = umax", "W6 tile max")
        self.w_bm_upd = one(bmb, "bm")
        self.w_sh_iters, shb = the_loop(B, "o", "__shfl_xor", "W6 tile reduction")
        self.w_sh_upd = one(shb, "bm")
        self.w_eb, self.w_e, self.w_inv2e = one(B, "eb"), one(B, "e"), one(B, "inv2e")
        self.w_c_iters, cb = the_loop(B, "x", "w_code", "W6 codes")
        cn = norm_ws(cb)
        if one(cb, "c") != "w_code(v[x], inv2e)":
            raise CuError("W6 code call")
        self.w_bit, self.w_wd, self.w_o = one(cb, "bit"), one(cb, "wd"), one(cb, "o")
        m = re.search(r"d\[(\w+)\] \|= (.+?); if \((.+?)\) d\[(.+?)\] \|= (.+?);", cn)
        if not m:
            raise CuError("W6 code packing")
        self.w_pack = (m.group(1), m.group(2), m.group(3), m.group(4), m.group(5))
        z = if_statements(B, r"\{\s*for \(int i = 0; i < 6; \+\+i\) d\[i\] = 0u; \}")
        if len(z) != 1:
            raise CuError("W6 non-finite zeroing")
        self.w_zero = z[0][0]
        m = re.search(r"const int (S = [^;]*);", n)
        if not m:
            raise CuError("W6 map statement")
        self.w_map = [tuple(x.split(" = ", 1)) for x in split_top(m.group(1))]
        if [nm for nm, _ in self.w_map] != ["S", "q", "t", "ws_", "jj", "R"]:
            raise CuError(f"W6 map names {[nm for nm, _ in self.w_map]}")
        m = re.search(r"uint8_t\* const slice = w6 \+ (.+?);", n)
        mh = re.search(r"\*reinterpret_cast<v4u\*>\(slice \+ (.+?)\) = \(v4u\)\{d\[(\d)\], d\[(\d)\], d\[(\d)\], "
                       r"d\[(\d)\]\};", n)
        mt = re.search(r"\*reinterpret_cast<uint2\*>\(slice \+ (.+?)\) = make_uint2\(d\[(\d)\], d\[(\d)\]\);", n)
        ms = if_statements(B, r"w6s\[(.+?)\] = \(uint8_t\)eb;")
        if not (m and mh and mt and len(ms) == 1):
            raise CuError("W6 stores")
        self.w_slice, self.w_head = m.group(1), (mh.group(1), [int(mh.group(g)) for g in (2, 3, 4, 5)])
        self.w_tail = (mt.group(1), [int(mt.group(2)), int(mt.group(3))])
        self.w_s_cond, self.w_s_idx = ms[0][0], ms[0][1].group(1)
        _, E = find_function(self.text, "w_e8m0")
        self.w_e8m0 = []
        for st in split_top(E, ";"):
            st = norm_ws(st)
            if not st:
                continue
            mi = re.fullmatch(r"if \((.+)\) return (.+)", st)
            md = re.fullmatch(r"(?:int\s+)?(\w+) = (.+)", st)
            mr = re.fullmatch(r"return (.+)", st)
            if mi:
                self.w_e8m0.append(("ret_if", mi.group(1), mi.group(2)))
            elif md:
                self.w_e8m0.append(("set", md.group(1), md.group(2)))
            elif mr:
                self.w_e8m0.append(("ret", None, mr.group(1)))
            else:
                raise CuError(f"w_e8m0 statement {st!r}")


# =====================================================================================================================
# instruction models (vectorised over the 512 threads of a workgroup)
# =====================================================================================================================
def _u32cv(a: np.ndarray) -> CV:
    return CV(np.asarray(a, dtype=np.int64) & 0xFFFFFFFF, "u32")


def _umax(a: CV, b: CV) -> CV:
    return CV(np.maximum(wrap(a.v, "u32"), wrap(b.v, "u32")), "u32")


def _pk(fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Callable[[CV, CV], CV]:
    def f(a: CV, b: CV) -> CV:
        x, y = wrap(a.v, "u32"), wrap(b.v, "u32")
        lo = fn(x & 0xFFFF, y & 0xFFFF) & 0xFFFF
        hi = fn(x >> 16, y >> 16) & 0xFFFF
        return CV((hi << 16) | lo, "u32")
    return f


def _pkasr15(a: CV) -> CV:
    x = wrap(a.v, "u32")
    lo = np.where(x & 0x8000, 0xFFFF, 0)
    hi = np.where(x & 0x80000000, 0xFFFF, 0)
    return CV((hi << 16) | lo, "u32")


class Regs:
    """A v16u register set (or a uint32 array) viewed by the evaluator: R[d] -> the lane vector of dword d."""

    def __init__(self, arr: np.ndarray) -> None:
        self.a = arr

    def __getitem__(self, idx: CV) -> CV:
        return CV(self.a[:, idx.scalar()].astype(np.int64), "u32")


class Mem:
    """A uint32 array indexed per lane (red[] scratch, w6s)."""

    def __init__(self, arr: np.ndarray) -> None:
        self.a = arr

    def __getitem__(self, idx: CV) -> CV:
        return CV(self.a[np.asarray(idx.v, dtype=np.int64)].astype(np.int64), "u32")


def dpp_model(tid: np.ndarray, ctrl: int, row_mask: int, bank_mask: int, bound_ctrl: bool) -> np.ndarray:
    """Source lane of every lane for __builtin_amdgcn_mov_dpp: quad_perm (0x00-0xFF), row_ror (0x121-0x12F)."""
    if row_mask != 0xF or bank_mask != 0xF or not bound_ctrl:
        raise CuError("DPP with partial row / bank masks is not modelled")
    if 0 <= ctrl <= 0xFF:
        perm = np.array([(ctrl >> (2 * i)) & 3 for i in range(4)])
        return (tid & ~3) | perm[tid & 3]
    if 0x121 <= ctrl <= 0x12F:
        return (tid & ~15) | (((tid & 15) - (ctrl & 0xF)) & 15)
    raise CuError(f"DPP control {ctrl:#x} is not modelled")


def permlane16_swap(x: np.ndarray, y: np.ndarray, reversed_rows: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """V_PERMLANE16_SWAP_B32 vdst=x, src0=y (App. B): the odd 16-lane rows of x <-> the even rows of y, per 32 lanes.
    reversed_rows: mutant (even rows of x <-> odd rows of y)."""
    nt = x.shape[0]
    x = x.reshape(nt // 32, 2, 16).copy()
    y = y.reshape(nt // 32, 2, 16).copy()
    xs, ys = (0, 1) if reversed_rows else (1, 0)
    t = y[:, ys].copy()
    y[:, ys] = x[:, xs]
    x[:, xs] = t
    return x.reshape(nt), y.reshape(nt)


def _bank_max(word_addr: np.ndarray, groups: Sequence[Sequence[int]], width: int) -> int:
    """Max lanes-per-bank over the given lane groups (64 banks x 4 B); width = dwords per lane access."""
    worst = 0
    for w in range(word_addr.shape[0] // 64):
        a = word_addr[w * 64:(w + 1) * 64]
        for gl in groups:
            banks = np.concatenate([(a[list(gl)] + d) % 64 for d in range(width)])
            worst = max(worst, int(np.bincount(banks, minlength=64).max()))
    return worst


_PHASE_CONSEC = [list(range(0, 16)), list(range(16, 32)), list(range(32, 48)), list(range(48, 64))]
_PHASE_MEASURED = [[0, 1, 2, 3, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27],
                   [4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 28, 29, 30, 31],
                   [32, 33, 34, 35, 44, 45, 46, 47, 52, 53, 54, 55, 56, 57, 58, 59],
                   [36, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 51, 60, 61, 62, 63]]
STANDIN_CODE = 0x1F  # [H] what the converter leaves for an Inf / NaN block (anything non-zero): the kernel must zero it
POISON_BF16 = 0x5F5F  # dout rows >= M in the emulated memory (finite, large): a read past num_records shows up
G6_GUARD = 4096
_DECL_TYPES = {"int": "i32", "uint32_t": "u32", "long long": "i64", "bool": "i32"}


def decl_type(text: str, name: str) -> Optional[str]:
    """C type of the declaration that introduces `name` (the statement's leading type), None for a plain assignment."""
    for pos, _ in assign_positions(text, name):
        k = pos
        while k > 0 and text[k - 1] not in ";{}":
            k -= 1
        st = norm_ws(text[k:pos])
        m = re.match(r"(?:const\s+)?(long long|uint32_t|int|bool)\b", st)
        if m:
            return _DECL_TYPES[m.group(1)]
    return None


def _as(cv: CV, t: Optional[str]) -> CV:
    if t is None or t == cv.t:
        return cv
    if t == "i32" and cv.t == "i32":
        return cv
    return CV(wrap(cv.v, t), t)


class Emulator:
    """Lane-level emulation of outproj_dual_mx6_kernel (all bands of one launch) and of outproj_w6_shadow_kernel, from
    a CuModel. mutants: instruction-model mutants of DESIGN App. B ('permlane_reversed', 'fix_neg0')."""

    def __init__(self, model: CuModel, mutants: Sequence[str] = ()) -> None:
        self.m = model
        self.mut = set(mutants)
        bad = self.mut - {"permlane_reversed", "fix_neg0"}
        if bad:
            raise ValueError(f"unknown instruction mutants {sorted(bad)}")
        self.NT = model.consts["NT"]
        if self.NT % 64:
            raise CuError("NT is not a whole number of waves")
        self.tid = np.arange(self.NT, dtype=np.int64)
        self.dpp_src = {nm: dpp_model(self.tid, *spec) for nm, spec in model.dpp.items()}
        _, K = find_function(model.text, "outproj_dual_mx6_kernel")
        _, PB = find_function(model.text, "pass_body")
        self.k_text, self.pb_text = K, PB
        lane_struct = re.search(r"struct Lane \{([^}]*)\};", model.text)
        self.lane_types: Dict[str, str] = {}
        for decl in lane_struct.group(1).split(";"):
            d = norm_ws(decl)
            m = re.match(r"(int|uint32_t) (.+)", d)
            if m:
                for f in m.group(2).split(","):
                    self.lane_types["L." + f.strip()] = _DECL_TYPES[m.group(1)]

    # ------------------------------------------------------------------------------------------------ env
    def _fns(self) -> Dict[str, Any]:
        env: Dict[str, Any] = {"umax": _umax, "pkmax": _pk(np.maximum), "pkadd": _pk(lambda a, b: a + b),
                               "pkasr15": _pkasr15, "__uint_as_float": lambda x: CV(wrap(x.v, "u32"), "u32"),
                               "__float_as_uint": lambda x: CV(wrap(x.v, "u32"), "u32")}

        def mk_dpp(src: np.ndarray) -> Callable[[CV], CV]:
            return lambda x: CV(wrap(np.broadcast_to(np.asarray(x.v), (self.NT,))[src], "u32"), "u32")

        for nm, src in self.dpp_src.items():
            env[nm] = mk_dpp(src)

        def shfl(x: CV, mk: CV) -> CV:
            k = mk.scalar()
            if not 0 <= k < 64:
                raise CuError("__shfl_xor across waves")
            return CV(np.broadcast_to(np.asarray(x.v), (self.NT,))[self.tid ^ k], x.t)

        env["__shfl_xor"] = shfl
        return env

    def _vec(self, cv: CV) -> np.ndarray:
        return np.broadcast_to(np.asarray(cv.v, dtype=np.int64), (self.NT,)).copy()

    # ------------------------------------------------------------------------------------------------ memory
    @staticmethod
    def _range(off: np.ndarray, size: int, nrec: int, what: str) -> np.ndarray:
        o = np.asarray(off, dtype=np.int64) & 0xFFFFFFFF
        if np.any(o % size):
            raise Fail(f"{what}: offset not {size}-B aligned")
        inr = o + size <= nrec
        if np.any(~inr & (o < nrec)):
            raise Fail(f"{what}: access straddles num_records ({nrec})")
        return inr

    def _load(self, st: Dict[str, Any], desc: str, off: CV, size: int) -> Tuple[np.ndarray, bool]:
        mem, base, nrec = st["desc"][desc]
        o = self._vec(off) & 0xFFFFFFFF
        inr = self._range(o, size, nrec, "buffer load")
        out = np.zeros((self.NT, size // 4), dtype=np.uint32)
        if inr.any():
            idx = (base + o[inr])[:, None] + np.arange(size)[None, :]
            out[inr] = np.ascontiguousarray(mem[idx]).view(np.uint32).reshape(-1, size // 4)
        return out, bool(inr.any())

    def _store(self, st: Dict[str, Any], desc: str, off: CV, data: np.ndarray, active: Optional[np.ndarray] = None,
               what: str = "buffer store") -> None:
        mem, base, nrec = st["desc"][desc]
        cnt = st["cnt"][desc]
        size = data.shape[1] * 4
        o = self._vec(off) & 0xFFFFFFFF
        act = np.ones(self.NT, bool) if active is None else active
        inr = self._range(o[act], size, nrec, what)
        lanes = np.nonzero(act)[0][inr]
        if lanes.size:
            idx = (base + o[lanes])[:, None] + np.arange(size)[None, :]
            if idx.max() >= mem.size:
                raise Fail(f"{what}: outside the emulated allocation")
            mem[idx.reshape(-1)] = np.ascontiguousarray(data[lanes].astype(np.uint32)).view(np.uint8).reshape(-1)
            np.add.at(cnt, idx.reshape(-1), 1)

    # ------------------------------------------------------------------------------------------------ one launch
    def chain_lut(self, inv_b: int, tmul_b: int) -> np.ndarray:
        """The kernel's per-element dout_t chain (pinned: s = inv * x, st = tmul * s, med448, pk fp8 at scale 1.0) on
        all 65,536 bf16 patterns, under the kernel's MODE (FP16_OVFL setreg; DN1 setreg -> subnormal products flush)."""
        ftz = self.m.dn1_setreg
        s = f32_mul(np.uint32(inv_b), _ALL16 << 16, ftz)
        t = f32_mul(np.uint32(tmul_b), s, ftz)
        q = v_med3_f32(t, F32_NEG_E4M3_MAX, F32_POS_E4M3_MAX)
        return cvt_f32_to_fp8(q, "e4m3fn", self.m.fp16_ovfl_setreg)

    def run(self, bits: np.ndarray, inv: Any, t_shift: int, m_pad: Optional[int] = None) -> Dict[str, Any]:
        m = self.m
        BAND = m.consts["BAND"]
        bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16))
        M = int(bits.shape[0])
        MP = m_pad if m_pad is not None else (M + 127) // 128 * 128
        nb = ev(m.grid_expr, {"M": cv_i32(M)}).scalar()
        inv_b, tmul_b = int(inv_bits_of(inv)), int(t_mul_bits(t_shift))
        lut4 = self.chain_lut(inv_b, tmul_b)
        dmem = np.full(((nb + 1) * BAND, KD), POISON_BF16, dtype=np.uint16)
        dmem[:M] = bits
        n_dt = MP // 128 * 65536
        mems = {"dout": dmem.view(np.uint8).reshape(-1),
                "g6": np.full(M * REC + G6_GUARD, 0xA5, dtype=np.uint8),
                "dout_t": np.full(n_dt + 65536, 0x5A, dtype=np.uint8)}
        cnts = {"g6": np.zeros(mems["g6"].size, np.int32), "dout_t": np.zeros(mems["dout_t"].size, np.int32)}
        part = np.full(4 * nb + 4, 0xDEADBEEF, dtype=np.uint32)
        pc = np.zeros(part.size, np.int32)
        iu = np.full(4, 0xDEADBEEF, dtype=np.uint32)
        iuc = np.zeros(4, np.int32)
        stats: Dict[str, int] = {}
        for bx in range(nb):
            self._band(bx, nb, M, MP, inv_b, tmul_b, lut4, mems, cnts, part, pc, iu, iuc, stats)
        return dict(g6=mems["g6"], g6c=cnts["g6"], dt=mems["dout_t"], dtc=cnts["dout_t"], part=part, pc=pc, iu=iu,
                    iuc=iuc, stats=stats, MP=MP, nb=nb, M=M)

    def _band(self, bx: int, nb: int, M: int, MP: int, inv_b: int, tmul_b: int, lut4: np.ndarray,
              mems: Dict[str, np.ndarray], cnts: Dict[str, np.ndarray], part: np.ndarray, pc: np.ndarray,
              iu: np.ndarray, iuc: np.ndarray, stats: Dict[str, int]) -> None:
        m, K = self.m, self.k_text
        env = self._fns()
        env.update({nm: cv_i32(v) for nm, v in m.consts.items()})
        env.update({"blockIdx.x": cv_u32(bx), "gridDim.x": cv_u32(nb), "M": cv_i32(M), "M_PAD": cv_i32(MP)})
        for nm in ("band", "band0", "rvl", "rpl", "rows_valid", "rows_pad"):
            expr = {"band": m.band_expr, "band0": m.band0_expr, "rvl": m.rvl, "rpl": m.rpl,
                    "rows_valid": m.rows_valid, "rows_pad": m.rows_pad}[nm]
            env[nm] = _as(ev(expr, env), decl_type(K, nm))
        st: Dict[str, Any] = {"desc": {}, "cnt": {}, "regs": {}, "pending": {}, "step": 0, "last_write": -1,
                              "last_read": -1, "last_sync": -1, "lut4": lut4, "stats": stats}
        ptrs = {"dout": "dout", "g6": "g6", "dout_t": "dout_t"}
        for name, (ptr, off, nrec) in m.rsrc.items():
            if ptr not in ptrs:
                raise CuError(f"descriptor {name} on {ptr}")
            st["desc"][name] = (mems[ptr], ev(off, env).scalar(), ev(nrec, env).scalar() & 0xFFFFFFFF)
            st["cnt"][name] = cnts.get(ptr)
        env["threadIdx.x"] = CV(self.tid, "u32")
        for nm, ex in m.lane:
            env[nm] = _as(ev(ex, env), self.lane_types.get(nm, decl_type(K, nm)))
        st["acc"] = CV(self._vec(_as(ev(m.acc_init, env), "u32")), "u32")
        st["lds"] = np.full(m.consts["LDS_BYTES"], 0x77, dtype=np.uint8)
        st["ldsc"] = np.zeros(m.consts["LDS_T"], dtype=np.int32)
        env["t1"] = _as(ev(m.t1_expr, env), "i32")
        self._exec(m.schedule, env, st)
        for reg, (p, touched) in st["pending"].items():
            if touched:
                raise Fail(f"band {bx}: the unconsumed prefetch into {reg} (pass {p}) read memory")
        # PARTIAL: lane reduce -> LDS scratch -> barrier -> threads 0..3
        e = dict(env)
        e["a"] = st["acc"]
        for expr in m.red_updates:
            e["a"] = _as(ev(expr, e), "u32")
        rb = ev(m.red_base, e).scalar()
        if rb < m.consts["LDS_T"] or rb + 4 * 32 > m.consts["LDS_BYTES"]:
            raise Fail("PARTIAL scratch overlaps the staging image or leaves the LDS")
        red = np.full((m.consts["LDS_BYTES"] - rb) // 4, 0xDEADBEEF, dtype=np.uint32)
        wm = np.asarray(ev(m.red_w_cond, e).v) != 0
        wm = np.broadcast_to(wm, (self.NT,))
        widx = self._vec(ev(m.red_w_idx, e))[wm]
        if widx.size != np.unique(widx).size or widx.min() < 0 or widx.max() >= red.size:
            raise Fail("PARTIAL scratch writes collide or leave the scratch")
        red[widx] = self._vec(e["a"])[wm]
        rm = np.broadcast_to(np.asarray(ev(m.red_r_cond, e).v) != 0, (self.NT,))
        sub = {k: (CV(np.broadcast_to(np.asarray(v.v), (self.NT,))[rm], v.t)
                   if isinstance(v, CV) and np.ndim(v.v) == 1 else v) for k, v in e.items()}
        sub["red"] = Mem(red)
        sub["m"] = _as(ev(f"red[{m.red_r0}]", sub), "u32")
        for w in m.red_ws:
            sub["w"] = cv_i32(w)
            sub["m"] = _as(ev(f"umax(m, red[{m.red_rw}])", sub), "u32")
        pidx = np.asarray(ev(m.p_idx, sub).v, dtype=np.int64)
        pval = wrap(ev(m.p_val, sub).v, "u32")
        if np.any(pidx < 0) or np.any(pidx >= part.size):
            raise Fail("PARTIAL store outside the allocation")
        part[pidx] = pval
        np.add.at(pc, pidx, 1)
        # INV_USED (raw bits of INV[0])
        e2 = dict(env)
        e2.update({"inv_used": cv_i32(1), "nullptr": cv_i32(0)})
        im = np.broadcast_to(np.asarray(ev(m.iu_cond, e2).v) != 0, (self.NT,))
        if im.any():
            ix = np.broadcast_to(np.asarray(ev(m.iu_idx, e2).v), (self.NT,))[im]
            iu[ix] = inv_b
            np.add.at(iuc, ix, 1)

    def _exec(self, ops: List[Tuple[Any, ...]], env: Dict[str, Any], st: Dict[str, Any]) -> None:
        m = self.m
        for op in ops:
            st["step"] += 1
            if op[0] == "decl":
                for r in op[1]:
                    st["regs"][r] = np.full((self.NT, 16), 0xDEADBEEF, dtype=np.uint32)
            elif op[0] == "if":
                if ev(op[1], env).scalar():
                    self._exec(op[2], env, st)
            elif op[1] == "__syncthreads":
                st["last_sync"] = st["step"]
            elif op[1] == "load_pass":
                a = dict(zip(m.lp_params, op[2]))
                self._load_pass(env, st, a["P"], a["Q"], a["rs"], ev(a["p"], env).scalar(), env[a["g"]])
            elif op[1] == "pass_body":
                a = dict(zip(m.pb_params, op[2]))
                if a["rg6"] != "rg6" or a["acc"] != "acc" or a["inv"] != "inv" or a["tmul"] != "tmul":
                    raise CuError(f"pass_body call arguments {op[2]}")
                if st["last_read"] > st["last_sync"]:
                    raise Fail("a pass rewrites the staging image while a copy-out may still read it (no barrier)")
                self._pass_body(env, st, a["P"], a["Q"], ev(a["p"], env).scalar(), env[a["g"]])
                st["last_write"] = st["step"]
            elif op[1] == "copy_out":
                a = dict(zip(m.co_params, op[2]))
                if a["rdt"] != "rdt":
                    raise CuError(f"copy_out call arguments {op[2]}")
                if st["last_write"] > st["last_sync"]:
                    raise Fail("a copy-out reads the staging image without a barrier after the pass writes")
                ldsc = st["ldsc"]
                if not ((ldsc == 1).sum() == 65536 and ldsc.max() == 1):
                    raise Fail(f"stage tile not written exactly once ({int((ldsc == 1).sum())} bytes once, max "
                               f"{int(ldsc.max())})")
                self._copy_out(env, st, ev(a["tile_off"], env).scalar(), env[a["tid"]])
                ldsc[:] = 0
                st["last_read"] = st["step"]
            else:
                raise CuError(f"schedule call {op[1]} is not modelled")

    def _load_pass(self, env: Dict[str, Any], st: Dict[str, Any], Pn: str, Qn: str, rs: str, p: int, g: CV) -> None:
        m = self.m
        e = dict(env)
        e["p"], e["g"] = cv_i32(p), g
        e["r0"] = _as(ev(m.lp_r0, e), "u32")
        e["vo"] = _as(ev(m.lp_vo, e), "i32")
        regs = {"P": st["regs"][Pn].copy(), "Q": st["regs"][Qn].copy()}
        touched = False
        for i in m.lp_iters:
            e["i"] = cv_i32(i)
            got = {}
            for var, (desc, off) in m.lp_loads.items():
                if desc != "rs":
                    raise CuError("load_pass reads a descriptor other than its rs parameter")
                got[var], t = self._load(st, rs, ev(off, e), 16)
                touched |= t
            target = m.lp_rsel[1] if ev(m.lp_rsel[0], e).scalar() else m.lp_rsel[2]
            e["base"] = cv_i32(ev(m.lp_base, e).scalar())
            for ix, var, comp in m.lp_fill:
                regs[target][:, ev(ix, e).scalar()] = got[var][:, "xyzw".index(comp)]
        st["regs"][Pn], st["regs"][Qn] = regs["P"], regs["Q"]
        st["pending"][Pn] = (p, touched)
        st["pending"][Qn] = (p, touched)

    def _lds_write(self, st: Dict[str, Any], addr: np.ndarray, val: np.ndarray) -> None:
        if np.any(addr % 4) or np.any(addr < 0) or np.any(addr + 4 > self.m.consts["LDS_T"]):
            raise Fail("staging write misaligned or outside the 16 x 33 x 128 image")
        words = addr // 4
        st["stats"]["stage_write"] = max(st["stats"].get("stage_write", 0), _bank_max(words, [list(range(64))], 1))
        idx = (addr[:, None] + np.arange(4)[None, :]).reshape(-1)
        st["lds"][idx] = val.astype(np.uint32).view(np.uint8)
        np.add.at(st["ldsc"], idx, 1)

    def _pass_body(self, env: Dict[str, Any], st: Dict[str, Any], Pn: str, Qn: str, p: int, g: CV) -> None:
        m = self.m
        st["pending"].pop(Pn, None)
        st["pending"].pop(Qn, None)
        e = dict(env)
        e["p"], e["g"] = cv_i32(p), g
        X = {"P": st["regs"][Pn], "Q": st["regs"][Qn]}
        lut4 = st["lut4"]
        wbase = self._vec(ev(m.pb_wbase, e))
        for jj in m.pb_jj:
            e["jj"] = cv_i32(jj)
            e["dw"] = cv_i32(ev(m.pb_dw, e).scalar())
            d = [X[reg][:, ev(ix, e).scalar()] for reg, ix in m.pb_pick]
            hi = ev(m.pb_half, e).scalar() != 0
            h = [(dd >> 16) if hi else (dd & 0xFFFF) for dd in d]
            val = np.zeros(self.NT, dtype=np.uint32)
            for j, src in enumerate(m.pb_byte_src):
                val |= lut4[h[src]].astype(np.uint32) << np.uint32(8 * j)
            self._lds_write(st, wbase + self._vec(ev(m.pb_st_off, e)), val)
        rev = "permlane_reversed" in self.mut
        for dd in m.pb_swap_iters:
            e["d"] = cv_i32(dd)
            for (r1, i1), (r2, i2), (o0r, o0i), (o1r, o1i) in m.pb_swaps:
                x, y = X[r1][:, ev(i1, e).scalar()].copy(), X[r2][:, ev(i2, e).scalar()].copy()
                nx, ny = permlane16_swap(x, y, rev)
                X[o0r][:, ev(o0i, e).scalar()] = nx
                X[o1r][:, ev(o1i, e).scalar()] = ny
        e["P"], e["Q"] = Regs(X["P"]), Regs(X["Q"])
        PB = self.pb_text
        for s in (0, 1):
            e[f"m{s}"] = _as(ev(m.pb_m_init[s], e), "u32")
        for dd in m.pb_m_iters:
            e["d"] = cv_i32(dd)
            for nm, ex in m.pb_m_upd.items():
                e[nm] = _as(ev(ex, e), "u32")
        for s in (0, 1):
            sl = m.pb_slot[s]
            e[f"a{s}"] = _as(ev(sl["a"], e), decl_type(PB, f"a{s}"))
            e[f"b{s}"] = _as(ev(sl["b"], e), decl_type(PB, f"b{s}"))
            e[f"b{s}"] = _as(ev(sl["b_fix"], e), e[f"b{s}"].t)
            for k in ("bs", "e", "v"):
                e[f"{k}{s}"] = _as(ev(sl[k], e), decl_type(PB, f"{k}{s}"))
        nfl = np.broadcast_to(np.asarray(ev(m.pb_nf, e).v) != 0, (self.NT,))
        nf = np.repeat(nfl.reshape(-1, 64).any(axis=1), 64)
        for cond, var, reg, inside in m.pb_nan:
            mask = np.broadcast_to(np.asarray(ev(cond, e).v) != 0, (self.NT,)) & (nf if inside else True)
            if mask.any():
                e[var] = CV(np.where(mask, self._nanfree(X[reg]), self._vec(e[var])), "u32")
        codes = {}
        for var, reg, bs in m.pb_codes:
            codes[var] = self._g6_codes(X[reg], self._vec(e[bs]))
        for cond, var, inside in m.pb_zero:
            mask = np.broadcast_to(np.asarray(ev(cond, e).v) != 0, (self.NT,)) & (nf if inside else True)
            codes[var][mask] = 0
        st["acc"] = e["acc"] = CV(self._vec(_as(ev(m.pb_acc, dict(e, acc=st["acc"])), "u32")), "u32")
        e["rr"] = _as(ev(m.pb_rr, e), decl_type(PB, "rr"))
        e["ro"] = _as(ev(m.pb_ro, e), decl_type(PB, "ro"))
        for s in (0, 1):
            e[f"u{s}"] = _as(ev(m.pb_u_init[s], e), "u32")
        for var, fn in m.pb_u_dpp:
            e[var] = CV(wrap(self._vec(e[var]) | self._vec(e[fn](e[var])), "u32"), "u32")
        for kind, what, idxs, off, cond in m.pb_stores:
            if kind == "vec":
                self._store(st, "rg6", ev(off, e), codes[what][:, idxs], what="G6 store")
            else:
                act = np.broadcast_to(np.asarray(ev(cond, e).v) != 0, (self.NT,))
                data = self._vec(_as(ev(what, e), "u32"))[:, None]
                self._store(st, "rg6", ev(off, e), data, act, what="G6 E8M0 store")

    def _nanfree(self, R: np.ndarray) -> np.ndarray:
        m = self.m
        e = self._fns()
        e[m.nf_param] = Regs(R)
        e["m"] = _as(ev(m.nf_init, e), "u32")
        for dd in m.nf_iters:
            e["d"] = cv_i32(dd)
            for nm, ex in m.nf_body:
                e[nm] = _as(ev(ex, e), "u32")
        return self._vec(_as(ev(m.nf_ret, e), "u32"))

    def _g6_codes(self, R: np.ndarray, bs: np.ndarray) -> np.ndarray:
        """The FP6 convert of one row slot (16 dwords = 32 elements in element order) at the kernel's scale operand."""
        el = np.stack([R & 0xFFFF, R >> 16], axis=2).reshape(self.NT, 32).astype(np.int64)
        a16 = (el & 0x7FFF).max(axis=1)
        byte = e8m0_byte_np(a16)
        fin = byte != BYTE_NONFINITE
        env = self._fns()
        env[self.m.gc_bs] = CV(bs, "u32")
        sbits = wrap(ev(self.m.gc_scale, env).v, "u32")
        codes = np.full((self.NT, 32), STANDIN_CODE, dtype=np.int64)
        if self.m.g6_f32:
            want = (254 - np.where(fin & (a16 != 0), byte, 127)) << 23
            if not np.array_equal(sbits, want):
                raise Fail("G6_F32 pre-scale is not 2^-e of SPEC_G6's scale byte (zero / non-finite blocks: 1.0)")
            y = f32_mul((el << 16).astype(np.uint32), sbits[:, None].astype(np.uint32), self.m.dn1_setreg)
            with np.errstate(invalid="ignore"):
                a = np.abs((y & 0x7FFFFFFF).astype(np.uint32).view(np.float32).astype(np.float64))
            a = np.where(np.isfinite(a), a, 0.0)
            c = np.where(a < 2.0, np.rint(a * 8.0), np.where(a < 4.0, 8.0 + np.rint(a * 4.0), 16.0 + np.rint(a * 2.0)))
            c = np.minimum(c, 31.0).astype(np.int64) | ((y.astype(np.int64) >> 26) & 0x20)
            codes = np.where(fin[:, None], c, codes)
        else:
            if not np.array_equal(sbits, scale_operand_bits(byte).astype(np.int64)):
                raise Fail("scale operand != SPEC_G6 scale_operand_bits (zero / non-finite blocks need 1.0)")
            normal = (sbits & 0x7FFFFF) == 0
            if not np.all(normal | (sbits == 0x00400000)):
                raise Fail("scale operand is not a power of two")
            sbyte = np.where(sbits == 0x00400000, 0, sbits >> 23)
            if fin.any():
                try:
                    codes[fin] = e2m3_code_np(el[fin], sbyte[fin][:, None])
                except ValueError as err:
                    raise Fail(f"convert input outside the block's E2M3 range: {err}")
        if "fix_neg0" in self.mut:
            codes = np.where((codes & 31) == 0, 0, codes)
        return pack24_dwords(codes).astype(np.uint32)

    def _copy_out(self, env: Dict[str, Any], st: Dict[str, Any], tile_off: int, tid: CV) -> None:
        m = self.m
        e = dict(env)
        e["tid"], e["tile_off"] = tid, cv_u32(tile_off)
        for nm, ex in m.co_decl:
            e[nm] = _as(ev(ex, e), "i32")
        e["lane_off"] = _as(ev(m.co_lane_off, e), "u32")
        lds = st["lds"]
        for it in m.co_iters:
            e["it"] = cv_i32(it)
            e["b"] = _as(ev(m.co_b, e), "i32")
            e["grp"] = _as(ev(m.co_grp, e), "i32")
            la = self._vec(ev(m.co_rd, e))
            if np.any(la % 16) or np.any(la < 0) or np.any(la + 16 > m.consts["LDS_T"]):
                raise Fail("copy-out read misaligned or outside the staging image")
            words = la // 4
            st["stats"]["copy_read_consec"] = max(st["stats"].get("copy_read_consec", 0),
                                                  _bank_max(words, _PHASE_CONSEC, 4))
            st["stats"]["copy_read_measured"] = max(st["stats"].get("copy_read_measured", 0),
                                                    _bank_max(words, _PHASE_MEASURED, 4))
            v = np.ascontiguousarray(lds[la[:, None] + np.arange(16)[None, :]]).view(np.uint32).reshape(self.NT, 4)
            if m.co_swap is not None and ev(m.co_swap[0], e).scalar():
                v = v[:, m.co_swap[1]]
            off = ev(m.co_st, e)
            _, _, nrec = st["desc"]["rdt"]
            if np.any((self._vec(off) & 0xFFFFFFFF) + 16 > nrec):
                raise Fail("copy-out store outside the rdt descriptor")
            self._store(st, "rdt", off, v, what="dout_t store")

    # ------------------------------------------------------------------------------------------------ W6 kernel
    def run_w6(self, w: np.ndarray) -> Dict[str, Any]:
        """outproj_w6_shadow_kernel<BF16> on weight `w` (fp32 or bf16 numpy [1536, 512]): W6 [589,824] + W6S [768]
        with per-byte write counts (guards included)."""
        m = self.m
        bf16 = np.asarray(w).dtype == np.uint16
        nthr = m.w6_grid * m.w6_block
        if nthr % 64 or m.w6_block % 64:
            raise CuError("W6 launch is not whole waves")
        gtid = np.arange(nthr, dtype=np.int64)
        env = self._fns()
        env.update({nm: cv_i32(v) for nm, v in m.consts.items()})
        env["threadIdx.x"] = CV(gtid % m.w6_block, "u32")
        env["blockIdx.x"] = CV(gtid // m.w6_block, "u32")

        def shfl(x: CV, mk: CV) -> CV:
            k = mk.scalar()
            if not 0 <= k < 64:
                raise CuError("__shfl_xor across waves")
            return CV(np.broadcast_to(np.asarray(x.v), (nthr,))[gtid ^ k], x.t)

        env["__shfl_xor"] = shfl
        for nm, ex in m.w_decl:
            env[nm] = _as(ev(ex, env), "i32")
        v = np.zeros((nthr, 32), dtype=np.uint32)
        vc = np.zeros((nthr, 32), dtype=np.int32)
        if bf16:
            mem = np.ascontiguousarray(w, dtype=np.uint16).reshape(-1)
            off = np.broadcast_to(np.asarray(ev(m.w_off16, env).v, dtype=np.int64), (nthr,))
            for i in m.w_it16:
                base = off + 8 * i
                if np.any((base * 2) % 16):
                    raise Fail("W6 bf16 load not 16-B aligned")
                d = mem[base[:, None] + np.arange(8)[None, :]].astype(np.uint32)
                d = d[:, 0::2] | (d[:, 1::2] << 16)
                env["i"] = cv_i32(i)
                for k in m.w_k16:
                    env["k"] = cv_i32(k)
                    lo, hi = ev(m.w_lo16, env).scalar(), ev(m.w_hi16, env).scalar()
                    v[:, lo] = d[:, k] << 16
                    v[:, hi] = d[:, k] & 0xFFFF0000
                    vc[:, lo] += 1
                    vc[:, hi] += 1
        else:
            mem = np.ascontiguousarray(w, dtype=np.float32).view(np.uint32).reshape(-1)
            off = np.broadcast_to(np.asarray(ev(m.w_off32, env).v, dtype=np.int64), (nthr,))
            for i in m.w_it32:
                base = off + 4 * i
                if np.any((base * 4) % 16):
                    raise Fail("W6 fp32 load not 16-B aligned")
                t = mem[base[:, None] + np.arange(4)[None, :]]
                env["i"] = cv_i32(i)
                for ix, comp in m.w_f32:
                    x = ev(ix, env).scalar()
                    v[:, x] = t[:, comp]
                    vc[:, x] += 1
        if not (vc == 1).all():
            raise Fail("W6: the 32 values of a lane are not each loaded exactly once")
        env["v"] = Regs(v)
        env["bm"] = cv_u32(0)
        for x in m.w_bm_iters:
            env["x"] = cv_i32(x)
            env["bm"] = _as(ev(m.w_bm_upd, env), "u32")
        for o in m.w_sh_iters:
            env["o"] = cv_i32(o)
            env["bm"] = _as(ev(m.w_sh_upd, env), "u32")
        eb_env = dict(env)
        res = np.zeros(nthr, dtype=np.int64)
        done = np.zeros(nthr, dtype=bool)
        for kind, a, b in m.w_e8m0:
            if kind == "ret_if":
                c = (np.broadcast_to(np.asarray(ev(a, eb_env).v), (nthr,)) != 0) & ~done
                res = np.where(c, np.broadcast_to(ev(b, eb_env).arr(), (nthr,)), res)
                done |= c
            elif kind == "set":
                eb_env[a] = _as(ev(b, eb_env), "i32")
            else:
                res = np.where(done, res, np.broadcast_to(ev(b, eb_env).arr(), (nthr,)))
                done[:] = True
        env["eb"] = cv_i32(res)
        env["e"] = _as(ev(m.w_e, env), "i32")
        inv2e = wrap(ev(m.w_inv2e, env).v, "u32")
        d = np.zeros((nthr, 6), dtype=np.int64)
        for x in m.w_c_iters:
            env["x"] = cv_i32(x)
            c = w_code_model(v[:, x], inv2e)
            env["c"] = CV(c, "u32")
            env["bit"] = _as(ev(m.w_bit, env), "i32")
            env["wd"] = _as(ev(m.w_wd, env), "i32")
            env["o"] = _as(ev(m.w_o, env), "i32")
            i0, v0, cond, i1, v1 = m.w_pack
            d[:, ev(i0, env).scalar()] |= wrap(ev(v0, env).v, "u32")
            if ev(cond, env).scalar():
                d[:, ev(i1, env).scalar()] |= wrap(ev(v1, env).v, "u32")
        d &= 0xFFFFFFFF
        zm = np.broadcast_to(np.asarray(ev(m.w_zero, env).v) != 0, (nthr,))
        d[zm] = 0
        for nm, ex in m.w_map:
            env[nm] = _as(ev(ex, env), "i32")
        w6 = np.full(589824 + 4096, 0xEE, dtype=np.uint8)
        w6c = np.zeros(w6.size, np.int32)
        sl = np.broadcast_to(np.asarray(ev(m.w_slice, env).v, dtype=np.int64), (nthr,))
        for (expr, idxs) in (m.w_head, m.w_tail):
            o = sl + np.broadcast_to(np.asarray(ev(expr, env).v, dtype=np.int64), (nthr,))
            size = 4 * len(idxs)
            if np.any(o % size) or np.any(o < 0) or np.any(o + size > w6.size):
                raise Fail("W6 store misaligned or outside the allocation")
            idx = (o[:, None] + np.arange(size)[None, :]).reshape(-1)
            w6[idx] = np.ascontiguousarray(d[:, idxs].astype(np.uint32)).view(np.uint8).reshape(-1)
            np.add.at(w6c, idx, 1)
        w6s = np.full(768 + 256, 0xEE, dtype=np.uint8)
        w6sc = np.zeros(w6s.size, np.int32)
        sm = np.broadcast_to(np.asarray(ev(m.w_s_cond, env).v) != 0, (nthr,))
        si = np.broadcast_to(np.asarray(ev(m.w_s_idx, env).v, dtype=np.int64), (nthr,))[sm]
        if np.any(si < 0) or np.any(si >= w6s.size):
            raise Fail("W6S store outside the allocation")
        w6s[si] = (res[sm] & 0xFF).astype(np.uint8)
        np.add.at(w6sc, si, 1)
        return dict(w6=w6, w6c=w6c, w6s=w6s, w6sc=w6sc)


def w_code_model(y_bits: np.ndarray, inv2e_bits: np.ndarray) -> np.ndarray:
    """outproj_dual_mx6.cu w_code (pinned text): v = fl32(y * inv2e); a = min(|v|, 7.5) (NaN stays NaN); E2M3 by
    rintf (RNE) per binade; NaN -> 0; sign = signbit(v) (0x20)."""
    v = f32_mul(_u32(y_bits), _u32(np.broadcast_to(inv2e_bits, np.shape(y_bits))))
    with np.errstate(invalid="ignore"):
        a = (v & 0x7FFFFFFF).astype(np.uint32).view(np.float32).astype(np.float64)
        a = np.where(a > 7.5, 7.5, a)
        c = np.where(a < 2.0, np.rint(a * 8.0), np.where(a < 4.0, np.rint(a * 4.0) + 8.0, np.rint(a * 2.0) + 16.0))
    c = np.where(np.isnan(a), 0.0, c).astype(np.int64)
    return c | ((v.astype(np.int64) >> 26) & 0x20)


# =====================================================================================================================
# comparisons (DESIGN App. B compare(), extended: INV_USED, PARTIAL write-once, bank-conflict verdict)
# =====================================================================================================================
def compare(emu: Emulator, bits: np.ndarray, inv: Any, t: int, label: str = "") -> Dict[str, Any]:
    bits = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16))
    m = int(bits.shape[0])
    r = emu.run(bits, inv, t)
    ref = encode_g6_int(bits).reshape(-1)
    tri = dual_cast(bits, inv, t, True)
    g6, g6c, dt, dtc = r["g6"], r["g6c"], r["dt"], r["dtc"]
    n_dt = r["MP"] // 128 * 65536
    pb, tb = r["part"][:4 * r["nb"]], tri["partial_bits"]
    pn, tn = is_nan32(pb), is_nan32(tb)
    res = dict(case=label, M=m, t=int(t), inv=f"{int(inv_bits_of(inv)):#010x}",
               g6_mis=int((g6[:m * REC] != ref).sum()), g6_guard=int((g6[m * REC:] != 0xA5).sum()),
               g6_once=bool((g6c[:m * REC] == 1).all()) and int(g6c[m * REC:].sum()) == 0,
               dt_mis=int((dt[:n_dt] != tri["dout_t"].reshape(-1)).sum()), dt_guard=int((dt[n_dt:] != 0x5A).sum()),
               dt_once=bool((dtc[:n_dt] == 1).all()) and int(dtc[n_dt:].sum()) == 0,
               p_mis=int(((pn != tn) | (~pn & (pb != tb))).sum()) if pb.size == tb.size else max(pb.size, tb.size),
               p_len=int(pb.size), p_once=bool((r["pc"][:pb.size] == 1).all()) and int(r["pc"][pb.size:].sum()) == 0,
               p_nan=int(tn.sum()), iu_ok=bool(r["iuc"][0] == 1 and int(r["iuc"][1:].sum()) == 0
                                             and int(r["iu"][0]) == int(inv_bits_of(inv))),
               stage_write_maxway=r["stats"].get("stage_write"), copy_read_consec=r["stats"].get("copy_read_consec"),
               copy_read_measured=r["stats"].get("copy_read_measured"))
    res["ok"] = (res["g6_mis"] == 0 and res["g6_guard"] == 0 and res["g6_once"] and res["dt_mis"] == 0
                 and res["dt_guard"] == 0 and res["dt_once"] and res["p_mis"] == 0 and res["p_once"]
                 and pb.size == tb.size and res["iu_ok"])
    res["banks_ok"] = all(res[k] == 1 for k in ("stage_write_maxway", "copy_read_consec", "copy_read_measured"))
    return res


def compare_safe(emu: Emulator, bits: np.ndarray, inv: Any, t: int, label: str = "") -> Dict[str, Any]:
    """compare() with an emulated rule violation (Fail) or a model refusal (CuError) reported as a failing case."""
    try:
        return compare(emu, bits, inv, t, label)
    except (Fail, CuError) as err:
        return dict(case=label, M=int(np.asarray(bits).shape[0]), t=int(t), ok=False, banks_ok=False,
                    error=f"{type(err).__name__}: {err}")


# =====================================================================================================================
# cases: the in-tree vectors, lab rows (edge classes + adversarial rows at head / boundaries / tail), C4 tiles
# =====================================================================================================================
_VEC: Dict[str, np.ndarray] = {}


def vectors() -> Dict[str, np.ndarray]:
    """outproj_dual_mx6_vectors.npz (copied from r14 iter66 U1 spec/g6_vectors_v1.npz; see the test for the pins)."""
    if not _VEC:
        with np.load(VECTORS_PATH) as z:
            _VEC.update({k: z[k] for k in z.files})
    return _VEC


def npz30_rows() -> np.ndarray:
    return vectors()["dout_bits"].astype(np.uint16)


_GRID = [c / 8.0 if c < 16 else ((c - 8) / 4.0 if c < 24 else (c - 16) / 2.0) for c in range(32)]
_MIDS = [(_GRID[i] + _GRID[i + 1]) / 2.0 for i in range(31)]
EDGE_CLASSES = ("normal", "zero", "negzero", "subn", "tiny", "deep", "outlier", "tie", "amax75", "amax75p", "huge",
                "nonfinite")
_EDGE_P = (0.0, 0.03, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.002, 0.001)


def edge_rows(m: int, seed: int = 4) -> np.ndarray:
    """uint16 [m, 512] bf16 patterns: production-like rows (N(0, 1) x a log-normal row scale ~1e-4 x a column factor)
    with r16 REF synth_case 'edge' block classes (the _inject_edge class list and rates, re-implemented with numpy:
    zero, -0, bf16 subnormals, tiny, deep, outlier, RNE ties, amax 7.5 * 2^e, next-bf16, huge, Inf / NaN) and 0.5 %
    all-zero rows. Deterministic (PCG64 seeded by (seed, m))."""
    rng = np.random.default_rng([int(seed), int(m), 0x6F646D36])
    f32 = np.float32
    z = rng.standard_normal((m, KD)).astype(np.float32)
    rs = np.exp(np.log(1e-4) + rng.standard_normal((m, 1))).clip(1e-6, 2.5e-3).astype(np.float32)
    cf = np.exp(0.3 * rng.standard_normal((1, KD))).astype(np.float32)
    x = (z * rs * cf).astype(np.float32)
    p = np.array(_EDGE_P, dtype=np.float64)
    p[0] = 1.0 - p[1:].sum()
    cls = rng.choice(len(EDGE_CLASSES), size=(m, NBLK), p=p)
    xb = x.reshape(m, NBLK, BLK)
    mids, grid = np.array(_MIDS, dtype=np.float32), np.array(_GRID, dtype=np.float32)
    subn_bits = None
    for ci, name in enumerate(EDGE_CLASSES):
        if name == "normal":
            continue
        sel = np.argwhere(cls == ci)
        k = int(len(sel))
        if not k:
            continue
        rr, bb, ar = sel[:, 0], sel[:, 1], np.arange(k)
        sg = np.where(rng.random((k, BLK)) < 0.5, f32(-1.0), f32(1.0)).astype(np.float32)
        zz = rng.standard_normal((k, BLK)).astype(np.float32)
        pos = rng.integers(0, BLK, k)
        rsv = rs[rr, 0][:, None]
        sc = np.ldexp(np.ones(k, dtype=np.float32), rng.integers(-24, -4, k)).astype(np.float32)[:, None]
        if name == "zero":
            v = np.zeros((k, BLK), dtype=np.float32)
        elif name == "negzero":
            v = np.full((k, BLK), -0.0, dtype=np.float32)
        elif name == "subn":
            v = None
            subn_bits = (rr, bb, (np.where(sg < 0, 0x8000, 0) | rng.integers(1, 128, (k, BLK))).astype(np.uint16))
        elif name == "tiny":
            v = zz * f32(2.0 ** -122)
        elif name == "deep":
            v = zz * f32(2.0 ** -129)
        elif name == "outlier":
            v = zz * rsv * f32(2.0 ** -6)
            v[ar, pos] = sg[:, 0] * (f32(1.0) + rng.random(k).astype(np.float32)) * rsv[:, 0] * f32(2.0 ** 6)
        elif name == "tie":
            tt = np.where(rng.random((k, BLK)) < 0.7, mids[rng.integers(0, 31, (k, BLK))],
                          grid[rng.integers(0, 32, (k, BLK))]).astype(np.float32)
            v = sg * tt * sc
            v[ar, pos] = sg[:, 1] * f32(7.5) * sc[:, 0]
        elif name in ("amax75", "amax75p"):
            base = (np.clip(zz, -3.0, 3.0) * f32(2.0) * sc).astype(np.float32)
            v = (rne_bf16_bits(base).astype(np.uint32) << 16).view(np.float32).copy()
            v[ar, pos] = sg[:, 2] * (f32(7.5) if name == "amax75" else f32(7.53125)) * sc[:, 0]
        elif name == "huge":
            v = zz * f32(2.0 ** 100)
        else:
            v = zz * rsv
            v[ar, pos] = np.array([np.inf, -np.inf, np.nan], dtype=np.float32)[rng.integers(0, 3, k)]
        if v is not None:
            xb[rr, bb] = v.astype(np.float32)
    zrow = rng.random(m) < 0.005
    with np.errstate(over="ignore", invalid="ignore"):
        bits = rne_bf16_bits(x)
    if subn_bits is not None:
        bits.reshape(m, NBLK, BLK)[subn_bits[0], subn_bits[1]] = subn_bits[2]
    bits[zrow] = 0
    return bits


def lab_rows(m: int, seed: int = 4) -> np.ndarray:
    """U1 lab_case_bits' composition on in-tree rows: edge_rows(m) with the 30 adversarial rows (npz30) written over
    rows [0, 30), over the 8 rows straddling 128, 256, 384, 512, 768 and 1024, and over the last min(30, m) rows."""
    bits = edge_rows(m, seed)
    adv = npz30_rows()
    nr = adv.shape[0]

    def put(r0: int, n: int, a0: int = 0) -> None:
        for i in range(n):
            r = r0 + i
            if 0 <= r < m:
                bits[r] = adv[(a0 + i) % nr]

    put(0, min(nr, m))
    for t in (128, 256, 384, 512, 768, 1024):
        put(t - 4, 8, t)
    put(m - min(nr, m), min(nr, m), 7)
    return bits


def c4_tile(m: int) -> np.ndarray:
    a = np.full((m, KD), 0x3F00, dtype=np.uint16)
    a[:, 128:] = 0x3E80
    return a


def c4_cases() -> List[Tuple[str, np.ndarray]]:
    """Critique C4's PARTIAL tiles (DESIGN App. B): P1 one NaN + 2.0, P2 NaN + Inf, P3 all-NaN real band with masked
    rows (M 100), P4 all-NaN full tile, P3b (M 128), P3c (an all-sNaN-free 0x7FC1 128-row real band at M 384)."""
    out = []
    c = c4_tile(256)
    c[5, 7], c[9, 100] = 0x7FC0, 0x4000
    out.append(("C4-P1", c))
    c = c4_tile(256)
    c[3, 3], c[4, 4] = 0x7FC0, 0x7F80
    out.append(("C4-P2", c))
    c = c4_tile(100)
    c[:, :128] = 0x7FC0
    out.append(("C4-P3", c))
    c = c4_tile(256)
    c[:, :128] = 0x7FC0
    out.append(("C4-P4", c))
    c = c4_tile(128)
    c[:, :128] = 0x7FC0
    out.append(("C4-P3b M=128", c))
    c = c4_tile(384)
    c[256:, :128] = 0x7FC1
    out.append(("C4-P3c M=384", c))
    return out


T13_M = (1, 3, 64, 100, 127, 128, 129, 255, 256, 257, 336, 383, 384, 408, 411, 513, 769, 1000)
INV_CASES = (("publish(143)", 0x424880E5), ("NaN", 0x7FC00000), ("+Inf", 0x7F800000), ("0", 0x00000000),
             ("-1.0", 0xBF800000))

# DESIGN App. B's 10 mutants: 'cu' = an edit of the .cu text (then re-extracted: the emulator follows the edit);
# 'emu' = an instruction-model mutant (the edit has no .cu counterpart).
MUTANTS: Tuple[Tuple[str, str, Any], ...] = (
    ("C original zero-block scale (b0 before the zero test)", "cu",
     (("const uint32_t bs0 = (a0 - 1u) >= 0x7f7fu ? 127u : (uint32_t)b0;", "const uint32_t bs0 = (uint32_t)b0;"),
      ("const uint32_t bs1 = (a1 - 1u) >= 0x7f7fu ? 127u : (uint32_t)b1;", "const uint32_t bs1 = (uint32_t)b1;"))),
    ("permlane16_swap rows reversed", "emu", "permlane_reversed"),
    ("no dword-pair swap in copy-out", "cu", (("if ((it >> 1) & 1) v = (v4u){v.z, v.w, v.x, v.y};", ""),)),
    ("PARTIAL from the NaN-dominant max", "cu",
     (("if (a0 > 0x7f80u) v0 = nanfree_max1(P);", ""), ("if (a1 > 0x7f80u) v1 = nanfree_max1(Q);", ""))),
    ("band key init 0", "cu", (("uint32_t acc = rows_valid < BAND ? 1u : 0u;", "uint32_t acc = 0u;"),)),
    ("E8M0 dword in [k][q] order", "cu", (("ro + 400 * L.k + 384 + 4 * L.q", "ro + 400 * L.k + 384 + 4 * L.k"),)),
    ("fix_neg0 (0x20 -> 0x00)", "emu", "fix_neg0"),
    ("no byte-255 zeroing", "cu",
     (("if (a0 >= 0x7f80u) c0 = (v6u){0u, 0u, 0u, 0u, 0u, 0u};", ""),
      ("if (a1 >= 0x7f80u) c1 = (v6u){0u, 0u, 0u, 0u, 0u, 0u};", ""))),
    ("staging swizzle without 16*hb", "cu",
     (("L.sw = 2u * (uint32_t)(L.b >> 1) + 16u * (uint32_t)L.hb;", "L.sw = 2u * (uint32_t)(L.b >> 1);"),)),
    ("PARTIAL strip from q (row_ror 1/2)", "cu",
     (("(int)v, 0x124, 0xF, 0xF, true", "(int)v, 0x121, 0xF, 0xF, true"),
      ("(int)v, 0x128, 0xF, 0xF, true", "(int)v, 0x122, 0xF, 0xF, true"))),
)

# Further single-constant edits of the DESIGN sec. 5 maps (T13: "a source edit that breaks a map fails on CPU").
SOURCE_EDITS: Tuple[Tuple[str, str, str], ...] = (
    ("load map 32hb -> 16hb", "L.ld_off = 64u * L.b + 32u * L.hb;", "L.ld_off = 64u * L.b + 16u * L.hb;"),
    ("quad index g1 = 16 + g0 -> 8 + g0", "g1 = 16 + g0;", "g1 = 8 + g0;"),
    ("staging slot 33 -> 32 (write)", "(uint32_t)(33 * L.b + 16 * L.hb) * 128u", "(uint32_t)(32 * L.b + 16 * L.hb) * 128u"),
    ("copy-out read slot 33 -> 32", "lds + (33 * b + cw) * 128 + 16 * grp", "lds + (32 * b + cw) * 128 + 16 * grp"),
    ("copy-out group 4hbw -> 2hbw", "(hp ^ ((b >> 2) + 4 * hbw)) & 7", "(hp ^ ((b >> 2) + 2 * hbw)) & 7"),
    ("copy-out column stride 4096 -> 2048", "4096u * (uint32_t)it", "2048u * (uint32_t)it"),
    ("G6 row 2hb -> hb", "+ 2u * (uint32_t)L.hb;", "+ 1u * (uint32_t)L.hb;"),
    ("G6 tail 8b -> 8q", "ro + 256 + 8 * L.b", "ro + 256 + 8 * L.q"),
    ("G6 record 400 -> 384", "const int ro = (int)(rr * 400u);", "const int ro = (int)(rr * 384u);"),
    ("E8M0 quad_perm [1,0,3,2] -> identity", "(int)v, 0xB1, 0xF, 0xF, true", "(int)v, 0xE4, 0xF, 0xF, true"),
    ("PARTIAL xor 16 -> 8", "__shfl_xor((int)a, 16)", "__shfl_xor((int)a, 8)"),
    ("PARTIAL wave read w * 4 + tid -> w * 4 + (tid ^ 1)", "red[w * 4 + L.tid]", "red[w * 4 + (L.tid ^ 1)]"),
    ("G6 descriptor rows_valid -> rows_pad", "(uint32_t)rows_valid * 400u", "(uint32_t)rows_pad * 400u"),
    ("dout descriptor rows_valid -> rows_pad", "(uint32_t)rows_valid * 1024u", "(uint32_t)rows_pad * 1024u"),
    ("swap pairs P[8 + d] -> P[4 + d]", "permlane16_swap(P[d], P[8 + d], false, false); P[d] = r[0]; P[8 + d] = r[1];",
     "permlane16_swap(P[d], P[4 + d], false, false); P[d] = r[0]; P[4 + d] = r[1];"),
    ("dout_t pick P[8 + dw] -> P[dw + 1]", "d2 = P[8 + dw]", "d2 = P[dw + 1]"),
)


def apply_edits(src: str, edits: Sequence[Tuple[str, str]]) -> str:
    for old, new in edits:
        if src.count(old) != 1:
            raise ValueError(f"edit target occurs {src.count(old)} times: {old!r}")
        src = src.replace(old, new)
    return src


def read_cu(path: str = CU_PATH) -> str:
    with open(path, "r") as f:
        return f.read()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
