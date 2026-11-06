# Grouped FlyDSL LayerNorm(512) dx-sum backward with per-problem output modes:
# e5m2 dx + amax partials for E-M3a's FP8 L2 GEMMs (problems 0, 1), bf16 dx for
# the action problem (2). Arbor actckpt iteration 53, ledger entry
# A-em3a-em2-grouped-e5m2 (stacks E-M3a on E-M2 / P7 m1b).
"""ln_bwd_grouped3_fp8: ln_bwd_grouped.py's grouped kernel with an e5m2 dx epilogue.

E-M3a (ops/preproc_fp8_l2.py, DLRM_HSTU_FP8_PREPROC_L2=1) runs the FP8 L2
GEMMs of the content (0) and additional (1) MLPs, whose backward needs the
LayerNorm(512) dx as e5m2 bytes plus amax partials of the fp32 dx (P3,
triton_preproc_fp8.ln512_bwd_dxsum_emit, one call per problem), while the
action MLP (2) stays bf16 (one PR #7 call). This module does the three in ONE
launch that reads dy once: 6 KB/row (dy 1, x 3x1, dx 0.5 + 0.5 + 1) instead
of 2 x P3 + 1 x PR #7 = 8 KB/row.

The main kernel (ln_bwd_flydsl_grp3f8_d512_dxsum_x{R}_c{C}) is
ln_bwd_grouped.build_ln_bwd_grouped op for op (same grid G, rows per
iteration, XCD block order, slot-major/problem-minor order, nine fp32
accumulators, LDS reduction and PART layout, finalize
ln_bwd_grouped.build_finalize_grouped unchanged); only the dx epilogue of the
problems whose MODES entry is 1 differs:
  * the fp32 dx that feeds the dx-sum accumulator is the same value (the
    quantization works on a separate temporary);
  * s = dx * inv_p (inv_p: the fp32 scalar INV_p[0], loaded once per block);
    s = minimum(maximum(s, -57344), 57344) (llvm.maximum/minimum: a
    NaN-propagating clamp, v_maximum3_f32 + v_minimum3_f32, exactly what
    P3's tl.maximum/minimum(propagate_nan=ALL) compiles to); 8 elements -> 8
    e5m2 bytes with four v_cvt_scalef32_pk_bf8_f32 at scale 1.0 (2 x f32 ->
    2 bytes, RNE; P3's conversion instruction) -> one 8-byte buffer store at
    byte row*512 + col of DX8_p, whose buffer resource spans M*512 bytes from
    the runtime M. Every NaN becomes byte 0xFE (the hardware's; P3's too;
    torch's .to(float8_e5m2) writes 0x7F | sign instead), +-inf -> +-57344;
  * amax: per lane a running fp32 max of |dx| (maxnum: NaN-ignoring, from 0,
    as P3's tl.maximum) over its 8 elements, slots and iterations; per wave a
    DPP/swizzle/bpermute max (the ln_bwd._xor_sum tree with max); the waves'
    maxima go to a small LDS area before the existing first reduction
    barrier, after which thread 0 stores one partial per block:
    AMAX_p[lb] (G entries, lb the logical block; the same partial count as
    P3's sms*8 grid at production N, so site_steady sees the same length).

BITWISE CONTRACT (per problem p, against ln_bwd_grouped.ln_bwd_grouped3 with
the same options):
  * p with MODES[p] == 0 (action): bf16 dx bitwise;
  * every p: dW_p, dB_p, dx-sum_p bitwise;
  * p with MODES[p] == 1: the bytes are e5m2(clamp_nanprop(fp32_dx * inv_p,
    +-57344)) where fp32_dx is exactly the fp32 value ln_bwd_grouped3 rounds
    to bf16 and e5m2() is P3's Triton conversion (RNE, OCP e5m2, subnormals
    kept, every NaN -> 0xFE); the partials' max over non-NaN entries is the
    max over non-NaN |fp32_dx| (never NaN: maxnum from 0; inf if dx has inf).
Against P3 itself the bytes are NOT bitwise: P3's fp32 dx is computed in
Triton's (x - mean) * rstd order with its own grid, so it differs from this
fp32 dx at the ulp level. Iteration-53 lab: the bytes differ in ~6-7e-7 of
the elements on model-like data (up to ~2e-6 on heavy-tailed data, more for
signed zeros at extreme inv), and a differing byte can be several codes
apart near cancellation; the dequantized SNR vs an fp64 oracle is
identical; dW/dB/dx-sum within 7e-7 of the vector max; partial maxima equal
or 1 ulp apart.
Verified in the Arbor lab (kernel lab results:
/scratch/arbor_sessions/dlrmv4_20261005_actckpt/probes/iter53/kernel/results53.json;
independent bit check:
/scratch/arbor_sessions/dlrmv4_20261005_actckpt/probes/iter53/verify_bits/):
the conversion against P3's exact Triton sequence (0 mismatches) and torch
(NaN bytes only) on every e5m2 value, midpoints +-1 ulp, subnormals,
+-0/inf/overflow, 64 NaN payloads, f32 denormals and 23M random
values/products with non-power-of-two inv (conv53.py); the kernel against
ln_bwd_grouped3 and a DEBUG_F32 build (fp32 dx stored as two 16-bit planes),
76 cases (three runs) at N in {0, 1, 17, 4099, 100003, 1990001, 2097152,
2097153, 2333000, 2500001, 2776000} with model-like, dy*1000, zero-dy,
constant-x and NaN/inf rows, all bitwise / byte-exact, deterministic, and
MODES=(0,0,0) bitwise to ln_bwd_grouped3 (t53k.py).
Resources (gfx950): 234 VGPR (grp3: 222), 0 spills, no scratch, 24.1 KB LDS,
2 waves/SIMD like grp3.

DEBUG_F32 (verification only): an FP8 problem's output is an int16 [2N, 512]
tensor holding the fp32 dx bits (rows 0..N-1 the high halves, N..2N-1 the low
halves; two resources of N*1024 bytes, so any N under ln_bwd's bounds works).
NO_AMAX (timing decomposition only): no amax arithmetic, AMAX untouched.

Buffer bounds come from the runtime M (or G) or explicit byte counts only:
the flyc.jit cache keys tensors by type, not shape (ln_bwd.py docstring).
"""

import functools
from typing import List, Optional, Sequence, Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl.compiler.kernel_function import CompilationContext
try:
    # FlyDSL 0.2.4 (ROCm 10.2 UTD image) renamed fly_values to
    # extract_to_ir_values; same contract: the list of ir.Values of a DSL
    # value, [0] being the tensor's memref.
    from flydsl.compiler.protocol import extract_to_ir_values as fly_values
except ImportError:  # FlyDSL 0.1.1 (rocm/primus:v26.3)
    from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, buffer_ops, gpu, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from generative_recommenders.ops.flydsl import ln_bwd as _v2
from generative_recommenders.ops.flydsl.ln_bwd import (
    CDNA_BUFFER_FLAGS, _DPP_STEPS, _SWZ_XOR16, _as_i32, _elems, _f32, _i32, _idx, _rsrc, _u, _v8f,
    _xor_sum,
)
from generative_recommenders.ops.flydsl.ln_bwd_grouped import (  # noqa: F401  (NPROB re-exported)
    D_GROUPED, GROUPED_CFG, NPROB, build_finalize_grouped, grouped_kernel_options, grouped_reason,
)

# per-problem output mode in GroupedTail order (0 content, 1 additional, 2 action):
# 1 = e5m2 dx + amax partials, 0 = bf16 dx. The only mode set the API serves.
MODES_PROD = (1, 1, 0)
E5M2_MAX = 57344.0
# e5m2 conversion of the FP8 epilogue: (instruction, clamp), chosen by the
# iteration-53 conversion study (conv53.py; results53.json step2_conversion):
#   instruction "pk"  v_cvt_pk_bf8_f32 (rocdl.cvt.pk.bf8.f32), 2 x f32 -> 2 bytes
#               "sc1" v_cvt_scalef32_pk_bf8_f32 with scale 1.0 (P3's instruction)
#   clamp "sel"  select(isnan(s), s, minnum(maxnum(s, -L), L)) (v_med3 + v_cmp_u + v_cndmask)
#         "mm"   minimum(maximum(s, -L), L): llvm.maximum/minimum, NaN-propagating,
#                v_maximum3_f32 + v_minimum3_f32 on gfx950 (P3's instructions)
#         "num"  minnum(maxnum(s, -L), L) (NaN-ignoring; study only)
#         "none" no clamp (study only)
# pk/sc1 x sel/mm all gave P3's bytes on every probe; sc1+mm is P3's own
# instruction sequence (its ISA: v_maximum3_f32, v_minimum3_f32,
# v_cvt_scalef32_pk_bf8_f32 ..., 1.0) at 2 clamp ops per element.
_CVT = ("sc1", "mm")


# ----------------------------------------------------------------------------- epilogue helpers
def _clamp_e5m2(s, mode):
    """vec8 f32 -> vec8 f32 clamped to +-E5M2_MAX (mode: see _CVT)."""
    if mode == "none":
        return _u(s)
    lo = arith.constant_vector(-E5M2_MAX, _v8f())
    hi = arith.constant_vector(E5M2_MAX, _v8f())
    if mode == "mm":
        return arith.minimumf(arith.maximumf(_u(s), _u(lo)), _u(hi))
    c = arith.minnumf(arith.maxnumf(_u(s), _u(lo)), _u(hi))
    if mode == "num":
        return c
    assert mode == "sel", mode
    nan = arith.cmpf(arith.CmpFPredicate.UNO, _u(s), _u(s))
    return arith.select(nan, _u(s), c)


def _e5m2_pack8(c, inst):
    """vec8 f32 (already clamped) -> vec2 i32: bytes e0..e7 in element order."""
    e = _elems(c)
    words = []
    for h in range(2):
        a = [_u(e[4 * h + j]) for j in range(4)]
        if inst == "pk":
            w = _u(_i32(0))
            w = rocdl_raw.cvt_pk_bf8_f32(T.i32, a[0], a[1], w, False)
            w = rocdl_raw.cvt_pk_bf8_f32(T.i32, a[2], a[3], w, True)
        else:
            assert inst == "sc1", inst
            one = _u(_f32(1.0))
            v = _u(arith.constant_vector(0, T.vec(2, T.i16)))
            v = rocdl_raw.cvt_scalef32_pk_bf8_f32(T.vec(2, T.i16), v, a[0], a[1], one, False)
            v = rocdl_raw.cvt_scalef32_pk_bf8_f32(T.vec(2, T.i16), v, a[2], a[3], one, True)
            w = vector.extract(vector.bitcast(T.vec(1, T.i32), v), static_position=[0],
                               dynamic_position=[])
        words.append(_u(w))
    return vector.from_elements(T.vec(2, T.i32), words)


def _amax8(acc, v):
    """Running max of |v_j| (maxnum: a NaN element leaves acc unchanged). |x|
    as a sign-bit clear (LLVM folds it into fabs / the max3 abs modifiers)."""
    bits = vector.bitcast(T.vec(8, T.i32), _u(v))
    mag = arith.andi(_u(bits), _u(arith.constant_vector(0x7FFFFFFF, T.vec(8, T.i32))))
    a = vector.bitcast(_v8f(), _u(mag))
    for x in _elems(a):
        acc = arith.maxnumf(_u(acc), _u(x))
    return acc


def _xor_max(s, lane, width, dpp=False):
    """ln_bwd._xor_sum's lane tree with maxnum instead of add (order-free)."""
    k = 1
    if dpp:
        while k < width and k in _DPP_STEPS:
            si = llvm.BitcastOp(T.i32, _u(s)).result
            pi = rocdl.update_dpp(T.i32, si, si, _DPP_STEPS[k], 0xF, 0xF, True)
            s = arith.maxnumf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
            k *= 2
        if k == 16 and k < width:
            si = llvm.BitcastOp(T.i32, _u(s)).result
            pi = rocdl.ds_swizzle(T.i32, si, _u(_i32(_SWZ_XOR16)))
            s = arith.maxnumf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
            k *= 2
    while k < width:
        addr = arith.ArithValue(arith.xori(_u(lane), _u(_i32(k)))) * _i32(4)
        pi = rocdl.ds_bpermute(T.i32, _u(addr), llvm.BitcastOp(T.i32, _u(s)).result)
        s = arith.maxnumf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
        k *= 2
    return s


def _rsrc_at(tensor, byte_off_i64, num_bytes_i64):
    """Buffer resource based `byte_off_i64` bytes into `tensor`, num_records =
    num_bytes_i64 (ln_bwd._shifted_rsrc with byte units and an i64 size)."""
    ptr_ty = ir.Type.parse("!llvm.ptr")
    base = fly.extract_aligned_pointer_as_index(ptr_ty, fly_values(tensor)[0])
    base_i64 = llvm.PtrToIntOp(T.i64, base).result
    new_i64 = llvm.AddOp(_u(base_i64), _u(byte_off_i64), llvm.IntegerOverflowFlags(0)).result
    new_ptr = llvm.IntToPtrOp(ptr_ty, new_i64).result
    return rocdl_raw.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"), new_ptr,
        _u(arith.constant(0, type=T.i16)), _u(num_bytes_i64), _u(_i32(CDNA_BUFFER_FLAGS)),
    ).result


# ----------------------------------------------------------------------------- main kernel
@functools.lru_cache(maxsize=None)
def build_ln_bwd_grouped_fp8(D: int = 512, WAVES: int = 4, UNROLL: int = 4, CACHE: int = 0,
                             DPP: bool = True, REMAP: int = 0, STORE_CACHE: int = -1,
                             MODES: Tuple[int, ...] = MODES_PROD, DEBUG_F32: bool = False,
                             NO_AMAX: bool = False, CVT: Optional[Tuple[str, str]] = None):
    """ln_bwd_grouped.build_ln_bwd_grouped with per-problem output MODES (1 =
    e5m2 dx + amax partials, 0 = bf16 dx). CACHE / STORE_CACHE / REMAP as
    there (the e5m2 stores use the dx store policy). DEBUG_F32 / NO_AMAX: see
    the module docstring. CVT: the conversion (default _CVT; study only)."""
    assert D in (256, 512), D
    assert not isinstance(REMAP, bool) and REMAP >= 0, REMAP
    MODES = tuple(int(m) for m in MODES)
    assert len(MODES) == NPROB and all(m in (0, 1) for m in MODES), MODES
    CVT = tuple(CVT) if CVT else _CVT
    F8 = [p for p in range(NPROB) if MODES[p]]
    NF8 = len(F8)
    AMAX = NF8 > 0 and not NO_AMAX
    SCACHE = CACHE if STORE_CACHE < 0 else STORE_CACHE
    NXCD = _v2.N_XCD             # a closure value: part of the jit cache key
    LPR = D // 8
    RPW = 64 // LPR
    RPI = WAVES * RPW
    RPS = RPI * UNROLL
    SLOTS = WAVES * RPW
    THREADS = 64 * WAVES
    VEC = D // THREADS
    assert VEC * THREADS == D, (D, THREADS)
    NACC = 3
    # ln_bwd_grouped.py's naming with grp3f8; variant tags after P7's suffixes
    name = (f"ln_bwd_flydsl_grp{NPROB}f8_d{D}_dxsum" + (f"_x{REMAP}" if REMAP else "")
            + (f"_c{CACHE}" if CACHE else "") + (f"_s{SCACHE}" if SCACHE != CACHE else "")
            + ("" if MODES == MODES_PROD else "_m" + "".join(str(m) for m in MODES))
            + ("_dbg32" if DEBUG_F32 else "") + ("_noamax" if NO_AMAX else "")
            + ("" if CVT == _CVT else f"_{CVT[0]}{CVT[1]}"))
    allocator = SmemAllocator(None, arch=get_rocm_arch(),
                              global_sym_name=f"{name}_w{WAVES}_u{UNROLL}_p{int(DPP)}_smem")
    off_red = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_red + NACC * SLOTS * D * 4
    off_amx = 0
    if AMAX:
        off_amx = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_amx + NF8 * WAVES * 4

    def vadd(a, b):
        return arith.addf(_u(a), _u(b))

    def vmul(a, b):
        return arith.mulf(_u(a), _u(b))

    def bcast(s):
        return vector.broadcast(_v8f(), _u(s))

    def vfma(a, b, c):
        return vector.FMAOp(_u(a), _u(b), _u(c)).result

    def vneg(a):
        return arith.negf(_u(a))

    def hsum8(v):
        h4 = arith.addf(vector.shuffle(_u(v), _u(v), [0, 1, 2, 3]),
                        vector.shuffle(_u(v), _u(v), [4, 5, 6, 7]))
        h2 = arith.addf(vector.shuffle(h4, h4, [0, 1]), vector.shuffle(h4, h4, [2, 3]))
        e = _elems(h2, 2)
        return arith.addf(_u(e[0]), _u(e[1]))

    def emit(DXS, DY, XS, WS, MEANS, RSTDS, INVS, AMAXS, PART, M_ARG, ITERS_ARG, G_ARG):
        tid = arith.ArithValue(_as_i32(fx.thread_idx.x))
        blk = arith.ArithValue(_as_i32(fx.block_idx.x))
        M = arith.ArithValue(_u(M_ARG))
        ITERS = arith.ArithValue(_u(ITERS_ARG))
        G = arith.ArithValue(_u(G_ARG))
        if REMAP:
            GR = NXCD * REMAP
            lb = ((blk // _i32(GR)) * _i32(GR) + (blk % _i32(NXCD)) * _i32(REMAP)
                  + (blk // _i32(NXCD)) % _i32(REMAP))
            ok = arith.cmpi(arith.CmpIPredicate.eq, _u(G % _i32(GR)), _u(_i32(0)))
            blk = arith.ArithValue(arith.select(ok, _u(lb), _u(blk)))
        lane = tid % _i32(64)
        wave = tid // _i32(64)
        sub = lane // _i32(LPR)
        col = (lane % _i32(LPR)) * _i32(8)

        m64 = arith.ArithValue(arith.extui(T.i64, _u(M)))
        rbytes = m64 * arith.constant(D * 2, type=T.i64)
        sbytes = m64 * arith.constant(4, type=T.i64)
        rDY = _rsrc(DY, rbytes)
        rX = [_rsrc(X, rbytes) for X in XS]
        rDX = []
        for p in range(NPROB):
            if not MODES[p]:
                rDX.append(_rsrc(DXS[p], rbytes))                  # bf16 [M, D]
            elif DEBUG_F32:                                       # int16 [2M, D]: hi, lo planes
                rDX.append((_rsrc(DXS[p], rbytes), _rsrc_at(DXS[p], rbytes, rbytes)))
            else:                                                 # e5m2 [M, D] bytes
                rDX.append(_rsrc(DXS[p], m64 * arith.constant(D, type=T.i64)))
        rMEAN = [_rsrc(MN, sbytes) for MN in MEANS]
        rRSTD = [_rsrc(RS, sbytes) for RS in RSTDS]
        invv = {}
        if not DEBUG_F32:
            for p in F8:
                rI = buffer_ops.create_buffer_resource(INVS[p], max_size=False, num_records_bytes=4)
                invv[p] = bcast(buffer_ops.buffer_load(rI, _i32(0), vec_width=1, dtype=T.f32))

        def load_param(P):
            r = buffer_ops.create_buffer_resource(P, max_size=True)
            lo = buffer_ops.buffer_load(r, col, vec_width=4, dtype=T.f32)
            hi = buffer_ops.buffer_load(r, col + _i32(4), vec_width=4, dtype=T.f32)
            return vector.shuffle(_u(lo), _u(hi), list(range(8)))

        ws = [load_param(W) for W in WS]

        def row_sum(s):
            return _xor_sum(s, lane, LPR, DPP)

        def load_slot(r):
            off = r * _i32(D) + col
            return dict(
                off=off,
                dy=buffer_ops.buffer_load(rDY, off, vec_width=8, dtype=T.bf16,
                                          cache_modifier=CACHE),
                x=[buffer_ops.buffer_load(rX[p], off, vec_width=8, dtype=T.bf16,
                                          cache_modifier=CACHE) for p in range(NPROB)],
                mean=[buffer_ops.buffer_load(rMEAN[p], r, vec_width=1, dtype=T.f32)
                      for p in range(NPROB)],
                rstd=[buffer_ops.buffer_load(rRSTD[p], r, vec_width=1, dtype=T.f32)
                      for p in range(NPROB)],
            )

        def compute(ld, p, acc, wv, amx):
            # ln_bwd_grouped.py compute, op for op, up to the dx store
            x = arith.extf(_v8f(), _u(ld["x"][p]))
            dy = arith.extf(_v8f(), _u(ld["dy"]))
            rstd = ld["rstd"][p]
            rstd_v = bcast(rstd)
            nmr = arith.negf(arith.mulf(_u(ld["mean"][p]), _u(rstd)))
            xhat = vfma(x, rstd_v, bcast(nmr))
            g = vmul(wv, dy)
            acc_dw = vfma(dy, xhat, acc[0])
            acc_db = vadd(acc[1], dy)
            c1 = arith.mulf(_u(row_sum(hsum8(vmul(xhat, g)))), _u(_f32(1.0 / D)))
            c2 = arith.mulf(_u(row_sum(hsum8(g))), _u(_f32(1.0 / D)))
            a1 = arith.mulf(_u(c1), _u(rstd))
            a2 = arith.mulf(_u(c2), _u(rstd))
            dx = vfma(g, rstd_v, vneg(vfma(xhat, bcast(a1), bcast(a2))))
            if not MODES[p]:
                buffer_ops.buffer_store(arith.truncf(T.vec(8, T.bf16), _u(dx)), rDX[p], ld["off"],
                                        cache_modifier=SCACHE)
            elif DEBUG_F32:
                bits = vector.bitcast(T.vec(8, T.i32), _u(dx))
                sh = arith.constant_vector(16, T.vec(8, T.i32))
                hi = arith.trunci(T.vec(8, T.i16), arith.shrui(_u(bits), _u(sh)))
                lo = arith.trunci(T.vec(8, T.i16), _u(bits))
                buffer_ops.buffer_store(hi, rDX[p][0], ld["off"], cache_modifier=SCACHE)
                buffer_ops.buffer_store(lo, rDX[p][1], ld["off"], cache_modifier=SCACHE)
            else:
                # separate temporary: dx itself still feeds the dx sum below
                q = _e5m2_pack8(_clamp_e5m2(vmul(dx, invv[p]), CVT[1]), CVT[0])
                buffer_ops.buffer_store(q, rDX[p], ld["off"], cache_modifier=SCACHE,
                                        offset_is_bytes=True)
            if MODES[p] and AMAX:
                amx = _amax8(amx, dx)
            return [acc_dw, acc_db, vadd(acc[2], dx)], amx

        zero = arith.constant_vector(0.0, _v8f())
        nflat = NACC * NPROB
        namx = NF8 if AMAX else 0
        loop = scf.ForOp(_idx(_i32(0)), _idx(ITERS), _u(arith.constant(1, index=True)),
                         [_u(zero)] * nflat + [_u(_f32(0.0))] * namx)
        with ir.InsertionPoint(loop.body):
            flat = list(loop.inner_iter_args[:nflat])
            amxs = list(loop.inner_iter_args[nflat:nflat + namx])
            acc = [flat[NACC * p:NACC * p + NACC] for p in range(NPROB)]
            it = arith.ArithValue(arith.index_cast(T.i32, loop.induction_variable))
            base = (it * G + blk) * _i32(RPS) + wave * _i32(RPW) + sub
            slots = [load_slot(base + _i32(u * RPI)) for u in range(UNROLL)]
            for ld in slots:
                for p in range(NPROB):
                    i8 = F8.index(p) if (AMAX and MODES[p]) else None
                    acc[p], a_new = compute(ld, p, acc[p], ws[p], amxs[i8] if i8 is not None else None)
                    if i8 is not None:
                        amxs[i8] = a_new
            scf.YieldOp([_u(v) for p in range(NPROB) for v in acc[p]] + [_u(a) for a in amxs])
        res = list(loop.results[:nflat])
        res_amx = list(loop.results[nflat:nflat + namx])
        res_acc = [res[NACC * p:NACC * p + NACC] for p in range(NPROB)]

        red = SmemPtr(allocator.get_base(), off_red, T.f32, shape=(NACC * SLOTS * D,)).get()
        slot = wave * _i32(RPW) + sub
        rP = buffer_ops.create_buffer_resource(PART, max_size=True)
        c0 = tid * _i32(VEC)
        if AMAX:
            # per wave max, parked in LDS before the first reduction barrier
            amx_l = SmemPtr(allocator.get_base(), off_amx, T.f32, shape=(NF8 * WAVES,)).get()
            wmax = [_xor_max(v, lane, 64, DPP) for v in res_amx]
            is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, _u(lane), _u(_i32(0)))
            guard = scf.IfOp(is_lane0, results_=[], has_else=False)
            with ir.InsertionPoint(guard.then_block):
                for i in range(NF8):
                    vector.store(vector.from_elements(T.vec(1, T.f32), [_u(wmax[i])]), amx_l,
                                 [_idx(_i32(i * WAVES) + wave)], alignment=4)
                scf.YieldOp([])
        for p in range(NPROB):
            if p:
                gpu.barrier()       # the previous problem's LDS reads are done
            for o in range(NACC):
                vector.store(_u(res_acc[p][o]), red,
                             [_idx(_i32(o * SLOTS * D) + slot * _i32(D) + col)], alignment=16)
            gpu.barrier()
            if p == 0 and AMAX:
                # every wave's maxima are in LDS: thread 0 stores the block's partials
                is_t0 = arith.cmpi(arith.CmpIPredicate.eq, _u(tid), _u(_i32(0)))
                g64 = arith.ArithValue(arith.extui(T.i64, _u(G)))
                for i, q in enumerate(F8):
                    tot = None
                    for k in range(WAVES):
                        v = vector.load_op(T.vec(1, T.f32), amx_l, [_idx(_i32(i * WAVES + k))])
                        v = vector.extract(_u(v), static_position=[0], dynamic_position=[])
                        tot = v if tot is None else arith.maxnumf(_u(tot), _u(v))
                    rA = _rsrc(AMAXS[q], g64 * arith.constant(4, type=T.i64))
                    buffer_ops.buffer_store(_u(tot), rA, blk, mask=is_t0)
            for o in range(NACC):
                tot = None
                for k in range(SLOTS):
                    v = vector.load_op(T.vec(VEC, T.f32), red,
                                       [_idx(_i32((o * SLOTS + k) * D) + c0)])
                    tot = v if tot is None else arith.addf(_u(tot), _u(v))
                poff = (_i32(p * NACC + o) * G + blk) * _i32(D) + c0
                if VEC == 1:
                    tot = vector.extract(_u(tot), static_position=[0], dynamic_position=[])
                buffer_ops.buffer_store(_u(tot), rP, poff)

    @flyc.kernel(name=name)
    def kern(DX0: fx.Tensor, DX1: fx.Tensor, DX2: fx.Tensor, DY: fx.Tensor,
             X0: fx.Tensor, X1: fx.Tensor, X2: fx.Tensor,
             W0: fx.Tensor, W1: fx.Tensor, W2: fx.Tensor,
             MEAN0: fx.Tensor, MEAN1: fx.Tensor, MEAN2: fx.Tensor,
             RSTD0: fx.Tensor, RSTD1: fx.Tensor, RSTD2: fx.Tensor,
             INV0: fx.Tensor, INV1: fx.Tensor, INV2: fx.Tensor,
             AMAX0: fx.Tensor, AMAX1: fx.Tensor, AMAX2: fx.Tensor,
             PART: fx.Tensor, M_ARG: fx.Int32, ITERS_ARG: fx.Int32, G_ARG: fx.Int32):
        emit([DX0, DX1, DX2], DY, [X0, X1, X2], [W0, W1, W2], [MEAN0, MEAN1, MEAN2],
             [RSTD0, RSTD1, RSTD2], [INV0, INV1, INV2], [AMAX0, AMAX1, AMAX2], PART,
             M_ARG, ITERS_ARG, G_ARG)

    @flyc.jit
    def launch(DX0: fx.Tensor, DX1: fx.Tensor, DX2: fx.Tensor, DY: fx.Tensor,
               X0: fx.Tensor, X1: fx.Tensor, X2: fx.Tensor,
               W0: fx.Tensor, W1: fx.Tensor, W2: fx.Tensor,
               MEAN0: fx.Tensor, MEAN1: fx.Tensor, MEAN2: fx.Tensor,
               RSTD0: fx.Tensor, RSTD1: fx.Tensor, RSTD2: fx.Tensor,
               INV0: fx.Tensor, INV1: fx.Tensor, INV2: fx.Tensor,
               AMAX0: fx.Tensor, AMAX1: fx.Tensor, AMAX2: fx.Tensor,
               PART: fx.Tensor, M: fx.Int32, ITERS: fx.Int32, G: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(DX0, DX1, DX2, DY, X0, X1, X2, W0, W1, W2, MEAN0, MEAN1, MEAN2,
             RSTD0, RSTD1, RSTD2, INV0, INV1, INV2, AMAX0, AMAX1, AMAX2, PART, M, ITERS, G).launch(
            grid=(G, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.rows_per_iter = RPS
    return launch


# ----------------------------------------------------------------------------- host side
def grouped_fp8_reason(dy: torch.Tensor, xs: Sequence[torch.Tensor],
                       weights: Sequence[torch.Tensor], means: Sequence[torch.Tensor],
                       rstds: Sequence[torch.Tensor], inv_scales) -> Optional[str]:
    """None when ln_bwd_grouped3_fp8 covers these inputs (MODES_PROD), else a
    stable, N-free reason: ln_bwd_grouped.grouped_reason's coverage, plus
    inv_scales = [fp32 1-element CUDA tensor on dy's device] * 2 + [None]."""
    why = grouped_reason(dy, xs, weights, means, rstds)
    if why is not None:
        return why
    try:
        n = len(inv_scales)
    except TypeError:
        return "inv_scales is not a sequence"
    if n != NPROB:
        return f"{n} inv_scales (the grouped fp8 kernel takes {NPROB})"
    for p, m in enumerate(MODES_PROD):
        s = inv_scales[p]
        if not m:
            if s is not None:
                return f"inv_scale_{p} given for a bf16 problem (only modes {MODES_PROD} are served)"
            continue
        if not isinstance(s, torch.Tensor) or s.dtype != torch.float32 or s.numel() != 1 \
                or not s.is_cuda or s.device != dy.device:
            return f"inv_scale_{p} is not a 1-element fp32 tensor on dy's device"
    return None


def _launch(dy, xs, weights, means, rstds, inv_scales, *, modes=MODES_PROD, debug_f32=False,
            no_amax=False, cvt=None, cache=None, remap=None, store_cache=None, stream=None,
            out=None):
    """Shared launcher (the public wrapper and the lab checks). Returns
    (outs, fouts, amax, G): outs[p] e5m2 [N, D] / bf16 [N, D] / for debug_f32
    int16 [2N, D]; fouts 9 fp32 [D]; amax[p] fp32 [G] or None."""
    out = out or {}
    dy = dy.detach()
    xs = [x.detach() for x in xs]
    weights = [w.detach() for w in weights]
    means = [m.detach() for m in means]
    rstds = [r.detach() for r in rstds]
    M, D = dy.shape
    modes = tuple(int(m) for m in modes)
    waves, unroll, tile_mult, *rest = GROUPED_CFG
    dpp = bool(rest[0]) if rest else False
    remap, cache, store_cache = grouped_kernel_options(cache, remap, store_cache)
    G = _v2.grid_for(M, dy.device, waves, tile_mult)
    dev = dy.device
    if "dx" in out:
        outs = list(out["dx"])
    else:
        outs = []
        for p in range(NPROB):
            if not modes[p]:
                outs.append(torch.empty_like(xs[p]))
            elif debug_f32:
                outs.append(torch.empty((2 * M, D), dtype=torch.int16, device=dev))
            else:
                outs.append(torch.empty((M, D), dtype=torch.float8_e5m2, device=dev))
    fouts = list(out["fin"]) if "fin" in out else [
        torch.empty((D,), dtype=torch.float32, device=dev) for _ in range(3 * NPROB)]
    if M == 0:
        for t in fouts:
            t.zero_()
        amax = [torch.zeros((1,), dtype=torch.float32, device=dev) if modes[p] else None
                for p in range(NPROB)]
        return outs, fouts, amax, G
    amax = [(out["amax"][p] if "amax" in out else torch.empty((G,), dtype=torch.float32, device=dev))
            if modes[p] else None for p in range(NPROB)]
    if stream is None:
        stream = torch.cuda.current_stream(dev)
    part = (out["part"][: 3 * NPROB * G * D].view(3 * NPROB, G, D) if "part" in out
            else torch.empty((3 * NPROB, G, D), dtype=torch.float32, device=dev))
    main = build_ln_bwd_grouped_fp8(D, waves, unroll, cache, dpp, remap, store_cache, modes,
                                    bool(debug_f32), bool(no_amax), tuple(cvt) if cvt else None)
    fin_kernel = build_finalize_grouped(D)
    iters = (M + G * main.rows_per_iter - 1) // (G * main.rows_per_iter)
    # the kernel sees raw bytes: e5m2 outputs go in as uint8 views (DLPack-safe)
    kd = [o.view(torch.uint8) if o.dtype == torch.float8_e5m2 else o for o in outs]
    f8 = [p for p in range(NPROB) if modes[p]]
    ph_inv = inv_scales[f8[0]].detach().reshape(1) if (f8 and not debug_f32) else means[0]
    ph_amx = amax[f8[0]] if f8 else part
    invs = [inv_scales[p].detach().reshape(1) if (modes[p] and not debug_f32) else ph_inv
            for p in range(NPROB)]
    amxs = [amax[p] if modes[p] else ph_amx for p in range(NPROB)]
    # _v2._fast_call: FlyDSL 0.2.4's per-call jit overhead (ln_bwd.py module top).
    _v2._fast_call(main, kd[0], kd[1], kd[2], dy, xs[0], xs[1], xs[2], weights[0], weights[1],
                   weights[2], means[0], means[1], means[2], rstds[0], rstds[1], rstds[2],
                   invs[0], invs[1], invs[2], amxs[0], amxs[1], amxs[2], part, M, iters, G,
                   fx.Stream(stream))
    _v2._fast_call(fin_kernel, part, *fouts, G, fx.Stream(stream))
    return outs, fouts, amax, G


def ln_bwd_grouped3_fp8(dy, xs, weights, biases, means, rstds, inv_scales, *, cache=None,
                        remap=None, store_cache=None, stream=None
                        ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                        Optional[torch.Tensor]]]:
    """[(out_p, dw_p, db_p, dxsum_p, amax_partials_p) for p in 0..2], MODES_PROD.
    p in {0, 1}: out_p float8_e5m2 [N, 512] = e5m2(clamp_nanprop(fp32 dx_p *
    inv_scales[p][0], +-57344)), amax_partials_p fp32 [G] (N == 0: zeros(1));
    p = 2: bf16 dx, None. dw/db/dxsum and the action dx bitwise equal to
    ln_bwd_grouped.ln_bwd_grouped3(dy, xs, ...) with the same cache / remap /
    store_cache options (default: ln_bwd.py's production options); the fp32
    dx_p that is quantized is the value ln_bwd_grouped3 rounds to bf16.
    biases are not read. Out of place only. The caller checks
    grouped_fp8_reason() first."""
    outs, fouts, amax, _G = _launch(dy, xs, weights, means, rstds, inv_scales, cache=cache,
                                    remap=remap, store_cache=store_cache, stream=stream)
    return [(outs[p], fouts[3 * p], fouts[3 * p + 1], fouts[3 * p + 2], amax[p])
            for p in range(NPROB)]


def stacked_kernel_name() -> str:
    """The production main kernel's name (arming line, traces, census role)."""
    remap, cache, store_cache = grouped_kernel_options()
    waves, unroll, _tile_mult, *rest = GROUPED_CFG
    return build_ln_bwd_grouped_fp8(D_GROUPED, waves, unroll, cache,
                                    bool(rest[0]) if rest else False, remap, store_cache,
                                    MODES_PROD).kernel_name


def prebuild_grouped_fp8(device=None, M: int = 1024) -> None:
    """Startup prebuild (pre-RUN_START): JIT-compile, load and launch once the
    production main kernel and the grouped finalize on synthetic tensors. M,
    ITERS and G are runtime arguments and the jit cache key holds no shape, so
    a small M builds the production specialization."""
    device = device or torch.device("cuda", torch.cuda.current_device())
    dy = torch.zeros((M, D_GROUPED), dtype=torch.bfloat16, device=device)
    xs = [torch.zeros_like(dy) for _ in range(NPROB)]
    ws = [torch.ones((D_GROUPED,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    bs = [torch.zeros_like(w) for w in ws]
    ms = [torch.zeros((M,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    rs = [torch.ones((M,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    inv = [torch.ones((1,), dtype=torch.float32, device=device) if m else None for m in MODES_PROD]
    ln_bwd_grouped3_fp8(dy, xs, ws, bs, ms, rs, inv)
