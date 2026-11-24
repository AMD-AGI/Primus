# The FlyDSL weighted LayerNorm backward behind DLRM_HSTU_FP8_WLN_BWD_FLYDSL
# (ops/triton/triton_layer_norm.py `_wln_bwd_flydsl`, gin DLRM_HSTU_FP8_WLN_BWD_FLYDSL; GitHub PR #7).
# Developed in a kernel-development repository 2f5897d, tools/kernels/dlrmv4_ln_bwd:
# its driver (fp64 oracle, HIP-graph timing) and ITERATIONS.md record how each
# part was measured.
"""FlyDSL weighted LayerNorm backward for DLRMv4 training on gfx950 (MI350X).

It replaces the training repository's Triton `_weighted_layer_norm_bwd_dx` and
its column partials (generative_recommenders/ops/triton/triton_layer_norm.py)
for the four variants the training step runs (Plan 05 section 2.1, plus the
targets-only STU norm of yambda_5b.gin item (20e)):

    STU      D=512, residual add, no swish, no dx sum
    STU tgt  D=512, no residual (item (20e) TGT_SPARSE_RESID), no swish, no dx sum
    MLP out  D=512, dx sum, no swish
    MLP hid  D=256, dx sum, swish (x * sigmoid(LN(x)))

Per row, with xhat = (x - mean) * rstd and wdy = w * dy (all fp32):
    plain   c1 = sum(xhat*wdy)/D, c2 = sum(wdy)/D, dx = (wdy - (xhat*c1 + c2))*rstd
            dW += dy*xhat, dB += dy
    swish   s = sigmoid(xhat*w + b), t = dy*x*s*(1-s), x_ = w*t
            c1 = sum(xhat*x_)/D, c2 = sum(x_)/D, dx = (x_ - (xhat*c1 + c2))*rstd + dy*s
            dW += t*xhat, dB += t
    residual: dx = bf16(dx) + residual  (rounded first, as the Triton kernel does)
    dx sum: dXS += dx (the fp32 dx, before it is stored)
    mscale (residual variant only; the layer-0 input-LayerNorm backward that
            yambda_5b.gin item (20)(d) DROPOUT_MSCALE_FOLD folds the input
            dropout's backward into): dx = (float(mask) * float(bf16(dx))) * scale
            in fp32 after the residual add -- the op order of aten's
            masked_scale_kernel and of the Triton v2 kernel's MASK_SCALE
            epilogue -- so dx equals native_dropout_backward applied to the
            residual variant's stored dx, bit for bit (signed zeros included).
            The bool mask [M, D] is read as 8 bytes per lane per row; dW/dB do
            not see the mask. The fp32 scale is a runtime Int32 argument
            holding its bits (a Float32 argument took FlyDSL 0.1.1's slow
            per-call argument path).
            mscale bits (DLRM_HSTU_FP8_DROPOUT_MASK_BITS, R-C7: the forward
            stores the mask bit-packed, uint8 [M, D/8], bit k of byte j of a
            row = column 8j + k; input_dropout_fold.pack_dropout_mask_bits):
            a lane's 8 contiguous columns are exactly one byte, so it reads 1
            byte per row (the 64 lanes of a D=512 row read the row's 64
            contiguous bytes) instead of 8, and keep = float((byte >> k) & 1)
            is the same 0.0 / 1.0 factor the bool byte gives: the epilogue,
            dx, dW and dB are bitwise the bool-mask call's.

Operation order (`tri_order` in build_ln_bwd): the residual variant and the
plain variant (no residual, no dx sum, no swish) evaluate xhat and dx in
Triton's order, xhat = (x - mean) * rstd and dx = (g - (xhat*c1 + c2)) * rstd.
The plain variant's bf16 dx plus the residual (one fp32 add, rounded to bf16,
as item (20e)'s index_add does) therefore equals the residual variant's dx up
to the sign of an exact zero, with the same dW/dB. The dx-sum and swish
variants fold the per-row scalars into FMAs: xhat = fma(x, rstd, -mean*rstd),
dx = fma(g, rstd, -fma(xhat, c1*rstd, c2*rstd)).

Two kernels:
  * main (`ln_bwd_flydsl_d{D}...`): 8 bf16 per lane (one 16-byte buffer load per
    tensor per row); a row takes LPR = D/8 lanes, so a wave holds RPW = 64/LPR
    rows per slot. Row sums run over the row's lanes (`_xor_sum`: DPP moves,
    ds_swizzle and ds_bpermute). Block b of G takes rows (it*G + b)*RPS + ...,
    so each pass of the grid covers one
    contiguous window. Buffer resources span the whole tensor with exact sizes,
    so rows past the end load zero (which contributes exactly zero to every
    output) and their stores are dropped.
    dW, dB and the dx sum stay in registers across the runtime row loop, are
    reduced across the block's waves in LDS, and stored as one fp32 partial row
    per block: PART[o, b, :].
    XCD order (v2.1; build_ln_bwd REMAP = R, production XCD_REMAP_CFG): the
    hardware hands workgroups to the N_XCD = 8 XCDs round-robin (physical block
    b runs on XCD b % 8), so in v2 each XCD saw every 8th chunk of a pass. With
    REMAP = R, physical block b acts as logical block
        lb = (b // 8R) * 8R + (b % 8) * R + (b // 8) % R
    when G % 8R == 0 (lb = b otherwise; chosen in-kernel, so one build serves
    every G), and lb replaces b in BOTH the row mapping and the PART row: XCD k
    then streams runs of R contiguous logical blocks (R*RPS rows) of every pass.
    R = G/8 is one run per XCD and pass; shorter runs (16-32) were faster
    (probes/iter7/v21). lb is a permutation of 0..G-1 and logical block lb
    takes exactly v2's block-lb rows in v2's order, so dx and PART, and with
    them dW, dB and the dx sum, are bitwise identical to v2.
    Cache policy (v2.1, STREAM_CACHE_CFG): the aux bits of the streamed x, dy
    and residual loads and, separately, of the dx store, per variant; no value
    changes.
    Kill switches (v2.1, read once at import; any value but "0" = on): the env
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL_XCD_REMAP=0 turns the XCD order off and
    DLRM_HSTU_FP8_WLN_BWD_FLYDSL_STREAM_CACHE=0 the cache policy; both off is
    v2's kernel (same name, same ISA). Neither changes any output bit.
  * finalize (`ln_bwd_flydsl_fin_...`): one block per (8 columns, output);
    256 threads stride the G partial rows, then an xor-shuffle tree per wave
    and LDS across waves. dW and dB are written in the weight's dtype, the dx
    sum in fp32.

Specialization: the flyc.jit cache keys tensor arguments by type (FlyDSL
0.1.1) or by dtype, rank and unit-stride axis (0.2.4), never by shape, and
M/ITERS/G are runtime Int32 arguments, so the kernel compiled at
the first call's M serves every M (the startup prebuild compiles at a small M
on purpose). Every buffer bound in these kernels therefore comes from the
runtime M or an explicit byte count. Never build a buffer resource from a
tensor's static shape (buffer_ops.create_buffer_resource(max_size=False)
without num_records_bytes): that would freeze the first call's size into the
kernel and silently drop later rows. Dtypes are fixed per build (asserted in
ln_bwd; w_f32 is a build argument).

Contract: keep `ln_bwd(...)` and its return values; launch on the stream the
caller passes (default: torch's current stream).

E-RM producer emit (EXPERIMENT, Arbor r14 iter55 erm_prod; DY85 ASSUMPTION
dy85-erm-rowmajor-emit-w3tr; default OFF: build_ln_bwd EMIT=False is the
shipped kernel, byte for byte, and `ln_bwd` never sets it). `ln_bwd_erm`
runs the residual variant (D=512, no swish / dx sum / mscale) with EMIT=True:
the same kernel body (v2.1 16-row logical blocks, REMAP, cache policy, dx,
PART), plus three outputs computed from x32 = fp32(bf16 dx), the bits the dx
store writes (contract I1):
    G8    float8_e5m2   [M, 512] row-major: e5m2(clampN(fl32(x32*inv), 57344))
          = bwd_fp8._quantize_amax_dual_kernel's DST bytes (bwd_fp8.py:296-299)
    XRM   float8_e4m3fn [M, 512] row-major: e4m3fn(clampN(fl32(s*2^-t), 448)),
          s = fl32(x32*inv), = the dual cast's DST_T values on its SEPARATE_T
          path (bwd_fp8.py:301-303), stored row-major instead of token-major
    AMAXP fp32 [G]: max over the non-NaN |x32| of the rows a logical block
          emits (its waves meet in a 16-B LDS slot at the existing block
          barrier), one plain store per block (no atomics); max(AMAXP)
          is the dual cast's amax on every input without an all-NaN 256x128
          cast tile (actckpt A5's exempt case, where the cast gives NaN).
clampN(v, L) = minnum(maxnum(v, -L), L) (tl.maximum / tl.minimum defaults:
NaN -> -L, +-Inf -> +-L); the fp8 conversions are v_cvt_scalef32_pk_{bf8,fp8}_f32
at scale 1.0, the dual cast's instructions. inv is the consumer site's device
fp32 state.inv_scale[0], loaded in-kernel (never read on the host); t the
site's t_shift, passed as the fp32 bits of 2^-t. The amax runs on the x32
bits through an exact i32 key (_erm_amax_key), K = (bits << 1) + 0x00FE0000
mod 2^32: the shift drops the sign, bf16 magnitudes m <= 0x7F80 map increasingly
to [0x00FE0000, 0xFFFE0000], NaN to [0, 0x00FC0000], so a u32 max is maxNum
over |x| ignoring every NaN, sNaN too; decode (K - 0x00FE0000) >> 1. Whole-row
stores: one dwordx2 per lane and row, 512 contiguous bytes per wave and row.
Kernel name at the shipped options: ln_bwd_flydsl_d512_res_erm_x32_c3 (the
_erm variant tag before the option tags, as _mscaleb).
"""

import functools
import inspect
import os
import struct

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
try:
    # FlyDSL 0.2.4 (ROCm 10.2 UTD image) renamed fly_values to
    # extract_to_ir_values; same contract: the list of ir.Values of a DSL
    # value, [0] being the tensor's memref.
    from flydsl.compiler.protocol import extract_to_ir_values as fly_values
except ImportError:  # FlyDSL 0.1.1 (rocm/primus:v26.3)
    from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, rocdl
from generative_recommenders.ops.flydsl._compat import buffer_ops, rocdl_raw, vector  # FlyDSL 0.2 API on 0.2 / >= 0.3
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils import env as _fly_env
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# FlyDSL 0.2.4 compile time: its per-op source-location capture, memoized per
# code object (flydsl_loc.py; same locations, byte-identical IR and kernels).
# Every FlyDSL kernel module of the trainer imports this module (hstu_fwd /
# hstu_bwd_dkdv take _fast_call from it) before it builds anything.  A no-op
# under FlyDSL 0.1.1.
from generative_recommenders.ops.flydsl import flydsl_loc as _flydsl_loc  # noqa: E402

_flydsl_loc.install()

# Host launch path.  FlyDSL 0.2.4 (ROCm 10.2 UTD image) re-derives a
# @flyc.jit call's cache key from every argument and re-snapshots every module
# global the launcher reaches (its "changed since first compile" check) on
# each call: ln_bwd main + finalize took 137-158 us of host time per call
# through the plain jit on a cluster A MI355X node, against 27-31 us under FlyDSL
# 0.1.1 (median per call; the C-ABI fill and launch themselves take ~6 us per
# launch).  flyc.compile(jit, *args) launches once and returns the same
# compiled kernel behind a positional-only dispatcher: 23-26 us per LN call.
# _fast_call keeps one per launcher and per what the 0.2.4 jit key varies with
# for these arguments -- each tensor's dtype, rank and first stride-1 axis,
# each other argument's type -- so a call differing in any of them gets its
# own entry exactly where the jit would compile anew; M / ITERS / G and the
# tensors' pointers, shapes and strides are filled per call either way.  Not
# re-checked per call: the globals (the launchers' captured globals, T and
# _DPP_STEPS, are never rebound) and the FLYDSL_COMPILE_* variables the jit
# key also reads (fixed for the process; nothing here sets them).  FlyDSL
# 0.1.1 (no flyc.compile) and COMPILE_ONLY=1 (no loaded code object to
# dispatch to) keep the plain jit call.  The FlyDSL attention fwd / bwd call()
# wrappers use it too (14-24 us per call, against 68-109 us through the plain
# 0.2.4 jit and 110-206 us under 0.1.1).
_FLYC_COMPILE = getattr(flyc, "compile", None)
_FAST_CALLS = {}
_JIT_SIGS = {}


def _fast_call(jit, *args, **kw):
    """jit(*args, **kw) through a cached flyc.compile dispatcher where FlyDSL
    has one (see above).  The dispatcher takes every parameter positionally,
    so keywords and defaults are bound first, as the jit call itself does."""
    if _FLYC_COMPILE is None or _fly_env.compile.compile_only:
        return jit(*args, **kw)
    sig = _JIT_SIGS.get(jit)
    if sig is None:
        sig = _JIT_SIGS[jit] = inspect.signature(jit.func)
    if kw or len(args) != len(sig.parameters):
        bound = sig.bind(*args, **kw)
        bound.apply_defaults()
        args = tuple(bound.arguments.values())
    try:
        key = (jit,) + tuple((a.dtype, a.dim(), a.stride().index(1))
                             if isinstance(a, torch.Tensor) else type(a) for a in args)
    except ValueError:  # a tensor without a stride-1 axis: the jit raises its own error
        return jit(*args)
    fn = _FAST_CALLS.get(key)
    if fn is None:
        _FAST_CALLS[key] = _FLYC_COMPILE(jit, *args)    # compiles if new; launches once
        return None
    return fn(*args)


# (waves per block, row slots per wave per iteration, blocks per CU, DPP row sums),
# per (D, swish, residual, dx sum); from the round-robin sweep (ITERATIONS.md, 3).
# (512, False, False, False) is the targets-only STU norm (item (20e)); it must
# keep the residual variant's values: the same grid and rows per block keep its
# dW/dB partial order identical to the residual call's (see `tri_order`).
DEFAULT_CFG = {(512, False, True, False): (4, 4, 8, 1),
               (512, False, False, False): (4, 4, 8, 1),
               (512, False, False, True): (4, 4, 8, 1),
               (256, True, False, True): (4, 4, 8, 1)}
# cache policy for the streamed tensors: the buffer instructions' aux bits on
# gfx950, 0 = default, 2 = nt, 3 = sc0 nt, as (loads of x, dy and residual,
# store of dx). v2 used 0 for every variant: non-temporal (2) and the other
# settings were no faster in the driver (ITERATIONS.md 5-6). v2.1 chooses it
# per (D, swish, residual, dx sum), as DEFAULT_CFG; STREAM_CACHE serves any
# other variant. STREAM_CACHE_ON False (env
# DLRM_HSTU_FP8_WLN_BWD_FLYDSL_STREAM_CACHE=0) uses STREAM_CACHE everywhere.
STREAM_CACHE_CFG = {(512, False, True, False): (3, 3),
                    (512, False, False, False): (3, 3),
                    (512, False, False, True): (3, 3),
                    (256, True, False, True): (3, 3)}
STREAM_CACHE = 0
STREAM_CACHE_ON = os.environ.get("DLRM_HSTU_FP8_WLN_BWD_FLYDSL_STREAM_CACHE", "1") != "0"
# XCD-aware block order (module docstring; build_ln_bwd REMAP), bitwise neutral:
# the run length R in blocks (0 = off) per variant as DEFAULT_CFG, applied while
# XCD_REMAP is True (False, env DLRM_HSTU_FP8_WLN_BWD_FLYDSL_XCD_REMAP=0, turns
# it off everywhere).
XCD_REMAP = os.environ.get("DLRM_HSTU_FP8_WLN_BWD_FLYDSL_XCD_REMAP", "1") != "0"
XCD_REMAP_CFG = {(512, False, True, False): 32,
                 (512, False, False, False): 16,
                 (512, False, False, True): 16,
                 (256, True, False, True): 16}
N_XCD = 8
# cache policy of the dropout mask load (mscale variant): non-temporal (2); it
# is read once, in the epilogue, and changes no value
MASK_CACHE = 2
# the packed (bit) mask's byte loads (mscale bits variant): default (0). A
# 128-byte line holds the mask of two D=512 rows, which two waves of a block
# read; non-temporal (2) loses the line between them and was 0.05 ms/call
# slower at M=2.35M (MI350X, probes/iter13/m2abits)
MASK_BITS_CACHE = 0
FALLBACK_CFG = (4, 4, 8, 1)
FIN_THREADS = 256
LOG2E = 1.4426950408889634
CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)   # as hstu_fwd.py
# E-RM emit (EMIT=True only; module docstring): the dual cast's limits
# (bwd_fp8._E5M2_MAX = LIMIT of the e5m2 copy, _E4M3_MAX = LIMIT_T of the e4m3
# copy) and the amax key of +0 (i32 key K = (bits << 1) + 0x00FE0000 on x32)
ERM_E5M2_MAX = 57344.0
ERM_E4M3_MAX = 448.0
ERM_KEY0 = 0x00FE0000        # amax key of +0 (_erm_amax_key)
# cache policy of the G8 / XRM stores (aux bits as STREAM_CACHE_CFG; -1 = the
# dx store's): streamed, written once, read by L11 / W3 later in the step
ERM_STORE_CACHE = -1


def _u(v):
    return arith.unwrap(v)


def _i32(v):
    return arith.constant(int(v), type=T.i32)


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _idx(v):
    return arith.index_cast(T.index, _u(v))


def _as_i32(v):
    raw = _u(v)
    if isinstance(raw.type, ir.IndexType):
        return arith.index_cast(T.i32, raw)
    return raw


def _shifted_rsrc(tensor, elem_offset_i64, elem_bytes, num_bytes_i32):
    """Buffer resource based at `elem_offset_i64` elements into `tensor`
    (computed in i64: a whole-tensor offset exceeds i32 at these sizes), with
    num_records = num_bytes_i32. Copied from the training tree's
    ops/flydsl/hstu_fwd.py `_shifted_rsrc`."""
    ptr_ty = ir.Type.parse("!llvm.ptr")
    base = fly.extract_aligned_pointer_as_index(ptr_ty, fly_values(tensor)[0])
    base_i64 = llvm.PtrToIntOp(T.i64, base).result
    byte_i64 = arith.ArithValue(_u(elem_offset_i64)) * arith.constant(int(elem_bytes), type=T.i64)
    new_i64 = llvm.AddOp(_u(base_i64), _u(byte_i64), llvm.IntegerOverflowFlags(0)).result
    new_ptr = llvm.IntToPtrOp(ptr_ty, new_i64).result
    nrec = arith.extsi(T.i64, _u(num_bytes_i32))
    return rocdl_raw.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"), new_ptr,
        _u(arith.constant(0, type=T.i16)), _u(nrec), _u(_i32(CDNA_BUFFER_FLAGS)),
    ).result


def _rsrc(tensor, num_bytes_i64):
    """Buffer resource over the whole tensor with num_records = num_bytes_i64.
    Offsets are unsigned 32-bit to the hardware, so a tensor under 4 GiB is
    addressed whole even where the i32 byte offset reads negative; rows past
    the end load zero and their stores are dropped."""
    ptr_ty = ir.Type.parse("!llvm.ptr")
    base = fly.extract_aligned_pointer_as_index(ptr_ty, fly_values(tensor)[0])
    return rocdl_raw.MakeBufferRsrcOp(
        ir.Type.parse("!llvm.ptr<8>"), base,
        _u(arith.constant(0, type=T.i16)), _u(num_bytes_i64), _u(_i32(CDNA_BUFFER_FLAGS)),
    ).result


def _v8f():
    return T.vec(8, T.f32)          # types need the trace-time MLIR context


def _elems(v, n=8):
    return [vector.extract(_u(v), static_position=[j], dynamic_position=[]) for j in range(n)]


# DPP controls that pair lane i with i^1, i^2 (quad_perm) and, once quads and
# octets hold equal sums, with the other quad (row_half_mirror) and the other
# octet (row_mirror). Any pairing across the two halves sums the same.
_DPP_STEPS = {1: 0xB1, 2: 0x4E, 4: 0x141, 8: 0x140}
_SWZ_XOR16 = 0x1F | (0x10 << 10)     # ds_swizzle bitmask mode: lane ^ 16 within 32


def _xor_sum(s, lane, width, dpp=False):
    """All-reduce a scalar f32 over aligned groups of `width` lanes. ds_bpermute
    for every step, or with dpp: DPP moves for 1..8, ds_swizzle for 16, and
    ds_bpermute only for 32."""
    k = 1
    if dpp:
        while k < width and k in _DPP_STEPS:
            si = llvm.BitcastOp(T.i32, _u(s)).result
            pi = rocdl.update_dpp(T.i32, si, si, _DPP_STEPS[k], 0xF, 0xF, True)
            s = arith.addf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
            k *= 2
        if k == 16 and k < width:
            si = llvm.BitcastOp(T.i32, _u(s)).result
            pi = rocdl.ds_swizzle(T.i32, si, _u(_i32(_SWZ_XOR16)))
            s = arith.addf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
            k *= 2
    while k < width:
        addr = arith.ArithValue(arith.xori(_u(lane), _u(_i32(k)))) * _i32(4)
        pi = rocdl.ds_bpermute(T.i32, _u(addr), llvm.BitcastOp(T.i32, _u(s)).result)
        s = arith.addf(_u(s), llvm.BitcastOp(T.f32, _u(pi)).result)
        k *= 2
    return s


# ---- E-RM emit helpers (traced only when build_ln_bwd EMIT=True) -----------

def _erm_xor_maxu(s, lane, width):
    """All-reduce an i32 with an unsigned max over aligned groups of `width`
    lanes, the _xor_sum(dpp=True) exchange pattern (DPP 1..8, ds_swizzle 16,
    ds_bpermute 32). Every value reduced here is the fp32 bit pattern of a
    non-negative, non-NaN float, so the unsigned max is the float max."""
    k = 1
    while k < width and k in _DPP_STEPS:
        p = rocdl.update_dpp(T.i32, _u(s), _u(s), _DPP_STEPS[k], 0xF, 0xF, True)
        s = arith.maxui(_u(s), _u(p))
        k *= 2
    if k == 16 and k < width:
        p = rocdl.ds_swizzle(T.i32, _u(s), _u(_i32(_SWZ_XOR16)))
        s = arith.maxui(_u(s), _u(p))
        k *= 2
    while k < width:
        addr = arith.ArithValue(arith.xori(_u(lane), _u(_i32(k)))) * _i32(4)
        p = rocdl.ds_bpermute(T.i32, _u(addr), _u(s))
        s = arith.maxui(_u(s), _u(p))
        k *= 2
    return s


def _erm_cvt_pk8(fn, v):
    """8 fp32 (vector<8xf32>, already clamped to the format's range) -> 8 fp8
    bytes as vector<2xi32>, byte j = element j: four
    v_cvt_scalef32_pk_{bf8,fp8}_f32 at scale 1.0 (the instruction the dual cast
    compiles to on gfx950), two per dword (low word, then high word). The low
    word's convert starts from an undef dword: its high word is always
    overwritten by the second convert, so no zeroing move is needed."""
    e = _elems(v)
    z = llvm.UndefOp(T.vec(2, T.i16)).result
    one = _u(_f32(1.0))

    def pair(old, a, b, hi):
        r = fn(T.vec(2, T.i16), _u(old), _u(a), _u(b), one, hi)
        return r.result if hasattr(r, "result") else r

    w0 = pair(pair(z, e[0], e[1], False), e[2], e[3], True)
    w1 = pair(pair(z, e[4], e[5], False), e[6], e[7], True)
    return vector.shuffle(vector.bitcast(T.vec(1, T.i32), _u(w0)),
                          vector.bitcast(T.vec(1, T.i32), _u(w1)), [0, 1])


def _erm_amax_key(x32, acc):
    """Fold the 8 elements of x32 = fp32(bf16 dx) (vector<8xf32>, bit pattern
    h << 16 per element) into the lane's i32 amax key `acc`.
    key = (bits << 1) + 0x00FE0000 mod 2^32 = (m << 17) + 0x00FE0000 with m the
    15-bit magnitude of the bf16 (the shift drops the sign): magnitudes
    m <= 0x7F80 (finite, Inf) map increasingly to [0x00FE0000, 0xFFFE0000],
    NaN magnitudes 0x7F81..0x7FFF wrap to [0, 0x00FC0000]. An unsigned max over
    keys is therefore maxNum over |x| with every NaN (quiet or signaling)
    ignored; key(+0) = ERM_KEY0 is the identity and the loop's initial value.
    Decode: (key - ERM_KEY0) >> 1 = m << 16, the fp32 bits of max |x|."""
    xi = vector.bitcast(T.vec(8, T.i32), _u(x32))
    key = arith.addi(arith.shli(_u(xi), _u(arith.constant_vector(1, T.vec(8, T.i32)))),
                     _u(arith.constant_vector(ERM_KEY0, T.vec(8, T.i32))))
    e = _elems(key)
    t0 = arith.maxui(_u(arith.maxui(_u(e[0]), _u(e[1]))), _u(e[2]))
    t1 = arith.maxui(_u(arith.maxui(_u(e[3]), _u(e[4]))), _u(e[5]))
    t2 = arith.maxui(_u(arith.maxui(_u(e[6]), _u(e[7]))), _u(acc))
    return arith.maxui(_u(arith.maxui(_u(t0), _u(t1))), _u(t2))


@functools.lru_cache(maxsize=None)
def build_ln_bwd(D: int, is_swish: bool, add_residual: bool, sum_dx: bool, w_f32: bool,
                 WAVES: int = 4, UNROLL: int = 1, CACHE: int = 0, DPP: bool = False,
                 REMAP: int = 0, STORE_CACHE: int = -1, MSCALE: bool = False,
                 MASK_BITS: bool = False, EMIT: bool = False, EMIT_CACHE: int = -1):
    assert D in (256, 512), D
    assert not isinstance(REMAP, bool) and REMAP >= 0, REMAP   # a run length, 0 = off
    # E-RM emit (module docstring): the residual variant at D=512 only; with
    # EMIT False nothing below traces a single extra op (the shipped kernel)
    assert not EMIT or (D == 512 and add_residual and not is_swish and not sum_dx
                        and not MSCALE and not w_f32), \
        (EMIT, D, add_residual, is_swish, sum_dx, MSCALE, w_f32)
    # the dropout mask/scale epilogue is defined for the residual call only
    # (triton_layer_norm._triton_weighted_layer_norm_bwd_mscale_impl)
    assert not MSCALE or (add_residual and not is_swish and not sum_dx), \
        (MSCALE, add_residual, is_swish, sum_dx)
    # MASK_BITS: the mscale epilogue reads the bit-packed mask (module docstring)
    assert not MASK_BITS or MSCALE, (MASK_BITS, MSCALE)
    # CACHE: the policy of the streamed loads (x, dy, residual); STORE_CACHE:
    # the dx store's, -1 = CACHE (v2.1's single policy)
    SCACHE = CACHE if STORE_CACHE < 0 else STORE_CACHE
    NXCD = N_XCD                 # a closure value: part of the jit cache key
    # likewise (MASK_BITS too: emit reads it, so the bool and the packed build
    # never share a jit cache entry)
    MCACHE = MASK_BITS_CACHE if MASK_BITS else MASK_CACHE
    LPR = D // 8                 # lanes per row
    RPW = 64 // LPR              # rows per wave per row slot
    RPI = WAVES * RPW            # rows per block per row slot
    RPS = RPI * UNROLL           # rows per block per iteration
    SLOTS = WAVES * RPW          # partial rows per block before the LDS reduction
    THREADS = 64 * WAVES
    VEC = D // THREADS           # columns per thread in the block reduction
    assert VEC * THREADS == D, (D, THREADS)
    NACC = 3 if sum_dx else 2
    # Triton's operation order for xhat and dx (module docstring) in the
    # residual variant and in the plain variant (no residual, no dx sum, no
    # swish); the dx-sum and swish variants keep the FMA-folded form.
    tri_order = add_residual or not (sum_dx or is_swish)
    # v2's names, plus _x{R} for REMAP = R, _c{CACHE} for a non-default cache
    # policy and _s{SCACHE} for a store policy that differs from the loads', so
    # traces stay attributable per option
    # E-RM: the G8 / XRM store policy, -1 = the dx store's (SCACHE); the
    # variant tag _erm sits with the other variant tags (as _mscale), before
    # the option tags, and _e{ECACHE} marks a store policy unlike the dx store's
    ECACHE = (SCACHE if EMIT_CACHE < 0 else EMIT_CACHE) if EMIT else SCACHE
    name = (f"ln_bwd_flydsl_d{D}" + ("_swish" if is_swish else "")
            + ("_res" if add_residual else "") + ("_dxsum" if sum_dx else "")
            + (("_mscaleb" if MASK_BITS else "_mscale") if MSCALE else "")
            + ("_erm" if EMIT else "")
            + (f"_x{REMAP}" if REMAP else "") + (f"_c{CACHE}" if CACHE else "")
            + (f"_s{SCACHE}" if SCACHE != CACHE else "")
            + (f"_e{ECACHE}" if ECACHE != SCACHE else ""))

    allocator = SmemAllocator(None, arch=get_rocm_arch(),
                              global_sym_name=f"{name}_w{WAVES}_u{UNROLL}_c{CACHE}"
                                              + (f"_s{SCACHE}" if SCACHE != CACHE else "")
                                              + f"_p{int(DPP)}_smem")
    off_red = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_red + NACC * SLOTS * D * 4
    if EMIT:
        # E-RM: one i32 per wave, the waves' amax values for the block partial
        off_am = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_am + WAVES * 4

    def vadd(a, b):
        return arith.addf(_u(a), _u(b))

    def vsub(a, b):
        return arith.subf(_u(a), _u(b))

    def vmul(a, b):
        return arith.mulf(_u(a), _u(b))

    def bcast(s):
        return vector.broadcast(_v8f(), _u(s))

    def vfma(a, b, c):
        return vector.FMAOp(_u(a), _u(b), _u(c)).result

    def vneg(a):
        return arith.negf(_u(a))

    def hsum8(v):
        # halves as packed adds: 8 -> 4 -> 2 -> 1
        h4 = arith.addf(vector.shuffle(_u(v), _u(v), [0, 1, 2, 3]),
                        vector.shuffle(_u(v), _u(v), [4, 5, 6, 7]))
        h2 = arith.addf(vector.shuffle(h4, h4, [0, 1]), vector.shuffle(h4, h4, [2, 3]))
        e = _elems(h2, 2)
        return arith.addf(_u(e[0]), _u(e[1]))

    # The whole body is a plain helper: FlyDSL's AST rewriter turns Python
    # for/if inside a @flyc.kernel function into scf ops, and every loop and
    # branch here is meant to unroll or specialise at trace time.
    def emit(DX, DY, X, RES, W, B, MEAN, RSTD, PART, M_ARG, ITERS_ARG, G_ARG,
             DMASK=None, SCALE_ARG=None, G8=None, XRM=None, AMAXP=None, INV=None,
             TBITS_ARG=None):
        tid = arith.ArithValue(_as_i32(fx.thread_idx.x))
        blk = arith.ArithValue(_as_i32(fx.block_idx.x))
        M = arith.ArithValue(_u(M_ARG))
        ITERS = arith.ArithValue(_u(ITERS_ARG))
        G = arith.ArithValue(_u(G_ARG))
        if REMAP:
            # XCD order (module docstring): the logical block, used for the
            # rows AND the PART row; identity unless G % (NXCD*REMAP) == 0
            GR = NXCD * REMAP
            lb = ((blk // _i32(GR)) * _i32(GR) + (blk % _i32(NXCD)) * _i32(REMAP)
                  + (blk // _i32(NXCD)) % _i32(REMAP))
            ok = arith.cmpi(arith.CmpIPredicate.eq, _u(G % _i32(GR)), _u(_i32(0)))
            blk = arith.ArithValue(arith.select(ok, _u(lb), _u(blk)))
        lane = tid % _i32(64)
        wave = tid // _i32(64)
        sub = lane // _i32(LPR)
        col = (lane % _i32(LPR)) * _i32(8)

        # Rows are interleaved across blocks as in the Triton kernel: pass `it`
        # of the grid covers one contiguous window of G*RPS rows, so the
        # blocks' reads stay adjacent instead of G streams a chunk apart.
        m64 = arith.ArithValue(arith.extui(T.i64, _u(M)))
        rbytes = m64 * arith.constant(D * 2, type=T.i64)
        rX = _rsrc(X, rbytes)
        rDY = _rsrc(DY, rbytes)
        rDX = _rsrc(DX, rbytes)
        rRES = _rsrc(RES, rbytes) if add_residual else None
        if MSCALE and MASK_BITS:
            # packed mask, one bit per element: M*D/8 bytes, rows past M load
            # 0; a lane's 8 columns are byte lane % LPR of its row. The scale
            # arrives as its fp32 bits.
            rMASK = _rsrc(DMASK, m64 * arith.constant(D // 8, type=T.i64))
            mcol = lane % _i32(LPR)
            scale_v = bcast(llvm.BitcastOp(T.f32, _u(SCALE_ARG)).result)
        elif MSCALE:
            # bool mask, one byte per element: M*D bytes, rows past M load 0;
            # each lane reads its 8 bytes as two dwords. The scale arrives as
            # its fp32 bits.
            rMASK = _rsrc(DMASK, m64 * arith.constant(D, type=T.i64))
            mcol = (lane % _i32(LPR)) * _i32(2)
            scale_v = bcast(llvm.BitcastOp(T.f32, _u(SCALE_ARG)).result)
        rMEAN = _rsrc(MEAN, m64 * arith.constant(4, type=T.i64))
        rRSTD = _rsrc(RSTD, m64 * arith.constant(4, type=T.i64))
        if EMIT:
            # E-RM outputs (module docstring), exact sizes: G8 / XRM M*512
            # bytes (rows past M are dropped like the dx store's), AMAXP one
            # fp32 per logical block. inv: the consumer site's device
            # scale, one uniform load made scalar; 2^-t arrives as fp32 bits.
            rG8 = _rsrc(G8, m64 * arith.constant(D, type=T.i64))
            rXRM = _rsrc(XRM, m64 * arith.constant(D, type=T.i64))
            g64 = arith.ArithValue(arith.extui(T.i64, _u(G)))
            rAM = _rsrc(AMAXP, g64 * arith.constant(4, type=T.i64))
            rINV = _rsrc(INV, arith.constant(4, type=T.i64))
            inv_ld = buffer_ops.buffer_load(rINV, _i32(0), vec_width=1, dtype=T.f32)
            inv_s = rocdl.readfirstlane(T.i32, llvm.BitcastOp(T.i32, _u(inv_ld)).result)
            inv_v = bcast(llvm.BitcastOp(T.f32, _u(inv_s)).result)
            tmul_v = bcast(llvm.BitcastOp(T.f32, _u(TBITS_ARG)).result)
            e5_lo = arith.constant_vector(-ERM_E5M2_MAX, _v8f())
            e5_hi = arith.constant_vector(ERM_E5M2_MAX, _v8f())
            e4_lo = arith.constant_vector(-ERM_E4M3_MAX, _v8f())
            e4_hi = arith.constant_vector(ERM_E4M3_MAX, _v8f())

        def load_param(P):
            r = buffer_ops.create_buffer_resource(P, max_size=True)
            if w_f32:
                lo = buffer_ops.buffer_load(r, col, vec_width=4, dtype=T.f32)
                hi = buffer_ops.buffer_load(r, col + _i32(4), vec_width=4, dtype=T.f32)
                return vector.shuffle(_u(lo), _u(hi), list(range(8)))
            return arith.extf(_v8f(), _u(buffer_ops.buffer_load(r, col, vec_width=8, dtype=T.bf16)))

        w = load_param(W)
        if is_swish:
            # sigmoid(z) = 1 / (1 + 2^(-z*log2e)): fold -log2e into w and b once
            nl2e = arith.constant_vector(-LOG2E, _v8f())
            w2 = vmul(w, nl2e)
            b2 = vmul(load_param(B), nl2e)

        def row_sum(s):
            return _xor_sum(s, lane, LPR, DPP)

        def load_slot(r):
            # raw loads only: every slot's loads are issued before any arithmetic
            off = r * _i32(D) + col
            return dict(
                off=off,
                x=buffer_ops.buffer_load(rX, off, vec_width=8, dtype=T.bf16,
                                         cache_modifier=CACHE),
                dy=buffer_ops.buffer_load(rDY, off, vec_width=8, dtype=T.bf16,
                                          cache_modifier=CACHE),
                res=(buffer_ops.buffer_load(rRES, off, vec_width=8, dtype=T.bf16,
                                            cache_modifier=CACHE)
                     if add_residual else None),
                mean=buffer_ops.buffer_load(rMEAN, r, vec_width=1, dtype=T.f32),
                rstd=buffer_ops.buffer_load(rRSTD, r, vec_width=1, dtype=T.f32),
                mask=((buffer_ops.buffer_load(rMASK, r * _i32(D // 8) + mcol, vec_width=1,
                                              dtype=T.i8, cache_modifier=MCACHE)
                       if MASK_BITS else
                       buffer_ops.buffer_load(rMASK, r * _i32(D // 4) + mcol, vec_width=2,
                                              dtype=T.i32, cache_modifier=MCACHE))
                      if MSCALE else None),
            )

        def compute_slot(ld, acc):
            # without tri_order, explicit FMAs (packed on gfx950); the per-row
            # scalars are folded first: xhat = x*rstd - mean*rstd,
            # dx = g*rstd - (xhat*(c1*rstd) + c2*rstd)
            x = arith.extf(_v8f(), _u(ld["x"]))
            dy = arith.extf(_v8f(), _u(ld["dy"]))
            rstd = ld["rstd"]
            rstd_v = bcast(rstd)
            if tri_order:
                # Triton's order: dx is rounded to bf16 before the residual is
                # added, and where the residual cancels it, a last-bit difference
                # in dx flips that rounding (driver: stu M=4099 dy*1000, it4c).
                # The plain variant shares it, so a plain call plus a separate
                # residual add (item (20e)) equals the residual call.
                xhat = vmul(vsub(x, bcast(ld["mean"])), rstd_v)
            else:
                nmr = arith.negf(arith.mulf(_u(ld["mean"]), _u(rstd)))
                xhat = vfma(x, rstd_v, bcast(nmr))
            if is_swish:
                z = _elems(vfma(xhat, w2, b2))
                sg = []
                for j in range(8):
                    ex = rocdl.exp2(T.f32, _u(z[j]))
                    sg.append(rocdl.rcp(T.f32, _u(arith.addf(_u(ex), _u(_f32(1.0))))))
                sig = vector.from_elements(_v8f(), [_u(v) for v in sg])
                deriv = vfma(vneg(sig), sig, sig)    # sig*(1 - sig)
                t = vmul(vmul(dy, x), deriv)
                g = vmul(w, t)                       # x_ in the Triton kernel
                acc_dw = vfma(t, xhat, acc[0])
                acc_db = vadd(acc[1], t)
            else:
                g = vmul(w, dy)
                acc_dw = vfma(dy, xhat, acc[0])
                acc_db = vadd(acc[1], dy)
            c1 = arith.mulf(_u(row_sum(hsum8(vmul(xhat, g)))), _u(_f32(1.0 / D)))
            c2 = arith.mulf(_u(row_sum(hsum8(g))), _u(_f32(1.0 / D)))
            if tri_order:
                dx = vmul(vsub(g, vadd(vmul(xhat, bcast(c1)), bcast(c2))), rstd_v)
            else:
                a1 = arith.mulf(_u(c1), _u(rstd))
                a2 = arith.mulf(_u(c2), _u(rstd))
                dx = vfma(g, rstd_v, vneg(vfma(xhat, bcast(a1), bcast(a2))))
            if is_swish:
                dx = vfma(dy, sig, dx)
            if add_residual:
                dx = vadd(arith.extf(_v8f(), arith.truncf(T.vec(8, T.bf16), _u(dx))),
                          arith.extf(_v8f(), _u(ld["res"])))
            if MSCALE:
                # aten masked_scale on the bf16 dx the unfolded path stores:
                # (float(mask) * float(bf16 dx)) * scale, two fp32 multiplies
                # (a multiply, not a select: a dropped -x gives -0 as aten does)
                if MASK_BITS:
                    # column col + k <- bit k of the lane's byte: the same
                    # 0.0 / 1.0 factor as the bool byte
                    bv = vector.broadcast(T.vec(8, T.i32), arith.extui(T.i32, _u(ld["mask"])))
                    sh = vector.from_elements(T.vec(8, T.i32), [_u(_i32(k)) for k in range(8)])
                    one = vector.broadcast(T.vec(8, T.i32), _u(_i32(1)))
                    keep = arith.uitofp(_v8f(), arith.andi(arith.shrui(_u(bv), _u(sh)), _u(one)))
                else:
                    keep = arith.uitofp(_v8f(), vector.bitcast(T.vec(8, T.i8), _u(ld["mask"])))
                dxb = arith.extf(_v8f(), arith.truncf(T.vec(8, T.bf16), _u(dx)))
                dx = vmul(vmul(keep, dxb), scale_v)
            dxs = arith.truncf(T.vec(8, T.bf16), _u(dx))
            buffer_ops.buffer_store(dxs, rDX, ld["off"], cache_modifier=SCACHE)
            out = [acc_dw, acc_db]
            if sum_dx:
                out.append(vadd(acc[2], dx))
            if EMIT:
                # E-RM (module docstring), from the stored bits (contract I1):
                # x32 = fp32(bf16 dx); s = fl32(x32*inv) (bwd_fp8.py:296);
                # G8 = e5m2(clampN(s, 57344)) (:297-299); XRM =
                # e4m3fn(clampN(fl32(s*2^-t), 448)) (:301-303). clampN is
                # minnum(maxnum(v, -L), L), tl.minimum(tl.maximum(...)).
                # x32 is unpacked from the four stored dwords with integer ops
                # (element 2k = dword k << 16, element 2k+1 = dword k &
                # 0xFFFF0000; the exact fp32 widening of a bf16), so the
                # compiler reuses the store's packed converts instead of
                # re-rounding every element (fpext(fptrunc) per element).
                w4 = vector.bitcast(T.vec(4, T.i32), _u(dxs))
                lo4 = arith.shli(_u(w4), _u(arith.constant_vector(16, T.vec(4, T.i32))))
                hi4 = arith.andi(_u(w4), _u(arith.constant_vector(-65536, T.vec(4, T.i32))))
                x32 = vector.bitcast(_v8f(), vector.shuffle(_u(lo4), _u(hi4),
                                                            [0, 4, 1, 5, 2, 6, 3, 7]))
                s = vmul(x32, inv_v)
                g8 = _erm_cvt_pk8(rocdl_raw.cvt_scalef32_pk_bf8_f32,
                                  arith.minnumf(_u(arith.maxnumf(_u(s), _u(e5_lo))), _u(e5_hi)))
                st = vmul(s, tmul_v)
                xr = _erm_cvt_pk8(rocdl_raw.cvt_scalef32_pk_fp8_f32,
                                  arith.minnumf(_u(arith.maxnumf(_u(st), _u(e4_lo))), _u(e4_hi)))
                # one fp8 byte per element: the element offset IS the byte
                # offset (r*512 + col < M*512 <= 2^31 under ln_bwd's guards)
                buffer_ops.buffer_store(g8, rG8, ld["off"], cache_modifier=ECACHE,
                                        offset_is_bytes=True)
                buffer_ops.buffer_store(xr, rXRM, ld["off"], cache_modifier=ECACHE,
                                        offset_is_bytes=True)
                out.append(_erm_amax_key(x32, acc[NACC]))
            return out

        zero = arith.constant_vector(0.0, _v8f())
        init = [_u(zero)] * NACC
        if EMIT:
            # the lane's amax key (_erm_amax_key), key(+0) = ERM_KEY0
            init.append(_u(_i32(ERM_KEY0)))
        loop = scf.ForOp(_idx(_i32(0)), _idx(ITERS), _u(arith.constant(1, index=True)),
                         init)
        with ir.InsertionPoint(loop.body):
            acc = list(loop.inner_iter_args[:len(init)])
            it = arith.ArithValue(arith.index_cast(T.i32, loop.induction_variable))
            base = (it * G + blk) * _i32(RPS) + wave * _i32(RPW) + sub
            slots = [load_slot(base + _i32(u * RPI)) for u in range(UNROLL)]
            for ld in slots:
                acc = compute_slot(ld, acc)
            scf.YieldOp([_u(v) for v in acc])
        res_acc = list(loop.results[:NACC])

        # reduce the block's SLOTS partial rows in LDS, store one row per block
        red = SmemPtr(allocator.get_base(), off_red, T.f32, shape=(NACC * SLOTS * D,)).get()
        slot = wave * _i32(RPW) + sub
        for o in range(NACC):
            vector.store(_u(res_acc[o]), red, [_idx(_i32(o * SLOTS * D) + slot * _i32(D) + col)],
                         alignment=16)
        if EMIT:
            # E-RM amax, wave part: decode the lane's key max to the fp32 bits
            # of max |x| over its non-NaN elements ((key - ERM_KEY0) >> 1 =
            # m << 16; +0 when none), reduce the wave with an unsigned max
            # (non-negative, non-NaN fp32 bits order as integers), and lane 0
            # parks the wave's value in LDS before the block barrier below.
            bits = arith.shrui(_u(arith.subi(_u(loop.results[NACC]), _u(_i32(ERM_KEY0)))),
                               _u(_i32(1)))
            bits = _erm_xor_maxu(bits, lane, 64)
            am = SmemPtr(allocator.get_base(), off_am, T.i32, shape=(WAVES,)).get()
            lane0 = arith.cmpi(arith.CmpIPredicate.eq, _u(lane), _u(_i32(0)))
            guard = scf.IfOp(lane0, results_=[], has_else=False)
            with ir.InsertionPoint(guard.then_block):
                vector.store(vector.from_elements(T.vec(1, T.i32), [_u(bits)]), am,
                             [_idx(wave)], alignment=4)
                scf.YieldOp([])
        gpu.barrier()
        rP = buffer_ops.create_buffer_resource(PART, max_size=True)
        c0 = tid * _i32(VEC)
        for o in range(NACC):
            tot = None
            for k in range(SLOTS):
                v = vector.load_op(T.vec(VEC, T.f32), red, [_idx(_i32((o * SLOTS + k) * D) + c0)])
                tot = v if tot is None else arith.addf(_u(tot), _u(v))
            poff = (_i32(o) * G + blk) * _i32(D) + c0
            if VEC == 1:
                tot = vector.extract(_u(tot), static_position=[0], dynamic_position=[])
            buffer_ops.buffer_store(_u(tot), rP, poff)
        if EMIT:
            # E-RM amax, block part: the max of the WAVES parked values is the
            # logical block's partial; thread 0 stores AMAXP[blk] (plain store,
            # one per block, no atomics; exact num_records drops the masked lanes)
            wv = _elems(vector.load_op(T.vec(WAVES, T.i32), am, [_idx(_i32(0))]), WAVES)
            bm = wv[0]
            for k in range(1, WAVES):
                bm = arith.maxui(_u(bm), _u(wv[k]))
            first = arith.cmpi(arith.CmpIPredicate.eq, _u(tid), _u(_i32(0)))
            buffer_ops.buffer_store(_u(bm), rAM, blk, mask=first)

    if EMIT:
        # G8 e5m2 [M, 512], XRM e4m3fn [M, 512] (as uint8 views), AMAXP fp32
        # [G], INV fp32 [1] (the consumer site's inv_scale), TBITS the fp32
        # bits of 2^-t as an Int32 (the jit's fast argument path)
        @flyc.kernel(name=name)
        def kern_erm(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
                     W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
                     PART: fx.Tensor, G8: fx.Tensor, XRM: fx.Tensor, AMAXP: fx.Tensor,
                     INV: fx.Tensor, M_ARG: fx.Int32, ITERS_ARG: fx.Int32, G_ARG: fx.Int32,
                     TBITS_ARG: fx.Int32):
            emit(DX, DY, X, RES, W, B, MEAN, RSTD, PART, M_ARG, ITERS_ARG, G_ARG,
                 G8=G8, XRM=XRM, AMAXP=AMAXP, INV=INV, TBITS_ARG=TBITS_ARG)

        @flyc.jit
        def launch_erm(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
                       W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
                       PART: fx.Tensor, G8: fx.Tensor, XRM: fx.Tensor, AMAXP: fx.Tensor,
                       INV: fx.Tensor, M: fx.Int32, ITERS: fx.Int32, G: fx.Int32,
                       TBITS: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            kern_erm(DX, DY, X, RES, W, B, MEAN, RSTD, PART, G8, XRM, AMAXP, INV,
                     M, ITERS, G, TBITS).launch(
                grid=(G, 1, 1), block=(THREADS, 1, 1), stream=stream)

        launch_erm.kernel_name = name
        launch_erm.rows_per_iter = RPS
        launch_erm.amax_partials_per_block = 1
        return launch_erm

    if MSCALE:
        # DMASK: the bool mask viewed as uint8, or with MASK_BITS the packed
        # uint8 [M, D/8] mask; SCALE: the fp32 scale's bits as an Int32 (Int32
        # arguments keep the jit on its fast launch path)
        @flyc.kernel(name=name)
        def kern_ms(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
                    W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
                    PART: fx.Tensor, DMASK: fx.Tensor, M_ARG: fx.Int32,
                    ITERS_ARG: fx.Int32, G_ARG: fx.Int32, SCALE_ARG: fx.Int32):
            emit(DX, DY, X, RES, W, B, MEAN, RSTD, PART, M_ARG, ITERS_ARG, G_ARG,
                 DMASK, SCALE_ARG)

        @flyc.jit
        def launch_ms(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
                      W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
                      PART: fx.Tensor, DMASK: fx.Tensor, M: fx.Int32, ITERS: fx.Int32,
                      G: fx.Int32, SCALE: fx.Int32, stream: fx.Stream = fx.Stream(None)):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            kern_ms(DX, DY, X, RES, W, B, MEAN, RSTD, PART, DMASK, M, ITERS, G, SCALE).launch(
                grid=(G, 1, 1), block=(THREADS, 1, 1), stream=stream)

        launch_ms.kernel_name = name
        launch_ms.rows_per_iter = RPS
        return launch_ms

    @flyc.kernel(name=name)
    def kern(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
             W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
             PART: fx.Tensor, M_ARG: fx.Int32, ITERS_ARG: fx.Int32, G_ARG: fx.Int32):
        emit(DX, DY, X, RES, W, B, MEAN, RSTD, PART, M_ARG, ITERS_ARG, G_ARG)

    @flyc.jit
    def launch(DX: fx.Tensor, DY: fx.Tensor, X: fx.Tensor, RES: fx.Tensor,
               W: fx.Tensor, B: fx.Tensor, MEAN: fx.Tensor, RSTD: fx.Tensor,
               PART: fx.Tensor, M: fx.Int32, ITERS: fx.Int32, G: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(DX, DY, X, RES, W, B, MEAN, RSTD, PART, M, ITERS, G).launch(
            grid=(G, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.rows_per_iter = RPS
    return launch


@functools.lru_cache(maxsize=None)
def build_finalize(D: int, nacc: int, w_f32: bool):
    """PART[o, 0:G, D] fp32 -> OUTDW[D], OUTDB[D] (weight dtype) for o = 0, 1 and
    OUTS[D] fp32 for o = 2. Separate tensors: the training repository's STU
    entry point is a torch custom op, whose outputs must not alias."""
    NCG = D // 8                 # column groups of 8
    WAVES = FIN_THREADS // 64
    name = f"ln_bwd_flydsl_fin_d{D}_n{nacc}" + ("_wf32" if w_f32 else "")
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name=f"{name}_smem")
    off_red = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_red + WAVES * 8 * 4

    def emit(PART, OUTDW, OUTDB, OUTS, G_ARG):
        tid = arith.ArithValue(_as_i32(fx.thread_idx.x))
        bid = arith.ArithValue(_as_i32(fx.block_idx.x))
        G = arith.ArithValue(_u(G_ARG))
        lane = tid % _i32(64)
        wave = tid // _i32(64)
        cg = bid % _i32(NCG)
        o = bid // _i32(NCG)
        col = cg * _i32(8)
        o64 = arith.ArithValue(arith.extsi(T.i64, _u(o)))
        g64 = arith.ArithValue(arith.extsi(T.i64, _u(G)))
        rP = _shifted_rsrc(PART, o64 * g64 * arith.constant(D, type=T.i64), 4, G * _i32(D * 4))
        n_iter = (G + _i32(FIN_THREADS * 4 - 1)) // _i32(FIN_THREADS * 4)
        zero = arith.constant_vector(0.0, _v8f())
        loop = scf.ForOp(_idx(_i32(0)), _idx(n_iter), _u(arith.constant(1, index=True)), [_u(zero)])
        with ir.InsertionPoint(loop.body):
            acc = loop.inner_iter_args[0]
            it = arith.ArithValue(arith.index_cast(T.i32, loop.induction_variable))
            vals = []
            for u in range(4):
                r = it * _i32(FIN_THREADS * 4) + _i32(u * FIN_THREADS) + tid
                off = r * _i32(D) + col
                lo = buffer_ops.buffer_load(rP, off, vec_width=4, dtype=T.f32)
                hi = buffer_ops.buffer_load(rP, off + _i32(4), vec_width=4, dtype=T.f32)
                vals.append(vector.shuffle(_u(lo), _u(hi), list(range(8))))
            for v in vals:
                acc = arith.addf(_u(acc), _u(v))
            scf.YieldOp([_u(acc)])
        acc = loop.results[0]
        # wave tree, then LDS across waves
        part = [_xor_sum(e, lane, 64) for e in _elems(acc)]
        red = SmemPtr(allocator.get_base(), off_red, T.f32, shape=(WAVES * 8,)).get()
        is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, _u(lane), _u(_i32(0)))
        guard = scf.IfOp(is_lane0, results_=[], has_else=False)
        with ir.InsertionPoint(guard.then_block):
            vector.store(vector.from_elements(_v8f(), [_u(p) for p in part]), red,
                         [_idx(wave * _i32(8))], alignment=16)
            scf.YieldOp([])
        gpu.barrier()
        j = tid % _i32(8)
        tot = None
        for k in range(WAVES):
            v = vector.load_op(T.vec(1, T.f32), red, [_idx(_i32(k * 8) + j)])
            v = vector.extract(_u(v), static_position=[0], dynamic_position=[])
            tot = v if tot is None else arith.addf(_u(tot), _u(v))
        first8 = arith.cmpi(arith.CmpIPredicate.slt, _u(tid), _u(_i32(8)))
        # exact sizes: a masked store moves its offset to 0x7FFFFFFF, which is
        # dropped only beyond num_records (max_size would write 2 GB past it)
        wval = _u(tot) if w_f32 else arith.truncf(T.bf16, _u(tot))
        for oi, OUT in ((0, OUTDW), (1, OUTDB)):
            is_o = arith.andi(first8, arith.cmpi(arith.CmpIPredicate.eq, _u(o), _u(_i32(oi))))
            rO = buffer_ops.create_buffer_resource(OUT, max_size=False,
                                                   num_records_bytes=D * (4 if w_f32 else 2))
            buffer_ops.buffer_store(wval, rO, col + j, mask=is_o)
        if nacc == 3:
            is_s = arith.andi(first8, arith.cmpi(arith.CmpIPredicate.eq, _u(o), _u(_i32(2))))
            rS = buffer_ops.create_buffer_resource(OUTS, max_size=False, num_records_bytes=D * 4)
            buffer_ops.buffer_store(_u(tot), rS, col + j, mask=is_s)

    @flyc.kernel(name=name)
    def kern(PART: fx.Tensor, OUTDW: fx.Tensor, OUTDB: fx.Tensor, OUTS: fx.Tensor,
             G_ARG: fx.Int32):
        emit(PART, OUTDW, OUTDB, OUTS, G_ARG)

    @flyc.jit
    def launch(PART: fx.Tensor, OUTDW: fx.Tensor, OUTDB: fx.Tensor, OUTS: fx.Tensor,
               G: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(PART, OUTDW, OUTDB, OUTS, G).launch(
            grid=(NCG * nacc, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream)

    return launch


def grid_for(M: int, device, waves: int, tile_mult: int) -> int:
    cus = torch.cuda.get_device_properties(device).multi_processor_count
    return max(1, min(cus * tile_mult, (M + waves - 1) // waves))


def offsets_fit(M: int, D: int, device, is_swish: bool, add_residual: bool, sum_dx: bool,
                cfg=None) -> bool:
    """True when every row the persistent grid ADDRESSES fits 32-bit buffer
    offsets. The main loop walks iters * G * rows_per_iter rows and masks the
    tail by row index, but the tail rows' byte offsets are still formed; past
    2**32 they wrap back in bounds (x/dy/residual read real rows, mean/rstd read
    0, dx races a real row's store), so the bound is on the padded row count,
    not on M. The training dispatch falls back to Triton when this is False.
    (On MI350X, G=2048 pads the largest per-rank M of the production batch,
    4,193,280 rows at D=512, to exactly 2**22 rows, which fits; a 240-CU part,
    G=1920, pads the same M to 4,208,640 rows, which would not.)"""
    waves, unroll, tile_mult, *_ = cfg or DEFAULT_CFG.get(
        (D, bool(is_swish), bool(add_residual), bool(sum_dx)), FALLBACK_CFG)
    G = grid_for(M, device, waves, tile_mult)
    rpi = waves * (64 // (D // 8)) * unroll
    padded = ((M + G * rpi - 1) // (G * rpi)) * G * rpi
    return padded * D * 2 <= 2 ** 32


def kernel_options(D: int, is_swish: bool, add_residual: bool, sum_dx: bool,
                   cache=None, remap=None, store_cache=None):
    """(remap run length, load cache policy, store cache policy) that ln_bwd
    uses for this variant: the overrides, else the module configuration
    (STREAM_CACHE_CFG / STREAM_CACHE_ON, XCD_REMAP_CFG / XCD_REMAP). The
    dispatch logs it in its arming line; the startup prebuild builds it."""
    key = (int(D), bool(is_swish), bool(add_residual), bool(sum_dx))
    if cache is None:
        c = STREAM_CACHE_CFG.get(key, STREAM_CACHE) if STREAM_CACHE_ON else STREAM_CACHE
        ld_c, st_c = c if isinstance(c, tuple) else (c, c)
    else:
        ld_c = st_c = cache
    if store_cache is not None:
        st_c = store_cache
    if remap is None:
        remap = XCD_REMAP_CFG.get(key, 0) if XCD_REMAP else 0
    for v in (remap, ld_c, st_c):
        if isinstance(v, bool):   # True would silently become a run length of 1
            raise TypeError(f"ln_bwd option must be an int, not {v!r}")
    remap, ld_c, st_c = int(remap), int(ld_c), int(st_c)
    assert remap >= 0 and ld_c >= 0 and st_c >= 0, (remap, ld_c, st_c)
    return remap, ld_c, st_c


def ln_bwd(dy, x, weight, bias, mean, rstd, residual=None, *, is_swish=False,
           sum_dx=False, cfg=None, stream=None, out=None, cache=None, remap=None,
           store_cache=None, dropout_mask=None, dropout_scale=1.0):
    """dx, dweight, dbias (weight's dtype), and with sum_dx the fp32 dx sum.
    `out` may hold preallocated "dx", "dw", "db" and "outs" [D] and a flat fp32
    "part" (at least 3*G*D): the driver's sweep reuses one set across
    configurations, so they compare on the same memory. `cache` (both the
    loads' and the dx store's policy), `store_cache` (the dx store's alone)
    and `remap` (a run length in blocks, 0 = off) override STREAM_CACHE_CFG
    and XCD_REMAP / XCD_REMAP_CFG (kernel_options; cache=0, remap=0 is v2's
    main kernel); none of them changes any output bit.
    dropout_mask (bool [M, D], contiguous; residual calls only) and
    dropout_scale (a Python float, rounded to fp32 as aten's and Triton's
    scale) fold the following dropout's backward into dx (module docstring,
    mscale); the residual variant's cfg, options, grid and finalize are kept,
    so dW/dB equal the unmasked call's. A uint8 [M, D/8] dropout_mask
    (contiguous) is the bit-packed mask (module docstring, mscale bits): the
    outputs are bitwise those of the bool mask it packs."""
    out = out or {}
    # The training step calls this inside autograd's backward, where the saved
    # activation and the weights still require grad; FlyDSL passes tensors by
    # DLPack, which refuses those ("Can't export tensors that require gradient",
    # job 59872). detach() launches nothing.
    dy, x, weight, mean, rstd = dy.detach(), x.detach(), weight.detach(), mean.detach(), rstd.detach()
    bias = bias.detach() if bias is not None else None
    residual = residual.detach() if residual is not None else None
    mscale = dropout_mask is not None
    mask_bits = mscale and dropout_mask.dtype == torch.uint8
    if mask_bits:
        assert residual is not None and not is_swish and not sum_dx
        # packed bits: bit k of byte j of a row = column 8j + k
        assert x.dim() == 2 and x.shape[1] % 8 == 0, tuple(x.shape)
        assert dropout_mask.shape == (x.shape[0], x.shape[1] // 8), (
            dropout_mask.dtype, tuple(dropout_mask.shape))
        assert dropout_mask.is_contiguous()
        dropout_mask = dropout_mask.detach()
        scale_bits = struct.unpack("<i", struct.pack("<f", float(dropout_scale)))[0]
    elif mscale:
        assert residual is not None and not is_swish and not sum_dx
        assert dropout_mask.dtype == torch.bool and dropout_mask.shape == x.shape, (
            dropout_mask.dtype, tuple(dropout_mask.shape))
        assert dropout_mask.is_contiguous()
        # one byte per element (the uint8 view Triton's mscale launch passes)
        # and the fp32 scale's bits
        dropout_mask = dropout_mask.detach().view(torch.uint8)
        scale_bits = struct.unpack("<i", struct.pack("<f", float(dropout_scale)))[0]
    M, D = x.shape
    assert x.dtype == torch.bfloat16 and dy.dtype == torch.bfloat16, (x.dtype, dy.dtype)
    assert x.is_contiguous() and dy.is_contiguous()
    assert mean.dtype == torch.float32 and rstd.dtype == torch.float32
    w_f32 = weight.dtype == torch.float32
    add_residual = residual is not None
    nacc = 3 if sum_dx else 2
    key = (D, bool(is_swish), add_residual, bool(sum_dx))
    waves, unroll, tile_mult, *rest = cfg or DEFAULT_CFG.get(key, FALLBACK_CFG)
    dpp = bool(rest[0]) if rest else False
    remap, cache, store_cache = kernel_options(D, is_swish, add_residual, sum_dx,
                                               cache, remap, store_cache)
    G = grid_for(M, x.device, waves, tile_mult)
    assert M * D * 2 < 2 ** 32, (M, D)      # 32-bit buffer offsets
    assert offsets_fit(M, D, x.device, is_swish, add_residual, sum_dx, cfg), (M, D)
    dx = out["dx"] if "dx" in out else torch.empty_like(x)
    dw = out["dw"] if "dw" in out else torch.empty((D,), dtype=weight.dtype, device=x.device)
    db = out["db"] if "db" in out else torch.empty((D,), dtype=weight.dtype, device=x.device)
    if sum_dx:
        outs = out["outs"] if "outs" in out else torch.empty((D,), dtype=torch.float32, device=x.device)
    else:
        outs = dw   # placeholder argument: the nacc=2 finalize never stores OUTS
    if M == 0:
        dw.zero_()
        db.zero_()
        outs.zero_()
    else:
        if stream is None:
            stream = torch.cuda.current_stream(x.device)
        part = (out["part"][: nacc * G * D].view(nacc, G, D) if "part" in out
                else torch.empty((nacc, G, D), dtype=torch.float32, device=x.device))
        main = build_ln_bwd(D, bool(is_swish), add_residual, bool(sum_dx), w_f32, waves, unroll,
                            cache, dpp, remap, store_cache, mscale, mask_bits)
        iters = (M + G * main.rows_per_iter - 1) // (G * main.rows_per_iter)
        # _fast_call: FlyDSL 0.2.4's per-call jit overhead (module top).
        if mscale:
            _fast_call(main, dx, dy, x, residual, weight, bias if bias is not None else weight,
                       mean, rstd, part, dropout_mask, M, iters, G, scale_bits,
                       fx.Stream(stream))
        else:
            _fast_call(main, dx, dy, x, residual if add_residual else x, weight,
                       bias if bias is not None else weight, mean, rstd, part,
                       M, iters, G, fx.Stream(stream))
        fin = build_finalize(D, nacc, w_f32)
        _fast_call(fin, part, dw, db, outs, G, fx.Stream(stream))
    if sum_dx:
        return dx, dw, db, outs
    return dx, dw, db


def erm_tbits(t_shift: int) -> int:
    """The fp32 bits of 2^-t_shift as a signed Int32 kernel argument (exact for
    every shift the site uses, 4..7, and any |t| <= 126)."""
    t = int(t_shift)
    assert -126 <= t <= 126, t
    return struct.unpack("<i", struct.pack("<f", 2.0 ** -t))[0]


def ln_bwd_erm(dy, x, weight, bias, mean, rstd, residual, inv_scale, t_shift, *,
               cfg=None, stream=None, out=None, cache=None, remap=None, store_cache=None,
               emit_cache=None):
    """E-RM (module docstring; EXPERIMENT, default off: nothing calls this unless
    asked). `ln_bwd(dy, x, weight, bias, mean, rstd, residual)` for the residual
    variant at D=512 that ALSO returns the two FP8 copies of the stored dx that
    bwd_fp8.quantize_row_major_dual(dx, inv_scale, t_dtype=e4m3fn,
    t_inv_mul=2^-t_shift, t_limit=448) would write, both ROW-MAJOR, plus the
    amax partials: (dx, dw, db, g8 [M,512] e5m2, xrm [M,512] e4m3fn,
    amaxp [G] fp32, one per logical block).
      g8   == the dual cast's row-major DST, byte for byte;
      xrm  == its token-major DST_T transposed back to [M, 512] (DST_T[c, t] ==
              xrm[t, c]; the cast's zero padding / step tiling is not produced:
              the E-RM W3 entry range-masks tokens >= M);
      amaxp: max(amaxp) == the cast's amax unless an in-range 256x128 cast tile
              is all NaN (then the cast's amax is NaN and this is the max over
              the non-NaN |dx|; actckpt A5's exempt case). Reduce it with the
              HELPERS_FUSED site program or torch.amax, as the cast's partials.
    inv_scale: the consumer site's fp32 [1] device tensor (state.inv_scale),
    read in-kernel at execution time; t_shift: the site's host int. cfg, cache,
    remap, store_cache: as ln_bwd (same grid, rows per block and finalize, so
    dx, dw, db are the residual call's); emit_cache: the G8 / XRM store policy
    (None = ERM_STORE_CACHE, -1 = the dx store's)."""
    out = out or {}
    dy, x, weight, mean, rstd = dy.detach(), x.detach(), weight.detach(), mean.detach(), rstd.detach()
    bias = bias.detach() if bias is not None else None
    assert residual is not None, "E-RM is the residual variant only"
    residual = residual.detach()
    inv_scale = inv_scale.detach()
    M, D = x.shape
    assert D == 512, D
    assert x.dtype == torch.bfloat16 and dy.dtype == torch.bfloat16, (x.dtype, dy.dtype)
    assert residual.dtype == torch.bfloat16 and residual.shape == x.shape, (residual.dtype, residual.shape)
    assert x.is_contiguous() and dy.is_contiguous() and residual.is_contiguous()
    assert mean.dtype == torch.float32 and rstd.dtype == torch.float32
    assert weight.dtype == torch.bfloat16, weight.dtype     # EMIT builds w_f32=False only
    assert inv_scale.dtype == torch.float32 and inv_scale.numel() == 1, (inv_scale.dtype, inv_scale.shape)
    assert inv_scale.is_contiguous() and inv_scale.device == x.device
    key = (D, False, True, False)
    waves, unroll, tile_mult, *rest = cfg or DEFAULT_CFG.get(key, FALLBACK_CFG)
    dpp = bool(rest[0]) if rest else False
    remap, cache, store_cache = kernel_options(D, False, True, False, cache, remap, store_cache)
    ecache = ERM_STORE_CACHE if emit_cache is None else int(emit_cache)
    G = grid_for(M, x.device, waves, tile_mult)
    assert M * D * 2 < 2 ** 32, (M, D)      # 32-bit buffer offsets (bf16 tensors; fp8 is half)
    assert offsets_fit(M, D, x.device, False, True, False, cfg), (M, D)
    dx = out["dx"] if "dx" in out else torch.empty_like(x)
    dw = out["dw"] if "dw" in out else torch.empty((D,), dtype=weight.dtype, device=x.device)
    db = out["db"] if "db" in out else torch.empty((D,), dtype=weight.dtype, device=x.device)
    g8 = out["g8"] if "g8" in out else torch.empty((M, D), dtype=torch.float8_e5m2, device=x.device)
    xrm = out["xrm"] if "xrm" in out else torch.empty((M, D), dtype=torch.float8_e4m3fn, device=x.device)
    assert g8.dtype == torch.float8_e5m2 and g8.shape == (M, D) and g8.is_contiguous()
    assert xrm.dtype == torch.float8_e4m3fn and xrm.shape == (M, D) and xrm.is_contiguous()
    if M == 0:
        dw.zero_()
        db.zero_()
        return dx, dw, db, g8, xrm, torch.zeros((0,), dtype=torch.float32, device=x.device)
    if stream is None:
        stream = torch.cuda.current_stream(x.device)
    nacc = 2
    part = (out["part"][: nacc * G * D].view(nacc, G, D) if "part" in out
            else torch.empty((nacc, G, D), dtype=torch.float32, device=x.device))
    main = build_ln_bwd(D, False, True, False, False, waves, unroll, cache, dpp, remap,
                        store_cache, False, False, True, ecache)
    npart = G * main.amax_partials_per_block
    amaxp = (out["amaxp"][:npart] if "amaxp" in out
             else torch.empty((npart,), dtype=torch.float32, device=x.device))
    # r14 E-RM integration (iter62 review F2): the kernel stores npart fp32
    # partials (num_records = G * 4 bytes) whatever tensor backs AMAXP, so a
    # short, non-fp32, strided or foreign-device out["amaxp"] fails here, on the
    # host, before any launch (host-only; the kernel is unchanged).
    assert (amaxp.dtype == torch.float32 and amaxp.numel() == npart and npart >= G
            and amaxp.is_contiguous() and amaxp.device == x.device), (
        amaxp.dtype, amaxp.numel(), npart, G, amaxp.stride(), amaxp.device)
    assert dx.shape == x.shape and dx.dtype == x.dtype and dx.is_contiguous(), (dx.shape, dx.dtype)
    iters = (M + G * main.rows_per_iter - 1) // (G * main.rows_per_iter)
    # _fast_call: FlyDSL 0.2.4's per-call jit overhead (module top).
    _fast_call(main, dx, dy, x, residual, weight, bias if bias is not None else weight, mean,
               rstd, part, g8, xrm, amaxp, inv_scale, M, iters, G, erm_tbits(t_shift),
               fx.Stream(stream))
    fin = build_finalize(D, nacc, False)
    _fast_call(fin, part, dw, db, dw, G, fx.Stream(stream))
    return dx, dw, db, g8, xrm, amaxp
