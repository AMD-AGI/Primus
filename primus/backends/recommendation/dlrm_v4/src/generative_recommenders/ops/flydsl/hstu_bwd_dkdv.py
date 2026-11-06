"""HSTU jagged-attention BACKWARD dk/dv pass written in FlyDSL for gfx950.

Structure (mirrors the Triton production split: a dk/dv pass + a separate dq pass):

    per (batch, head, n-block):
      load K,V tile (BN x D) once into registers
      for m0 in [n0, seq_len) step BM:
          stage Q,DO tiles in LDS (row-major + transposed)
          qk   = q  @ k^T                 (mfma, acc in f32)
          silu = qk*sigmoid(qk)/MAX_SEQ   (masked: causal + m<seq_len)
          dv  += silu^T @ do              (silu^T is the f32 acc reinterpreted
                                           as an mfma A-operand -- free transpose)
          dp   = do @ v^T
          ds   = dp * sigmoid'(...)       (masked)
          dk  += ds^T @ q
      store dk*alpha, dv

This covers four of the five HSTU backward GEMMs.  dq is NOT computed here.

Offsets: the per-(batch,head) base pointer is advanced in 64-bit before the
buffer resource is built, so every in-kernel voffset stays < 2^31 even though
the full tensors are multi-GB.
"""

import functools
import inspect
import math
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref
from flydsl._mlir.dialects import rocdl as rocdl_raw
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
try:
    # FlyDSL 0.2.4 (ROCm 10.2 UTD image) renamed fly_values to
    # extract_to_ir_values; same contract: the list of ir.Values of a DSL
    # value, [0] being the tensor's memref.
    from flydsl.compiler.protocol import extract_to_ir_values as fly_values
except ImportError:  # FlyDSL 0.1.1 (rocm/primus:v26.3)
    from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from generative_recommenders.ops.flydsl import current_stream_launch
# FlyDSL 0.2.4's per-call @flyc.jit overhead: see ln_bwd.py _fast_call.
from generative_recommenders.ops.flydsl.ln_bwd import _fast_call

LOG2E = 1.4426950408889634

# The shipping configuration, in one place so the benchmark, the correctness
# gate and the autograd wrapper cannot drift apart.
#
# dq16="bf16p" accumulates dq_acc in bf16 through packed atomics.  It is not a
# micro-optimisation: the fused kernel's runtime IS its dq atomic stream, the
# atomic unit is limited per 64 B line touched (atomic_roof.hip), and bf16p is
# the only 16-bit layout that actually halves the lines rather than just the
# bytes.  51.45 -> 37.29 ms.  It costs dq precision: 55.5 -> 45.2 dB against the
# dense fp32 reference, still clear of the 40 dB per-output gate, and dq is a
# bf16 output either way.
#
# sched="iglp2": iglp0 was tuned on the dk/dv-only body, which is now two thirds
# of this one.  Re-swept after bf16p moved the bottleneck off the atomic stream:
# iglp0 36.98, iglp1 36.13, iglp2 35.04 ms.
SHIPPED = {"BN": 128, "fused": True, "dq16": "bf16p", "sched": "iglp2"}


def dq_acc_dtype(dq16=None):
    """torch dtype the dq accumulator must have for a given dq16 mode."""
    import torch

    mode = SHIPPED["dq16"] if dq16 is None else dq16
    return {"": torch.float32, "f16": torch.float16}.get(mode, torch.bfloat16)

def cvt_scale_is_exact(scale):
    """Does `scale` survive `cvt_scalef32_pk_bf16_fp8`'s scale operand intact?

    Only if it is an exact power of two.  That operand is **exponent-only** on
    gfx950 (E8M0): the hardware keeps the biased exponent and discards the
    mantissa, so 2.234e-4 becomes 2**-13 = 1.2207e-4 and every value converted
    through it comes back scaled by a constant 1/1.830.  Nothing traps, nothing
    warns -- the kernel simply returns wrong numbers.  Measured on the
    `q_bf16` + `scale_in_cvt` configuration with a calibrated `amax/448` scale:
    dq/dk/dv = 3.1/6.9/6.9 dB, against 28.1/31.3/31.3 dB for the identical build
    with the scale rounded to a power of two, and 3.08 dB predicted in advance
    from the truncation alone.

    `frexp` returns a mantissa in [0.5, 1), so an exact power of two -- and only
    an exact power of two, subnormals included -- gives exactly 0.5.
    """
    try:
        s = float(scale)
    except (TypeError, ValueError):
        # Better to refuse than to wave through a boxed value we cannot read;
        # a scale that reaches the cvt unchecked is a silent wrong answer.
        return False
    return math.isfinite(s) and s > 0.0 and math.frexp(s)[0] == 0.5


# valu "tsc" (R15 iter74 f2; range R15 iter75 tscv2): the k range the token
# accepts.  Every derived factor (tau = 2^k*qk, 2^-k*om', the cast operand
# 2^k/(mu2_ssc*sc)) is a normal f32 there and the E8M0 byte 127+k is
# 87..120.  -7 is the largest k under vf3p's guard s <= 0.25 (tsc needs vf4,
# vf4 needs vf3p; s_-7 = 0.2475 <= 0.25 < s_-6 = 0.3500); -40 is the extent
# of the iter74 exactness evidence.  MUST equal ops/fp8_qkv_spf_bwd.py
# BWD_J_MIN/BWD_J_MAX: the launcher (DLRM_HSTU_FP8_LAB_BWD_SPF) refuses to
# arm when they differ.
TSC_K_MIN, TSC_K_MAX = -40, -7


def tsc_exponent(alpha, qkv_scale):
    """k if the bwd kernel's own f32 constant chain gives
    c_t = f32(f32(f32(alpha) * f32(s*s)) * f32(-log2e)) == -2**k EXACTLY
    (s = f32(qkv_scale)) with TSC_K_MIN <= k <= TSC_K_MAX, else None.

    valu "tsc" folds t = qk*c_t into the score MFMA's E8M0 scale, which is
    exact only for such a qkv_scale.  Host floats: the product of two f32 is
    exact in a double, so one rounding to f32 is IEEE f32 multiplication.
    """
    def f32(x):
        return struct.unpack("<f", struct.pack("<f", x))[0]

    try:
        s = f32(float(qkv_scale))
        ct = f32(f32(f32(float(alpha)) * f32(s * s)) * f32(-LOG2E))
    except (TypeError, ValueError, OverflowError):
        return None
    if not (s > 0.0 and math.isfinite(ct) and ct < 0.0):
        return None
    m, e = math.frexp(-ct)
    k = e - 1
    return k if m == 0.5 and TSC_K_MIN <= k <= TSC_K_MAX else None


def _require_pow2_cvt_scale(scale):
    """Fail loudly rather than return silently-rescaled gradients.

    Checked at every call, not at compile time, because `scale_in_cvt` is a
    build-time flag while `qkv_scale` is a runtime argument -- a cached kernel
    built under one scale is handed a different one on the next step.
    """
    if not cvt_scale_is_exact(scale):
        raise ValueError(
            "scale_in_cvt=True folds qkv_scale into "
            "cvt_scalef32_pk_bf16_fp8, whose gfx950 scale operand is "
            f"exponent-only, but qkv_scale={scale!r} is not an exact power of "
            "two. It would be truncated to 2**floor(log2(qkv_scale)) and every "
            "gradient would be silently rescaled (measured 3.1 dB). Quantize "
            "q/k/v with a power-of-two scale, or build with scale_in_cvt=False "
            "to apply the scale in the score/epilogue math instead."
        )


PTR_TY = "!llvm.ptr"
LDS_PTR_TY = "!llvm.ptr<3>"
RSRC_TY = "!llvm.ptr<8>"
CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)
GEP_DYNAMIC = -2147483648  # kDynamicIndex marker in llvm.getelementptr
CAT8 = [0, 1, 2, 3, 4, 5, 6, 7]  # concatenate two 4-wide fragments into K=32


def _lds_gep(mem, elem_off, elem_ty):
    """`!llvm.ptr<3>` at a dynamic element offset into an LDS memref.

    rocdl.ds.read.tr16.b64 takes a raw LDS pointer, not the (memref, index) pair
    the vector ops accept, so the view has to be reduced to a pointer by hand.
    """
    ptr_ty = ir.Type.parse(LDS_PTR_TY)
    # fly.extract_aligned_pointer_as_index only accepts global tensors, so the
    # LDS view goes through memref's version: index -> i32 -> ptr<3> (an LDS
    # pointer is 32 bits on AMDGPU).
    idx = memref.ExtractAlignedPointerAsIndexOp(mem).result
    base = llvm.IntToPtrOp(ptr_ty, arith.index_cast(T.i32, idx)).result
    return llvm.GEPOp(
        ptr_ty, base, [arith.unwrap(elem_off)], [GEP_DYNAMIC], elem_ty,
        llvm.GEPNoWrapFlags(0),
    ).result


def _atomic_fadd(val, rsrc, byte_off, aux=0):
    """Global atomic add through a buffer resource.

    flydsl's buffer_ops has loads and stores but no atomics, so this goes
    straight to the ROCDL op.  Offset is in BYTES, like the raw op wants.
    `aux` is the cache-policy field (bit0 sc0, bit1 nt, bit4 sc1).
    """
    rocdl_raw.RawPtrBufferAtomicFaddOp(
        arith.unwrap(val),
        rsrc,
        arith.unwrap(byte_off),
        arith.unwrap(_i32(0)),
        arith.unwrap(_i32(aux)),
    )


DPP_QUAD_SWAP_ADJ = 0xB1  # quad_perm[1,0,3,2]: lane l reads lane l^1


def _dpp_swap_adj(val):
    """Lane l's copy of lane (l^1)'s value, as a DPP modifier on a v_mov.

    Costs no extra instruction slot beyond the mov and never touches the LDS
    pipe, unlike ds_bpermute.
    """
    return rocdl_raw.DPPUpdateOp(
        T.f32, arith.unwrap(val), arith.unwrap(val),
        DPP_QUAD_SWAP_ADJ, 0xF, 0xF, True,
    ).result


def _current_stream(t):
    """fx.Stream for the caller's current PyTorch stream on t's device.

    The jitted launch()'s default fx.Stream(None) is HIP's null stream, which
    does not order against PyTorch's non-blocking side streams, so a launch
    issued while a side stream is current raced its own producers.  On the
    default stream this is the same handle as before.  Only called when
    current_stream_launch() ($DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM) is on.
    """
    import torch

    return fx.Stream(torch.cuda.current_stream(t.device))


def _i32(v):
    return arith.constant(int(v), type=T.i32)


def _i64(v):
    return arith.constant(int(v), type=T.i64)


def _to_i64(v):
    v = arith.unwrap(v)
    return v if v.type == T.i64 else arith.extsi(T.i64, v)


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _idx(v):
    return arith.index_cast(T.index, v)


def _as_i32(v):
    """thread/block index -> i32 ArithValue.

    Module scope, not nested in the @flyc.kernel bodies: FlyDSL 0.2.4's AST
    rewriter turns every non-const_expr `if` of a kernel body (nested defs
    included) into a dispatched closure, so an early `return` inside the `if`
    only leaves the closure and is silently dropped.
    """
    raw = arith.unwrap(v)
    if not isinstance(raw.type, ir.IndexType):
        return arith.ArithValue(raw)
    return arith.ArithValue(arith.index_cast(T.i32, raw))


def _dq_ws_2d(t):
    """1-D dq-split workspace -> [numel / W, W] view of the same bytes
    (W = gcd(numel, 128): hstu_dq_split's workspace is a multiple of 256 B,
    i.e. of 128 bf16).  Other ranks (the fused dq accumulator) pass through.
    See call(): FlyDSL 0.2.4 packs each dynamic dim of a tensor argument as
    int32, and a production workspace has more than 2^31 elements."""
    if getattr(t, "dim", None) is None or t.dim() != 1:
        return t
    return t.view(-1, math.gcd(t.numel(), 128) or 128)


# FlyDSL 0.2.4's KernelLauncher.launch raises for a static block of more than
# 256 threads unless the @flyc.kernel declares known_block_size (msplit's
# 512-thread kernel); FlyDSL 0.1.1's kernel() has no such keyword and launches
# any block size.
_FLYC_KNOWN_BLOCK_SIZE = (
    "known_block_size" in inspect.signature(flyc.kernel).parameters)


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _shifted_rsrc(tensor, elem_offset, elem_bytes, num_bytes=None):
    """Buffer resource whose base is advanced by elem_offset.

    `elem_offset` must ALREADY be i64 when it can exceed 2^31.  It is only
    widened here, so a product formed in i32 has already wrapped by the time it
    arrives: at production's 2.77 M tokens a row stride of 2048 reaches 5.7e9
    elements, and even the packed 512 reaches 1.4e9, within 1.5x of the i32
    ceiling.  Callers build it with _to_i64 for that reason.

    `num_bytes`, when given, becomes the descriptor's num_records: a raw buffer
    load whose voffset reaches it returns zero instead of reading.  That turns
    "row past the end of this sequence" into an exact algebraic zero and removes
    the need for an in-loop bounds mask.
    """
    ptr_ty = ir.Type.parse(PTR_TY)
    base = fly.extract_aligned_pointer_as_index(ptr_ty, fly_values(tensor)[0])
    base_i64 = llvm.PtrToIntOp(T.i64, base).result
    off_i64 = arith.ArithValue(_to_i64(elem_offset))
    byte_i64 = off_i64 * arith.constant(int(elem_bytes), type=T.i64)
    new_i64 = llvm.AddOp(
        arith.unwrap(base_i64), arith.unwrap(byte_i64), llvm.IntegerOverflowFlags(0)
    ).result
    new_ptr = llvm.IntToPtrOp(ptr_ty, new_i64).result
    if num_bytes is None:
        nrec = arith.constant(0xFFFFFFFF, type=T.i64)
    else:
        nrec = arith.extsi(T.i64, arith.unwrap(num_bytes))
    return rocdl_raw.MakeBufferRsrcOp(
        ir.Type.parse(RSRC_TY),
        new_ptr,
        arith.unwrap(arith.constant(0, type=T.i16)),
        arith.unwrap(nrec),
        arith.unwrap(_i32(CDNA_BUFFER_FLAGS)),
    ).result


def _set_flat_wg_size(threads):
    """Stamp rocdl.flat_work_group_size on the kernel being emitted.

    Kept outside the @flyc.kernel body so FlyDSL's AST rewriter does not turn
    the parent walk into an scf loop.  (Same helper as iter17 b1_regdiet.)
    """
    fop = ir.InsertionPoint.current.block.owner.operation
    while fop is not None and fop.name not in ("gpu.func", "llvm.func",
                                               "func.func"):
        fop = fop.parent
    assert fop is not None, "no enclosing kernel function"
    fop.attributes["rocdl.flat_work_group_size"] = ir.StringAttr.get(
        f"{threads},{threads}")


@functools.lru_cache(maxsize=64)
def compile_hstu_bwd_dkdv(
    heads: int,
    dim: int,
    alpha: float,
    BM: int = 64,
    BN: int = 128,
    abl: str = "",
    lds_pad: int = 0,
    # Bank-conflict padding for the tr16 Q/dO tiles.  The tr16 layout is
    # flat[d_block][m][d%16] with a d_block stride of BM*16 = 1024 halfwords =
    # 2048 B, an exact multiple of the 128 B (32-bank) LDS period, so all 8
    # d_blocks alias the same banks.  write_tiles spreads consecutive threads
    # across d_blocks (st_db), so its b128 stores collide ~16-way.  Adding
    # tr_pad halfwords per d_block (16 B, keeps every d_block 16 B-aligned for
    # the b128 store/load) rotates each d_block onto a distinct bank group and
    # takes the write pass from ~16 cycles to the ideal 4.  Reads (a_frag b128,
    # b_frag_tr) are already conflict-free and stay so.  +256 B LDS total.
    tr_pad: int = 8,
    SQ: int = 136,
    ST: int = 68,
    rm_align: int = 8,
    # iglp1 measured 0.8 ms better on the fused body but trips an assertion in
    # LLVM's AMDGPUIGroupLP on some parameter combinations, so it is not the
    # default.  iglp0 is stable and best for the dk/dv-only body.
    sched: str = "iglp0",
    tr16: bool = True,
    DQC: int = 1,
    fused: bool = False,
    dq16: str = "",
    dq_scale: float = 1.0,
    gorder: str = "xcd",
    rev: bool = False,
    dq_aux: int = 0,
    # R15 design B (default off): instead of the in-kernel dq MFMA and the
    # bf16p dq atomics, every workgroup stores its e4m3 ds tile (the exact
    # dq-MFMA A fragments, m-major [m 64][n 128]) to a
    # triangular workspace passed in the DQ slot; hstu_dq_split computes
    # dq = sum_n ds K afterwards in f32 with one plain store per row.  Same
    # products, f32 instead of bf16-atomic accumulation across n-blocks.
    dqsplit: bool = False,
    NXCD: int = 8,
    rs: int = 0,
    rs_in: int = 0,
    rs_out: int = 0,
    input_fp8: bool = False,
    scale_in_cvt: bool = False,
    q_bf16: bool = False,
    rs_q: int = 0,
    # ---- true fp8 MFMA on the qk gemm -----------------------------------
    # qk = q @ k^T is the ONLY one of the five gemms whose two operands are
    # both already e4m3 in global memory.  e4m3 -> bf16 is lossless (3 mantissa
    # bits into 7), so today the kernel widens both operands and runs a bf16
    # MFMA over values that are exactly fp8 -- the products are identical
    # either way and only the f32 accumulation order changes.  That makes this
    # one gemm the only numerically FREE place to put an fp8 MFMA: every other
    # gemm has a genuinely-bf16 operand (dO) or an f32-computed one (ds), and
    # rounding those to e4m3 costs 13-21 dB (bench/sim_fp8_numerics.py).
    #
    #   ""     incumbent: bf16 16x16x32 everywhere.
    #   "k32"  mfma_f32_16x16x32_fp8_fp8 for qk.  Same (lane, element) -> k
    #          mapping as the bf16 op, so it is a drop-in once both operands
    #          are raw bytes.  The MFMA rate is only 1.06x (bench/mfma_peak.py)
    #          -- the win this mode is really testing is the LDS one: the Q
    #          A-operand read halves from b128 to b64 and the staged Q bytes
    #          halve.
    #   "k128" mfma_scale_f32_16x16x128_f8f6f4 for qk, 2.17x the bf16 rate and
    #          4x the K depth, so KSTEP collapses from 4 mfma to 1 and the four
    #          b128 A-operand reads become two.  Same LDS tile as "k32".
    #
    # Both modes keep the bf16 tr16 Q tile, because the dk gemm (dk += ds^T @ q)
    # needs q as a bf16 B-operand and ds cannot leave f32/bf16.  Reading the
    # fp8 tile transposed (ds_read_tr8_b64) and widening in registers was
    # costed and rejected: the Q tile is read 8x more than it is written, so
    # converting at read time quadruples the cvt count instead of removing it.
    fp8_mfma: str = "",
    # Bank-conflict padding, in BYTES, for the fp8 Q tile's 32-byte d-block
    # stride.  Same role as tr_pad for the bf16 tile: BM*32 = 2048 B is an
    # exact multiple of the 128 B LDS period, so without it all four d_blocks
    # alias the same banks and write_tiles (which spreads consecutive threads
    # across d_blocks via st_d // 32) collides 4-way.  32 B rotates each block
    # by 8 banks and keeps every block 16 B-aligned for the b128 reads "k128"
    # issues.
    q8_pad: int = 32,
    # ---- TRACK U1: m-unroll x2 so the dv/dk out-gemms reach K=128 ---------
    # dv = silu^T @ dO and dk = ds^T @ q contract over m.  One m tile is BM=64
    # deep, which is why they are stuck on the bf16 16x16x32 op: the only
    # profitable fp8 shape on gfx950 is mfma_scale_f32_16x16x128_f8f6f4, and
    # that needs contraction depth 128.  Running TWO m tiles per loop body
    # forms K=128.
    #   ""      incumbent.
    #   "bf16"  m-unroll only, out-gemms still bf16 K=32.  CONTROL: isolates
    #           the register/schedule cost of the unroll from the fp8 gemm.
    #   "k128"  m-unroll + mfma_scale_f32_16x16x128_f8f6f4 on dv and dk.
    # The out-gemm mfma count falls 4x (128 -> 32 per two m tiles); the A
    # operands (silu, ds) are f32 in registers, so unlike dO/V they can be
    # RESCALED by a power of two before the cast, which is the one degree of
    # freedom the closed dp-k128 experiment did not have.
    mu2: str = "",
    mu2_fmt: str = "e4m3",
    mu2_ssc: float = 1.0,   # silu pre-scale (dv A operand)
    mu2_dsc: float = 1.0,   # ds   pre-scale (dk A operand)
    mu2_osc: float = 1.0,   # dO pre-scale (dv B operand)
    mu2_qsc: float = 1.0,   # q  pre-scale (dk B operand); q is already e4m3
                            # in LDS, so 1.0 is a LOSSLESS reinterpretation
    # ---- R1: full-fp8 backward (default OFF) -----------------------------
    #   ""     incumbent.
    #   "all"  ALL FIVE gemms on mfma_scale_f32_16x16x128_f8f6f4: qk (as k128),
    #          dp = dO V^T, dv = silu^T dO, dk = ds^T q, dq = ds k.  dO is an
    #          e4m3 INPUT tensor (producer-emitted), there are no bf16 Q/dO
    #          tiles at all: Q and dO are staged once as e4m3 in the 32 B
    #          d-block layout and the out-gemms' transposed B operands come
    #          straight out of them with ds_read_b64_tr_b8.  ds is exchanged
    #          through LDS as e4m3 ([n][m] bytes, tr8 back), K for dq is staged
    #          once as e4m3 in the bf16p interleave.  Needs k128 + mu2=k128 +
    #          fused + bf16p.  Stage-1 scales: unit except the existing mu2_*sc
    #          pre-scales and qkv_scale; dO's own scale is folded by the caller.
    f8b: str = "",
    ds8_pad: int = 8,       # bytes of padding per n row of the e4m3 ds tile
    # ---- R1 stage 2: runtime dO scale (default OFF, f8b only) -------------
    # When set, the launcher passes a 2-element fp32 DEVICE tensor
    # do_scale = [s_do, 1/s_do] (s_do a power of two) and the e4m3 dO tensor
    # holds dO * s_do.  dp and ds then carry the factor s_do, and the static
    # mu2_dsc is chosen for the NORMALIZED dO (amax in (224, 448]), so the
    # e4m3 ds tile sees a range that no longer depends on the loss scale.
    # All three epilogues (dv, dk, dq) multiply by 1/s_do -- exact, pow2.
    # No host sync: the scale never leaves the device.
    # RESEARCH-ONLY since the C8 fold (2026-10-03): the production wrapper
    # always passes do_dyn=False/do_blk=True, and no in-tree producer of the
    # [s_do, 1/s_do] scale remains (quantize_do_e4m3 was removed).  Kept for
    # kernel A/B probes that supply their own scale.
    do_dyn: bool = False,
    # ---- C8 option B: per-(sequence, head, token-group) dO scale -----------
    # (default OFF, f8b only, exclusive with do_dyn).  The launcher passes a
    # uint8 tensor E8 [T, H] in the do_scale slot; E8[p, h] = 127 - e_g where
    # dO8 = e4m3(dO * 2**e_g) and the group of seq-relative token p is the
    # HARDWARE MX K-block: G(p) = (p // 64, (p % 16) // 8).  dp / ds stay in
    # the per-group scaled domain in registers (the static mu2_dsc is still
    # right: each group's amax is normalized like the global one was), and
    # the 2**-e_g is undone INSIDE the out-gemm MFMAs through their E8M0 scale
    # operands (mfma_scale_f32_16x16x128_f8f6f4, factor 2**(byte-127)):
    #   dv = silu^T dO8  (K = m): B scale per K-block = that block's group byte
    #   dk = ds^T  q     (K = m): A scale per K-block = group byte
    #   dq = ds    k     (K = n): A scale per ROW m   = row's group byte
    # The unscaled side passes a RUNTIME 127 (a runtime 0 means 2**-127; only
    # a literal 0 is folded to the unscaled op).  No inv_s_do epilogue.
    do_blk: bool = False,
    # ---- R1 stage 2b: saturating e4m3 cast of silu/ds (f8b only) ----------
    # cvt_scalef32_pk_fp8_f32 does NOT saturate: |v*scale| > 448 becomes NaN
    # (iter-17 in-model probe 51366: silu*ssc reached 1e4 and dv went NaN on
    # the first armed call).  The cvt's own VOP3 clamp bit is NOT available:
    # gfx950's assembler rejects 'clamp' on v_cvt_scalef32_pk_fp8_f32
    # ("invalid operand for instruction", iter-17 probe 51407).
    # f8_sat clamps each element to +-448/scale first
    # (one v_min + one v_max, or a v_med3, per element) so an out-of-range
    # tile degrades to saturation instead of poisoning the weights.
    f8_sat: bool = False,
    # ---- silu / grad-coefficient epilogue form (roadmap R3) ---------------
    #   "legacy"  silu = (x*sig)*inv_max;
    #             gc   = sig*(1 + x*(1-sig))*inv_max;  ds = dp*vis*gc
    #   "reuse"   hoist the uniform alpha*score_input_scale product out of
    #             the per-element chain and share sig/N between SiLU and its
    #             derivative:  sig_inv = sig*inv_max;  silu = x*sig_inv;
    #             gc = fma(silu, 1-sig, sig_inv)  (= SiLU'(x)/N exactly in
    #             real arithmetic).  Same math, different f32 rounding order
    #             (~1 ulp); fewer VALU ops in the VALU-bound epilogue.
    #   "reuse_pk" (R15; default OFF, env-armed): "reuse" with every non-transcendental
    #             op of the chain (x, exp2 argument, e+1, sig/N, silu, 1-sig,
    #             the gc fma, dp*vis, *gc) issued on gfx950's double-rate
    #             packed f32 (v_pk_mul/add/fma_f32) over two consecutive
    #             accumulator elements; exp2/rcp stay scalar.  Same ops, same
    #             order, IEEE per lane -> bitwise equal to "reuse".  Same
    #             technique as hstu_fwd's epi_pk.
    # Sept-22 prototype (job42237): bwd_dkdv 54.47 -> 53.02 ms/rank/step
    # in-model, VGPR 420 -> 416, max paired train-loss diff 3.13e-7.
    epilogue: str = "reuse",
    # R15 iter11 intra-pair lookahead (dqsplit + mu2 only; 0 = off, the
    # unchanged kernel).  Both m tiles of an mu2 pair are staged in LDS up
    # front (a second q8/o8 buffer), and the qk/dp MFMAs of tile B are issued
    # in the same scheduling region as tile A's silu/ds VALU chain, which they
    # do not depend on.  Same operations on every element; only the program
    # order, the LDS staging and the out-gemm B-operand read point move.
    #   1: next pair's two tiles are prefetched at the top of the body.
    #   2: as 1, but the second tile's prefetch is issued after tile A's ds
    #      exchange (shorter live range of the prefetch registers).
    #   3: as 2, and the out-gemm MFMAs of the last NSTEP/2 d tiles are issued
    #      after the tile-B ds-exchange barrier, next to its readback / ws
    #      store (their q/dO B operands are still read before the barrier,
    #      which is what frees the q8/o8 buffers for the next body).
    #   4: as 3 with all NSTEP d tiles' out-gemm MFMAs after the barrier.
    #   5: as 2, and the out-gemm q/dO B-operand LDS reads of the first
    #      NSTEP/2 d tiles move ahead of tile B's silu/ds chain (same region),
    #      so their latency hides under that VALU work.
    #   6: as 5 with all NSTEP d tiles' B-operand reads.
    pipe: int = 0,
    # ---- R15 VALU diet (iter11 design, ported iter17; default "" = OFF, the
    # unchanged kernel and kernel name).  Comma set, every member bitwise
    # neutral (same f32 ops on every element; only register class / issue
    # order / a dead start value change):
    #   "undef"  the e4m3 pk4 casts start from an UNDEF dword instead of a
    #            zero vector (cvt_scalef32_pk_fp8_f32 writes one 16-bit half
    #            and keeps the other; both halves are always written).
    #   "rfl"    dqsplit: v_readfirstlane the per-batch workspace prefix (a
    #            workgroup-uniform buffer_load) so the workspace descriptor is
    #            SGPR-resident and the ds-tile buffer_stores leave their
    #            readfirstlane/v_cmp/saveexec waterfall loops.
    #   "ilp"    epilogue=reuse_pk: the packed silu/ds chain issued
    #            stage-major across all element pairs of an n-tile row.
    #   "nobr"   dk/dv epilogue: bound the rDK/rDV descriptors at the sequence
    #            end (num_records = seq_len rows) and store unconditionally,
    #            instead of one compare + saveexec branch per stored element;
    #            rows >= seq_len are dropped by the buffer unit's range check.
    #            Same values to the same addresses.
    valu: str = "",
    # ---- R15 iter17r3 B1 m-split (default 0 = OFF: the unchanged kernel,
    # kernel name and LDS layout).  dq-split C8 only; replaces the mu2 pair
    # body (and any `pipe`).  EIGHT waves per BN=128 workgroup (2 per SIMD):
    # waves w and w+4 own the SAME 32 n rows (NT=2, so every staged Q/dO
    # fragment still feeds two n tiles -- LDS bytes per flop unchanged, unlike
    # WAVES8/W8 which doubled them), but
    #   * score + silu/ds chain: wave group g = w//4 does tile g of each mu2
    #     pair (m split), and publishes its e4m3 silu/ds to LDS;
    #   * K=128 out-gemm: group g owns d tiles [4g, 4g+4) of dk/dv for those
    #     n rows (d split), reading both tiles' silu/ds back from LDS.
    # So each wave carries only half the dk/dv accumulators (64 regs instead
    # of 128) and no wave holds a private partial: every dk/dv element is
    # produced by exactly the MFMA sequence the stack issues (same operands,
    # same pair order) -> dq/dk/dv bitwise vs the stack, no end reduction.
    # The out-gemm of pair k-1 runs in the same barrier interval as the
    # score/act of pair k (2 barriers per pair; tiles and exchange double
    # buffered by pair parity, ~140 KB LDS, 1 workgroup/CU).
    #   1: both groups issue out-gemm(k-1) then score/act(k).
    #   2: anti-phase: group 0 issues out-gemm(k-1) then score/act(k), group 1
    #      score/act(k) then out-gemm(k-1) (scalar branch on the wave group),
    #      so the two waves sharing a SIMD (w, w+4) overlap one's MFMA stream
    #      with the other's VALU chain instead of running both in lock-step.
    msplit: int = 0,
    # R15 iter19 ms1tune: msplit scheduling variants (comma tokens, default
    # "" = unchanged).  so: score MFMAs before the out-gemm in program order;
    # anti: per-slot scalar branch, group g runs out-gemm in slot g and
    # score/act in the other slot (MSPLIT=2 order without duplicating the
    # whole step); prio: group 0 waves s_setprio(1) for the whole kernel;
    # iglpN / noiglp: loop sched override; sgK: sched_group_barrier pattern
    # interleaving each out-gemm MFMA with K VALU ops.
    # mid: score MFMAs of pair k first, then act(n tile 0) + its cast and
    # exchange store, then the out-gemm of pair k-1, then act(n tile 1): the
    # out-gemm MFMAs overlap the wave's own VALU chain instead of preceding
    # it.  sb (needs mid): sched_barrier(0) before the out-gemm, so the
    # scheduler cannot hoist it back above the act.  wsl (needs mid): the dq
    # ws store of pair k-1 moves after its out-gemm (into the second region).
    # sgvK: in that region, (1 MFMA, K VALU) x16 sched_group_barrier.
    # Shipped mid,sb,wsl,sgv5: chosen by lab microbench (R15 iter19), confirmed
    # by in-model A/B (R15 iter24 job 61268; ship8 R1->R2).  ntm/oh/anti/so/sgK
    # timed or spilled worse (kept for the record).  Same values, MFMA operands
    # and per-accumulator order in every variant: bitwise vs msplit=1.
    ms_sched: str = "",
    # R15 iter24 ws track ($DLRM_HSTU_FP8_BWD_WS_AUX, default 0 = off): the
    # cache-policy aux field of the dq-split e4m3 ds workspace stores
    # (gfx940/950 cpol: bit0 sc0, bit1 nt, bit4 sc1).  The ws is written once
    # here and read once by hstu_dq_split, so a streaming / non-temporal store
    # may keep L2 for the Q/K/dO tiles.  Same bytes stored: bitwise.
    ws_aux: int = 0,
    # ws_c64 ($DLRM_HSTU_FP8_BWD_WS_LAYOUT=c64, default off): each of the two
    # ws store instructions writes one CONTIGUOUS 64 B half of every row
    # (lane l4 at byte 16*l4 + 64*q4) instead of 4 x 16 B at stride 32 B.
    # The row's n order in memory becomes 16 B chunk c_n at position
    # 4*(c_n % 2) + c_n // 2; hstu_dq_split(ws_c64=True) loads through the
    # inverse, so the dq kernel sees the same values: bitwise.
    ws_c64: bool = False,
    # ws_l128 ($DLRM_HSTU_FP8_BWD_WS_LAYOUT=l128, default off): natural ws
    # layout, but each store instruction writes 8 FULL 128 B rows: lanes
    # l16 >= 8 hand their chunk 2*l4 to lane l16 - 8 and take its chunk
    # 2*l4 + 1 (one v_mov_dpp row_shr/row_shl:8 per dword, bound_ctrl off),
    # so store 0 covers rows 0..7 and store 1 rows 8..15 of the wave.  Same
    # bytes at the same addresses: bitwise, dq kernel unchanged.
    ws_l128: bool = False,
    # ws_l128t ($DLRM_HSTU_FP8_BWD_WS_LAYOUT=l128t, default off; R15 iter25
    # wsla): the same full-128 B-row stores as ws_l128, but with NO lane
    # exchange.  The four tr8 reads of the ws tile get per-lane addresses such
    # that receiver half h=1 (lanes l16 >= 8) is served the rows n + 16 that
    # ws_l128 moves over with DPP: read (q, p) uses row t8_n + 16*(l16 % 2)
    # + 8*p and column 8*q of the wave's m window, so the result is already
    # the l128 store payload (store q = reads (q,0),(q,1) = rows 8q..8q+7).
    # Same bytes at the same addresses: bitwise, dq kernel unchanged; zero
    # extra VALU in the loop (the DPP movs/copies and their hazard nops go).
    ws_l128t: bool = False,
    # R15 M-E1 ($ no env; launcher argument dkdv_fp8_sink, default "" = OFF:
    # the unchanged kernel, kernel name, signature and LDS layout).
    # e8emit="only": the dk/dv epilogue writes the e5m2 copy of its BF16
    # dV/dK INSTEAD of the BF16 dk/dv stores, into a [Mp, 4*H*D] dz8 buffer
    # (cols H*D + h*D + d for dV, 3*H*D + ... for dK; the u|v|q|k layout
    # bwd_fp8.quantize_amax_colsum casts), with the delayed inv_scale:
    # bf16 -> f32 * inv -> clamp +-57344 -> v_cvt_scalef32_pk_bf8_f32
    # (scale 1.0), the cast's exact op sequence, plus amax partials
    # (max |bf16|, 32 per workgroup) and fp32 column-sum partials (one
    # [2*H*D] row slice per (b, n block)).
    e8emit: str = "",
):
    assert int(ws_c64) + int(ws_l128) + int(ws_l128t) <= 1
    assert ws_aux in (0, 1, 2, 3, 16, 17, 18, 19), ws_aux
    # `abl` is a comma-separated set of ablation switches.  Ablated kernels are
    # numerically WRONG on purpose; they exist only to attribute time to a term.
    abl_set = {s for s in abl.split(",") if s}
    ABL_ACT = "act" in abl_set  # drop silu / grad-coeff math
    ABL_MASK = "mask" in abl_set  # drop causal + bounds selects
    ABL_TR = "tr" in abl_set  # feed out-gemms from the row-major tile
    ABL_SCORE = "score" in abl_set  # drop the qk / dp gemms
    ABL_OUT = "out" in abl_set  # drop the dv / dk gemms
    ABL_PREFETCH = "nopf" in abl_set  # no global-load prefetch
    # Attribution only: collapse every LDS read of one operand class onto a
    # single address so CSE leaves exactly one ds_read.  The mfma stream, the
    # elementwise chain and the global traffic are untouched, so the time delta
    # is the cost of that class's LDS read bandwidth and nothing else.
    ABL_LDSA = "ldsa" in abl_set  # score-gemm A operands (q_rm / o_rm)
    ABL_LDSB = "ldsb" in abl_set  # out-gemm B operands (q_tr / o_tr)
    # Clean footprint-only variants of ldsa / ldsb.  `ldsa`/`ldsb` replace the
    # step index by a literal 0, so every read of the class becomes the SAME
    # SSA expression and CSE deletes all but one -- the delta then mixes read
    # INSTRUCTIONS with read BYTES and over-states the bandwidth term.
    # `ldsa2`/`ldsb2` instead multiply the step index by an OPAQUE ZERO (a
    # select on a runtime predicate LLVM cannot fold), so each read keeps a
    # distinct SSA address, every ds_read survives, the operand registers and
    # the mfma stream are untouched, and only the runtime FOOTPRINT collapses
    # onto one tile's worth of lines.  Lane-dependent terms are left alone, so
    # the bank pattern of a single read is identical to the real one.
    #   delta(ldsa2)  = LDS read bandwidth / L1-of-LDS locality of that class
    #   delta(ldsa) - delta(ldsa2) = the ds_read ISSUE cost of that class
    ABL_LDSA2 = "ldsa2" in abl_set
    ABL_LDSB2 = "ldsb2" in abl_set
    # Register-pressure probe for a FUSED backward.  Adds the fifth gemm
    # (dq = ds @ k), its BM x D f32 accumulator and its atomic flush to the
    # loop, with deliberately wrong operand orientations: the result is
    # garbage, but the mfma count, the accumulator size, the address
    # arithmetic and the atomic stream are the ones the real design needs.
    # DQC splits the accumulator into that many d-chunks, flushed in turn, to
    # trade peak register pressure against instruction-level parallelism.
    ABL_DQ = "dq" in abl_set
    # Attribution switches for the fused dq path.  Each collapses exactly one
    # term and leaves the rest of the instruction stream alone, so the time
    # delta is that term's cost.  All of them make dq numerically wrong.
    ABL_DQA = "dqa" in abl_set  # plain store instead of atomic (no RMW read)
    ABL_DQF = "dqf" in abl_set  # flush 1 of every 4 elements (quarter volume)
    ABL_DQR = "dqr" in abl_set  # collapse the ds transpose reads to one address
    ABL_DQW = "dqw" in abl_set  # collapse the ds writes to one address
    ABL_DQB = "dqb" in abl_set  # drop the ds-exchange barrier
    # Drop m0 from the atomic address, so every m iteration of every workgroup
    # read-modify-writes the SAME BM x D tile of dq_acc.  Instruction count,
    # atomic count and coalescing are all unchanged; only the address stream
    # collapses, so the traffic becomes cache-resident.  This is the exact
    # analogue of CK's level-2 dq ablation (`store_tile` without advancing the
    # window), which is what makes the two attributions comparable.
    ABL_DQZ = "dqz" in abl_set
    # MFMA-sensitivity probe for the fp8-MFMA track.  Issues every score-gemm
    # mfma TWICE into the same accumulator: +40% MFMA instructions, and nothing
    # else changes -- same LDS reads, same operands, same register pressure, no
    # dead code for the compiler to remove.  The time delta is the EXPOSED cost
    # of that MFMA issue, which is what a faster MFMA could actually buy back;
    # the analytic issue-time model in bench/mfma_fraction.py only bounds it
    # from above, because MFMA that hides under LDS latency is free to remove
    # and free to add.  Numerically wrong on purpose.
    ABL_MFMA2X = "mfma2x" in abl_set
    # Same probe, for the other two gemm classes, because the hi+lo error
    # compensation of H4 does NOT double the MFMA count uniformly -- it doubles
    # exactly the gemms whose inexact operand gets split, and the three classes
    # sit in different parts of the schedule (score gemms wait on the k/v LDS
    # reads, the out gemms on the tr16 transpose, the dq gemm on the ds
    # round-trip through LDS plus its barrier).  An MFMA that hides under one
    # of those stalls is free to add; one that does not is not.  Each switch
    # re-issues that class's mfma into the same accumulator, so operands,
    # register pressure, LDS traffic and the atomic stream are untouched and
    # the time delta is that class's EXPOSED marginal MFMA cost -- which is
    # exactly the quantity a 2x-MFMA scheme has to pay.  Numerically wrong.
    ABL_MFMA2XO = "mfma2xo" in abl_set  # out gemms (dv, dk)
    ABL_MFMA2XQ = "mfma2xq" in abl_set  # dq gemm
    # ---- TRACK X1 CEILING PROBE ----------------------------------------
    # The whole fp8-LDS-staging + ds_read_tr8_b64 escape exists to delete ONE
    # thing: the v_cvt_scalef32_pk_fp8_bf16 that narrow the k128 out-gemms' B
    # operands (dO^T, q^T) in the inner loop, NSTEP times per element instead
    # of once at write_tiles.  Rather than spend the build on the LDS rewrite
    # and then discover the cvts were not the cost, price them first: keep the
    # identical k128 kernel, keep the LDS reads and their data dependence, and
    # replace only the cvt with a register slice of the bf16 fragment that
    # already arrived.  Zero VALU, same loads, same mfma count, same operand
    # widths -- the ONLY delta is the 512 cvts.  Numerically wrong on purpose:
    # it reinterprets bf16 halfwords as fp8 bytes.  The resulting time is the
    # FLOOR the real escape could reach on the B-operand axis (the real thing
    # additionally halves the tr reads and may relieve the spill, so it can go
    # a little lower -- but if this probe cannot beat the baseline, nothing
    # built on top of it can either).
    ABL_BCVT0 = "bcvt0" in abl_set
    # Prices the e4m3->bf16 widening (_widen_raw8). See the branch there.
    ABL_WCVT0 = "wcvt0" in abl_set
    assert dim == 128, "kernel is specialised for D=128"
    assert BM == 64 and BN % 64 == 0
    D = dim
    H = heads
    HD = H * D
    # Elements between consecutive tokens of q/k/v (and dq/dk/dv).  A packed
    # [N, H, D] tensor has H*D, but production never builds one: the UVQK gemm
    # emits a single [N, 2*H*hidden + 2*H*attn] buffer and hands the backward
    # COLUMN SLICES of it, so one token's q sits `rs` elements from the next.
    # Passing that stride is what lets the kernel read the slice in place; with
    # rs unset the caller is promising a packed tensor and gets the old math.
    #
    # The column offset needs no parameter -- a torch slice view carries it in
    # its data_ptr, which is what _shifted_rsrc starts from.  dO is NOT a slice
    # of that buffer (it is the attention output's gradient, its own contiguous
    # tensor), so it keeps the packed stride and is addressed separately.
    RS_IN = rs_in or rs or HD
    RS_OUT = rs_out or rs or HD
    RS_Q = rs_q or RS_IN
    # Eight-wide input loads are 16 B for BF16 and 8 B for FP8. Output gradient
    # loads/stores remain BF16. Both production row strides satisfy the stricter
    # eight-element alignment.
    assert RS_IN % 8 == 0, (
        f"input row stride {RS_IN} must be a multiple of 8 elements"
    )
    assert RS_OUT % 8 == 0, (
        f"output row stride {RS_OUT} must be a multiple of 8 elements"
    )
    assert RS_Q % 8 == 0, (
        f"Q row stride {RS_Q} must be a multiple of 8 elements"
    )
    assert RS_IN >= HD, (
        f"input row stride {RS_IN} cannot be narrower than one H*D row"
    )
    assert RS_OUT >= HD, (
        f"output row stride {RS_OUT} cannot be narrower than one H*D row"
    )
    assert RS_Q >= HD, (
        f"Q row stride {RS_Q} cannot be narrower than one H*D row"
    )
    assert not q_bf16 or input_fp8
    # "k32w" is an attribution probe, not a candidate: k128's TWO b128 A-operand
    # reads feeding k32's FOUR 16x16x32 mfma.  k128 changes two things at once
    # (4 mfma -> 1, and 4 b64 reads -> 2 b128), and this holds the read side at
    # k128 while leaving the mfma side at k32, so the difference k32w - k128 is
    # the mfma-rate term alone and k32 - k32w is the read term alone.
    assert fp8_mfma in ("", "k32", "k32w", "k128"), fp8_mfma
    if fp8_mfma:
        # Both qk operands have to BE e4m3 in memory.  q_bf16 stages a genuinely
        # bf16 Q, so there is nothing to reinterpret and rounding it here would
        # cost precision the incumbent does not pay.
        assert input_fp8 and not q_bf16, (
            "fp8_mfma needs e4m3 q and k (input_fp8=True, q_bf16=False)"
        )
        assert not (ABL_SCORE or ABL_MFMA2X or ABL_LDSA or ABL_DQ), (
            "score-gemm ablations are defined against the bf16 qk path"
        )
        assert tr16, "fp8_mfma reuses the tr16 staging schedule"
    FP8_QK = fp8_mfma
    assert mu2 in ("", "bf16", "k128"), mu2
    MU2 = mu2
    assert mu2_fmt in ("e4m3", "e5m2"), mu2_fmt
    if MU2:
        assert BM == 64 and tr16, "mu2 unrolls the BM=64 tr16 body"
        assert not (ABL_OUT or ABL_TR or ABL_MFMA2XO), (
            "out-gemm ablations are defined against the K=32 path"
        )
        for s in (mu2_ssc, mu2_dsc, mu2_osc, mu2_qsc):
            assert cvt_scale_is_exact(s) or s == 1.0, (
                f"mu2 pre-scale {s} must be an exact power of two: it is "
                "folded into cvt_scalef32's E8M0 (exponent-only) scale field"
            )
    assert f8b in ("", "all"), f8b
    F8B = f8b == "all"
    assert not f8_sat or F8B, "f8_sat needs f8b=all"
    assert epilogue in ("legacy", "reuse", "reuse_pk"), (
        f"unknown epilogue {epilogue!r}")
    EPI_REUSE = epilogue in ("reuse", "reuse_pk")
    EPI_PK = epilogue == "reuse_pk"
    valu_set = {v for v in valu.split(",") if v}
    assert not (valu_set - {"undef", "rfl", "ilp", "nobr", "h16", "vf", "vf2",
                         "msat", "vf3", "vf3s", "vf3p", "vf4", "tsc"}), (
        f"unknown valu {valu!r}")
    VALU_UNDEF = "undef" in valu_set
    VALU_RFL = "rfl" in valu_set
    VALU_ILP = "ilp" in valu_set
    VALU_NOBR = "nobr" in valu_set
    # iter22 f16act (default OFF, declared numerics change, ilp path only):
    #   "vf"  f32 constant fold: t = qk*(sas*-log2e), silu = qk*(sig*inv*sas),
    #         gc*vis = fma(silu, vis-vis*sig, sig*inv*vis), ds = dp*(gc*vis):
    #         x and dv multiplies gone (8 pk ops per pair instead of 9).
    #   "h16" silu/ds tail in packed f16 in a 2^-6 scaled e4m3 domain:
    #         exp/rcp stay f32; x, sig, dp*(vis*inv*dsc/64) -> v_cvt_pk_f16_f32;
    #         u=x*sig, S=u*(inv*ssc/64), g=fma(u,1-sig,sig), D=dv*g in
    #         v_pk_*_f16; sat via v_pk_min/max_f16 at +-7 (=448/64);
    #         v_cvt_scalef32_pk_fp8_f16 with scale 2^-6 (dst = src*64).
    VALU_H16 = "h16" in valu_set
    VALU_VF = "vf" in valu_set
    #   "vf2" deeper fold (declared numerics change): sinv = sig*inv,
    #         silu' = qk*sinv = silu/sas, gc = fma(silu', sas-sas*sig, sinv),
    #         ds' = dp*gc = ds/s.  The launcher re-centres mu2_ssc by
    #         2^round(log2 sas) and mu2_dsc by 2^round(log2 s) and multiplies
    #         the dq-split scale by s; the dv epilogue multiplies by sas and
    #         the dk epilogue by s (uniform scalars).  7 pk ops per pair.
    VALU_VF2 = "vf2" in valu_set
    if VALU_VF2:
        VALU_VF = True
    # R15 iter43 a8 "vf3" (default OFF): 7-op act chain at the DEFAULT cast
    #   scales (pure f32 reassociation, no e4m3 grid shift):
    #   t = qk*(sas*-log2e); x' = qk*(alpha*s); e = exp2(t);
    #   d = fma(e, N/s, N/s); r = rcp(d) (= s*sig/N); silu = x'*r;
    #   om = fma(r, -N, s) (= s*(1-sig)); g = fma(silu, om, r) (= s*gc);
    #   ds = dp*g.  N = 1/inv_max.  Uniform constants c_t, alpha*s, N/s are
    #   v_readfirstlane'd to SGPRs; -N stays a VGPR splat ("vf3") or is also
    #   readfirstlane'd ("vf3s").
    VALU_VF3S = "vf3s" in valu_set
    # "vf3p": same math as vf3; the four uniform constants are packed as two
    # <2 x f32> pairs (c_t, alpha*s) and (N/s, -N) and each pair is
    # readfirstlane'd as ONE 64-bit uniform value, so v_pk ops can take an
    # SGPR pair with op_sel instead of four VGPR-pair splats.
    VALU_VF3P = "vf3p" in valu_set
    VALU_VF3 = ("vf3" in valu_set) or VALU_VF3S or VALU_VF3P
    assert not VALU_VF3 or (VALU_ILP and not (VALU_VF or VALU_H16)), (
        "vf3 needs valu ilp and excludes vf/vf2/h16")
    # R15 iter51 "vf4" (default OFF, DECLARED PRECISION CHANGE: the silu e4m3
    #   cast is re-rounded on a shifted grid, dS stays ulp-level vs vf3p).
    #   vf3p minus its x' multiply, 6 pk ops per element pair:
    #   t = qk*c_t; e = exp2(t); d = fma(e, N/s, N/s); r = rcp(d) (= s*sig/N);
    #   silu' = qk*r (= silu/(alpha*s)); om' = fma(r, -(alpha*s)*N, sas)
    #   (= sas*(1-sig)); g = fma(silu', om', r) (= vf3p's g up to rounding);
    #   ds = dp*g.  The uniform constants are packed (c_t, N/s) and
    #   (-(alpha*s)*N, sas), each pair readfirstlane'd as one <2 x f32>.
    #   silu' lacks alpha*s, so its e4m3 cast scale is re-centred by the pow2
    #   sc = 2^ka, ka = round(log2(alpha*s)) (vf2's recipe: the cast scale
    #   field is E8M0, pow2 only), and the dv epilogue (bf16 and e8)
    #   multiplies by 1/(mu2_ssc*sc*mu2_osc)*(alpha*s) instead of
    #   1/(mu2_ssc*mu2_osc) (one uniform factor, one multiply per element,
    #   as without vf4).  R15 iter52 landing form: sc is computed IN THE
    #   KERNEL from the runtime qkv_scale (_vf4_cast_scale: integer ops on
    #   the f32 alpha*s, once per workgroup, uniform), so mu2_ssc stays the
    #   compile-time base scale and one build serves every qkv_scale (the
    #   iter51 probe folded 2^ka into mu2_ssc in the launcher: one build per
    #   alpha*s binade).  Every factor is a power of two or alpha*s itself,
    #   so the cast operand and the undo factor are bitwise the probe's.
    VALU_VF4 = "vf4" in valu_set
    assert not VALU_VF4 or (VALU_VF3P and F8B and MU2 == "k128"
                            and not do_dyn), (
        "vf4 needs vf3p and the f8b=all k128 path (no do_dyn)")
    # vf4 needs msat (iter52): the silu' cast saturates in hardware, so the
    # runtime cast scale needs no runtime clamp limit (VALU_MSAT is defined
    # below; checked there).
    VF4_TIE = False
    if VALU_VF4:
        # log2(mu2_ssc) as an int: the runtime cast operand 2^-(E+ka) and the
        # undo factor stay normal f32 for |E| <= 50 and the launcher's guard
        # 2^-50 <= alpha*qkv_scale <= 2^50 (|ka| <= 50).
        VF4_ESSC = math.frexp(mu2_ssc)[1] - 1
        assert 2.0 ** VF4_ESSC == mu2_ssc and abs(VF4_ESSC) <= 50, (
            f"vf4 needs a power-of-two mu2_ssc within 2^-50..2^50, got "
            f"{mu2_ssc}")
        # alpha a half-integer power of two at f32 precision (the in-model
        # 128^-0.5): then alpha*s is an exact tie of round(log2()) for every
        # power-of-two s, the probe's host formula (double log2, Python
        # round) rounds it half-to-even, and _vf4_cast_scale does the same.
        VF4_TIE = (struct.unpack("<I", struct.pack("<f", alpha))[0]
                   & 0x7FFFFF) in (0x3504F3, 0x3504F4)
    assert not (VALU_H16 or VALU_VF) or VALU_ILP, "h16/vf need valu ilp"
    assert not (VALU_H16 and VALU_VF), "h16 and vf are exclusive"
    assert not VALU_H16 or (F8B and f8_sat), "h16 needs f8b + f8_sat"
    # msat (B iter22 satcvt): saturate the f8_sat e4m3 casts of silu/ds in
    # HARDWARE instead of the per-element v_min/v_med3.  With MODE.FP16_OVFL
    # (MODE bit 23) set, v_cvt_scalef32_pk_fp8_f32 returns +-448 instead of
    # NaN on overflow (incl. +-inf) -- exhaustive 2^32 HW proof (probes/
    # iter22/satcvt/hw/modesat.hip) bitwise vs v_min/v_med3 + cvt for every
    # non-NaN f32 at S=2^-3..2^21.  The bit is set once at kernel entry; the
    # kernel's only other narrow-format VALU op (v_cvt_pk_bf16_f32, dk/dv
    # epilogue) is NOT mode-invariant near f32 max (hw/modebf16.hip), so the
    # bit is cleared again before the epilogue.  NaN silu/ds: the clamp
    # mapped NaN to +448 (0x7e); msat propagates it as fp8 NaN (0xff).
    VALU_MSAT = "msat" in valu_set
    assert not VALU_MSAT or f8_sat, "valu msat needs f8_sat"
    assert not VALU_VF4 or VALU_MSAT, "vf4 needs valu msat"
    assert not (VALU_MSAT and VALU_H16), "msat is untested with h16"
    SAT_HI = False if VALU_MSAT else ("hi" if f8_sat else False)
    SAT_BOTH = False if VALU_MSAT else ("both" if f8_sat else False)
    assert not VALU_ILP or (EPI_PK and not ABL_ACT), "valu ilp needs reuse_pk"
    if F8B:
        assert (fp8_mfma == "k128" and mu2 == "k128" and mu2_fmt == "e4m3"
                and fused and dq16 == "bf16p" and input_fp8 and not q_bf16
                and not rev and not scale_in_cvt), (
            "f8b=all needs k128 + mu2=k128(e4m3) + fused bf16p + e4m3 inputs")
        assert not (abl_set - {"bcvt0", "wcvt0"}), "f8b: no ablations"
        # dO / q reach the out-gemms as stored e4m3 bytes, with no re-cast to
        # carry a pre-scale.
        assert mu2_osc == 1.0 and mu2_qsc == 1.0
        assert ds8_pad % 8 == 0
    DQSPLIT = bool(dqsplit)
    MSPLIT = int(msplit)
    assert MSPLIT in (0, 1, 2), msplit
    # R15 iter74 f2 "tsc" (LAB, default OFF; vf4 + msat on the msplit=1 body
    #   only): fold vf4's t = qk*c_t into the score MFMA.  If qkv_scale is
    #   snapped so that the kernel's own f32 constant
    #   c_t = f32(f32(f32(alpha)*f32(s*s))*f32(-log2e)) is EXACTLY -2^k
    #   (tsc_exponent), then t = -(2^k*qk) exactly, and:
    #   * the 8 S = Q K^T MFMAs per pair become v_mfma_scale with E8M0 A scale
    #     127+k (B scale 127): tau = 2^k*qk (HW assumption: exactly 2^k times
    #     the unscaled accumulator); the dP MFMAs stay unscaled;
    #   * the t multiply is gone: exp2 reads -tau (VOP3 neg source modifier);
    #   * silu'' = tau*r = 2^k*silu' is cast with the operand 2^k/(mu2_ssc*sc)
    #     (cvt divides by it), om'' = fma(r, 2^-k*(-(alpha*s)*N), 2^-k*sas) =
    #     2^-k*om', so g = fma(silu'', om'', r) and ds = dp*g are vf4's: every
    #     factor is a power of two, the silu/dS e4m3 bytes are vf4's bytes at
    #     the same (snapped) qkv_scale.  ds cast and dk/dv/E1 epilogues unchanged.
    #   k is not a compile parameter: c_t's f32 exponent field IS the E8M0
    #   byte 127+k, read once per workgroup from the runtime c_t, so one build
    #   serves every snapped qkv_scale; call() refuses a qkv_scale whose c_t
    #   is not exactly -2^k with k in [TSC_K_MIN, TSC_K_MAX].
    VALU_TSC = "tsc" in valu_set
    assert not VALU_TSC or (VALU_VF4 and VALU_MSAT and MSPLIT == 1
                            and FP8_QK == "k128"), (
        "valu tsc needs vf4 + msat on the msplit=1 k128 body")
    ms_tok = [t for t in ms_sched.split(",") if t]
    assert not ms_tok or MSPLIT, "ms_sched needs msplit"
    MS_SO = "so" in ms_tok
    MS_ANTI = "anti" in ms_tok
    MS_PRIO = "prio" in ms_tok
    MS_IGLP = next((t for t in ms_tok if t.startswith("iglp")
                    or t == "noiglp"), None)
    MS_SG = next((int(t[2:]) for t in ms_tok if t.startswith("sg")
                  and not t.startswith("sgv")), None)
    MS_SGV = next((int(t[3:]) for t in ms_tok if t.startswith("sgv")), None)
    assert not (MS_ANTI and MSPLIT != 1)
    MS_MID = "mid" in ms_tok
    MS_SB = "sb" in ms_tok
    # R15 iter25 pmc / iter27 nk: nk = skip the K byte staging (K -> kt8_lds,
    # its 4 HBM loads, waits and barrier) in the prologue.  It only feeds
    # kt_frag, the in-kernel dq B operand, which dqsplit never reads; the
    # region is then overwritten by the ds/silu exchange.
    MS_NK = "nk" in ms_tok
    # R15 iter27 nk track: wh = issue the dqsplit workspace header load
    # (prefix[batch], the per-batch tile base) right after the seq offset
    # loads, before the n-block guard, so its HBM round trip overlaps the
    # seq-offset one instead of serialising in front of the prologue tile
    # loads.  Same value (bounded at prefix[nbatch]; the guard rejects any
    # padded batch slot before the value is used).
    MS_WH = "wh" in ms_tok
    # mc (needs valu msat): clear MODE.FP16_OVFL at the top of the drain
    # instead of after it, fenced by sched_barrier(0) on both sides.  The
    # drain (ws store + last out-gemm) has no fp8 cvt, so every saturating
    # cast still runs with the bit set, and the drain MFMAs and the dk/dv
    # epilogue share one scheduling region (with valu nobr they interleave).
    # Without it, valu nobr lets the scheduler hoist epilogue bf16 cvts
    # above the late clear (inline asm is not a scheduling fence).
    MS_MC = "mc" in ms_tok
    MS_NTM = "ntm" in ms_tok
    MS_WSL = "wsl" in ms_tok  # mid: dq ws store of pair k-1 after its out-gemm
    MS_OH = "oh" in ms_tok    # mid,sb: out-gemm d tiles 0..1 before the split
    MS_IA = "ia" in ms_tok    # mid,sb: iglp_opt(2) in the score/act-tile-0 region
    # R15 iter24 ws track (default off): wse = mid: the ws store of pair k-1
    # after act n tile 1 (end of the step); wss = mid,wsl: its tr8 LDS
    # readback at the step start, the global stores where wsl puts them.
    MS_WSE = "wse" in ms_tok
    MS_WSS = "wss" in ms_tok
    # wsx = mid,wsl: tr8 readback where wsl puts the store, global stores
    # after act n tile 1 (end of the step).
    MS_WSX = "wsx" in ms_tok
    assert not (MS_WSX and (MS_WSS or not (MS_MID and MS_WSL)))
    assert not (MS_WSE and (MS_WSL or not MS_MID))
    assert not (MS_WSS and not (MS_MID and MS_WSL))
    # R15 iter24 bar (default off): work moved ahead of the top-of-pair
    # barrier B1 (after ms_write_tile, before gpu.barrier()), so it overlaps
    # the barrier drain instead of following it.  Only reads of buffers that
    # are complete since the previous closing barrier and are not written
    # until the next one, or global loads -- same values, same addresses:
    #   wsb: the dq ws store of pair k-1 (reads exchange parity k-1).
    #   lb:  the global prefetch loads of pair k+1.
    #   xe:  the out-gemm(k-1) silu/ds exchange readback (a_dv/a_dk).
    MS_WSB = "wsb" in ms_tok
    MS_LB = "lb" in ms_tok
    MS_XE = "xe" in ms_tok
    # The closing barrier B2 of pair k only orders out-gemm(k-1)'s reads of
    # tile parity k-1 before W(k+1) overwrites it (the exchange RAW/WAR are
    # already ordered by B1 of pair k+1).  b2e (needs mid): B2 moves to right
    # after out-gemm(k-1) (+ its ws store), mid-pair; act(n tile 1) runs after
    # it, and one barrier follows the loop for the drain.  b2ew: as b2e and
    # W(k+1) is written at the end of pair k (after act tile 1) instead of at
    # the top of pair k+1.  la (needs mid): the k+1 prefetch loads are issued
    # after the score-gemm A-operand reads instead of before them.
    MS_B2EW = "b2ew" in ms_tok
    MS_B2E = "b2e" in ms_tok or MS_B2EW
    # lo: those loads after out-gemm(k-1) (+ its ws store); le: after the
    # act of n tile 1; ls: tile A's loads at the la point, tile B's at lo.
    # la2: between the score A-operand LDS reads and the score MFMAs; la4:
    # as la2 with a sched_barrier(0) after the loads.
    _LP = ("la", "lo", "le", "ls", "la2", "la4")
    MS_LPOS = next((t for t in ms_tok if t in _LP), None)
    MS_LA = MS_LPOS is not None
    assert sum(t in _LP for t in ms_tok) <= 1
    assert not ((MS_B2E or MS_LA) and not MS_MID)
    assert not (MS_LA and MS_LB)
    assert not ((MS_WSL or MS_IA) and not MS_MID)
    assert not (MS_SB and not MS_MID)
    assert not (MS_OH and not (MS_MID and MS_SB and not MS_NTM))
    assert not ((MS_WSB or MS_B2E) and (MS_WSE or MS_WSS or MS_WSX))
    # R15 iter25 addr track (default off; loop address arithmetic only,
    # same values to the same bytes: bitwise):
    #   xt: the silu exchange region holds each lane's 4 m-subtile dwords
    #       contiguously (dword 4*l4 + ms of row n) instead of the (n, m)
    #       byte layout, so its store / out-gemm read back are one
    #       ds_write_b128 / ds_read_b128 per n tile (no read2/write2 8-bit
    #       offset bases).  Nothing else reads the silu region.  (iter19
    #       valu4 xt, re-tried on the 242-reg la body.)
    #   rs2: the loop's q / dO / e8 prefetch loads use buffer descriptors
    #       whose base is advanced by the tile row m (SALU) and whose
    #       num_records shrinks by the same bytes (clamped at 0), so every
    #       per-lane voffset is loop invariant: no per-iteration v_add.  The
    #       range check sees the same distance to the sequence end.  With
    #       "la" there is ONE descriptor per tensor per pair (load_pair_rs2);
    #       without "la" each tile gets its own (load_tiles(.., rs=True)).
    #   lc: the m loop runs on an i32 induction variable (scalar compare)
    #       instead of index/i64 (v_cmp_lt_u64 on VALU).
    MS_XT = "xt" in ms_tok
    #   ek: the out-gemm K-block scale select (l4 < 2 ? e8 A : e8 B) of the
    #       NEXT pair is formed once where its bytes arrive and carried as
    #       one value instead of two.
    MS_RS2 = "rs2" in ms_tok
    MS_RS = MS_RS2
    MS_EK = "ek" in ms_tok
    MS_LC = "lc" in ms_tok
    # R15 iter34 energy track (default off; every token keeps dq/dk/dv
    # bit-identical, see each):
    #   dz:  drain store drop.  The ws store of a padding tile (last pair,
    #        tile 1 starting at/after seq_len) is pushed out of the descriptor
    #        range (dropped by the buffer unit): dq never reads those rows.
    #        The +2^31 push needs nb*(nb+1)*8 KiB < 2^31, i.e. nb <= 511
    #        (N <= 65408; the launcher asserts it when dz/dzr is armed).
    #   dzr: dz per ws ROW (every last-pair row >= seq_len is dropped, not
    #        only whole padding tiles; l128t ws layout).  Store drop only.
    #   Neither token skips any MFMA or VALU work: every score/act/out-gemm
    #   instruction still runs on every tile; only the store of padding rows
    #   that dq never reads is dropped.
    MS_DZR = "dzr" in ms_tok
    MS_DZ = "dz" in ms_tok or MS_DZR
    #   bst: score-MFMA program order B-stationary, K0 x4, V0 x4, K1 x4,
    #        V1 x4 (the loop-invariant K/V operand repeats on srcB).  Bitwise:
    #        one C=0 MFMA per accumulator, so the order is free.
    MS_BST = "bst" in ms_tok
    # R15 iter36 obb (default off; bitwise: out-gemm(k-1) B operands = the
    # tr8 reads of the dO^T / q^T tiles of parity pprev, stable from B1(k-1)
    # to W(k+1); only their issue point moves, every accumulator still gets
    # ONE mfma per pair with the same operands in the same (j, nt) order):
    #   ob0: B(0) of out-gemm(k-1) emitted in the n-tile-0 hook before the
    #        tile-0 cast + exchange store, i.e. ahead of the sb fence;
    #   obj: in the out-gemm, B(j+1) emitted before the MFMAs of j (one-j
    #        lookahead).  Measured together (ob0,obj): 256 VGPR, 0 spill.
    MS_OBJ = "obj" in ms_tok
    MS_OB0 = "ob0" in ms_tok
    assert not ((MS_OBJ or MS_OB0)
                and not (MSPLIT == 1 and MS_MID and MS_SB and not MS_NTM
                         and not MS_OH and not MS_XE)), (
        "ob0/obj need msplit=1 mid,sb (no ntm/oh/xe)")
    assert not (MS_DZ and not (MSPLIT == 1 and do_blk)), (
        "dz/dzr need msplit=1 + do_blk")
    # R15 iter73 gsplit (LAB, default off: without these tokens no code path
    # below changes).  gsp = G-split out-gemm: wave group 0 owns dV for ALL
    # NSTEP d tiles (2*NSTEP accumulators of one tensor), group 1 owns dK;
    # each group reads back only its own tensor's A operand (silu or ds, both
    # tiles) from the exchange and only its own B operand (dO^T or q^T) for
    # every d tile.  Every dk/dv accumulator receives the shipped MFMA
    # sequence (same pair order, same A/B bytes, same E8M0 scale operands);
    # only the issuing wave changes.  The exchange stores, the ws store and
    # the score/act half are unchanged.  The A readback is uniform: both
    # groups issue the same reads at a group-selected (SGPR) region base,
    # which needs the silu region in ds's plain byte(n, m) layout (xt's b128
    # silu store/readback is dropped under gsp).  gsb (needs gsp): instead a
    # uniform scalar BRANCH on the group around the A readback (silu keeps
    # xt).  gspr (needs gsp,gsb): the own-tile half of A from the lane's own
    # cast registers (carried across the pair) instead of the exchange
    # readback.  gsj2 (needs gsp,obj): B lookahead of two d tiles, not one.
    MS_GSP = "gsp" in ms_tok
    MS_GSPR = "gspr" in ms_tok
    MS_GSJ2 = "gsj2" in ms_tok
    MS_GSB = "gsb" in ms_tok
    # xtd (LAB, independent of gsp; needs msplit=1, xt, ws_l128t): the ds
    # exchange region also takes xt's per-lane contiguous dword layout
    # (dword 4*l4 + ms of row n), so the ds store and its out-gemm readback
    # are one b128 per (tile, n tile).  The ws tr8 reads keep their exact
    # addresses: in the xt layout the 8-byte run at column 16*wave + 8*q of
    # row n holds m = 16*(2q + b//4) + 4*wave + b%4 (b = byte), so each lane
    # still receives 16 consecutive n of ONE m; only the ws store row moves
    # to that m (dqsplit_ws_write).  Same bytes at the same ws addresses.
    MS_XTD = "xtd" in ms_tok
    assert not (MS_XTD and not (MSPLIT == 1 and MS_XT and ws_l128t)), (
        "xtd needs msplit=1, xt and the l128t ws layout")
    # gssN (needs gsp,sgvK): the post-fence sched_group_barrier pattern of
    # sgvK with N DS reads pulled in after every out-gemm MFMA, after an
    # initial group of 12 (the A readback + the first B), so the B tr8 reads
    # issue early instead of right before their MFMA.  Schedule only.
    MS_GSS = next((int(t[3:]) for t in ms_tok if t.startswith("gss")), None)
    MS_GSU = MS_GSP and not MS_GSB  # uniform readback, silu plain layout
    # osn (LAB, R15 iter78, needs gsp): snake EMISSION ORDER of the gsp
    # out-gemm -- odd d tiles jj issue nt = 1 then 0, so consecutive
    # out-gemm MFMAs alternate a shared B (same jj) and a shared A (same nt):
    # (A0,B0) (A1,B0) (A1,B1) (A0,B1) (A0,B2) ...  Each accumulator still
    # gets exactly one MFMA per pair with the same (A, B, scales, C): only
    # the order of independent MFMAs changes, so outputs are bitwise.
    # osnb (needs osn): additionally pin that MFMA order against the
    # scheduler with sched_barrier(0x7F6) between consecutive out-gemm
    # MFMAs (VALU/SALU/TRANS/VMEM/DS may cross, MFMA may not).  Schedule only.
    MS_OSN = "osn" in ms_tok
    MS_OSNB = "osnb" in ms_tok
    assert not (MS_OSN and not MS_GSP), "osn needs gsp"
    assert not (MS_OSNB and not MS_OSN), "osnb needs osn"
    assert not (MS_GSP and not (MSPLIT == 1 and MS_MID and MS_SB
                                and not (MS_NTM or MS_OH or MS_XE or MS_SO
                                         or MS_ANTI))), (
        "gsp needs msplit=1 mid,sb (no ntm/oh/xe/so/anti)")
    assert not ((MS_GSPR or MS_GSJ2 or MS_GSB) and not MS_GSP), (
        "gspr/gsj2/gsb need gsp")
    assert not (MS_GSJ2 and not MS_OBJ), "gsj2 needs obj"
    assert not (MS_GSPR and not MS_GSB), "gspr needs gsb (branch readback)"
    assert MS_GSS is None or (MS_GSP and MS_SGV is not None), "gssN needs gsp,sgvK"
    assert set(ms_tok) <= {"mc", "nk", "wh", "so", "anti", "prio", "mid", "sb", "ntm", "wsl", "ia", "oh", "wsb", "lb", "xe", "b2e", "b2ew", "la", "lo", "le", "ls", "la2", "la4", "wse", "wss", "wsx", "xt", "rs2", "ek", "lc", "dz", "dzr", "bst", "ob0", "obj", "gsp", "gspr", "gsj2", "gsb", "xtd", "osn", "osnb", MS_IGLP,
                           None if MS_SG is None else f"sg{MS_SG}",
                           None if MS_SGV is None else f"sgv{MS_SGV}",
                           None if MS_GSS is None else f"gss{MS_GSS}"}, ms_tok
    if DQSPLIT:
        assert F8B and do_blk and BN == 128 and BM == 64, (
            "dqsplit needs the C8 path (f8b=all, do_blk) at BN=128, BM=64")
    # e8emit = "" (off) | "only" (the e5m2 dV/dK epilogue, no BF16 dk/dv).
    E8 = e8emit
    assert E8 in ("", "only"), e8emit
    if E8:
        # The emission epilogue is written for the m0e1/ob0j msplit body: the
        # dead parity-(1-plast) exchange block hosts the column-sum exchange
        # (needs the closing barrier of every pair, i.e. no b2e/b2ew), the
        # MODE.FP16_OVFL bit is clear before the drain (mc), and the dk/dv
        # descriptors stop at seq_len (nobr).
        assert (MSPLIT == 1 and DQSPLIT and VALU_NOBR and VALU_MSAT
                and "mc" in ms_tok and "b2e" not in ms_tok
                and "b2ew" not in ms_tok and BN == 128 and not do_dyn), (
            "e8emit needs msplit=1 + dqsplit + valu nobr,msat + ms mc "
            "(no b2e/b2ew) at BN=128")
    PIPE = int(pipe)
    assert PIPE in (0, 1, 2, 3, 4, 5, 6), pipe
    if MSPLIT:
        assert (DQSPLIT and MU2 == "k128" and fused and not rev
                and not abl_set and not PIPE and BN == 128 and BM == 64), (
            "msplit needs dqsplit + mu2=k128 at BN=128/BM=64, ascending, "
            "no ablations, pipe=0 (it replaces the pair body)")
    if PIPE:
        assert DQSPLIT and MU2 == "k128" and fused and not rev and not abl_set, (
            "pipe needs dqsplit + mu2=k128, ascending, no ablations")
    assert not do_dyn or F8B, "do_dyn needs f8b=all"
    assert not do_blk or (F8B and not do_dyn), (
        "do_blk needs f8b=all and excludes do_dyn")
    # Per-group dO scales put ~1e-6 of ds above 448 on real tensors, and the
    # unclamped cvt_scalef32_pk_fp8_f32 returns NaN above 464 (probe P1).
    # The production C8 path (hstu_flydsl_mha) always passes both.
    assert not do_blk or f8_sat, (
        "do_blk needs f8_sat: without the clamp the rare ds > 448 becomes a "
        "NaN gradient")
    # The K-block <-> token-group identity below needs m tiles that start on a
    # multiple of 64 seq-relative tokens (m0 = n0 + 64*it, n0 = BN*n_blk) and
    # pairs (pend, cur) = (m0 - 64, m0): BM=64, BN a multiple of 128.
    assert not do_blk or (BM == 64 and BN % 128 == 0)
    DO_BYTES = 1 if F8B else 2
    INPUT_BYTES = 1 if input_fp8 else 2
    Q_BYTES = 2 if q_bf16 else INPUT_BYTES
    # WAVES = n-owning waves (each owns 16*NT rows of the BN block).  msplit
    # launches two waves per n-owner (WAVES_HW = 8); everything indexed by the
    # n-owner keeps using WAVES.
    WAVES = 4
    WAVES_HW = 8 if MSPLIT else WAVES
    THREADS = WAVES_HW * 64
    # Q/dO rows staged per thread (BM rows x 16 threads per row): 4, or 2 for
    # msplit's 512 threads.
    LROWS = BM * 16 // THREADS
    E8S = slice(2 * LROWS, 2 * LROWS + 2)  # load_tiles' two e8 bytes
    NSTEP = D // 16  # 8 d-tiles of the dk/dv accumulators
    MSTEP = BM // 16  # 4 m-tiles
    KSTEP = D // 32  # 4 k-tiles of the score gemms (double-rate 16x16x32)
    # n-tiles owned by one wave.  Every LDS read in the body is indexed by (m, d)
    # or (m, k) and never by n, so widening BN reuses the same staged Q/dO tile
    # against NT times as much math: LDS bytes per flop are 5/BN.  The cost is
    # registers -- accumulators, K/V fragments and score tiles all scale with NT.
    NT = BN // (WAVES * 16)
    NPEEL = BN // BM  # leading m iterations whose tile overlaps the n range
    # ---- fused dq -------------------------------------------------------
    # dq[m][d] = sum_n ds[m][n] k[n][d] contracts over n, so a wave that owns
    # only 16*NT of the block's n rows can produce nothing but a partial, and
    # four partials per workgroup would mean four atomic streams into dq_acc --
    # 4x the global read-modify-write traffic, which is the one thing this
    # design cannot afford.  So dq partitions the waves by D instead: wave w
    # takes d in [DQD*16*w, +DQD*16) and contracts over the FULL BN.  That
    # needs every wave to see every wave's ds, i.e. an exchange through LDS.
    # The exchange is not overhead: ds arrives from the mfma accumulator with n
    # in the lane index and m in the element index, and dq needs the opposite,
    # so the tile has to be transposed anyway.  Writing it in the tr16 layout
    # and reading it back with ds_read_b64_tr_b16 does the exchange and the
    # transpose in one round trip.
    DQD = D // 16 // WAVES  # 16-wide d tiles per wave
    DQK = BN // 32  # 32-wide n slices, one double-rate mfma each
    assert not fused or (D % (16 * WAVES) == 0 and BN % 32 == 0)
    XKIND = {"xcd": 1, "xcdn": 2}.get(gorder, 0)
    # dq_acc element type.  The fused kernel is bandwidth-bound on dq_acc's
    # read-modify-write traffic -- 61 GB each way at BN=128 -- so halving the
    # element halves the dominant cost.  gfx950 has no packed f32 atomic add,
    # but it has buffer_atomic_pk_add_bf16 and _f16, which add two 16-bit
    # values in one dword.  Reaching them needs the accumulator TRANSPOSED so
    # that a lane's four elements are four consecutive d rather than four m;
    # that is one mfma operand swap, see the flush below.
    # "bf16"/"f16" transpose the accumulator so a lane's own elements pair up.
    # That is simple but it moves m onto the lane axis, and m has stride HD in
    # memory, so one atomic then touches 16 cache lines instead of 4 -- measured
    # 99.4 ms against 59.0.  "bf16c" keeps the coalesced C[m][d] orientation and
    # gets the pair from the NEIGHBOURING LANE with a DPP quad swap instead;
    # the odd lanes then have nothing to contribute and are dropped by pointing
    # them past num_records.
    #
    # Both of those miss the point, because the atomic unit is limited per 64 B
    # LINE TOUCHED, not per byte or per element (atomic_roof.hip: f32,
    # pk_add_f16, pk_add_bf16 and add_f64 all land on 2.0e10 line-requests/s).
    # bf16c keeps 16 lanes' worth of addresses but only 8 live lanes, so each
    # request covers half a line and the request count is unchanged -- measured
    # 49.8 ms against 49.6, exactly as that model predicts.
    #
    # "bf16p" is the one that pays: it interleaves d in the staged K tile so
    # that a lane's TWO accumulators hold d and d+1, which lets all 16 lanes
    # pack a pair and span a whole 64 B line in one request.  Same bytes as
    # bf16c, half the line-requests.
    assert dq16 in ("", "bf16", "f16", "bf16c", "bf16p")
    # bf16p needs exactly two d tiles per wave to pair up.
    assert dq16 != "bf16p" or not fused or DQD == 2
    DQ_BYTES = 4 if not dq16 else 2
    # Resolved lazily: MLIR types need a live Context, which only exists once
    # the kernel body is being emitted.
    def dq_ty():
        return {"bf16": T.bf16, "bf16c": T.bf16, "bf16p": T.bf16,
                "f16": T.f16}.get(dq16, T.f32)

    def emit_sched():
        """Tell the machine scheduler how to pipeline the loop body.

        At BN=128 the kernel runs one wave per SIMD, so nothing else covers LDS
        latency: left alone, LLVM emits ds_read / s_waitcnt lgkmcnt(0) / v_mfma
        in lockstep and every mfma pays a full LDS round trip.  The hand-written
        assembly issues a whole tile's reads and then waits with a non-zero
        lgkmcnt.  `sched` picks how to ask for the same shape.
        """
        if not sched:
            return
        if sched.startswith("iglp"):
            rocdl.iglp_opt(int(sched[4:]))
            return
        if sched.startswith("sgb"):
            # sgb<reads>x<mfmas>: repeat a (DS_READ n, MFMA m) group until the
            # body's 64 reads and 128 mfmas are consumed.
            r, m = (int(x) for x in sched[3:].split("x"))
            for _ in range(min(64 // r, 128 // m)):
                rocdl.sched_dsrd(r)
                rocdl.sched_mfma(m)
            return
        raise ValueError(f"unknown sched {sched!r}")

    def ms_emit_sched():
        """R15 iter19 ms1tune loop schedule for the msplit body."""
        if MS_GSS is not None:
            # gss (gsp LAB): A readback + first B, then per out MFMA: 1 MFMA,
            # MS_GSS DS reads, MS_SGV VALU.
            rocdl.sched_group_barrier(0x100, 12, 0)
            for _ in range(16):
                rocdl.sched_group_barrier(0x8, 1, 0)
                rocdl.sched_group_barrier(0x100, MS_GSS, 0)
                rocdl.sched_group_barrier(0x2, MS_SGV, 0)
            return
        if MS_SGV is not None:
            # out MFMAs spread over the region's VALU stream, reads free.
            for _ in range(16):
                rocdl.sched_group_barrier(0x8, 1, 0)
                rocdl.sched_group_barrier(0x2, MS_SGV, 0)
            return
        if MS_SG is not None and MS_MID:
            # out-gemm region (after act n tile 0): each out MFMA with 3
            # LDS reads and MS_SG VALU ops of the n tile 1 act chain.
            rocdl.sched_group_barrier(0x100, 8, 0)
            for _ in range(16):
                rocdl.sched_group_barrier(0x8, 1, 0)
                rocdl.sched_group_barrier(0x100, 3, 0)
                rocdl.sched_group_barrier(0x2, MS_SG, 0)
            return
        if MS_SG is not None:
            # score A reads, score MFMAs (light interleave), then each
            # out-gemm MFMA with 3 LDS reads and MS_SG VALU ops.
            rocdl.sched_group_barrier(0x100, 16, 0)
            for _ in range(16):
                rocdl.sched_group_barrier(0x8, 1, 0)
                rocdl.sched_group_barrier(0x2, 2, 0)
            rocdl.sched_group_barrier(0x100, 8, 0)
            for _ in range(16):
                rocdl.sched_group_barrier(0x8, 1, 0)
                rocdl.sched_group_barrier(0x100, 3, 0)
                rocdl.sched_group_barrier(0x2, MS_SG, 0)
            return
        if MS_IGLP == "noiglp":
            return
        if MS_IGLP:
            rocdl.iglp_opt(int(MS_IGLP[4:]))
            return
        emit_sched()

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(f"hstu_dkdv_smem_{abl.replace(',', '_')}_{lds_pad}"
                         f"_{SQ}_{ST}_{rm_align}_{BN}_{int(tr16)}"
                         f"_{int(fused)}_{dq16}_{gorder}{NXCD}"
                         f"_{int(rev)}_{dq_aux}_{RS_IN}_{RS_OUT}"
                         f"_{int(input_fp8)}_{int(scale_in_cvt)}"
                         f"_{int(q_bf16)}_{RS_Q}"
                         f"_{fp8_mfma or 'bf16'}_{q8_pad}"
                         f"_mu{mu2 or 'off'}{mu2_fmt}_{mu2_ssc}_{mu2_dsc}_{mu2_osc}_{mu2_qsc}"
                         f"_f8b{f8b or 'off'}{ds8_pad}{'_dd' if do_dyn else ''}{'_db' if do_blk else ''}{'_sat' if f8_sat else ''}"
                         f"_epi{epilogue}"
                         + ("_dqs" if dqsplit else "")
                         + (f"_pipe{PIPE}" if PIPE else "")
                         + ("_valu" + "".join(sorted(valu_set))
                            if valu_set else "")
                         + (f"_ms{MSPLIT}" if MSPLIT else "")
                         + ("_mss" + "_".join(ms_tok) if ms_tok else "")
                         + (f"_wsa{ws_aux}" if ws_aux else "")
                         + ("_wsc64" if ws_c64 else "")
                         + ("_wsl128" if ws_l128 else "")
                         + ("_wsl128t" if ws_l128t else "")
                         + (("_e8" + E8) if E8 else "")),
    )
    # With tr16 there is one copy of each tile, in the layout gfx950's
    # ds_read_b64_tr_b16 wants:  halfword(m, d) = ((d/16)*BM + m)*16 + d%16.
    # A run of 16 halfwords is one m at one 16-wide d block, so 8 of them are
    # still contiguous for the natural read, and 4 consecutive m at one d are
    # exactly the transpose the instruction gathers.  Without tr16 the
    # transposed operand needs a second, hand-packed copy.
    RM = D if tr16 else SQ
    # Halfwords per Q/dO tile.  With tr16 the tile is (D/16) d_blocks each of
    # BM*16 halfwords, plus tr_pad halfwords of bank-conflict padding per block.
    DBS = BM * 16 + tr_pad  # d_block stride in halfwords (tr16)
    RM_HW = (D // 16) * DBS if tr16 else BM * RM
    if F8B:
        RM_HW = 8  # no bf16 tiles; keep a token allocation so the views exist
    off_q_rm = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_q_rm + RM_HW * 2
    off_o_rm = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_o_rm + RM_HW * 2
    # ---- fp8 Q tile ------------------------------------------------------
    # byte(m, d) = (d/32)*Q8BS + m*32 + (d%32): the byte analogue of tr16 but
    # with 32-byte d_blocks instead of 16.  32 is what makes BOTH qk modes
    # conflict-free and contiguous:
    #   k32  wants 8 bytes at d = 8*l4 + 32*ks, i.e. block ks, offset 8*l4
    #        inside the row -- 64 lanes then start at banks
    #        (8*l16 + 2*l4) mod 32, which covers all 32 banks exactly 4 deep,
    #        the floor for a 512 B wavefront read.
    #   k128 wants all 32 bytes of block l4 at one m -- one contiguous 32 B
    #        run, read as two b128.
    # A 16-byte d_block (the direct byte copy of tr16) would leave half the
    # banks idle on the k32 read, which is why the block is widened here.
    Q8BS = BM * 32 + q8_pad  # d_block stride in BYTES
    assert Q8BS % 16 == 0
    off_q8 = 0
    if FP8_QK:
        off_q8 = allocator._align(allocator.ptr, 16)
        # msplit: one block of 8 tiles [parity 2][tile 2][q, dO] at off_q8.
        allocator.ptr = off_q8 + (D // 32) * Q8BS * (8 if MSPLIT else 1)
    off_o8 = 0
    if F8B and MSPLIT:
        off_o8 = off_q8 + (D // 32) * Q8BS  # inside the msplit tile block
    elif F8B:
        off_o8 = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o8 + (D // 32) * Q8BS
    # pipe: second fp8 Q / dO tile pair (tile B of an mu2 pair).
    off_q8b = off_o8b = 0
    if PIPE:
        off_q8b = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_q8b + (D // 32) * Q8BS
        off_o8b = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o8b + (D // 32) * Q8BS
    # e4m3 ds exchange: byte(n, m) = n*DS8R + m.
    DS8R = BM + ds8_pad
    # msplit: the K-staging region afterwards holds 8 such [BN][DS8R] e4m3
    # exchange regions, [parity 2][ds, silu][tile 2].
    KT8_BYTES = max(BN * D, (8 if MSPLIT else 1) * BN * DS8R)
    assert not MSPLIT or DS8R % 4 == 0
    if not tr16:
        off_q_tr = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_q_tr + D * ST * 2
        off_o_tr = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_o_tr + D * ST * 2
    if fused:
        # One region serves two purposes in sequence.  Before the m loop it
        # holds K in the tr16 layout, just long enough to pull dq's B operands
        # into registers; from then on its first BM*BN halfwords are the ds
        # exchange buffer.  K needs BN*D and ds needs BM*BN, so the larger one
        # sizes it and the two never overlap in time.
        off_kt = allocator._align(allocator.ptr, 16)
        allocator.ptr = off_kt + (KT8_BYTES if F8B
                                  else max(BN * D, BM * BN) * 2)
    # Dead LDS, used only to probe how workgroups-per-CU affects this kernel.
    if lds_pad:
        allocator.ptr = allocator._align(allocator.ptr, 16) + lds_pad

    # ---- valu "vf4": runtime pow2 re-centring of the silu' e4m3 cast ------
    # sc = 2^ka, ka = round(log2(cx)), cx = f32(alpha) * qkv_scale (the f32
    # alpha*s the vf3p/vf4 constants already use; a positive normal f32 under
    # the launcher's guard).  Adding 0x800000 - 0x3504F4 to the bits of cx
    # carries into the exponent field exactly when its mantissa is >= sqrt(2)
    # (0x3504F4 is the smallest 23-bit fraction f with 1 + f/2^23 > sqrt(2);
    # no f32 equals sqrt(2)*2^k, so round(log2(cx)) has no ties), and masking
    # the sum to the exponent field gives the bits of 2^ka.  VF4_TIE (alpha a
    # half-integer power of two) and s a power of two: alpha*s is an exact tie
    # of the probe's host round(log2(alpha*s)) (Python, half-to-even), so the
    # exponent of cx is rounded to the odd biased value, i.e. ka to even (cx
    # is f32(alpha)*s exactly there).  Then 0x7F000000 minus the bits of 2^ka
    # is 2^-ka, and the cast operand (cvt_scalef32 divides by it) is
    # 1/(mu2_ssc*sc) = 2^-(E+ka), E = log2(mu2_ssc): products of powers of
    # two, exact.  Integer ops on uniform values, once per workgroup,
    # readfirstlane'd to an SGPR (the cast reads it where the probe read a
    # literal).
    VF4_RND = 0x800000 - 0x3504F4

    def _tsc_consts(ct):
        """valu tsc: (E8M0 byte 127+k as i32, 2^k, 2^-k) from c_t = -2^k.

        c_t is exactly -2^k (call() checked the qkv_scale), so its biased f32
        exponent field is 127+k; 2^k and 2^-k are built from that byte with
        integer ops (uniform, once per workgroup, SGPR)."""
        from flydsl._mlir.dialects import arith as _ar
        cb = rocdl.readfirstlane(T.i32, arith.unwrap(
            arith.bitcast(T.i32, arith.unwrap(ct))))
        b = arith.ArithValue(_ar.AndIOp(
            _ar.ShRUIOp(cb, arith.unwrap(_i32(23))).result,
            arith.unwrap(_i32(0xFF))).result)                   # 127+k

        def _pow2(eb):                                       # f32 2^(eb-127)
            return arith.ArithValue(arith.bitcast(T.f32, _ar.ShLIOp(
                arith.unwrap(eb), arith.unwrap(_i32(23))).result))
        return b, _pow2(b), _pow2(arith.ArithValue(_i32(254)) - b)

    def _vf4_cast_scale(cx, s, tsc_p2k=None):
        """(cast operand 1/(mu2_ssc*sc), dv undo factor), uniform SGPRs.

        tsc_p2k (valu tsc): 2^k; silu'' = 2^k*silu', so the cast operand is
        2^k/(mu2_ssc*sc) (pow2 x pow2, exact) and the undo factor is vf4's."""
        from flydsl._mlir.dialects import arith as _ar

        def _rfl(v):
            vi = arith.bitcast(T.i32, arith.unwrap(v))
            return arith.ArithValue(arith.bitcast(
                T.f32, rocdl.readfirstlane(T.i32, arith.unwrap(vi))))

        def _and(a, m):
            return arith.ArithValue(_ar.AndIOp(
                arith.unwrap(a), arith.unwrap(_i32(m))).result)
        cb = arith.ArithValue(arith.bitcast(T.i32, arith.unwrap(cx)))
        scb = _and(cb + _i32(VF4_RND), 0x7F800000)           # bits of 2^ka
        if VF4_TIE:
            sb = arith.ArithValue(arith.bitcast(T.i32, arith.unwrap(s)))
            tie = arith.cmpi(arith.CmpIPredicate.eq,
                             arith.unwrap(_and(sb, 0x7FFFFF)),
                             arith.unwrap(_i32(0)))
            odd = _and(arith.ArithValue(_ar.OrIOp(
                arith.unwrap(cb), arith.unwrap(_i32(0x800000))).result),
                0x7F800000)
            scb = arith.ArithValue(arith.select(tie, arith.unwrap(odd),
                                                arith.unwrap(scb)))
        isc = arith.ArithValue(arith.bitcast(T.f32, arith.unwrap(
            arith.ArithValue(_i32(0x7F000000)) - scb)))      # 2^-ka
        sop = isc * _f32(1.0 / mu2_ssc)                      # 1/(mu2_ssc*sc)
        # dv undo factor 1/(mu2_ssc*sc*mu2_osc) * (alpha*s): the probe's pow2
        # constant times f32(alpha)*s, the same two f32 factors.  Formed here
        # and kept in an SGPR, so the epilogues do not need alpha*s in a VGPR
        # across the m loop.
        u = sop if mu2_osc == 1.0 else sop * _f32(1.0 / mu2_osc)
        cop = sop if tsc_p2k is None else sop * tsc_p2k
        return _rfl(cop), _rfl(u * cx)

    def _e8_epilogue(res_dv, res_dk, plast, e8, n0, wave, grp, l16, l4, tid,
                     head, seq_start, seq_len, kt8_lds,
                     dk_input_scale, value_input_scale, score_alpha_scale,
                     inv_s_do, XR_B, vf4_undo=None):
        """R15 M-E1 (e8emit="only"): the dk/dv epilogue writing e5m2 into the
        sink's dz8 instead of BF16 dk/dv, plus the amax/colsum partials.

        Lane (l4, l16) of wave (grp, wave) holds res_*[nt*NDO + ds][i] =
        row n0 + 32*wave + 16*nt + 4*l4 + i, col 64*grp + 16*ds + l16.
        vf4_undo (valu vf4 only): the dv undo factor
        1/(mu2_ssc*sc*mu2_osc)*(alpha*s) (emit_body's prologue, uniform).
        """
        from flydsl._mlir.dialects import arith as _ar
        DZ8_, PAM_, PCS_, inv_v, prow0_v = e8
        i8_t = ir.IntegerType.get_signless(8)
        _tr8_op = getattr(rocdl_raw, "ds_read_tr8_b64_", None)
        assert _tr8_op is not None, "e8 epilogue needs rocdl ds_read_tr8_b64"
        NDO = NSTEP // 2
        v2f = T.vec(2, T.f32)
        v2bf = T.vec(2, T.bf16)
        v2i16 = T.vec(2, T.i16)
        RS8 = 4 * HD                 # dz8 row stride in bytes ([u|v|q|k])
        LIM = 57344.0
        inv = arith.ArithValue(inv_v)
        inv2 = vector.from_elements(v2f, [arith.unwrap(inv)] * 2)
        prow = arith.ArithValue(prow0_v) + arith.ArithValue(n0) // _i32(BN)
        if VALU_VF4:
            # vf4: the silu' cast scale lacks alpha*s; fold it into the
            # uniform undo factor 1/(mu2_ssc*sc*mu2_osc)*(alpha*s) (exact:
            # pow2 * f32), as in the bf16 epilogue.
            dv_undo_vf4 = vf4_undo

        def _vals(nt, ds):
            """dv/dk f32 values: the bf16 epilogue's op sequence verbatim."""
            fv, fk = [], []
            for i in range_constexpr(4):
                dvv = vector.extract(res_dv[nt * NDO + ds], static_position=[i],
                                     dynamic_position=[])
                dkv = arith.ArithValue(vector.extract(
                    res_dk[nt * NDO + ds], static_position=[i],
                    dynamic_position=[])) * _f32(alpha)
                if MU2 == "k128":
                    dvv = arith.unwrap(arith.ArithValue(dvv)
                                       * (dv_undo_vf4 if VALU_VF4 else
                                          _f32(1.0 / (mu2_ssc * mu2_osc))))
                    if do_dyn:
                        dvv = arith.unwrap(arith.ArithValue(dvv) * inv_s_do)
                        dkv = arith.ArithValue(dkv) * inv_s_do
                    dkv = arith.ArithValue(dkv) * _f32(1.0 / (mu2_dsc * mu2_qsc))
                if input_fp8:
                    dkv = arith.ArithValue(dkv) * dk_input_scale
                if VALU_VF2:
                    dkv = arith.ArithValue(dkv) * value_input_scale
                    dvv = arith.unwrap(arith.ArithValue(dvv) * score_alpha_scale)
                fv.append(arith.unwrap(dvv))
                fk.append(arith.unwrap(dkv))
            return fv, fk

        # bf16 pairs (i0,i1), (i2,i3): v_cvt_pk_bf16_f32, RNE per element ->
        # the same bf16 the shipped epilogue stores.
        bfp = {}
        for nt in range_constexpr(NT):
            for ds in range_constexpr(NDO):
                fv, fk = _vals(nt, ds)
                for t, f in ((0, fv), (1, fk)):
                    bfp[(t, nt, ds)] = [
                        _ar.TruncFOp(v2bf, arith.unwrap(vector.from_elements(
                            v2f, [f[2 * h], f[2 * h + 1]]))).result
                        for h in range(2)]

        mn = getattr(_ar, "minnumf", None) or (
            lambda a_, b_: _ar.MinNumFOp(a_, b_).result)
        mx = getattr(_ar, "maxnumf", None) or (
            lambda a_, b_: _ar.MaxNumFOp(a_, b_).result)
        hi_l = arith.unwrap(_f32(LIM))
        lo_l = arith.unwrap(_f32(-LIM))
        one_sc = arith.unwrap(_f32(1.0))
        absf = []

        def _x2(v2b):
            """bf16 pair -> f32 pair (exact), |x| collected for the amax."""
            x = _ar.ExtFOp(v2f, v2b).result
            for k in range(2):
                xe = vector.extract(x, static_position=[k], dynamic_position=[])
                xi = arith.bitcast(T.i32, arith.unwrap(xe))
                absf.append(arith.bitcast(T.f32, _ar.AndIOp(
                    arith.unwrap(xi),
                    arith.unwrap(_i32(0x7FFFFFFF))).result))
            return x

        def _cvt_y(acc, y, half, clamp):
            """y = x * inv pair -> 2 e5m2 bytes in half `half` of acc.
            clamp: maxnum/minnum to +-57344 first (the shipped cast's
            v_med3; NaN -> -57344); else the caller runs it under
            MODE.FP16_OVFL = 1 (saturating convert, same byte for every
            non-NaN y)."""
            y0 = vector.extract(y, static_position=[0], dynamic_position=[])
            y1 = vector.extract(y, static_position=[1], dynamic_position=[])
            if clamp:
                y0 = mn(mx(arith.unwrap(y0), lo_l), hi_l)
                y1 = mn(mx(arith.unwrap(y1), lo_l), hi_l)
            r = rocdl_raw.cvt_scalef32_pk_bf8_f32(v2i16, acc, y0, y1, one_sc,
                                                  half)
            return r.result if hasattr(r, "result") else r

        def _dpp(ty, v, ctrl):
            # Untied DPP (old = undef, bound_ctrl) for the amax row max: every
            # ctrl used (quad_perm, row_[half_]mirror) is a full permutation
            # within a row and all lanes are active, so no lane reads 'old'.
            return rocdl_raw.DPPUpdateOp(ty, llvm.UndefOp(ty).result,
                                         arith.unwrap(v), ctrl, 0xF, 0xF,
                                         True).result

        rDZ = _shifted_rsrc(
            DZ8_, arith.ArithValue(_to_i64(seq_start)) * _i64(RS8)
            + arith.ArithValue(_to_i64(head)) * _i64(D), 1,
            num_bytes=arith.ArithValue(seq_len) * _i32(RS8))
        def _colsum():
            """Column sums: one bf16 MFMA per (t, ds) with A = ones over the
            lane's 8 bf16 (rows 16*nt + 4*l4 + i): every lane gets the 32-row
            sum of col 16*ds + l16; lane keeps ds = l4.  Issued after the e5m2
            stores and the amax."""
            ones8 = vector.bitcast(T.vec(8, T.bf16), arith.constant_vector(
                0x3F803F80, T.vec(4, T.i32)))
            zero4 = arith.constant_vector(0.0, T.f32x4)
            csv = []
            for t in range_constexpr(2):
                parts = []
                for ds in range_constexpr(NDO):
                    b8 = vector.shuffle(
                        vector.shuffle(bfp[(t, 0, ds)][0], bfp[(t, 0, ds)][1],
                                       [0, 1, 2, 3]),
                        vector.shuffle(bfp[(t, 1, ds)][0], bfp[(t, 1, ds)][1],
                                       [0, 1, 2, 3]),
                        list(range(8)))
                    c4 = rocdl_raw.mfma_f32_16x16x32_bf16(
                        T.f32x4, arith.unwrap(ones8), arith.unwrap(b8),
                        arith.unwrap(zero4), 0, 0, 0).result
                    parts.append(vector.extract(c4, static_position=[0],
                                                dynamic_position=[]))
                v = parts[NDO - 1]
                for ds in range(NDO - 2, -1, -1):
                    v = arith.select(arith.cmpi(arith.CmpIPredicate.eq,
                                                arith.unwrap(l4),
                                                arith.unwrap(_i32(ds))),
                                     parts[ds], v)
                csv.append(v)
            return csv

        # ---- e5m2 payload: 16 dwords per lane (t, nt, ds) -> 4 dwordx4
        # stores.  The dword of (t, nt, ds) holds rows 4*l4 + 0..3 of col
        # c = 16*ds + l16 of the wave's tile ti = 2*t + nt (16 rows x 64
        # cols).  Each lane writes it column-major ([ti][c][16 rows] bytes)
        # into a wave-private 4 KB LDS block; within each 16-lane group,
        # ds_read_b64_tr_b8 hands lane j byte (j % 8) of the 8-byte runs
        # named by source lanes 2*e + j//8 as its byte e (measured, see
        # tr8), so with source lane j naming column 16*l4 + 8*q + j//2, rows
        # 8*(j%2) .. +7, two reads (q = 0, 1) give lane (l16, l4) row l16,
        # cols 16*l4 + 0..15 of the tile: one buffer_store_dwordx4.  The
        # block sits in the parity-(1-plast) exchange block (dead after the
        # last pair's closing barrier) after the colsum's first 4 KB:
        # 8 waves x 4 KB <= 4*XR_B - 4 KB.  LDS ops of one wave execute in
        # order, so no barrier.  Lane/wave indices are re-derived after the
        # loop from an opaque copy of tid: none of the address terms can be
        # merged into a value computed before the m loop (that kept such
        # values live through the loop and spilled them).
        keys = [(t, nt, ds) for t in range(2) for nt in range(NT)
                for ds in range(NDO)]
        assert 4 * XR_B >= 4096 * (1 + 2 * WAVES), XR_B

        def _payload(ys, clamp):
            tid_o = arith.ArithValue(llvm.inline_asm(
                T.i32, [arith.unwrap(tid)], "v_mov_b32 $0, $1", "=v,v",
                has_side_effects=False))
            l16_o = tid_o % _i32(16)
            l4_o = (tid_o // _i32(16)) % _i32(4)
            wave_o = (tid_o // _i32(64)) % _i32(WAVES)
            grp_o = tid_o // _i32(64 * WAVES)
            wave_hw_ = wave_o + grp_o * _i32(WAVES)
            stg = ((_i32(1) - arith.ArithValue(plast)) * _i32(XR_B)
                   + _i32(1024) + wave_hw_ * _i32(1024))      # dword index
            wr = stg + (arith.ArithValue(l16_o) * _i32(4) + l4_o)
            for key in keys:
                t, nt, ds = key
                acc = llvm.UndefOp(v2i16).result
                acc = _cvt_y(acc, ys[key][0], 0, clamp)
                acc = _cvt_y(acc, ys[key][1], 1, clamp)
                w = vector.extract(vector.bitcast(T.vec(1, T.i32), acc),
                                   static_position=[0], dynamic_position=[])
                vector.store(vector.from_elements(T.vec(1, T.i32), [
                    arith.unwrap(w)]), kt8_lds,
                    [_idx(wr + _i32((2 * t + nt) * 256 + 64 * ds))],
                    alignment=4)
            rd = (stg * _i32(4)
                  + (arith.ArithValue(l4_o) * _i32(16)
                     + arith.ArithValue(l16_o) // _i32(2)) * _i32(16)
                  + (arith.ArithValue(l16_o) % _i32(2)) * _i32(8))  # bytes
            row_o = n0 + wave_o * _i32(16 * NT) + l16_o
            col_o = grp_o * _i32(16 * NDO) + arith.ArithValue(l4_o) * _i32(16)
            for t in range_constexpr(2):
                for nt in range_constexpr(NT):
                    ti = 2 * t + nt
                    parts = []
                    for q in range_constexpr(2):
                        r_ = _tr8_op(T.vec(2, T.i32), _lds_gep(
                            kt8_lds, rd + _i32(ti * 1024 + 128 * q), i8_t))
                        parts.append(r_.result if hasattr(r_, "result")
                                     else r_)
                    v4 = vector.shuffle(parts[0], parts[1], [0, 1, 2, 3])
                    off = ((row_o + _i32(16 * nt)) * _i32(RS8) + col_o
                           + _i32((1 + 2 * t) * HD))
                    rocdl_raw.RawPtrBufferStoreOp(
                        arith.unwrap(v4), rDZ, arith.unwrap(off),
                        arith.unwrap(_i32(0)), arith.unwrap(_i32(0)))

        def _lane_max(vals, op):
            m_ = vals[0]
            for x_ in vals[1:]:
                m_ = op(m_, x_)
            return m_

        def _set_fp16_ovfl(bit):
            rocdl.sched_barrier(0)
            llvm.InlineAsmOp(
                None, [],
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), %d\n"
                "s_nop 7\ns_nop 7" % bit, "", has_side_effects=True)
            rocdl.sched_barrier(0)

        def _abs_from_bf16():
            """|x| in absf order, re-derived from the packed bf16 with
            v_perm_b32 (an op the fpext above cannot be CSE'd into, so x is
            not kept live into the NaN path)."""
            out = []
            for key in keys:
                for v2b in bfp[key]:
                    p = vector.extract(vector.bitcast(T.vec(1, T.i32), v2b),
                                       static_position=[0],
                                       dynamic_position=[])
                    for sel in (0x01000C0C, 0x03020C0C):   # lo, hi -> f32
                        r = llvm.call_intrinsic(
                            T.i32, "llvm.amdgcn.perm",
                            [arith.unwrap(p), arith.unwrap(p),
                             arith.unwrap(_i32(sel))], [], [])
                        r = r.result if hasattr(r, "result") else r
                        out.append(arith.bitcast(T.f32, _ar.AndIOp(
                            r, arith.unwrap(_i32(0x7FFFFFFF))).result))
            return out

        # NaN-guarded saturating converts.  Under MODE.FP16_OVFL = 1,
        # v_cvt_scalef32_pk_bf8_f32 gives every non-NaN y the byte of
        # cvt(clamp(y, +-57344)) (+-inf and f32-overflowed products
        # included; r15-e1sweep-ovfl-shortcut), so the 64 v_med3 clamps per
        # lane are only needed when some y is NaN, i.e. when some x is NaN or
        # inv is not a finite positive.  x and y = x * inv of every pair are
        # formed first; the lane max of |x| uses the NaN-propagating maximum
        # (== the maxnum chain when no |x| is NaN).  If no lane of the wave
        # sees a NaN and inv is finite positive (wave-uniform branch), the
        # payload runs clamp-free under FP16_OVFL = 1 (set after the bf16
        # casts above, cleared after the e5m2 stores); otherwise the shipped
        # clamp path runs and the amax is the maxnum chain re-derived from
        # the bf16 values.  Bytes and amax are bitwise in both paths.
        ys = {key: [_ar.MulFOp(_x2(v2b), arith.unwrap(inv2)).result
                    for v2b in bfp[key]] for key in keys}
        mxm = getattr(_ar, "maximumf", None) or (
            lambda a_, b_: _ar.MaximumFOp(a_, b_).result)
        am_nan = _lane_max(absf, mxm)
        P_ = _ar.CmpFPredicate
        inv_u = arith.unwrap(inv)
        bad = _ar.OrIOp(
            _ar.cmpf(P_.UNO, am_nan, am_nan),
            _ar.OrIOp(_ar.cmpf(P_.ULE, inv_u, arith.unwrap(_f32(0.0))),
                      _ar.cmpf(P_.UGE, inv_u,
                               arith.unwrap(_f32(float("inf"))))
                      ).result).result
        bal = llvm.call_intrinsic(T.i64, "llvm.amdgcn.ballot.i64", [bad],
                                  [], [])
        bal = bal.result if hasattr(bal, "result") else bal
        g_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, bal,
                                   arith.unwrap(_i64(0))),
                        results_=[T.f32], has_else=True)
        with ir.InsertionPoint(g_if.then_block):
            _set_fp16_ovfl(1)
            _payload(ys, clamp=False)
            _set_fp16_ovfl(0)
            scf.YieldOp([am_nan])
        with ir.InsertionPoint(g_if.else_block):
            _payload(ys, clamp=True)
            scf.YieldOp([_lane_max(_abs_from_bf16(), mx)])

        # ---- amax: lane max -> 16-lane row max (4 DPP steps) -> l16 == 0
        # lanes store 4 partials per wave (32 per workgroup).
        am = g_if.results[0]
        for ctrl in (0xB1, 0x4E, 0x141, 0x140):
            am = mx(am, _dpp(T.f32, am, ctrl))
        rPAM = _shifted_rsrc(PAM_, _to_i64(_i32(0)), 4)
        wave_hw = arith.ArithValue(wave) + grp * _i32(WAVES)
        am_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq,
                                    arith.unwrap(l16), arith.unwrap(_i32(0))),
                         results_=[], has_else=False)
        with ir.InsertionPoint(am_if.then_block):
            buffer_ops.buffer_store(
                am, rPAM,
                prow * _i32(H * 32) + arith.ArithValue(head) * _i32(32)
                + wave_hw * _i32(4) + l4)
            scf.YieldOp([])

        # ---- column sums.  Cross-wave (the 4 n-owner waves of this group):
        # the exchange block of parity 1 - plast is dead after the last
        # pair's closing barrier (the drain reads parity plast only).
        # [t][wave][128 cols] f32.
        csv = _colsum()
        fb = (_i32(1) - arith.ArithValue(plast)) * _i32(XR_B)
        colw = (grp * _i32(16 * NDO) + arith.ArithValue(l4) * _i32(16) + l16)
        for t in range_constexpr(2):
            idx = (fb + (arith.ArithValue(wave) + _i32(4 * t)) * _i32(128)
                   + colw)
            vector.store(vector.from_elements(T.vec(1, T.i32), [
                arith.unwrap(arith.bitcast(T.i32, arith.unwrap(csv[t])))]),
                kt8_lds, [_idx(idx)], alignment=4)
        gpu.barrier()
        rPCS = _shifted_rsrc(PCS_, _to_i64(_i32(0)), 4)
        cs_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.slt,
                                    arith.unwrap(tid), arith.unwrap(_i32(256))),
                         results_=[], has_else=False)
        with ir.InsertionPoint(cs_if.then_block):
            t_ = arith.ArithValue(tid) // _i32(128)
            c_ = arith.ArithValue(tid) % _i32(128)
            s = None
            for w_ in range_constexpr(4):
                idx = fb + (t_ * _i32(4) + _i32(w_)) * _i32(128) + c_
                p = arith.ArithValue(arith.bitcast(T.f32, arith.unwrap(
                    vector.extract(vector.load_op(T.vec(1, T.i32), kt8_lds,
                                                  [_idx(idx)]),
                                   static_position=[0], dynamic_position=[]))))
                s = p if s is None else s + p
            buffer_ops.buffer_store(
                s, rPCS,
                prow * _i32(2 * HD) + t_ * _i32(HD)
                + arith.ArithValue(head) * _i32(D) + c_)
            scf.YieldOp([])

    def _e8_epilogue_gsp(res_dv, res_dk, plast, e8, n0, wave, grp, l16, l4,
                         tid, head, seq_start, seq_len, kt8_lds,
                         dk_input_scale, value_input_scale, score_alpha_scale,
                         inv_s_do, XR_B, vf4_undo=None):
        """R15 iter73 gsp: _e8_epilogue for the G-split accumulator layout.

        Lane (l4, l16) of wave (grp, wave) holds res[nt*NSTEP + jj][i] of
        tensor grp (0 dV, 1 dK; res = res_dv + res_dk) = row n0 + 32*wave
        + 16*nt + 4*l4 + i, col 16*jj + l16.  Key (h, nt, ds), jj = NDO*h
        + ds, is the shipped key (t = grp, nt, ds) of wave (h, wave):
          * values: the shipped per-tensor f32 op sequence (_vals), chosen by
            a uniform branch on grp, packed to the same bf16 pairs;
          * e5m2 payload: staged as tile 2*h + nt, stored to dz8 col
            64*h + 16*l4 + (1 + 2*grp)*HD -- the shipped bytes and addresses;
          * colsum: the shipped bf16 MFMA per (h, ds) into LDS slot
            [t = grp][wave][64*h + 16*l4 + l16], same barrier and reduction;
          * amax: the shipped partial (g, wave, l4) is the max over BOTH
            tensors of column half g.  Each wave forms one partial per half,
            keeps h = grp and hands h = 1 - grp to its partner wave
            (1 - grp, wave) through the dead parity-(1 - plast) block (dwords
            AMX.. after the colsum and staging areas) across the colsum
            barrier; it stores maxnum(own, partner) in the shipped slot.
            Non-NaN maxima are exact, so the partial equals the shipped one
            whenever the partial's 1024 values are not all NaN.
        """
        from flydsl._mlir.dialects import arith as _ar
        DZ8_, PAM_, PCS_, inv_v, prow0_v = e8
        i8_t = ir.IntegerType.get_signless(8)
        _tr8_op = getattr(rocdl_raw, "ds_read_tr8_b64_", None)
        assert _tr8_op is not None, "e8 epilogue needs rocdl ds_read_tr8_b64"
        NDO = NSTEP // 2
        v2f = T.vec(2, T.f32)
        v2bf = T.vec(2, T.bf16)
        v2i16 = T.vec(2, T.i16)
        RS8 = 4 * HD                 # dz8 row stride in bytes ([u|v|q|k])
        LIM = 57344.0
        AMX = 1024 * (1 + 2 * WAVES)  # amax hand-over slots (dword index)
        assert 4 * XR_B >= 4 * AMX + 4 * 4 * 2 * WAVES, XR_B
        inv = arith.ArithValue(inv_v)
        inv2 = vector.from_elements(v2f, [arith.unwrap(inv)] * 2)
        prow = arith.ArithValue(prow0_v) + arith.ArithValue(n0) // _i32(BN)
        res = list(res_dv) + list(res_dk)
        keys = [(h, nt, ds) for h in range(2) for nt in range(NT)
                for ds in range(NDO)]

        def _val(t, x):
            """The shipped _vals op sequence of tensor t (0 dV, 1 dK)."""
            if t == 0:
                dvv = x
                if MU2 == "k128":
                    dvv = arith.unwrap(arith.ArithValue(dvv)
                                       * (vf4_undo if VALU_VF4 else
                                          _f32(1.0 / (mu2_ssc * mu2_osc))))
                    if do_dyn:
                        dvv = arith.unwrap(arith.ArithValue(dvv) * inv_s_do)
                if VALU_VF2:
                    dvv = arith.unwrap(arith.ArithValue(dvv)
                                       * score_alpha_scale)
                return arith.unwrap(dvv)
            dkv = arith.ArithValue(x) * _f32(alpha)
            if MU2 == "k128":
                if do_dyn:
                    dkv = arith.ArithValue(dkv) * inv_s_do
                dkv = arith.ArithValue(dkv) * _f32(1.0 / (mu2_dsc * mu2_qsc))
            if input_fp8:
                dkv = arith.ArithValue(dkv) * dk_input_scale
            if VALU_VF2:
                dkv = arith.ArithValue(dkv) * value_input_scale
            return arith.unwrap(dkv)

        def _bfp_list(t):
            out = []
            for (h, nt, ds) in keys:
                acc_v = res[nt * NSTEP + NDO * h + ds]
                f = [_val(t, vector.extract(acc_v, static_position=[i],
                                            dynamic_position=[]))
                     for i in range_constexpr(4)]
                for hh in range(2):
                    out.append(_ar.TruncFOp(v2bf, arith.unwrap(
                        vector.from_elements(v2f, [f[2 * hh],
                                                   f[2 * hh + 1]]))).result)
            return out

        vbr = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0)),
                       results_=[v2bf] * (2 * len(keys)), has_else=True)
        with ir.InsertionPoint(vbr.then_block):
            scf.YieldOp(_bfp_list(0))
        with ir.InsertionPoint(vbr.else_block):
            scf.YieldOp(_bfp_list(1))
        flat = list(vbr.results)
        bfp = {key: [flat[2 * k], flat[2 * k + 1]]
               for k, key in enumerate(keys)}

        mn = getattr(_ar, "minnumf", None) or (
            lambda a_, b_: _ar.MinNumFOp(a_, b_).result)
        mx = getattr(_ar, "maxnumf", None) or (
            lambda a_, b_: _ar.MaxNumFOp(a_, b_).result)
        hi_l = arith.unwrap(_f32(LIM))
        lo_l = arith.unwrap(_f32(-LIM))
        one_sc = arith.unwrap(_f32(1.0))
        absf = {0: [], 1: []}

        def _x2(v2b, h):
            """bf16 pair -> f32 pair (exact), |x| collected per half h."""
            x = _ar.ExtFOp(v2f, v2b).result
            for k in range(2):
                xe = vector.extract(x, static_position=[k], dynamic_position=[])
                xi = arith.bitcast(T.i32, arith.unwrap(xe))
                absf[h].append(arith.bitcast(T.f32, _ar.AndIOp(
                    arith.unwrap(xi),
                    arith.unwrap(_i32(0x7FFFFFFF))).result))
            return x

        def _cvt_y(acc, y, half, clamp):
            y0 = vector.extract(y, static_position=[0], dynamic_position=[])
            y1 = vector.extract(y, static_position=[1], dynamic_position=[])
            if clamp:
                y0 = mn(mx(arith.unwrap(y0), lo_l), hi_l)
                y1 = mn(mx(arith.unwrap(y1), lo_l), hi_l)
            r = rocdl_raw.cvt_scalef32_pk_bf8_f32(v2i16, acc, y0, y1, one_sc,
                                                  half)
            return r.result if hasattr(r, "result") else r

        def _dpp(ty, v, ctrl):
            return rocdl_raw.DPPUpdateOp(ty, llvm.UndefOp(ty).result,
                                         arith.unwrap(v), ctrl, 0xF, 0xF,
                                         True).result

        rDZ = _shifted_rsrc(
            DZ8_, arith.ArithValue(_to_i64(seq_start)) * _i64(RS8)
            + arith.ArithValue(_to_i64(head)) * _i64(D), 1,
            num_bytes=arith.ArithValue(seq_len) * _i32(RS8))

        def _colsum():
            """Per column half h: the shipped bf16 colsum MFMA (A = ones) of
            this wave's tensor; lane keeps ds = l4 (col 64*h + 16*l4 + l16)."""
            ones8 = vector.bitcast(T.vec(8, T.bf16), arith.constant_vector(
                0x3F803F80, T.vec(4, T.i32)))
            zero4 = arith.constant_vector(0.0, T.f32x4)
            csv = []
            for h in range_constexpr(2):
                parts = []
                for ds in range_constexpr(NDO):
                    b8 = vector.shuffle(
                        vector.shuffle(bfp[(h, 0, ds)][0], bfp[(h, 0, ds)][1],
                                       [0, 1, 2, 3]),
                        vector.shuffle(bfp[(h, 1, ds)][0], bfp[(h, 1, ds)][1],
                                       [0, 1, 2, 3]),
                        list(range(8)))
                    c4 = rocdl_raw.mfma_f32_16x16x32_bf16(
                        T.f32x4, arith.unwrap(ones8), arith.unwrap(b8),
                        arith.unwrap(zero4), 0, 0, 0).result
                    parts.append(vector.extract(c4, static_position=[0],
                                                dynamic_position=[]))
                v = parts[NDO - 1]
                for ds in range(NDO - 2, -1, -1):
                    v = arith.select(arith.cmpi(arith.CmpIPredicate.eq,
                                                arith.unwrap(l4),
                                                arith.unwrap(_i32(ds))),
                                     parts[ds], v)
                csv.append(v)
            return csv

        def _payload(ys, clamp):
            tid_o = arith.ArithValue(llvm.inline_asm(
                T.i32, [arith.unwrap(tid)], "v_mov_b32 $0, $1", "=v,v",
                has_side_effects=False))
            l16_o = tid_o % _i32(16)
            l4_o = (tid_o // _i32(16)) % _i32(4)
            wave_o = (tid_o // _i32(64)) % _i32(WAVES)
            grp_o = tid_o // _i32(64 * WAVES)
            wave_hw_ = wave_o + grp_o * _i32(WAVES)
            stg = ((_i32(1) - arith.ArithValue(plast)) * _i32(XR_B)
                   + _i32(1024) + wave_hw_ * _i32(1024))      # dword index
            wr = stg + (arith.ArithValue(l16_o) * _i32(4) + l4_o)
            for key in keys:
                h, nt, ds = key
                acc = llvm.UndefOp(v2i16).result
                acc = _cvt_y(acc, ys[key][0], 0, clamp)
                acc = _cvt_y(acc, ys[key][1], 1, clamp)
                w = vector.extract(vector.bitcast(T.vec(1, T.i32), acc),
                                   static_position=[0], dynamic_position=[])
                vector.store(vector.from_elements(T.vec(1, T.i32), [
                    arith.unwrap(w)]), kt8_lds,
                    [_idx(wr + _i32((2 * h + nt) * 256 + 64 * ds))],
                    alignment=4)
            rd = (stg * _i32(4)
                  + (arith.ArithValue(l4_o) * _i32(16)
                     + arith.ArithValue(l16_o) // _i32(2)) * _i32(16)
                  + (arith.ArithValue(l16_o) % _i32(2)) * _i32(8))  # bytes
            row_o = n0 + wave_o * _i32(16 * NT) + l16_o
            # gsp: the wave's tensor is grp (dV cols H*D.., dK cols 3*H*D..)
            col_o = (arith.ArithValue(l4_o) * _i32(16)
                     + (_i32(1) + grp_o * _i32(2)) * _i32(HD))
            for h in range_constexpr(2):
                for nt in range_constexpr(NT):
                    ti = 2 * h + nt
                    parts = []
                    for q in range_constexpr(2):
                        r_ = _tr8_op(T.vec(2, T.i32), _lds_gep(
                            kt8_lds, rd + _i32(ti * 1024 + 128 * q), i8_t))
                        parts.append(r_.result if hasattr(r_, "result")
                                     else r_)
                    v4 = vector.shuffle(parts[0], parts[1], [0, 1, 2, 3])
                    off = ((row_o + _i32(16 * nt)) * _i32(RS8) + col_o
                           + _i32(16 * NDO * h))
                    rocdl_raw.RawPtrBufferStoreOp(
                        arith.unwrap(v4), rDZ, arith.unwrap(off),
                        arith.unwrap(_i32(0)), arith.unwrap(_i32(0)))

        def _lane_max(vals, op):
            m_ = vals[0]
            for x_ in vals[1:]:
                m_ = op(m_, x_)
            return m_

        def _set_fp16_ovfl(bit):
            rocdl.sched_barrier(0)
            llvm.InlineAsmOp(
                None, [],
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), %d\n"
                "s_nop 7\ns_nop 7" % bit, "", has_side_effects=True)
            rocdl.sched_barrier(0)

        def _abs_from_bf16(h):
            out = []
            for key in keys:
                if key[0] != h:
                    continue
                for v2b in bfp[key]:
                    p = vector.extract(vector.bitcast(T.vec(1, T.i32), v2b),
                                       static_position=[0],
                                       dynamic_position=[])
                    for sel in (0x01000C0C, 0x03020C0C):   # lo, hi -> f32
                        r = llvm.call_intrinsic(
                            T.i32, "llvm.amdgcn.perm",
                            [arith.unwrap(p), arith.unwrap(p),
                             arith.unwrap(_i32(sel))], [], [])
                        r = r.result if hasattr(r, "result") else r
                        out.append(arith.bitcast(T.f32, _ar.AndIOp(
                            r, arith.unwrap(_i32(0x7FFFFFFF))).result))
            return out

        ys = {key: [_ar.MulFOp(_x2(v2b, key[0]), arith.unwrap(inv2)).result
                    for v2b in bfp[key]] for key in keys}
        mxm = getattr(_ar, "maximumf", None) or (
            lambda a_, b_: _ar.MaximumFOp(a_, b_).result)
        am_nan = [_lane_max(absf[h], mxm) for h in range(2)]
        P_ = _ar.CmpFPredicate
        inv_u = arith.unwrap(inv)
        bad = _ar.OrIOp(
            _ar.OrIOp(_ar.cmpf(P_.UNO, am_nan[0], am_nan[0]),
                      _ar.cmpf(P_.UNO, am_nan[1], am_nan[1])).result,
            _ar.OrIOp(_ar.cmpf(P_.ULE, inv_u, arith.unwrap(_f32(0.0))),
                      _ar.cmpf(P_.UGE, inv_u,
                               arith.unwrap(_f32(float("inf"))))
                      ).result).result
        bal = llvm.call_intrinsic(T.i64, "llvm.amdgcn.ballot.i64", [bad],
                                  [], [])
        bal = bal.result if hasattr(bal, "result") else bal
        g_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, bal,
                                   arith.unwrap(_i64(0))),
                        results_=[T.f32, T.f32], has_else=True)
        with ir.InsertionPoint(g_if.then_block):
            _set_fp16_ovfl(1)
            _payload(ys, clamp=False)
            _set_fp16_ovfl(0)
            scf.YieldOp(am_nan)
        with ir.InsertionPoint(g_if.else_block):
            _payload(ys, clamp=True)
            scf.YieldOp([_lane_max(_abs_from_bf16(h), mx) for h in range(2)])

        # ---- amax: per half, lane max -> 16-lane row max (4 DPP steps)
        am = list(g_if.results)
        for h in range(2):
            for ctrl in (0xB1, 0x4E, 0x141, 0x140):
                am[h] = mx(am[h], _dpp(T.f32, am[h], ctrl))
        g0 = arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0))
        am_own = arith.select(g0, am[0], am[1])   # half h = grp
        am_pub = arith.select(g0, am[1], am[0])   # half 1 - grp -> partner
        fb = (_i32(1) - arith.ArithValue(plast)) * _i32(XR_B)
        wl = arith.ArithValue(wave) * _i32(4) + l4
        pub_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq,
                                     arith.unwrap(l16), arith.unwrap(_i32(0))),
                          results_=[], has_else=False)
        with ir.InsertionPoint(pub_if.then_block):
            vector.store(vector.from_elements(T.vec(1, T.i32), [
                arith.unwrap(arith.bitcast(T.i32, arith.unwrap(am_pub)))]),
                kt8_lds, [_idx(fb + _i32(AMX) + grp * _i32(4 * WAVES) + wl)],
                alignment=4)
            scf.YieldOp([])

        # ---- column sums -> the shipped [t][wave][128 cols] LDS slots
        csv = _colsum()
        for h in range_constexpr(2):
            idx = (fb + (arith.ArithValue(wave) + grp * _i32(4)) * _i32(128)
                   + _i32(16 * NDO * h) + arith.ArithValue(l4) * _i32(16)
                   + l16)
            vector.store(vector.from_elements(T.vec(1, T.i32), [
                arith.unwrap(arith.bitcast(T.i32, arith.unwrap(csv[h])))]),
                kt8_lds, [_idx(idx)], alignment=4)
        gpu.barrier()
        rPAM = _shifted_rsrc(PAM_, _to_i64(_i32(0)), 4)
        wave_hw = arith.ArithValue(wave) + grp * _i32(WAVES)
        am_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq,
                                    arith.unwrap(l16), arith.unwrap(_i32(0))),
                         results_=[], has_else=False)
        with ir.InsertionPoint(am_if.then_block):
            pidx = (fb + _i32(AMX) + (_i32(1) - grp) * _i32(4 * WAVES) + wl)
            partner = arith.bitcast(T.f32, arith.unwrap(vector.extract(
                vector.load_op(T.vec(1, T.i32), kt8_lds, [_idx(pidx)]),
                static_position=[0], dynamic_position=[])))
            buffer_ops.buffer_store(
                arith.ArithValue(mx(arith.unwrap(am_own),
                                    arith.unwrap(partner))), rPAM,
                prow * _i32(H * 32) + arith.ArithValue(head) * _i32(32)
                + wave_hw * _i32(4) + l4)
            scf.YieldOp([])
        rPCS = _shifted_rsrc(PCS_, _to_i64(_i32(0)), 4)
        cs_if = scf.IfOp(arith.cmpi(arith.CmpIPredicate.slt,
                                    arith.unwrap(tid), arith.unwrap(_i32(256))),
                         results_=[], has_else=False)
        with ir.InsertionPoint(cs_if.then_block):
            t_ = arith.ArithValue(tid) // _i32(128)
            c_ = arith.ArithValue(tid) % _i32(128)
            s = None
            for w_ in range_constexpr(4):
                idx = fb + (t_ * _i32(4) + _i32(w_)) * _i32(128) + c_
                p = arith.ArithValue(arith.bitcast(T.f32, arith.unwrap(
                    vector.extract(vector.load_op(T.vec(1, T.i32), kt8_lds,
                                                  [_idx(idx)]),
                                   static_position=[0], dynamic_position=[]))))
                s = p if s is None else s + p
            buffer_ops.buffer_store(
                s, rPCS,
                prow * _i32(2 * HD) + t_ * _i32(HD)
                + arith.ArithValue(head) * _i32(D) + c_)
            scf.YieldOp([])

    def emit_body(
        Q,
        K,
        V,
        DO,
        DK,
        DV,
        DQ,
        inv_max,
        qkv_scale,
        tid,
        n0,
        head,
        seq_start,
        seq_len,
        DO_SC=None,
        batch_idx=None,
        nbatch=None,
        ws_pre_v=None,
        e8=None,
    ):
        mfma = rocdl.mfma_f32_16x16x16bf16_1k
        v4i16 = T.vec(4, T.i16)
        v4bf = T.vec(4, T.bf16)
        v8bf = T.vec(8, T.bf16)
        v2i32 = T.vec(2, T.i32)
        v4i32 = T.vec(4, T.i32)
        v8i32 = T.vec(8, T.i32)
        zero_acc = arith.constant_vector(0.0, T.f32x4)

        def load_raw8(resource, offset):
            """The 8 packed e4m3 bytes at `offset` (in elements), un-widened.

            Exactly the buffer_load load_qkv issues before its cvt chain; the
            fp8 paths keep the dwords instead of widening them.
            """
            return buffer_ops.buffer_load(
                resource,
                arith.ArithValue(offset) // _i32(4),
                vec_width=2,
                dtype=T.i32,
            )

        def _widen_raw8(packed):
            """8 packed e4m3 bytes (vec<2xi32>) -> vec<8xbf16>.

            e4m3 -> bf16 is exact (3 mantissa bits into 7), so this is a pure
            widening whenever conversion_scale is 1.0.
            """
            if ABL_WCVT0:
                # Ceiling probe for the e4m3->bf16 WIDENING, same trick as
                # ABL_BCVT0 does for the out-gemm B operands: keep the load, the
                # data dependence and the fragment width, and delete ONLY the
                # four cvt_scalef32_pk_bf16_fp8 ops by reinterpreting the two
                # dwords as bf16 halfwords and widening the fragment by shuffle.
                # Zero VALU, numerically garbage on purpose.  The resulting
                # time is the FLOOR any faster convert could reach here, because
                # it prices a convert that is infinitely fast.
                #
                # MEASURED (2026-09-13, gfx950, ISA-verified that the 128
                # v_cvt_scalef32_pk_bf16_fp8 ops are gone and ds_read/mfma counts
                # are unchanged): that floor is ZERO.  The shipped k128 schedule
                # gets SLOWER with the converts deleted -- 15.500 -> 15.812 ms at
                # batch 512, 31.019 -> 31.489 at 1024 -- because it has >=3 ms of
                # VALU slack and hides them completely.  Read that as "no gain
                # available", not as "converts are beneficial"; the negative
                # point estimate is the substitute perturbing the schedule.
                #
                # This is what closed FP6 and FP4: a 17.86x faster convert is
                # worth nothing when the incumbent convert already costs nothing.
                # Do NOT reach for this expecting the old `cvt_scalef32_pk8_*`
                # framing -- that family is gfx1250 and does not exist on gfx950,
                # so its "Cannot select" is correct behaviour, not a toolchain
                # gap awaiting a fix.
                #
                # Run this probe in the SHIPPED config before proposing any
                # narrower storage format for q/k/v.  An exposed cost measured in
                # a non-shipped arm is not a prize: the same converts price at
                # 3.0 ms in the bf16-MFMA arm and 0 ms here.
                as_bf = vector.bitcast(T.vec(4, T.bf16), packed)
                return vector.shuffle(as_bf, as_bf, CAT8)
            converted = []
            conversion_scale = (
                arith.ArithValue(qkv_scale)
                if scale_in_cvt
                else _f32(1.0)
            )
            for dword in range_constexpr(2):
                source = vector.extract(
                    packed,
                    static_position=[dword],
                    dynamic_position=[],
                )
                for high_word in (False, True):
                    converted.append(
                        rocdl_raw.cvt_scalef32_pk_bf16_fp8(
                            T.vec(2, T.bf16),
                            source,
                            arith.unwrap(conversion_scale),
                            high_word,
                        )
                    )
            low = vector.shuffle(converted[0], converted[1], [0, 1, 2, 3])
            high = vector.shuffle(converted[2], converted[3], [0, 1, 2, 3])
            return vector.shuffle(low, high, CAT8)

        def load_qkv(resource, offset, source_fp8=True):
            if input_fp8 and source_fp8:
                # Every call is eight-byte aligned. Loading two dwords avoids
                # vector<8xi8>, which SelectionDAG cannot legally split for the
                # raw-buffer intrinsic on gfx950.
                return _widen_raw8(load_raw8(resource, offset))
            return buffer_ops.buffer_load(
                resource, offset, vec_width=8, dtype=T.bf16
            )

        score_input_scale = (
            arith.ArithValue(qkv_scale)
            * (
                _f32(1.0)
                if q_bf16
                else arith.ArithValue(qkv_scale)
            )
            if input_fp8 and not scale_in_cvt
            else _f32(1.0)
        )
        value_input_scale = (
            arith.ArithValue(qkv_scale)
            if input_fp8 and not scale_in_cvt
            else _f32(1.0)
        )
        dk_input_scale = (
            _f32(1.0) if q_bf16 else value_input_scale
        )
        # epilogue="reuse": the uniform alpha*score_input_scale product is
        # paid once per kernel invocation instead of once per score element.
        score_alpha_scale = _f32(alpha) * arith.ArithValue(score_input_scale)
        # epilogue="reuse_pk": <2 x f32> splats of the uniform operands, built
        # once per kernel (LLVM keeps them in SGPR/VGPR pairs that
        # v_pk_*_f32 read directly).
        V2F = T.vec(2, T.f32)
        # valu vf4: the runtime silu' cast operand and dv undo factor (set
        # below).  valu tsc: the score MFMA's E8M0 byte 127+k (set below).
        vf4_sop = vf4_undo = None
        tsc_sa = None
        if EPI_PK:
            def _splat2(v):
                v = arith.unwrap(v)
                return vector.from_elements(V2F, [v, v])
            pk_sas2 = _splat2(score_alpha_scale)
            pk_nl2e2 = arith.constant_vector(-LOG2E, V2F)
            pk_one2 = arith.constant_vector(1.0, V2F)
            pk_inv2 = _splat2(inv_max)
            pk_vis2 = _splat2(value_input_scale)
            if VALU_VF or VALU_H16:
                pk_c1 = _splat2(score_alpha_scale * _f32(-LOG2E))
            if VALU_VF3:
                def _rfl32(v):
                    vi = arith.bitcast(T.i32, arith.unwrap(v))
                    return arith.ArithValue(arith.bitcast(
                        T.f32, rocdl.readfirstlane(T.i32, arith.unwrap(vi))))
                pk_c1 = _splat2(_rfl32(score_alpha_scale * _f32(-LOG2E)))
                pk_cx2 = _splat2(_rfl32(_f32(alpha)
                                        * arith.ArithValue(value_input_scale)))
                _nf = (arith.ArithValue(_f32(1.0))
                       / arith.ArithValue(inv_max))
                pk_k2 = _splat2(_rfl32(arith.ArithValue(_nf)
                                       / arith.ArithValue(value_input_scale)))
                _nn = arith.ArithValue(_nf) * _f32(-1.0)
                pk_nn2 = _splat2(_rfl32(_nn) if VALU_VF3S else _nn)
                if VALU_VF3P:
                    _ct = score_alpha_scale * _f32(-LOG2E)
                    _cx = _f32(alpha) * arith.ArithValue(value_input_scale)
                    _kk = arith.ArithValue(_nf) / arith.ArithValue(value_input_scale)
                    _p0 = vector.from_elements(V2F, [arith.unwrap(_ct), arith.unwrap(_cx)])
                    _p1 = vector.from_elements(V2F, [arith.unwrap(_kk), arith.unwrap(_nn)])
                    _s0 = rocdl.readfirstlane(V2F, arith.unwrap(_p0))
                    _s1 = rocdl.readfirstlane(V2F, arith.unwrap(_p1))
                    # gfx950 op_sel rule (opsel-fix): a splat of a pair's HIGH
                    # half (pk_cx2, pk_nn2, pk_ncxn2) is selected as the SGPR
                    # pair with op_sel = 1 (the low lane reads the high dword).
                    # On gfx950 that encoding is invalid in src1 of
                    # v_pk_mul/add_f32 and in src1/src2 of v_pk_fma_f32 (AMD
                    # LLVM 24's assembler rejects it, llvm/llvm-project#232016);
                    # it is valid in src0.  The LLVM bundled with FlyDSL 0.2.4
                    # (and 0.1.1) does not commute it there by itself, so every
                    # product below takes a high-half splat as its FIRST
                    # multiplicand: same IEEE product / fma (commutative in the
                    # two multiplicands), same instructions, op_sel:[1,0(,0)]
                    # -- the form a fixed LLVM emits from the old operand order.
                    # This removes every invalid form from the production
                    # vf3p+vf4 builds (F2 / F2e; checked by flydsl_pk_opsel_test).
                    # It does not steer how LLVM places other splats: a
                    # (non-production) vf3p build without vf4 still gets the
                    # VGPR-pair splat of pk_vis2 as a high-half fma ADDEND
                    # (op_sel:[0,0,1]); re-check the ISA before shipping it.
                    pk_c1 = vector.shuffle(_s0, _s0, [0, 0])
                    pk_cx2 = vector.shuffle(_s0, _s0, [1, 1])
                    pk_k2 = vector.shuffle(_s1, _s1, [0, 0])
                    pk_nn2 = vector.shuffle(_s1, _s1, [1, 1])
                    if VALU_VF4:
                        # vf4: vf3p's packing with the (N/s, .) pair's high
                        # half = -(alpha*s)*N; the om' addend sas comes from
                        # the VGPR-pair splat pk_sas2 (constant bus 1, as
                        # vf3p's s).  Reading both halves of ONE SGPR pair in
                        # the om' fma (iter51 first build) is rejected by
                        # LLVM: "VOP* instruction violates constant bus
                        # restriction".
                        _ncxn = (arith.ArithValue(_cx) * arith.ArithValue(_nf)
                                 * _f32(-1.0))
                        tsc_p2k = None
                        if VALU_TSC:
                            # tsc: om' constants times 2^-k (exact), so
                            # om'' = 2^-k*om' against silu'' = 2^k*silu'.
                            tsc_sa, tsc_p2k, _p2mk = _tsc_consts(_ct)
                            _ncxn = arith.ArithValue(_ncxn) * _p2mk
                        _p1 = vector.from_elements(
                            V2F, [arith.unwrap(_kk), arith.unwrap(_ncxn)])
                        _s1 = rocdl.readfirstlane(V2F, arith.unwrap(_p1))
                        pk_k2 = vector.shuffle(_s1, _s1, [0, 0])
                        pk_ncxn2 = vector.shuffle(_s1, _s1, [1, 1])
                        pk_sasc2 = pk_sas2
                        if VALU_TSC:
                            pk_sasc2 = _splat2(score_alpha_scale * _p2mk)
                        # The silu' cast operand 1/(mu2_ssc*sc) from the
                        # runtime alpha*s, uniform, once: one build serves
                        # every qkv_scale.
                        vf4_sop, vf4_undo = _vf4_cast_scale(
                            _cx, value_input_scale, tsc_p2k)
            if VALU_VF2:
                pk_nsas2 = _splat2(score_alpha_scale * _f32(-1.0))
            if VALU_VF and not VALU_VF2:
                pk_iq = _splat2(arith.ArithValue(inv_max) * score_alpha_scale)
                pk_iv = _splat2(arith.ArithValue(inv_max)
                                * arith.ArithValue(value_input_scale))
                pk_nvis = _splat2(arith.ArithValue(value_input_scale)
                                  * _f32(-1.0))
            if VALU_H16:
                V2H = T.vec(2, T.f16)
                pk_dvk = _splat2(arith.ArithValue(value_input_scale)
                                 * arith.ArithValue(inv_max)
                                 * _f32(mu2_dsc / 64.0))
                _ksh = arith.trunc_f(T.f16, arith.unwrap(
                    arith.ArithValue(inv_max) * _f32(mu2_ssc / 64.0)))
                h_ks2 = vector.from_elements(V2H, [arith.unwrap(_ksh)] * 2)
                h_one2 = arith.constant_vector(1.0, V2H)
        # R1 stage 2 runtime dO scale.  do_scale = [s_do, 1/s_do].
        inv_s_do = None
        if do_dyn:
            r_dsc = buffer_ops.create_buffer_resource(
                DO_SC, max_size=False,
                num_records_bytes=arith.unwrap(_i32(8)))
            inv_s_do = arith.ArithValue(buffer_ops.buffer_load(
                r_dsc, _i32(1), vec_width=1, dtype=T.f32))

        # Double-rate bf16 MFMA (K=32 per issue) for the score GEMMs.  Reached
        # through the raw ROCDL op: it consumes native v8bf16 A/B fragments and
        # keeps the same 16x16 f32 C layout as the half-rate 16x16x16 variant.
        def mfma2(a, b, acc):
            return rocdl_raw.mfma_f32_16x16x32_bf16(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                0,
                0,
                0,
            ).result

        # ---- fp8 MFMA for the qk gemm ------------------------------------
        # Both variants take the SAME (lane, element) -> k mapping the bf16
        # 16x16x32 op uses, so the operands are the raw e4m3 bytes in exactly
        # the order the widened bf16 fragments carried them.  No scale is
        # folded into the instruction: qkv_scale**2 is still applied once per
        # score in the epilogue (score_input_scale), which keeps the dequant
        # out of the E8M0-only scale field -- see the cvt_scalef32 finding.
        def mfma_qk_k32(a, b, acc):
            # a/b are i64 (8 packed e4m3).  vector.bitcast cannot produce a
            # scalar, so the dword pair goes through llvm.bitcast.
            return rocdl_raw.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4,
                llvm.BitcastOp(T.i64, arith.unwrap(a)).result,
                llvm.BitcastOp(T.i64, arith.unwrap(b)).result,
                arith.unwrap(acc),
                0,
                0,
                0,
            ).result

        z_i32 = arith.unwrap(arith.constant(0, type=T.i32))

        def mfma_qk_k128(a, b, acc):
            # a/b are vec<8xi32> (32 packed e4m3).  cbsz=0 / blgp=0 select e4m3
            # for both operands; scale=0 is E8M0 exponent bias 127, multiplier
            # 1.0, so the instruction's scale is a no-op.
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                0,
                0,
                0,
                z_i32,
                0,
                z_i32,
            ).result

        def mfma_qk_k128_tsc(a, b, acc):
            # valu tsc: the S = Q K^T MFMA with the uniform E8M0 A scale
            # 127+k and B scale 127 (a runtime 127, as do_blk's sc127: only a
            # literal 0 folds to the unscaled op), so it returns 2^k*qk.
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                0,
                0,
                0,
                arith.unwrap(tsc_sa),
                0,
                arith.unwrap(_i32(127)),
            ).result

        # ---- TRACK U1 helpers: fp8 packing + K=128 out-gemm ---------------
        # v_cvt_scalef32_pk8_{fp8,bf8}_{f32,bf16}: EIGHT elements -> eight fp8
        # bytes (two dwords) in ONE VALU op.  That is what makes the m-unroll
        # affordable at all -- the naive per-pair cvt would cost 4x as many
        # instructions as the mfma saving.  The scale operand is E8M0
        # (exponent-only), hence the power-of-two assert on mu2_*sc.
        MU2_BF8 = mu2_fmt == "e5m2"

        def _res(x):
            return x.result if hasattr(x, "result") else x

        # The pk8 (8-element) forms exist in the ROCDL dialect but are NOT
        # selectable in this toolchain (LLVM ERROR: Cannot select
        # llvm.amdgcn.cvt.scalef32.pk8.fp8.bf16), so the pairwise forms are
        # used: two elements per op, written into the lo or hi half of a
        # dword.  Four ops per four-byte piece pair instead of one.
        # ---- MEASURED SEMANTICS OF cvt_scalef32's SCALE OPERAND -------------
        # The DOWN-convert DIVIDES by the scale: dst = fp8(src / scale).  That
        # is the OCP-MX convention (the scale is the shared block exponent, and
        # the stored element is value/scale), and it is the opposite of the
        # up-convert (cvt_scalef32_pk_bf16_fp8, dst = fp8_to_bf16(src) * scale)
        # this file already used, which is where the sign error came from.
        #
        # Proof, not guesswork.  Two observations from the k128 arm pin it:
        #   * scale = 2**27 on silu (amax 2.47e-6) returned dk and dv
        #     IDENTICALLY ZERO -- 2.47e-6 / 2**27 = 1.8e-14, far under e4m3's
        #     smallest subnormal, so every element flushed.  Had the operand
        #     been multiplied, that arm would have been CORRECT, not zero.
        #   * feeding the reciprocals instead gave dk = -276.95 dB.  Under the
        #     divide reading, the cvt is then right but the epilogue undoes the
        #     scale the wrong way, leaving a constant factor (2**23 * 2**0)**2
        #     = 2**46; -20*log10(2**46) = -276.9476 dB.  Matches to 0.01 dB.
        # `mu2_*sc` therefore keeps its natural caller-facing meaning ("multiply
        # the operand by this before casting") and the reciprocal is formed
        # here.  Exact for a power of two, which is asserted at entry.
        def _cvt_sc(scale):
            return arith.unwrap(_f32(1.0 / scale))

        def _sat(x, lim, mode):
            # mode "hi": min(x, lim) -- silu >= -0.279/N can never reach -lim
            # for any K the launcher allows.  mode "both": min(max(x,-lim),lim)
            # for ds.  NaN-ignoring num-variants lower to v_min/v_max_f32
            # (or one fused v_med3_f32).
            from flydsl._mlir.dialects import arith as _ar
            mn = getattr(_ar, "minnumf", None) or (
                lambda a_, b_: _ar.MinNumFOp(a_, b_).result)
            mx = getattr(_ar, "maxnumf", None) or (
                lambda a_, b_: _ar.MaxNumFOp(a_, b_).result)
            hi = arith.unwrap(_f32(lim))
            if mode == "hi":
                return mn(x, hi)
            return mn(mx(x, arith.unwrap(_f32(-lim))), hi)

        def pk4_from_f32(v4f32, scale, sat=False, rt=None):
            """4 f32 -> one dword of 4 fp8, in element order.

            rt (valu vf4, the silu' casts only): the cast operand as a
            runtime uniform value; it replaces the compile-time 1/scale
            (see _vf4_cast_scale; vf4 needs msat, so sat is off)."""
            f = (rocdl_raw.cvt_scalef32_pk_bf8_f32 if MU2_BF8
                 else rocdl_raw.cvt_scalef32_pk_fp8_f32)
            if VALU_UNDEF:
                # Both halves are written below; the start value is dead.
                acc = llvm.UndefOp(T.vec(2, T.i16)).result
            else:
                acc = arith.unwrap(arith.constant_vector(0, T.vec(2, T.i16)))
            if VALU_H16:
                # iter22 f16act: v4f16 in the 2^-6 scaled domain.
                assert not MU2_BF8
                fh = rocdl_raw.cvt_scalef32_pk_fp8_f16
                sch = _cvt_sc(64.0)
                from flydsl._mlir.dialects import arith as _ar
                V2H_ = T.vec(2, T.f16)
                hi_ = arith.constant_vector(7.0, V2H_)
                lo_ = arith.constant_vector(-7.0, V2H_)
                for half in range_constexpr(2):
                    h2 = vector.shuffle(v4f32, v4f32, [2 * half, 2 * half + 1])
                    h2 = arith.unwrap(h2)
                    if sat:
                        if sat != "hi":
                            h2 = _ar.MaxNumFOp(h2, arith.unwrap(lo_)).result
                        h2 = _ar.MinNumFOp(h2, arith.unwrap(hi_)).result
                    acc = _res(fh(T.vec(2, T.i16), acc, h2, sch, half))
                return acc
            if rt is None:
                sc = _cvt_sc(scale)
            else:
                assert not MU2_BF8 and not sat
                sc = arith.unwrap(rt)
            lim = (57344.0 if MU2_BF8 else 448.0) / scale
            for half in range_constexpr(2):
                e0 = vector.extract(v4f32, static_position=[2 * half],
                                    dynamic_position=[])
                e1 = vector.extract(v4f32, static_position=[2 * half + 1],
                                    dynamic_position=[])
                if sat:
                    e0 = _sat(arith.unwrap(e0), lim, sat)
                    e1 = _sat(arith.unwrap(e1), lim, sat)
                acc = _res(f(T.vec(2, T.i16), acc, e0, e1, sc, half))
            return acc

        def pk4_from_bf16(v4bf16, scale):
            """4 bf16 -> one dword of 4 fp8, in element order."""
            f = (rocdl_raw.cvt_scalef32_pk_bf8_bf16 if MU2_BF8
                 else rocdl_raw.cvt_scalef32_pk_fp8_bf16)
            acc = arith.unwrap(arith.constant_vector(0, T.vec(2, T.i16)))
            sc = _cvt_sc(scale)
            for half in range_constexpr(2):
                acc = _res(f(
                    T.vec(2, T.i16), acc,
                    vector.shuffle(v4bf16, v4bf16, [2 * half, 2 * half + 1]),
                    sc, half))
            return acc

        # Pieces stay as halfword vectors all the way to the mfma: v2i16 (one
        # dword, 4 fp8 = one 16-row m sub-tile) -> v4i16 (a K=32 pair) ->
        # v8i16 (one BM=64 tile's half of the operand) -> v16i16 for the pair,
        # bitcast to the v8i32 the mfma_scale op takes.
        def pk8_from_f32(frag_a, frag_b, scale):
            return vector.shuffle(pk4_from_f32(frag_a, scale),
                                  pk4_from_f32(frag_b, scale), [0, 1, 2, 3])

        def pk8_from_bf16(frag_a, frag_b, scale):
            return vector.shuffle(pk4_from_bf16(frag_a, scale),
                                  pk4_from_bf16(frag_b, scale), [0, 1, 2, 3])

        # cbsz/blgp select the operand format class: 0 = e4m3, 1 = e5m2.
        MU2_FSEL = 1 if MU2_BF8 else 0

        def mfma_out_k128(a, b, acc):
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                MU2_FSEL,
                MU2_FSEL,
                0,
                z_i32,
                0,
                z_i32,
            ).result

        # do_blk: RUNTIME E8M0 scale operands (byte 0 of an i32, opsel 0).
        # 127 = x1 on the unscaled side; it must be a runtime value -- see the
        # do_blk flag comment (P0: runtime 0 = 2**-127).
        sc127 = arith.unwrap(_i32(127)) if do_blk else None

        def mfma_k128_sc(a, b, acc, sa, sb):
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4,
                arith.unwrap(a),
                arith.unwrap(b),
                arith.unwrap(acc),
                MU2_FSEL,
                MU2_FSEL,
                0,
                arith.unwrap(sa),
                0,
                arith.unwrap(sb),
            ).result

        # ---- R1 f8b: ds_read_b64_tr_b8 --------------------------------------
        # MEASURED on gfx950 (probes/iter16/probe_tr8.hip): within a 16-lane
        # group, lane j receives byte (j % 8) of the 8-byte runs addressed by
        # source lanes 2*e + j//8, e = 0..7, as its byte e.  So source lane j
        # names row e = j//2 for the receiver half k = j%2, and the receiver
        # half k reads 8 consecutive COLUMNS starting at that row address.
        i8_t = ir.IntegerType.get_signless(8)
        _tr8_op = getattr(rocdl_raw, "ds_read_tr8_b64_", None)

        def tr8(lds, byte_off):
            ptr = _lds_gep(lds, byte_off, i8_t)
            if _tr8_op is not None:
                return _res(_tr8_op(v2i32, ptr))
            addr = llvm.PtrToIntOp(T.i32, ptr).result
            return llvm.inline_asm(
                v2i32, [addr], "ds_read_b64_tr_b8 $0, $1", "=v,v",
                has_side_effects=False)

        # i64: seq_start * row-stride is a whole-tensor element offset and overflows i32
        # at production sizes (see _shifted_rsrc).  Only this base shift needs
        # the wider type -- every offset used inside the loop is relative to it
        # and bounded by one sequence.
        head_off = arith.ArithValue(_to_i64(head)) * _i64(D)
        tok_off_in = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_IN) + head_off
        )
        tok_off_q = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_Q) + head_off
        )
        tok_off_out = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(RS_OUT) + head_off
        )
        # dO always carries the packed H*D stride.
        tok_off_do = (
            arith.ArithValue(_to_i64(seq_start)) * _i64(HD) + head_off
        )
        # Q and dO are the only tensors read past the end of the sequence (the m
        # loop runs in whole BM tiles and prefetches one tile beyond the last).
        # Stopping their descriptors at the sequence end makes those reads return
        # zero, and a zero q/dO row contributes exactly zero to both outputs:
        # silu(0) = 0, and dp = 0 forces ds = 0.  So the bounds half of the
        # in-loop mask is unnecessary -- see the m loop below.
        q_bytes = arith.ArithValue(seq_len) * _i32(RS_Q * Q_BYTES)
        do_bytes = arith.ArithValue(seq_len) * _i32(HD * DO_BYTES)
        rQ = _shifted_rsrc(
            Q, tok_off_q, Q_BYTES, num_bytes=q_bytes
        )
        rK = _shifted_rsrc(K, tok_off_in, INPUT_BYTES)
        rV = _shifted_rsrc(V, tok_off_in, INPUT_BYTES)
        rDO = _shifted_rsrc(DO, tok_off_do, DO_BYTES, num_bytes=do_bytes)
        if do_blk:
            # E8 [T, H] uint8, based at (seq_start, head) and stopped at the
            # sequence end: byte offset p*H is token p's group byte, and a
            # token past seq_len reads 0 (2**-127) -- only ever applied to
            # all-zero data (its dO row, hence ds row, is zero).
            rE8 = _shifted_rsrc(
                DO_SC, arith.ArithValue(_to_i64(seq_start)) * _i64(H)
                + _to_i64(head), 1,
                num_bytes=arith.ArithValue(seq_len) * _i32(H))
        if VALU_NOBR:
            out_bytes = arith.ArithValue(seq_len) * _i32(RS_OUT * 2)
            rDK = _shifted_rsrc(DK, tok_off_out, 2, num_bytes=out_bytes)
            rDV = _shifted_rsrc(DV, tok_off_out, 2, num_bytes=out_bytes)
        else:
            rDK = _shifted_rsrc(DK, tok_off_out, 2)
            rDV = _shifted_rsrc(DV, tok_off_out, 2)
        if ABL_DQ:
            # The probe has no dq tensor of its own; it scribbles f32 atomics
            # over DK, bounded by DK's real extent so nothing lands outside the
            # allocation.  Same instruction stream, garbage result, by design.
            rDQ = _shifted_rsrc(
                DK,
                tok_off_out,
                2,
                num_bytes=arith.ArithValue(seq_len) * _i32(RS_OUT * 2),
            )
        if fused:
            # num_records stops the descriptor at this sequence's last token, so
            # the atomics for m rows past seq_len are dropped in hardware and
            # the flush needs no bounds branch.  Those rows carry exactly zero
            # anyway (a zero dO row forces dp = 0 hence ds = 0), so this only
            # avoids touching a neighbour's memory, it does not change values.
            rDQ = _shifted_rsrc(
                DQ, tok_off_out, DQ_BYTES,
                num_bytes=arith.ArithValue(seq_len)
                * _i32(RS_OUT * DQ_BYTES))
        if DQSPLIT:
            # Workspace (passed as DQ): header int32 prefix[B+1] of per-batch
            # tile counts, padded to WS_HDR bytes, then 8 KiB e4m3 ds^T tiles.
            # (b, h) owns H-major nb*(nb+1) tiles, nb = ceil(L/128); n-block j
            # owns tiles t = 2j .. 2nb-1 at slot 2*j*nb - j*(j-1) + t - 2j.
            # Must match hstu_dq_split.py.
            if ws_pre_v is not None:
                ws_pre = arith.ArithValue(ws_pre_v)
            else:
                r_hdr = _shifted_rsrc(DQ, _to_i64(_i32(0)), 1)
                ws_pre = arith.ArithValue(buffer_ops.buffer_load(
                    r_hdr, batch_idx, vec_width=1, dtype=T.i32))
            if VALU_RFL:
                # batch_idx is workgroup-uniform: same value in every lane,
                # only the register class (SGPR) changes, and with it rWS.
                ws_pre = arith.ArithValue(rocdl.readfirstlane(
                    T.i32, arith.unwrap(ws_pre)))
            ws_nb = (arith.ArithValue(seq_len) + _i32(127)) // _i32(128)
            ws_hdr = ((arith.ArithValue(nbatch) + _i32(1)) * _i32(4)
                      + _i32(255)) // _i32(256) * _i32(256)
            ws_tiles = ws_pre + arith.ArithValue(head) * ws_nb * (ws_nb + _i32(1))
            ws_j = n0 // _i32(128)
            # Bounded to this (b, h)'s nb*(nb+1) tiles: a stray offset is
            # dropped by the buffer unit instead of landing in a neighbour.
            rWS = _shifted_rsrc(
                DQ, arith.ArithValue(_to_i64(ws_tiles)) * _i64(8192)
                + _to_i64(ws_hdr), 1,
                num_bytes=ws_nb * (ws_nb + _i32(1)) * _i32(8192))
            ws_jbase = (_i32(2) * ws_j * ws_nb - ws_j * (ws_j - _i32(1))
                        - _i32(2) * ws_j)

        base = allocator.get_base()
        q_rm = SmemPtr(base, off_q_rm, T.bf16, shape=(RM_HW,)).get()
        o_rm = SmemPtr(base, off_o_rm, T.bf16, shape=(RM_HW,)).get()
        if FP8_QK:
            # Typed as i32 so every offset below is a dword index: the tile is
            # only ever touched 8 or 16 bytes at a time and never element-wise,
            # and a dword view keeps the address arithmetic in the same units
            # as the buffer_loads that fill it.
            q8_lds = SmemPtr(
                base, off_q8, T.i32, shape=((D // 32) * Q8BS // 4,)
            ).get()
        Q8BS_DW = Q8BS // 4
        if F8B:
            o8_lds = SmemPtr(
                base, off_o8, T.i32, shape=((D // 32) * Q8BS // 4,)
            ).get()
            kt8_lds = SmemPtr(base, off_kt, T.i32,
                              shape=(KT8_BYTES // 4,)).get()
        if MSPLIT:
            # The whole 8-tile block as one dword view; tiles are picked by a
            # (dynamic parity, wave group) dword offset.
            ms_t8 = SmemPtr(base, off_q8, T.i32,
                            shape=(8 * (D // 32) * Q8BS // 4,)).get()
        if PIPE:
            q8b_lds = SmemPtr(
                base, off_q8b, T.i32, shape=((D // 32) * Q8BS // 4,)).get()
            o8b_lds = SmemPtr(
                base, off_o8b, T.i32, shape=((D // 32) * Q8BS // 4,)).get()
        if not tr16:
            q_tr = SmemPtr(base, off_q_tr, T.bf16, shape=(D * ST,)).get()
            o_tr = SmemPtr(base, off_o_tr, T.bf16, shape=(D * ST,)).get()
        if fused:
            kt_lds = SmemPtr(base, off_kt, T.bf16, shape=(BN * D,)).get()
            ds_lds = SmemPtr(base, off_kt, T.bf16, shape=(BM * BN,)).get()

        lane = tid % _i32(64)
        wave = tid // _i32(64)
        grp = None
        if MSPLIT:
            # n-owner index (what every n-indexed address below uses) and the
            # wave group, made scalar so the group branch is uniform.
            grp = arith.ArithValue(rocdl.readfirstlane(
                T.i32, arith.unwrap(wave // _i32(WAVES))))
            wave = wave % _i32(WAVES)
        l16 = lane % _i32(16)
        l4 = lane // _i32(16)
        seq_last = seq_len - _i32(1)

        # Opaque zero for the ldsa2/ldsb2 footprint ablations: seq_len is a
        # runtime kernel value and is never 0 (the launcher skips empty
        # sequences), so this selects 0 every time, but LLVM cannot prove it
        # and therefore cannot constant-fold `oz * c` or CSE two reads whose
        # step constants differ.
        def _oz_mul(c):
            if c == 0:
                return _i32(0)
            nz = arith.cmpi(arith.CmpIPredicate.ne, seq_len, _i32(0))
            return arith.select(nz, _i32(0), _i32(c))

        # ---- K / V fragments, live across the whole m loop -------------------
        # Wave `wave` owns n rows [n0 + wave*16*NT, +16*NT), as NT 16-row tiles.
        kf = []
        vf = []
        kf8 = []
        vf8 = []  # F8B only: one v8i32 per n tile
        for nt in range_constexpr(NT):
            n_row = n0 + wave * _i32(16 * NT) + _i32(16 * nt) + l16
            n_row_safe = arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, n_row, seq_len), n_row, seq_last
            )
            kv_base = (
                arith.ArithValue(n_row_safe) * _i32(RS_IN) + l4 * _i32(8)
            )
            kf.append([])
            vf.append([])
            kf8.append([])
            for ks in range_constexpr(KSTEP):
                o = kv_base + _i32(32 * ks)
                # K feeds the qk gemm and nothing else (dq's B operands come
                # from the separately staged kt_lds), so under FP8_QK it never
                # needs widening: the cvt chain disappears and the fragment
                # halves from 4 to 2 VGPRs.
                if not FP8_QK:
                    kf[nt].append(load_qkv(rK, o))
                if not F8B:
                    vf[nt].append(load_qkv(rV, o))
            if FP8_QK in ("k32", "k32w"):
                # Same 8-bytes-at-d = 8*l4 + 32*ks slicing the bf16 op used.
                for ks in range_constexpr(KSTEP):
                    kf8[nt].append(load_raw8(rK, kv_base + _i32(32 * ks)))
            elif FP8_QK == "k128":
                # K = 128 wants lane l4 to hold d in [32*l4, 32*l4 + 32): one
                # 32 B run, which is two dwordx4 loads (a 256-bit buffer load
                # is not selectable on gfx950).
                kb = (arith.ArithValue(n_row_safe) * _i32(RS_IN)
                      + l4 * _i32(32)) // _i32(4)
                kf8[nt].append(vector.shuffle(
                    buffer_ops.buffer_load(rK, kb, vec_width=4, dtype=T.i32),
                    buffer_ops.buffer_load(rK, kb + _i32(4), vec_width=4,
                                           dtype=T.i32),
                    [0, 1, 2, 3, 4, 5, 6, 7]))
                if F8B:
                    # dp's B operand: V rows n, k = d, same 32 B run as K.
                    vf8.append(vector.shuffle(
                        buffer_ops.buffer_load(rV, kb, vec_width=4,
                                               dtype=T.i32),
                        buffer_ops.buffer_load(rV, kb + _i32(4), vec_width=4,
                                               dtype=T.i32),
                        [0, 1, 2, 3, 4, 5, 6, 7]))

        st_m = (tid // _i32(16)) * _i32(LROWS)
        st_d = (tid % _i32(16)) * _i32(8)

        def _rs_rsrc(tensor, tok_off, rs_el, el_bytes, nbytes, m):
            """MSPLIT_SCHED rs: descriptor based at row m of this sequence,
            num_records = max(nbytes - m*rs_el*el_bytes, 0)."""
            mb = arith.ArithValue(m) * _i32(rs_el * el_bytes)
            rem = arith.ArithValue(nbytes) - mb
            rem = arith.select(arith.cmpi(arith.CmpIPredicate.sgt, rem,
                                          _i32(0)), rem, _i32(0))
            return _shifted_rsrc(
                tensor, arith.ArithValue(_to_i64(tok_off))
                + arith.ArithValue(_to_i64(m)) * _i64(rs_el), el_bytes,
                num_bytes=rem)

        def _rs2_rsrc(tensor, tok_off, rs_el, el_bytes, nbytes, m):
            """rs2: base at row m via an i32 byte delta (fits: m*rs < 2^31)."""
            mb = arith.ArithValue(m) * _i32(rs_el * el_bytes)
            rem = arith.ArithValue(nbytes) - mb
            rem = arith.select(arith.cmpi(arith.CmpIPredicate.sgt, rem,
                                          _i32(0)), rem, _i32(0))
            return _shifted_rsrc(
                tensor, arith.ArithValue(_to_i64(tok_off)) * _i64(el_bytes)
                + arith.extui(T.i64, arith.unwrap(mb)), 1, num_bytes=rem)

        def load_pair_rs2(mA):
            """rs2: load_tiles(mA) + load_tiles(mA + BM) through one
            descriptor per tensor based at row mA."""
            rq = _rs2_rsrc(Q, tok_off_q, RS_Q, Q_BYTES, q_bytes, mA)
            rd = _rs2_rsrc(DO, tok_off_do, HD, DO_BYTES, do_bytes, mA)
            re = _rs2_rsrc(DO_SC, arith.ArithValue(_to_i64(seq_start))
                           * _i64(H) + _to_i64(head), H, 1,
                           arith.ArithValue(seq_len) * _i32(H), mA)
            res = []
            for tb in (0, BM):
                out = []
                for src, src_rs in ((rq, RS_Q), (rd, HD)):
                    for r in range_constexpr(LROWS):
                        m = arith.ArithValue(st_m + _i32(r + tb))
                        out.append(load_raw8(src, m * _i32(src_rs) + st_d))
                for tok in ((l4 % _i32(2)) * _i32(8) + _i32(tb),
                            (l16 // _i32(8)) * _i32(8) + _i32(tb)):
                    b = buffer_ops.buffer_load(
                        re, arith.ArithValue(tok) * _i32(H),
                        vec_width=1, dtype=T.i8)
                    out.append(arith.extui(T.i32, b))
                res.append(out)
            return res

        def load_tiles(m0, rs=False):
            """Issue the global loads for the Q and DO tiles at row offset m0.

            No clamp: rows at or past seq_len fall outside the descriptor's
            num_records and read back as zero, which is the value the math wants.
            """
            out = []
            if F8B and rs:
                assert do_blk
                srcs = ((_rs_rsrc(Q, tok_off_q, RS_Q, Q_BYTES, q_bytes, m0),
                         RS_Q),
                        (_rs_rsrc(DO, tok_off_do, HD, DO_BYTES, do_bytes, m0),
                         HD))
                for src, src_rs in srcs:
                    for r in range_constexpr(LROWS):
                        m = arith.ArithValue(st_m + _i32(r))
                        out.append(load_raw8(src, m * _i32(src_rs) + st_d))
                rE8m = _rs_rsrc(DO_SC, arith.ArithValue(_to_i64(seq_start))
                                * _i64(H) + _to_i64(head), H, 1,
                                arith.ArithValue(seq_len) * _i32(H), m0)
                for tok in ((l4 % _i32(2)) * _i32(8),
                            (l16 // _i32(8)) * _i32(8)):
                    b = buffer_ops.buffer_load(
                        rE8m, arith.ArithValue(tok) * _i32(H),
                        vec_width=1, dtype=T.i8)
                    out.append(arith.extui(T.i32, b))
                return out
            if F8B:
                # Q and dO both arrive as e4m3 and are staged raw; nothing is
                # widened anywhere in the loop.
                for src, src_rs in ((rQ, RS_Q), (rDO, HD)):
                    for r in range_constexpr(LROWS):
                        m = arith.ArithValue(m0 + st_m + _i32(r))
                        out.append(load_raw8(src, m * _i32(src_rs) + st_d))
                if do_blk:
                    # Two group bytes per lane for tile m0, prefetched with the
                    # tile (a load inside one_iter would queue behind the next
                    # tile's prefetch on vmcnt):
                    #   [8] out-gemm K-block byte: group (m0, l4 % 2), i.e. the
                    #       group's first token m0 + 8*(l4 % 2);
                    #   [9] dq row byte: row l16 -> group (m0, l16 // 8).
                    for tok in (m0 + (l4 % _i32(2)) * _i32(8),
                                m0 + (l16 // _i32(8)) * _i32(8)):
                        b = buffer_ops.buffer_load(
                            rE8, arith.ArithValue(tok) * _i32(H),
                            vec_width=1, dtype=T.i8)
                        out.append(arith.extui(T.i32, b))
                return out
            raws = []  # FP8_QK only: Q's un-widened bytes, appended after dO
            for source_index, (src, src_rs) in enumerate(
                ((rQ, RS_Q), (rDO, HD))
            ):
                for r in range_constexpr(4):
                    m = arith.ArithValue(m0 + st_m + _i32(r))
                    off = m * _i32(src_rs) + st_d
                    if input_fp8 and source_index == 0:
                        if FP8_QK:
                            # One load, two consumers.  The raw dwords go to the
                            # fp8 tile for qk; the widened copy still has to be
                            # produced because the dk gemm reads Q back
                            # transposed as a bf16 B-operand, and converting
                            # there instead would quadruple the cvt count (the
                            # tile is read ~8x more than it is written).
                            # Kept as a flat tail rather than a tuple: `staged`
                            # is carried through scf.for as iter_args.
                            raw = load_raw8(src, off)
                            raws.append(raw)
                            out.append(_widen_raw8(raw))
                        else:
                            out.append(
                                load_qkv(src, off, source_fp8=not q_bf16)
                            )
                    else:
                        out.append(
                            buffer_ops.buffer_load(
                                src, off, vec_width=8, dtype=T.bf16
                            )
                        )
            return out + raws

        # d block index and offset within it for this thread's 8-wide store.
        st_db = (tid % _i32(16)) // _i32(2)
        st_din = ((tid % _i32(16)) % _i32(2)) * _i32(8)

        # fp8 Q tile addressing: d_block = st_d // 32 at stride Q8BS_DW dwords,
        # m-row within it at stride 8 dwords (32 B), byte offset (st_d % 32)
        # inside the row -- always a multiple of 8 B because st_d is.
        st8_db = st_d // _i32(32)
        st8_in = (st_d % _i32(32)) // _i32(4)

        def write_tiles(vecs, bufs=None):
            if F8B:
                for t, lds8 in enumerate(bufs or (q8_lds, o8_lds)):
                    for r in range_constexpr(4):
                        lo = (arith.ArithValue(st8_db) * _i32(Q8BS_DW)
                              + (arith.ArithValue(st_m) + _i32(r)) * _i32(8)
                              + st8_in)
                        vector.store(vecs[4 * t + r], lds8, [_idx(lo)],
                                     alignment=8)
                return
            if FP8_QK:
                # vecs[8:12] are Q's un-widened bytes; publish them to the fp8
                # tile, then fall through with the leading 8 bf16 tiles.
                q_raw = list(vecs[8:12])
                for r in range_constexpr(4):
                    lo = (arith.ArithValue(st8_db) * _i32(Q8BS_DW)
                          + (arith.ArithValue(st_m) + _i32(r)) * _i32(8)
                          + st8_in)
                    vector.store(q_raw[r], q8_lds, [_idx(lo)], alignment=8)
            if tr16:
                for t, lds_rm in enumerate((q_rm, o_rm)):
                    rows = vecs[4 * t : 4 * t + 4]
                    for r in range_constexpr(4):
                        # d_block=st_db at stride DBS (padded), m-row within it
                        # at stride 16, d%16 half at st_din.
                        lo = (arith.ArithValue(st_db) * _i32(DBS)
                              + (arith.ArithValue(st_m) + _i32(r)) * _i32(16)
                              + st_din)
                        vector.store(rows[r], lds_rm, [_idx(lo)], alignment=16)
                return
            for t, (lds_rm, lds_tr) in enumerate(((q_rm, q_tr), (o_rm, o_tr))):
                rows = vecs[4 * t : 4 * t + 4]
                for r in range_constexpr(4):
                    lo = (st_m + _i32(r)) * _i32(SQ) + st_d
                    vector.store(rows[r], lds_rm, [_idx(lo)], alignment=rm_align)
                if ABL_TR:
                    continue
                for e in range_constexpr(8):
                    elems = [
                        vector.extract(rows[r], static_position=[e], dynamic_position=[])
                        for r in range_constexpr(4)
                    ]
                    vector.store(
                        vector.from_elements(v4bf, elems),
                        lds_tr,
                        [_idx((st_d + _i32(e)) * _i32(ST) + st_m)],
                        alignment=8,
                    )

        l4h = l4 // _i32(2)  # d block within a 32-wide k slice
        l4l = (l4 % _i32(2)) * _i32(8)  # offset inside that 16-wide d block
        # ds_read_b64_tr_b16, measured on gfx950 (isa_diff/probe_tr2.hip):
        # for lane 16*g + j, write k = j/4 and s = j%4.  Element e of that lane
        # is the halfword at (address given by lane 16*g + 4*e + k) + s.  So each
        # 16-lane group runs four independent 4x4 transposes, and lane j supplies
        # the address for row e = j/4 of sub-group k = j%4.
        l16q = l16 // _i32(4)
        l16r = (l16 % _i32(4)) * _i32(4)

        def tr_read(lds, row_base, inner, pad_off=0, oz=False):
            """One ds_read_b64_tr_b16 out of a tile in the tr16 layout.

            The layout is halfword(r, c) = ((c/16)*ROWS + r)*16 + c%16, and
            `row_base` selects the (c/16) block as (c/16)*ROWS.  The read hands
            lane j the four r values 4*l4 + 16*inner + 0..3 taken at column
            c = 16*(c/16) + j: a transposed 4-wide slice with the lane index on
            the c axis.  Three tiles use it -- Q and dO with (r, c) = (m, d),
            K with (n, d), and the ds exchange with (n, m).
            """
            cst = _oz_mul if oz else _i32
            if isinstance(row_base, int):
                row_base = cst(row_base)
            off = (arith.ArithValue(row_base) + l4 * _i32(4) + cst(16 * inner)
                   + arith.ArithValue(l16q)) * _i32(16) + l16r
            # pad_off (halfwords) rotates a padded d_block's base; row_base is
            # d_block*ROWS so the padding cannot fold into the *16 above.
            if pad_off:
                off = arith.ArithValue(off) + cst(pad_off)
            return rocdl_raw.ds_read_tr16_b64_(v4bf, _lds_gep(lds, off, T.bf16))

        def tr_pair(lds, row_base, inner2):
            """K=32 fragment: two tr reads whose r values differ by 16.

            The mfma only requires A and B to agree on which logical k each
            (lane, element) slot carries, not that the k values be contiguous,
            so the pair can be concatenated as it comes out.
            """
            return vector.shuffle(tr_read(lds, row_base, 2 * inner2),
                                  tr_read(lds, row_base, 2 * inner2 + 1), CAT8)

        def a_frag(lds, mstep, kstep):
            if ABL_LDSA:
                mstep = kstep = 0
            cst = _oz_mul if ABL_LDSA2 else _i32
            if tr16:
                # d = l4*8 + 32*kstep, so d/16 = l4/2 + 2*kstep and d%16 = 8*(l4%2)
                off = ((arith.ArithValue(l4h) + cst(2 * kstep)) * _i32(DBS)
                       + (l16 + cst(16 * mstep)) * _i32(16) + l4l)
                return vector.load_op(v8bf, lds, [_idx(off)])
            off = (l16 + cst(16 * mstep)) * _i32(SQ) + l4 * _i32(8) + cst(32 * kstep)
            return vector.load_op(v8bf, lds, [_idx(off)])

        def a_frag8(mstep, kstep, lds8=None):
            """Q A-operand as raw e4m3 out of the fp8 tile.

            k32:  8 B at d = 8*l4 + 32*kstep -> block kstep, byte 8*l4 in the
                  row.  One ds_read_b64, half the bytes of the bf16 b128 it
                  replaces.
            k128: all 32 B of block l4 at this m, as two b128 -- four b128
                  reads and four mfma collapse to two reads and one mfma.
            """
            row = (l16 + _i32(16 * mstep)) * _i32(8)
            if FP8_QK == "k32":
                off = (_i32(kstep * Q8BS_DW) + row + l4 * _i32(2))
                return vector.load_op(v2i32, q8_lds, [_idx(off)])
            if FP8_QK == "k32w":
                # k128's read, k32's mfma: pull the full 32 B once and hand out
                # the four K=32 slices.  d = 8*l4 + 32*ks lives in block ks at
                # byte 8*l4, so the four blocks a lane needs are Q8BS_DW apart
                # and each slice is dwords [2*l4, 2*l4+2) of its block -- which
                # is NOT what one 32 B run holds, so the probe reads the two
                # b128 that cover blocks 0..1 and 2..3 of THIS lane's l4 slot.
                off = arith.ArithValue(l4) * _i32(Q8BS_DW) + row
                wide = vector.shuffle(
                    vector.load_op(v4i32, q8_lds, [_idx(off)]),
                    vector.load_op(v4i32, q8_lds,
                                   [_idx(arith.ArithValue(off) + _i32(4))]),
                    [0, 1, 2, 3, 4, 5, 6, 7])
                return vector.shuffle(wide, wide, [2 * kstep, 2 * kstep + 1])
            src8 = q8_lds if lds8 is None else lds8
            off = arith.ArithValue(l4) * _i32(Q8BS_DW) + row
            return vector.shuffle(
                vector.load_op(v4i32, src8, [_idx(off)]),
                vector.load_op(v4i32, src8, [_idx(arith.ArithValue(off)
                                                  + _i32(4))]),
                [0, 1, 2, 3, 4, 5, 6, 7])

        # ---- F8B transposed operand addressing (see tr8 above) ------------
        # Source lane j = l16: row slot e = j//2, receiver half k = j%2.
        t8_e = l16 // _i32(2)
        t8_k8 = (l16 % _i32(2)) * _i32(8)
        # Out-gemm B (dO^T / q^T) for an mu2 HALF: element e is
        # m = 16*(2r + e//4) + 4*l4 + e%4 -- exactly the m order the silu / ds
        # A pieces carry (pk8 of m-subtiles 2r and 2r+1).
        t8_m = ((t8_e // _i32(4)) * _i32(16) + l4 * _i32(4)
                + t8_e % _i32(4))

        def b8_tr(lds8, r, ds):
            off = (_i32((ds // 2) * Q8BS + 16 * (ds % 2) + 32 * 32 * r)
                   + arith.ArithValue(t8_m) * _i32(32) + t8_k8)
            return vector.bitcast(T.vec(4, T.i16), tr8(lds8, off))

        # dq operands: k byte b of lane group l4 is n = 32*l4 + b.
        t8_n = l4 * _i32(32) + t8_e

        def dq8_a(ms):
            """ds as dq's A operand: row m = 16*ms + l16, 32 n per lane."""
            parts = [tr8(kt8_lds, (arith.ArithValue(t8_n) + _i32(8 * r))
                         * _i32(DS8R) + t8_k8 + _i32(16 * ms))
                     for r in range_constexpr(4)]
            return vector.shuffle(vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                                  vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                                  list(range(8)))

        def b_frag_tr(lds, mtile, dstep):
            """4 consecutive m at one d, straight out of the row-major tile.

            ds_read_b64_tr_b16 works on 16-lane groups: lanes 4e of a group supply
            the four row base addresses, and lane j gets element e = row_e + j.
            With the tr16 layout row_e is (m = 4*(lane/16) + 16*mtile + e) at d
            block `dstep`, so lane j receives m = q0..q0+3 at d = 16*dstep + j --
            exactly the transposed B operand, with no second LDS copy.
            """
            if ABL_LDSB:
                mtile = dstep = 0
            # q_rm/o_rm carry tr_pad halfwords of padding per d_block; add the
            # d_block's share (dstep*tr_pad) so the read matches write_tiles.
            return tr_read(lds, dstep * BM, mtile, pad_off=dstep * tr_pad,
                           oz=ABL_LDSB2)

        def b_frag(lds, mtile, dstep):
            cst = _oz_mul if ABL_LDSB2 else _i32
            off = (l16 + cst(16 * dstep)) * _i32(ST) + l4 * _i32(4) + cst(16 * mtile)
            return vector.bitcast(v4i16, vector.load_op(v4bf, lds, [_idx(off)]))

        def b_frag_bf(lds, mtile, dstep):
            # Same transposed B-operand slice as b_frag but left as native bf16
            # so two consecutive m-tiles can be concatenated into a K=32 fragment.
            if ABL_LDSB:
                mtile = dstep = 0
            cst = _oz_mul if ABL_LDSB2 else _i32
            off = (l16 + cst(16 * dstep)) * _i32(ST) + l4 * _i32(4) + cst(16 * mtile)
            return vector.load_op(v4bf, lds, [_idx(off)])

        n_idx_score = [
            n0 + wave * _i32(16 * NT) + _i32(16 * nt) + l16 for nt in range(NT)
        ]

        def act_tile(m0, qk, dp, masked, hook=None):
            """silu / ds of one m tile from its qk / dp accumulators.
            hook(nt, silu32[nt], ds32[nt]) runs after each n tile (ms1tune)."""
            silu_f = []
            ds_f = []
            silu32 = []  # MU2 only: the f32 fragments, before any narrowing
            ds32 = []
            ds_raw = []  # same fragments as ds_f, kept as bf16 for the LDS store
            for nt in range_constexpr(NT):
                silu_f.append([])
                ds_f.append([])
                silu32.append([])
                ds32.append([])
                ds_raw.append([])
                ilp_silu = ilp_ds = None
                if VALU_ILP:
                    # valu "ilp": the reuse_pk chain below, op for op, issued
                    # stage-major over all 2*MSTEP element pairs of this row.
                    def _ex2i(v2, j):
                        return arith.unwrap(vector.extract(
                            v2, static_position=[j], dynamic_position=[]))
                    prs = [(ms, h2) for ms in range(MSTEP) for h2 in range(2)]
                    sels = [[2 * h2, 2 * h2 + 1] for (_, h2) in prs]
                    if VALU_VF or VALU_H16 or VALU_VF3:
                        q2s = [vector.shuffle(qk[nt][ms], qk[nt][ms], sel)
                               for (ms, _), sel in zip(prs, sels)]
                        if VALU_TSC:
                            # tsc: q2 holds tau = 2^k*qk = -t exactly (the S
                            # MFMA's E8M0 scale): no t multiply, exp2 reads
                            # -tau (VOP3 neg source modifier) below.
                            t2s = None
                        else:
                            t2s = [arith.mulf(q2, pk_c1) for q2 in q2s]
                        if VALU_H16:
                            x2s = [arith.mulf(q2, pk_sas2) for q2 in q2s]
                        if VALU_VF3 and not VALU_VF4:
                            # high-half splat first (gfx950 op_sel rule, see
                            # pk_cx2): v_pk_mul_f32 op_sel:[1,0], not [0,1]
                            x2s = [arith.mulf(pk_cx2, q2) for q2 in q2s]
                    else:
                        x2s = [arith.mulf(vector.shuffle(qk[nt][ms],
                                                         qk[nt][ms], sel),
                                          pk_sas2)
                               for (ms, _), sel in zip(prs, sels)]
                        t2s = [arith.mulf(x2, pk_nl2e2) for x2 in x2s]
                    if VALU_TSC:
                        e2s = [vector.from_elements(V2F, [
                            rocdl.exp2(T.f32, arith.negf(_ex2i(q2, j)))
                            for j in range_constexpr(2)]) for q2 in q2s]
                    else:
                        e2s = [vector.from_elements(V2F, [
                            rocdl.exp2(T.f32, _ex2i(t2, j))
                            for j in range_constexpr(2)]) for t2 in t2s]
                    if VALU_VF3:
                        d2s = [llvm.intr_fma(arith.unwrap(e2),
                                             arith.unwrap(pk_k2),
                                             arith.unwrap(pk_k2))
                               for e2 in e2s]
                    else:
                        d2s = [arith.addf(e2, pk_one2) for e2 in e2s]
                    sig2s = [vector.from_elements(V2F, [
                        rocdl.rcp(T.f32, _ex2i(d2, j))
                        for j in range_constexpr(2)]) for d2 in d2s]
                    if VALU_VF or VALU_H16 or VALU_VF3:
                        p2s = [vector.shuffle(dp[nt][ms], dp[nt][ms], sel)
                               for (ms, _), sel in zip(prs, sels)]
                    if VALU_H16:
                        def _h(v):
                            return arith.trunc_f(V2H, arith.unwrap(v))
                        dk2s = [arith.mulf(p2, pk_dvk) for p2 in p2s]
                        xh = [_h(x2) for x2 in x2s]
                        sh = [_h(sg) for sg in sig2s]
                        dh = [_h(v) for v in dk2s]
                        u2s = [arith.mulf(a_, b_) for a_, b_ in zip(xh, sh)]
                        silu2s = [arith.mulf(u, h_ks2) for u in u2s]
                        oms2s = [arith.subf(h_one2, sg) for sg in sh]
                        g2s = [llvm.intr_fma(arith.unwrap(u), arith.unwrap(om),
                                             arith.unwrap(sg))
                               for u, om, sg in zip(u2s, oms2s, sh)]
                        ds2s = [arith.mulf(d_, g_) for d_, g_ in zip(dh, g2s)]
                    elif VALU_VF4:
                        # sig2s holds r = s*sig/N here.  silu' = qk*r =
                        # silu/(alpha*s) (the cast scale and the dv epilogue
                        # carry alpha*s); om' = sas - (alpha*s*N)*r =
                        # sas*(1-sig); g = silu'*om' + r = vf3p's s*gc.
                        silu2s = [arith.mulf(q2, r_) for q2, r_ in zip(q2s, sig2s)]
                        # high-half splat first (gfx950 op_sel rule, see
                        # pk_cx2): v_pk_fma_f32 op_sel:[1,0,0], not [0,1,0]
                        om2s = [llvm.intr_fma(arith.unwrap(pk_ncxn2),
                                              arith.unwrap(r_),
                                              arith.unwrap(pk_sasc2))
                                for r_ in sig2s]
                        gc2s = [llvm.intr_fma(arith.unwrap(sl),
                                              arith.unwrap(om),
                                              arith.unwrap(r_))
                                for sl, om, r_ in zip(silu2s, om2s, sig2s)]
                        ds2s = [arith.mulf(p2, gc) for p2, gc in zip(p2s, gc2s)]
                    elif VALU_VF3:
                        # sig2s holds r = s*sig/N here.
                        silu2s = [arith.mulf(x2, r_) for x2, r_ in zip(x2s, sig2s)]
                        # high-half splat first (gfx950 op_sel rule, see
                        # pk_cx2): v_pk_fma_f32 op_sel:[1,0,0], not [0,1,0]
                        om2s = [llvm.intr_fma(arith.unwrap(pk_nn2),
                                              arith.unwrap(r_),
                                              arith.unwrap(pk_vis2))
                                for r_ in sig2s]
                        gc2s = [llvm.intr_fma(arith.unwrap(sl),
                                              arith.unwrap(om),
                                              arith.unwrap(r_))
                                for sl, om, r_ in zip(silu2s, om2s, sig2s)]
                        ds2s = [arith.mulf(p2, gc) for p2, gc in zip(p2s, gc2s)]
                    elif VALU_VF2:
                        sq2s = [arith.mulf(sg, pk_inv2) for sg in sig2s]
                        silu2s = [arith.mulf(q2, sq)
                                  for q2, sq in zip(q2s, sq2s)]
                        om2s = [llvm.intr_fma(arith.unwrap(sg),
                                              arith.unwrap(pk_nsas2),
                                              arith.unwrap(pk_sas2))
                                for sg in sig2s]
                        gc2s = [llvm.intr_fma(arith.unwrap(sl),
                                              arith.unwrap(om),
                                              arith.unwrap(sq))
                                for sl, om, sq in zip(silu2s, om2s, sq2s)]
                        ds2s = [arith.mulf(p2, gc) for p2, gc in zip(p2s, gc2s)]
                    elif VALU_VF:
                        sq2s = [arith.mulf(sg, pk_iq) for sg in sig2s]
                        silu2s = [arith.mulf(q2, sq)
                                  for q2, sq in zip(q2s, sq2s)]
                        sv2s = [arith.mulf(sg, pk_iv) for sg in sig2s]
                        om2s = [llvm.intr_fma(arith.unwrap(sg),
                                              arith.unwrap(pk_nvis),
                                              arith.unwrap(pk_vis2))
                                for sg in sig2s]
                        gc2s = [llvm.intr_fma(arith.unwrap(sl),
                                              arith.unwrap(om),
                                              arith.unwrap(sv))
                                for sl, om, sv in zip(silu2s, om2s, sv2s)]
                        ds2s = [arith.mulf(p2, gc) for p2, gc in zip(p2s, gc2s)]
                    else:
                        sinv2s = [arith.mulf(sg, pk_inv2) for sg in sig2s]
                        silu2s = [arith.mulf(x2, si)
                                  for x2, si in zip(x2s, sinv2s)]
                        oms2s = [arith.subf(pk_one2, sg) for sg in sig2s]
                        gc2s = [llvm.intr_fma(arith.unwrap(sl),
                                              arith.unwrap(om),
                                              arith.unwrap(si))
                                for sl, om, si in zip(silu2s, oms2s, sinv2s)]
                        dv2s = [arith.mulf(vector.shuffle(dp[nt][ms],
                                                          dp[nt][ms], sel),
                                           pk_vis2)
                                for (ms, _), sel in zip(prs, sels)]
                        ds2s = [arith.mulf(dv, gc)
                                for dv, gc in zip(dv2s, gc2s)]
                    ilp_silu = [[None] * 4 for _ in range(MSTEP)]
                    ilp_ds = [[None] * 4 for _ in range(MSTEP)]
                    for k_, (ms, h2) in enumerate(prs):
                        for j in range_constexpr(2):
                            ilp_silu[ms][2 * h2 + j] = arith.ArithValue(
                                _ex2i(silu2s[k_], j))
                            ilp_ds[ms][2 * h2 + j] = arith.ArithValue(
                                _ex2i(ds2s[k_], j))
                for ms in range_constexpr(MSTEP):
                    s_el = []
                    d_el = []
                    pk_silu = [None] * 4
                    pk_ds = [None] * 4
                    if VALU_ILP:
                        pk_silu = ilp_silu[ms]
                        pk_ds = ilp_ds[ms]
                    elif EPI_PK and not ABL_ACT:
                        # The "reuse" chain below, two accumulator elements
                        # per issue.  Op for op (and operand order for the
                        # non-commutative 1-sig and the fma) identical to the
                        # scalar branch, so every lane rounds identically.
                        q4 = qk[nt][ms]
                        p4 = dp[nt][ms]

                        def _ex2(v2, j):
                            return arith.unwrap(vector.extract(
                                v2, static_position=[j], dynamic_position=[]))

                        for h2 in range_constexpr(2):
                            sel = [2 * h2, 2 * h2 + 1]
                            x2 = arith.mulf(vector.shuffle(q4, q4, sel),
                                            pk_sas2)
                            t2 = arith.mulf(x2, pk_nl2e2)
                            e2 = vector.from_elements(V2F, [
                                rocdl.exp2(T.f32, _ex2(t2, j))
                                for j in range_constexpr(2)])
                            d2 = arith.addf(e2, pk_one2)
                            sig2 = vector.from_elements(V2F, [
                                rocdl.rcp(T.f32, _ex2(d2, j))
                                for j in range_constexpr(2)])
                            sinv2 = arith.mulf(sig2, pk_inv2)
                            silu2 = arith.mulf(x2, sinv2)
                            oms2 = arith.subf(pk_one2, sig2)
                            gc2 = llvm.intr_fma(arith.unwrap(silu2),
                                                arith.unwrap(oms2),
                                                arith.unwrap(sinv2))
                            dv2 = arith.mulf(vector.shuffle(p4, p4, sel),
                                             pk_vis2)
                            ds2 = arith.mulf(dv2, gc2)
                            for j in range_constexpr(2):
                                pk_silu[2 * h2 + j] = arith.ArithValue(
                                    _ex2(silu2, j))
                                pk_ds[2 * h2 + j] = arith.ArithValue(
                                    _ex2(ds2, j))
                    for i in range_constexpr(4):
                        raw = arith.ArithValue(
                            vector.extract(qk[nt][ms], static_position=[i],
                                           dynamic_position=[])
                        )
                        dpv = arith.ArithValue(
                            vector.extract(dp[nt][ms], static_position=[i],
                                           dynamic_position=[])
                        )
                        if EPI_PK and not ABL_ACT:
                            silu = pk_silu[i]
                            dsv = pk_ds[i]
                        elif ABL_ACT:
                            silu = raw
                            dsv = dpv
                        else:
                            if EPI_REUSE:
                                x = raw * score_alpha_scale
                            else:
                                x = raw * _f32(alpha) * score_input_scale
                            e = rocdl.exp2(T.f32, arith.unwrap(x * _f32(-LOG2E)))
                            sig = arith.ArithValue(
                                rocdl.rcp(T.f32, arith.unwrap(arith.ArithValue(e) + _f32(1.0)))
                            )
                            # inv_max is a scalar KERNEL ARGUMENT, not a baked
                            # constant: production derives max_seq_len per batch
                            # as max_uih_len + num_candidates, so baking it into
                            # the lru_cache key would JIT a fresh kernel per
                            # batch.  It lands in an SGPR and the multiply is
                            # the same one that was there before, so this is
                            # bit-identical to the constant-folded version for
                            # every max_seq_len, powers of two or not -- unlike
                            # folding it into the epilogue, which would move the
                            # bf16 rounding point.
                            if EPI_REUSE:
                                # SiLU'(x)/N = sig/N + (x*sig/N)*(1-sig):
                                # share sig/N and the SiLU value itself.
                                sig_inv = sig * inv_max
                                silu = arith.ArithValue(x) * sig_inv
                                gc = arith.ArithValue(llvm.intr_fma(
                                    arith.unwrap(silu),
                                    arith.unwrap(_f32(1.0) - sig),
                                    arith.unwrap(sig_inv)))
                            else:
                                silu = arith.ArithValue(x * sig) * inv_max
                                gc = arith.ArithValue(
                                    sig * (_f32(1.0) + arith.ArithValue(x) * (_f32(1.0) - sig))
                                ) * inv_max
                            dsv = (
                                dpv
                                * value_input_scale
                                * arith.ArithValue(gc)
                            )
                        if masked and not ABL_MASK:
                            # Causal only.  The bounds test that used to sit here
                            # is gone: out-of-range q/dO rows arrive as zeros from
                            # the buffer descriptor and are already exactly zero.
                            m_idx = m0 + _i32(16 * ms) + l4 * _i32(4) + _i32(i)
                            ok = arith.cmpi(arith.CmpIPredicate.sge, m_idx,
                                            n_idx_score[nt])
                            z_ = (arith.constant(0.0, type=T.f16) if VALU_H16
                                  else _f32(0.0))
                            silu = arith.select(ok, silu, z_)
                            dsv = arith.select(ok, dsv, z_)
                        s_el.append(silu)
                        d_el.append(dsv)
                    # One vector truncf per fragment: v4f32 -> v4bf16 lowers to
                    # v_cvt_pk_bf16_f32 pairs that pack two f32 into two bf16
                    # halfwords directly, vs 4 scalar truncs + v_perm_b32 shuffles
                    # to assemble the v4i16.  The silu/ds values are already scalar
                    # out of the scalar-only exp2/rcp chain, so this tail pack does
                    # NOT fight the transcendental repack barrier -- it only removes
                    # the halfword-assembly perms from the issue-bound VALU stream.
                    _v4 = T.vec(4, T.f16) if VALU_H16 else T.f32x4
                    s_v = vector.from_elements(_v4, s_el)
                    d_v = vector.from_elements(_v4, d_el)
                    silu32[nt].append(s_v)
                    ds32[nt].append(d_v)
                    if VALU_H16:
                        # f16 tail feeds only the e4m3 casts (pk4_from_f32);
                        # the bf16 fragments below are dead on the f8b path.
                        from flydsl._mlir.dialects import arith as _ar
                        s_v = _ar.ExtFOp(T.f32x4, arith.unwrap(s_v)).result
                        d_v = _ar.ExtFOp(T.f32x4, arith.unwrap(d_v)).result
                    silu_bf = arith.trunc_f(v4bf, s_v)
                    ds_bf = arith.trunc_f(v4bf, d_v)
                    silu_f[nt].append(vector.bitcast(v4i16, silu_bf))
                    ds_raw[nt].append(ds_bf)
                    ds_f[nt].append(vector.bitcast(v4i16, ds_bf))
                if hook is not None:
                    hook(nt, silu32[nt], ds32[nt])
            return silu_f, ds_f, silu32, ds32, ds_raw

        def f8b_cast(silu32, ds32):
            s8, d8 = [], []
            for nt in range_constexpr(NT):
                s8.append([pk4_from_f32(silu32[nt][ms], mu2_ssc,
                                        SAT_HI, rt=vf4_sop)
                           for ms in range_constexpr(MSTEP)])
                d8.append([pk4_from_f32(ds32[nt][ms], mu2_dsc,
                                        SAT_BOTH)
                           for ms in range_constexpr(MSTEP)])
            return s8, d8

        def f8b_ds_store(d8):
            for nt in range_constexpr(NT):
                for ms in range_constexpr(MSTEP):
                    n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
                    off = (arith.ArithValue(n_loc) * _i32(DS8R)
                           + _i32(16 * ms) + l4 * _i32(4)) // _i32(4)
                    vector.store(
                        vector.bitcast(T.vec(1, T.i32), d8[nt][ms]),
                        kt8_lds, [_idx(off)], alignment=4)

        def dqsplit_ws_store(m0, xb=None, oob=None):
            """Read the exchanged ds tile back (tr8) and store it to the ws.

            xb (msplit only): byte base of the exchange region to read.
            oob (dz): uniform byte offset added to the store address (0, or
            past the descriptor's num_records to drop a dead store)."""
            dqsplit_ws_write(m0, dqsplit_ws_read(xb), oob)

        def dqsplit_ws_read(xb=None):
            def _xo(o):
                return o if xb is None else arith.ArithValue(o) + xb
            if ws_l128t:
                rb = arith.ArithValue(t8_n) + t8_k8 * _i32(2)
                parts = [tr8(kt8_lds, _xo((rb + _i32(8 * p)) * _i32(DS8R)
                                          + _i32(8 * q) + wave * _i32(16)))
                         for q in range_constexpr(2)
                         for p in range_constexpr(2)]
                return vector.shuffle(
                    vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                    vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                    list(range(8)))
            parts = [tr8(kt8_lds, _xo((arith.ArithValue(t8_n) + _i32(8 * r))
                                      * _i32(DS8R) + t8_k8 + wave * _i32(16)))
                     for r in range_constexpr(4)]
            a8 = vector.shuffle(
                vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                list(range(8)))
            return a8

        def dqsplit_ws_write(m0, a8, oob=None):
            if ws_l128:
                assert oob is None, "dz/dzr store drop: not wired for l128"
                return dqsplit_ws_write_l128(m0, a8)
            if ws_l128t:
                ws_t = arith.ArithValue(m0) // _i32(64)
                if MS_XTD:
                    # xtd: lane l16 of read q holds m = 16*(2q + (l16%8)//4)
                    # + 4*wave + l16%4 (see MS_XTD); rows step 32 per q.
                    m_r = ((l16 % _i32(8)) // _i32(4) * _i32(16)
                           + wave * _i32(4) + l16 % _i32(4))
                    q_rows = 32
                    off0 = ((ws_jbase + ws_t) * _i32(8192)
                            + arith.ArithValue(m_r) * _i32(128)
                            + l4 * _i32(32) + (l16 // _i32(8)) * _i32(16))
                else:
                    q_rows = 8
                    off0 = ((ws_jbase + ws_t) * _i32(8192)
                            + (wave * _i32(16) + l16 % _i32(8)) * _i32(128)
                            + l4 * _i32(32) + (l16 // _i32(8)) * _i32(16))
                rows = isinstance(oob, str) and oob == "rows"
                if oob is not None and not rows:
                    off0 = arith.ArithValue(off0) + oob
                for q4 in range_constexpr(2):
                    v = vector.shuffle(a8, a8, [4 * q4 + i for i in range(4)])
                    off_q = arith.ArithValue(off0) + _i32(128 * q_rows * q4)
                    if rows:
                        # dzr: this lane's 16 B belong to ws row m0 + 16*wave
                        # + l16%8 + 8*q4 (xtd: m0 + m_r + 32*q4); a row at/
                        # after seq_len is padding (dq never stores it) ->
                        # push past num_records.
                        if MS_XTD:
                            r_ = (arith.ArithValue(m0) + m_r
                                  + _i32(q_rows * q4))
                        else:
                            r_ = (arith.ArithValue(m0) + wave * _i32(16)
                                  + l16 % _i32(8) + _i32(8 * q4))
                        off_q = off_q + arith.select(
                            arith.cmpi(arith.CmpIPredicate.slt, r_, seq_len),
                            _i32(0), _i32(-(1 << 31)))
                    rocdl_raw.RawPtrBufferStoreOp(
                        arith.unwrap(v), rWS, arith.unwrap(off_q),
                        arith.unwrap(_i32(0)), arith.unwrap(_i32(ws_aux)))
                return
            ws_t = arith.ArithValue(m0) // _i32(64)
            ws_off = ((ws_jbase + ws_t) * _i32(8192)
                      + (wave * _i32(16) + l16) * _i32(128)
                      + l4 * _i32(16 if ws_c64 else 32))
            if oob is not None:
                ws_off = arith.ArithValue(ws_off) + oob
            for q4 in range_constexpr(2):
                v = vector.shuffle(a8, a8,
                                   [4 * q4 + i for i in range(4)])
                rocdl_raw.RawPtrBufferStoreOp(
                    arith.unwrap(v), rWS,
                    arith.unwrap(arith.ArithValue(ws_off)
                                 + _i32((64 if ws_c64 else 16) * q4)),
                    arith.unwrap(_i32(0)), arith.unwrap(_i32(ws_aux)))

        def dqsplit_ws_write_l128(m0, a8):
            ws_t = arith.ArithValue(m0) // _i32(64)
            hi = l16 // _i32(8)
            off0 = ((ws_jbase + ws_t) * _i32(8192)
                    + (wave * _i32(16) + l16 % _i32(8)) * _i32(128)
                    + l4 * _i32(32) + hi * _i32(16))
            a8i = a8  # vec<8xi32> (tr8 parts)

            def _dpp(old, src, ctrl):
                return rocdl_raw.DPPUpdateOp(
                    T.i32, arith.unwrap(old), arith.unwrap(src), ctrl,
                    0xF, 0xF, False).result
            r0 = [vector.extract(a8i, static_position=[i],
                                 dynamic_position=[]) for i in range(4)]
            r1 = [vector.extract(a8i, static_position=[4 + i],
                                 dynamic_position=[]) for i in range(4)]
            # row_shr:8 (0x118): lane l16 >= 8 reads lane l16 - 8, lanes
            # l16 < 8 keep old; row_shl:8 (0x108) the mirror image.
            v0 = [_dpp(r0[i], r1[i], 0x118) for i in range(4)]
            v1 = [_dpp(r1[i], r0[i], 0x108) for i in range(4)]
            for q4, vv in ((0, v0), (1, v1)):
                v = vector.from_elements(T.vec(4, T.i32), vv)
                rocdl_raw.RawPtrBufferStoreOp(
                    arith.unwrap(v), rWS,
                    arith.unwrap(arith.ArithValue(off0) + _i32(1024 * q4)),
                    arith.unwrap(_i32(0)), arith.unwrap(_i32(ws_aux)))

        def one_iter(m0, acc_dv, acc_dk, masked, pend=None, e8=None):
            # e8 (do_blk only): [kblock byte, dq row byte] of tile m0, as
            # loaded by load_tiles(m0) -- elements 8:10 of its result.
            assert (e8 is not None) == do_blk
            # A-operands of the score gemms and B-operands of the out gemms are
            # both independent of n, so they are read once here and fed to all NT
            # n-tiles.  That is the whole point of a wide BN.
            if FP8_QK in ("k32", "k32w"):
                a_qk_frag = [
                    [a_frag8(ms, ks) for ks in range_constexpr(KSTEP)]
                    for ms in range_constexpr(MSTEP)
                ]
            elif FP8_QK == "k128":
                a_qk_frag = [[a_frag8(ms, 0)] for ms in range_constexpr(MSTEP)]
            else:
                a_qk_frag = [
                    [a_frag(q_rm, ms, ks) for ks in range_constexpr(KSTEP)]
                    for ms in range_constexpr(MSTEP)
                ]
            if F8B:
                a_dp_frag = [[a_frag8(ms, 0, o8_lds)]
                             for ms in range_constexpr(MSTEP)]
            else:
                a_dp_frag = [[a_frag(o_rm, ms, ks)
                              for ks in range_constexpr(KSTEP)]
                             for ms in range_constexpr(MSTEP)]

            qk = []
            dp = []
            for nt in range_constexpr(NT):
                qk.append([])
                dp.append([])
                for ms in range_constexpr(MSTEP):
                    a_qk = zero_acc
                    a_dp = zero_acc
                    if FP8_QK:
                        # dp keeps the bf16 op: dO is genuinely bf16 and
                        # rounding it to e4m3 costs dq/dk 13 dB.
                        if FP8_QK == "k128":
                            a_qk = mfma_qk_k128(a_qk_frag[ms][0], kf8[nt][0],
                                                a_qk)
                        if F8B:
                            a_dp = mfma_qk_k128(a_dp_frag[ms][0], vf8[nt],
                                                a_dp)
                        for ks in range_constexpr(0 if F8B else KSTEP):
                            if FP8_QK in ("k32", "k32w"):
                                a_qk = mfma_qk_k32(a_qk_frag[ms][ks],
                                                   kf8[nt][ks], a_qk)
                            a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks], a_dp)
                    elif not ABL_SCORE:
                        for ks in range_constexpr(KSTEP):
                            a_qk = mfma2(a_qk_frag[ms][ks], kf[nt][ks], a_qk)
                            a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks], a_dp)
                            if ABL_MFMA2X:
                                a_qk = mfma2(a_qk_frag[ms][ks], kf[nt][ks],
                                             a_qk)
                                a_dp = mfma2(a_dp_frag[ms][ks], vf[nt][ks],
                                             a_dp)
                    else:
                        a_qk = mfma2(kf[nt][0], kf[nt][1], a_qk)
                        a_dp = mfma2(vf[nt][0], vf[nt][1], a_dp)
                    qk[nt].append(a_qk)
                    dp[nt].append(a_dp)

            silu_f, ds_f, silu32, ds32, ds_raw = act_tile(m0, qk, dp, masked)

            s8, d8 = [], []
            if F8B:
                s8, d8 = f8b_cast(silu32, ds32)
                f8b_ds_store(d8)
            elif fused:
                # Hoisted ds-exchange publish.  ds_raw is complete here and the
                # out-gemm mfmas below do not touch ds_lds, so issuing the LDS
                # writes NOW puts ~128 independent mfmas between them and the
                # cross-wave barrier.  The store is async; without work in
                # between, the s_barrier stalls on the full LDS write latency
                # (KB LDS Fix C, exposed write-read distance), and iglp cannot
                # hoist it itself because an s_barrier is a hard scheduling
                # boundary.  Same region, same barrier, same ds_raw liveness as
                # before, so this only moves the store earlier in program order.
                for nt in range_constexpr(NT):
                    for ms in range_constexpr(MSTEP):
                        if ABL_DQW:
                            off = _i32(0)
                        else:
                            off = (_i32(ms * BN) + wave * _i32(16 * NT)
                                   + _i32(16 * nt) + l16) * _i32(16) + l4 * _i32(4)
                        vector.store(ds_raw[nt][ms], ds_lds, [_idx(off)],
                                     alignment=8)

            new_dv = list(acc_dv)
            new_dk = list(acc_dk)
            cur = None
            if MU2 == "k128":
                # ---- TRACK U1: build this m tile's HALF of the K=128 operands.
                # A: silu / ds arrive as four f32 fragments per n-tile, one per
                #    16-row m sub-tile.  Two fragments = 8 f32 = one pk8 cvt =
                #    two dwords, so a whole BM=64 tile is 4 dwords per n-tile.
                # B: dO / q come out of the tr16 tile as v4bf slices, same
                #    (l4, element) -> m mapping as A, so the pieces concatenate
                #    in the same order and the two operands agree on k.
                v4i32_t = T.vec(4, T.i32)
                a_dv_h, a_dk_h = [], []
                for nt in range_constexpr(NT if F8B else 0):
                    def _cat(p):
                        return vector.shuffle(
                            vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                            vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                            list(range(8)))
                    a_dv_h.append(_cat(s8[nt]))
                    a_dk_h.append(_cat(d8[nt]))
                for nt in range_constexpr(0 if F8B else NT):
                    def _pack(frags, sc):
                        halves = [
                            pk8_from_f32(frags[nt][2 * j], frags[nt][2 * j + 1],
                                         sc)
                            for j in range_constexpr(MSTEP // 2)
                        ]
                        return vector.shuffle(halves[0], halves[1],
                                              list(range(8)))
                    a_dv_h.append(_pack(silu32, mu2_ssc))
                    a_dk_h.append(_pack(ds32, mu2_dsc))
                bo_h, bq_h = [], []
                for ds in range_constexpr(NSTEP if F8B else 0):
                    for lds8, out in ((o8_lds, bo_h), (q8_lds, bq_h)):
                        out.append(vector.shuffle(b8_tr(lds8, 0, ds),
                                                  b8_tr(lds8, 1, ds),
                                                  list(range(8))))
                for ds in range_constexpr(0 if F8B else NSTEP):
                    for lds, out, bsc in ((o_rm, bo_h, mu2_osc),
                                          (q_rm, bq_h, mu2_qsc)):
                        if ABL_BCVT0:
                            # v4i16 (4 bf16) + v4i16 -> v4i16 "8 fp8": a pure
                            # register select, no VALU, both LDS reads still
                            # feed it.  Ceiling probe only -- see ABL_BCVT0.
                            halves = [
                                vector.shuffle(b_frag_tr(lds, 2 * j, ds),
                                               b_frag_tr(lds, 2 * j + 1, ds),
                                               [0, 1, 4, 5])
                                for j in range_constexpr(MSTEP // 2)
                            ]
                        else:
                            halves = [
                                pk8_from_bf16(b_frag_tr(lds, 2 * j, ds),
                                              b_frag_tr(lds, 2 * j + 1, ds),
                                              bsc)
                                for j in range_constexpr(MSTEP // 2)
                            ]
                        out.append(vector.shuffle(halves[0], halves[1],
                                                  list(range(8))))
                cur = (a_dv_h, a_dk_h, bo_h, bq_h,
                       e8[0] if do_blk else None)
                if pend is not None:
                    p_dv, p_dk, p_bo, p_bq, p_e8 = pend
                    v8i32_t = T.vec(8, T.i32)
                    CAT32 = list(range(16))

                    def _wide(a, b):
                        return vector.bitcast(
                            v8i32_t, vector.shuffle(a, b, CAT32))
                    sc_kb = None
                    if do_blk:
                        # K=128 operand = [pend half | cur half].  Data (lane
                        # group g, half h) holds m%16 in [4g, 4g+4) of tile h
                        # (d8/s8 dword ms = acc rows 16ms+4g+i; b8_tr's t8_m is
                        # the same order) and is HW K-block 2h + g//2, scaled
                        # by scale lanes 16*(2h + g//2) + l16 (P0.c).  So
                        # scale lane group l4 = K-block = (tile l4//2, sub
                        # l4%2); each tile's load already picked sub l4%2.
                        sc_kb = arith.select(
                            arith.cmpi(arith.CmpIPredicate.ult, l4, _i32(2)),
                            p_e8, e8[0])
                    for ds in range_constexpr(NSTEP):
                        b_o = _wide(p_bo[ds], bo_h[ds])
                        b_q = _wide(p_bq[ds], bq_h[ds])
                        for nt in range_constexpr(NT):
                            i = nt * NSTEP + ds
                            if do_blk:
                                new_dv[i] = mfma_k128_sc(
                                    _wide(p_dv[nt], a_dv_h[nt]),
                                    b_o, acc_dv[i], sc127, sc_kb)
                                new_dk[i] = mfma_k128_sc(
                                    _wide(p_dk[nt], a_dk_h[nt]),
                                    b_q, acc_dk[i], sc_kb, sc127)
                                continue
                            new_dv[i] = mfma_out_k128(
                                _wide(p_dv[nt], a_dv_h[nt]),
                                b_o, acc_dv[i])
                            new_dk[i] = mfma_out_k128(
                                _wide(p_dk[nt], a_dk_h[nt]),
                                b_q, acc_dk[i])
                    cur = None
            # Concatenate two consecutive m-tiles' free-transpose score fragments
            # into a K=32 A-operand; the matching B-operand pair is contiguous in
            # M, so the double-rate mfma2 sums the same M values A and B carry.
            cat = [0, 1, 2, 3, 4, 5, 6, 7]
            for ds in range_constexpr(0 if MU2 == "k128" else NSTEP):
                bo_frag = []
                bq_frag = []
                for j in range_constexpr(MSTEP // 2):
                    if tr16:
                        bo_frag.append(vector.shuffle(
                            b_frag_tr(o_rm, 2 * j, ds),
                            b_frag_tr(o_rm, 2 * j + 1, ds), cat))
                        bq_frag.append(vector.shuffle(
                            b_frag_tr(q_rm, 2 * j, ds),
                            b_frag_tr(q_rm, 2 * j + 1, ds), cat))
                    elif ABL_TR:
                        bo_frag.append(a_frag(o_rm, 2 * j, ds))
                        bq_frag.append(a_frag(q_rm, 2 * j, ds))
                    else:
                        bo_frag.append(vector.shuffle(
                            b_frag_bf(o_tr, 2 * j, ds),
                            b_frag_bf(o_tr, 2 * j + 1, ds), cat))
                        bq_frag.append(vector.shuffle(
                            b_frag_bf(q_tr, 2 * j, ds),
                            b_frag_bf(q_tr, 2 * j + 1, ds), cat))
                for nt in range_constexpr(NT):
                    adv = acc_dv[nt * NSTEP + ds]
                    adk = acc_dk[nt * NSTEP + ds]
                    if ABL_OUT:
                        # Attribution only (Task C, scratch): the old form fed
                        # kf/vf, which are K=32 v8bf16 fragments since the fp8
                        # widening landed, into the half-rate 16x16x16 mfma and
                        # no longer verifies.  Keep j=0 only, so the out-gemm
                        # mfma count halves and nothing else changes.
                        adv = mfma2(
                            vector.bitcast(v8bf, vector.shuffle(
                                silu_f[nt][0], silu_f[nt][1], cat)),
                            bo_frag[0], adv)
                        adk = mfma2(
                            vector.bitcast(v8bf, vector.shuffle(
                                ds_f[nt][0], ds_f[nt][1], cat)),
                            bq_frag[0], adk)
                    else:
                        for j in range_constexpr(MSTEP // 2):
                            a_dv = vector.bitcast(
                                v8bf,
                                vector.shuffle(silu_f[nt][2 * j],
                                               silu_f[nt][2 * j + 1], cat),
                            )
                            a_dk = vector.bitcast(
                                v8bf,
                                vector.shuffle(ds_f[nt][2 * j],
                                               ds_f[nt][2 * j + 1], cat),
                            )
                            adv = mfma2(a_dv, bo_frag[j], adv)
                            adk = mfma2(a_dk, bq_frag[j], adk)
                            if ABL_MFMA2XO:
                                adv = mfma2(a_dv, bo_frag[j], adv)
                                adk = mfma2(a_dk, bq_frag[j], adk)
                    new_dv[nt * NSTEP + ds] = adv
                    new_dk[nt * NSTEP + ds] = adk

            if fused:
                # The ds-exchange publish (tr16 layout, (r, c) = (n, m)) was
                # hoisted above the out-gemm mfmas; only the cross-wave barrier
                # remains here so the ~128 mfmas cover the LDS write latency.
                if not ABL_DQB:
                    gpu.barrier()
                # Read it back transposed and contract over the full BN.  The
                # accumulator is complete for this m tile after one iteration,
                # so it is flushed here rather than carried.
                acc_dq = [zero_acc] * (MSTEP * DQD)
                if DQSPLIT:
                    # The ds tile is complete in LDS after the barrier above.
                    # dq8_a(ms) (here ms = wave) is ds[m = 16*ms + l16][n = 32*l4 .. 32*l4+31]
                    # in natural byte order (tr8 receiver byte e of part r is
                    # n = 32*l4 + 8r + e), so wave w stores rows 16w..16w+15
                    # of an m-major [m 64][n 128] e4m3 tile: the 4 l4 lanes
                    # of a row write one full 128 B line.
                    dqsplit_ws_store(m0)
                for ms in range_constexpr(MSTEP if (F8B and not DQSPLIT) else 0):
                    # One K=128 (= BN) fp8 mfma per (m tile, d tile) instead
                    # of DQK=4 bf16 K=32 ones.  bf16p orientation: ds is A.
                    a8 = dq8_a(ms)
                    for dt in range_constexpr(DQD):
                        if do_blk:
                            # A row m = 16*ms + l16 carries its group's
                            # 2**e_g in every K-block: A scale lane l (row
                            # l16, any K-block) = byte of group (m0, l16//8).
                            acc_dq[ms * DQD + dt] = mfma_k128_sc(
                                a8, kt_frag[dt], acc_dq[ms * DQD + dt],
                                e8[1], sc127)
                            continue
                        acc_dq[ms * DQD + dt] = mfma_qk_k128(
                            a8, kt_frag[dt], acc_dq[ms * DQD + dt])
                for ms in range_constexpr(0 if F8B else MSTEP):
                    for t in range_constexpr(DQK):
                        a_dq = vector.bitcast(
                            v8bf, tr_pair(ds_lds, 0 if ABL_DQR else ms * BN,
                                          0 if ABL_DQR else t))
                        for dt in range_constexpr(DQD):
                            # Operand order decides the accumulator's
                            # orientation.  f32 wants C[m][d], so ds supplies M;
                            # a packed 16-bit flush wants C[d][m], so that a
                            # lane's four elements are four CONSECUTIVE d and
                            # pair up into dwords -- then k supplies M instead.
                            a, b = ((kt_frag[dt][t], a_dq)
                                    if dq16 in ("bf16", "f16")
                                    else (a_dq, kt_frag[dt][t]))
                            acc_dq[ms * DQD + dt] = mfma2(
                                a, b, acc_dq[ms * DQD + dt])
                            if ABL_MFMA2XQ:
                                acc_dq[ms * DQD + dt] = mfma2(
                                    a, b, acc_dq[ms * DQD + dt])
                sc = _f32(alpha * dq_scale / (mu2_dsc if F8B else 1.0))
                if input_fp8:
                    sc = arith.ArithValue(sc) * value_input_scale
                if do_dyn:
                    sc = arith.ArithValue(sc) * inv_s_do
                if dq16 == "bf16p" and not DQSPLIT:
                    # The staged K tile was interleaved so that lane l16's two
                    # accumulators hold d = 32*wave + 2*l16 and +1.  Packing them
                    # into one dword makes the 16 lanes of a row cover
                    # [32*wave, 32*wave+32) in bf16 = 64 contiguous bytes, one
                    # aligned line per request, with every lane live.
                    dqt = dq_ty()
                    v2 = T.vec(2, dqt)
                    for ms in range_constexpr(MSTEP):
                        lo_a = acc_dq[ms * DQD + 0]
                        hi_a = acc_dq[ms * DQD + 1]
                        row = m0 + _i32(16 * ms) + l4 * _i32(4)
                        col = wave * _i32(32) + l16 * _i32(2)
                        for i in range_constexpr(1 if ABL_DQF else 4):
                            pk = vector.from_elements(v2, [
                                arith.trunc_f(dqt, arith.ArithValue(
                                    vector.extract(a, static_position=[i],
                                                   dynamic_position=[])) * sc)
                                for a in (lo_a, hi_a)
                            ])
                            off = ((arith.ArithValue(row) + _i32(i)) * _i32(RS_OUT)
                                   + col) * _i32(2)
                            _atomic_fadd(pk, rDQ, off, dq_aux)
                if dq16 == "bf16c":
                    # Lane l owns d = 16*dt + l16 and l16 runs 0..15, so the
                    # pair (d, d+1) that shares a dword straddles lanes l and
                    # l^1.  One DPP swap brings the partner's value in; the odd
                    # lanes are then redundant and are steered out of range.
                    dqt = dq_ty()
                    v2 = T.vec(2, dqt)
                    keep = arith.cmpi(arith.CmpIPredicate.eq,
                                      l16 % _i32(2), _i32(0))
                    for ms in range_constexpr(MSTEP):
                        for dt in range_constexpr(DQD):
                            a = acc_dq[ms * DQD + dt]
                            row = m0 + _i32(16 * ms) + l4 * _i32(4)
                            col = wave * _i32(16 * DQD) + _i32(16 * dt) + l16
                            # The DPP swap has to run with BOTH lanes of a pair
                            # enabled, so the pack is built before the odd lanes
                            # are masked off.
                            pks = []
                            for i in range_constexpr(1 if ABL_DQF else 4):
                                lo = arith.ArithValue(
                                    vector.extract(a, static_position=[i],
                                                   dynamic_position=[])) * sc
                                hi = _dpp_swap_adj(lo)
                                pks.append(vector.from_elements(v2, [
                                    arith.trunc_f(dqt, lo),
                                    arith.trunc_f(dqt, hi)]))
                            # Steering the odd lanes out of num_records instead
                            # of masking exec faults the queue: unlike loads,
                            # an out-of-range buffer ATOMIC is not discarded.
                            odd_off = scf.IfOp(keep, results_=[], has_else=False)
                            with ir.InsertionPoint(odd_off.then_block):
                                for i, pk in enumerate(pks):
                                    off = ((arith.ArithValue(row) + _i32(i))
                                           * _i32(RS_OUT) + col) * _i32(2)
                                    _atomic_fadd(pk, rDQ, off, dq_aux)
                                scf.YieldOp([])
                for ms in range_constexpr(
                        MSTEP if dq16 not in ("bf16c", "bf16p") else 0):
                    for dt in range_constexpr(DQD):
                        a = acc_dq[ms * DQD + dt]
                        if dq16:
                            m_idx = m0 + _i32(16 * ms) + l16
                            d_idx = (wave * _i32(16 * DQD) + _i32(16 * dt)
                                     + l4 * _i32(4))
                            base = (
                                arith.ArithValue(m_idx) * _i32(RS_OUT) + d_idx
                            )
                            dqt = dq_ty()
                            for p in range_constexpr(1 if ABL_DQF else 2):
                                pk = vector.from_elements(T.vec(2, dqt), [
                                    arith.trunc_f(dqt, arith.ArithValue(
                                        vector.extract(a, static_position=[2 * p + s],
                                                       dynamic_position=[])) * sc)
                                    for s in range(2)
                                ])
                                off = (base + _i32(2 * p)) * _i32(2)
                                _atomic_fadd(pk, rDQ, off, dq_aux)
                            continue
                        row = ((_i32(0) if ABL_DQZ else m0)
                               + _i32(16 * ms) + l4 * _i32(4))
                        col = wave * _i32(16 * DQD) + _i32(16 * dt) + l16
                        for i in range_constexpr(1 if ABL_DQF else 4):
                            off = ((arith.ArithValue(row) + _i32(i)) * _i32(RS_OUT)
                                   + col) * _i32(4)
                            val = arith.ArithValue(
                                vector.extract(a, static_position=[i],
                                               dynamic_position=[])) * sc
                            if ABL_DQA:
                                buffer_ops.buffer_store(
                                    arith.unwrap(val), rDQ, off,
                                    offset_is_bytes=True)
                            else:
                                _atomic_fadd(val, rDQ, off, dq_aux)

            if ABL_DQ:
                # dq[m][d] = sum_n ds[m][n] k[n][d].  A wave owns 16*NT of the
                # block's n rows, so its dq partial covers the FULL BM x D tile
                # and is complete for those n after one m iteration -- it is not
                # loop-carried, it is flushed by atomic fadd every iteration.
                # That is 4 partials per workgroup, hence 4 atomic streams.
                assert NSTEP % DQC == 0
                nd = NSTEP // DQC
                for c in range_constexpr(DQC):
                    dq = [zero_acc] * (MSTEP * nd)
                    for ms in range_constexpr(MSTEP):
                        # K = 32 contraction over this wave's n rows.
                        a_dq = vector.bitcast(
                            v8bf,
                            vector.shuffle(ds_f[0][ms], ds_f[NT - 1][ms], cat),
                        )
                        for dt in range_constexpr(nd):
                            # Every d tile must get a DISTINCT B operand or CSE
                            # folds the accumulators together and the probe
                            # understates the pressure it exists to measure.
                            e = c * nd + dt
                            dq[ms * nd + dt] = mfma2(
                                a_dq, kf[(e // KSTEP) % NT][e % KSTEP],
                                dq[ms * nd + dt],
                            )
                    for ms in range_constexpr(MSTEP):
                        for dt in range_constexpr(nd):
                            row = m0 + _i32(16 * ms) + l4 * _i32(4)
                            col = l16 + _i32(16 * (c * nd + dt))
                            for i in range_constexpr(4):
                                off = (
                                    (row + _i32(i)) * _i32(RS_OUT) + col
                                ) * _i32(4)
                                _atomic_fadd(
                                    vector.extract(dq[ms * nd + dt],
                                                   static_position=[i],
                                                   dynamic_position=[]),
                                    rDQ, off,
                                )
            return new_dv, new_dk, cur

        def pipe_score(lds_q8, lds_o8):
            """qk / dp accumulators of one m tile (F8B k128 path of one_iter)."""
            a_qk_frag = [a_frag8(ms, 0, lds_q8) for ms in range_constexpr(MSTEP)]
            a_dp_frag = [a_frag8(ms, 0, lds_o8) for ms in range_constexpr(MSTEP)]
            qk, dp = [], []
            for nt in range_constexpr(NT):
                qk.append([])
                dp.append([])
                for ms in range_constexpr(MSTEP):
                    qk[nt].append(mfma_qk_k128(a_qk_frag[ms], kf8[nt][0],
                                               zero_acc))
                    dp[nt].append(mfma_qk_k128(a_dp_frag[ms], vf8[nt],
                                               zero_acc))
            return qk, dp

        def pipe_pair(mA, acc_dv, acc_dk, masked, eA, eB, prefetch_b=None):
            """One mu2 pair (tiles mA, mA + BM) with both tiles already in LDS:
            tile A in (q8_lds, o8_lds), tile B in (q8b_lds, o8b_lds).

            Same per-element math as one_iter(A, pend=None) followed by
            one_iter(B, pend=A); only the program order moves:
              score(A), score(B) | act(A) cast store | bar | ws(A)
              | act(B) cast | bar | store(B) out-gemm(A+B) | bar | ws(B)
            score(B) shares the first scheduling region with act(A).
            """
            mB = mA + _i32(BM)
            v8i32_t = T.vec(8, T.i32)
            CAT32 = list(range(16))

            def _wide(a, b):
                return vector.bitcast(v8i32_t, vector.shuffle(a, b, CAT32))
            qkA, dpA = pipe_score(q8_lds, o8_lds)
            qkB, dpB = pipe_score(q8b_lds, o8b_lds)
            _, _, silA, dsA, _ = act_tile(mA, qkA, dpA, masked)
            s8A, d8A = f8b_cast(silA, dsA)
            f8b_ds_store(d8A)
            gpu.barrier()
            dqsplit_ws_store(mA)
            nb = prefetch_b() if prefetch_b is not None else None

            def _bop(ds):
                bo = [vector.shuffle(b8_tr(l8, 0, ds), b8_tr(l8, 1, ds),
                                     list(range(8))) for l8 in (o8_lds, o8b_lds)]
                bq = [vector.shuffle(b8_tr(l8, 0, ds), b8_tr(l8, 1, ds),
                                     list(range(8))) for l8 in (q8_lds, q8b_lds)]
                return (_wide(bo[0], bo[1]), _wide(bq[0], bq[1]))
            n_early = {5: NSTEP // 2, 6: NSTEP}.get(PIPE, 0)
            b_ops = [_bop(ds) for ds in range_constexpr(n_early)]
            _, _, silB, dsB, _ = act_tile(mB, qkB, dpB, masked)
            s8B, d8B = f8b_cast(silB, dsB)
            gpu.barrier()  # every wave has read tile A's ds back
            f8b_ds_store(d8B)

            def _cat(p):
                return vector.shuffle(
                    vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                    vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                    list(range(8)))
            sc_kb = arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, l4, _i32(2)), eA[0], eB[0])
            a_dv = [_wide(_cat(s8A[nt]), _cat(s8B[nt]))
                    for nt in range_constexpr(NT)]
            a_dk = [_wide(_cat(d8A[nt]), _cat(d8B[nt]))
                    for nt in range_constexpr(NT)]
            new_dv = list(acc_dv)
            new_dk = list(acc_dk)
            n_late = {3: NSTEP // 2, 4: NSTEP}.get(PIPE, 0)
            for j in range_constexpr(NSTEP - n_early):
                b_ops.append(_bop(n_early + j))

            def _out(ds):
                b_o, b_q = b_ops[ds]
                for nt in range_constexpr(NT):
                    i = nt * NSTEP + ds
                    new_dv[i] = mfma_k128_sc(a_dv[nt], b_o, acc_dv[i],
                                             sc127, sc_kb)
                    new_dk[i] = mfma_k128_sc(a_dk[nt], b_q, acc_dk[i],
                                             sc_kb, sc127)
            for ds in range_constexpr(NSTEP - n_late):
                _out(ds)
            gpu.barrier()
            dqsplit_ws_store(mB)
            for j in range_constexpr(n_late):
                _out(NSTEP - n_late + j)
            return new_dv, new_dk, nb

        # ---- R15 iter17r3 msplit (B1 m-split pair) -----------------------
        # LDS: tile block ms_t8 = [parity 2][tile 2][q, dO] fp8 tiles of
        # TILE_B bytes (same per-tile layout as q8_lds/o8_lds); exchange block
        # kt8_lds = [parity 2][ds, silu][tile 2] regions of XR_B bytes, each
        # byte(n, m) = n*DS8R + m (f8b_ds_store's layout).
        TILE_B = (D // 32) * Q8BS
        TILE_DW = TILE_B // 4
        XR_B = BN * DS8R
        NJ = NSTEP // 2  # d tiles per wave group in the out-gemm

        def ms_write_tile(vecs, par, tile):
            """write_tiles(vecs) into tile `tile` of parity `par` (dynamic)."""
            for t in range_constexpr(2):  # 0 = Q, 1 = dO
                tb = ((arith.ArithValue(par) * _i32(4) + _i32(2 * tile + t))
                      * _i32(TILE_DW))
                for r in range_constexpr(LROWS):
                    lo = (tb + arith.ArithValue(st8_db) * _i32(Q8BS_DW)
                          + (arith.ArithValue(st_m) + _i32(r)) * _i32(8)
                          + st8_in)
                    vector.store(vecs[LROWS * t + r], ms_t8, [_idx(lo)],
                                 alignment=8)

        def ms_a8(tb_dw, ms):
            """a_frag8(ms, 0, <tile at dword tb_dw>) (k128)."""
            row = (l16 + _i32(16 * ms)) * _i32(8)
            off = arith.ArithValue(tb_dw) + arith.ArithValue(l4) * _i32(Q8BS_DW) + row
            return vector.shuffle(
                vector.load_op(v4i32, ms_t8, [_idx(off)]),
                vector.load_op(v4i32, ms_t8,
                               [_idx(arith.ArithValue(off) + _i32(4))]),
                [0, 1, 2, 3, 4, 5, 6, 7])

        def ms_xt_layout(tensor):
            """Exchange region layout: xt (per-lane contiguous dwords) for
            silu under xt (dropped by gsp's uniform readback unless xtd), for
            ds under xtd; else the plain byte(n, m) layout."""
            if tensor == 1:
                return MS_XT and (not MS_GSU or MS_XTD)
            return MS_XTD

        def ms_x_store(v8, par, tensor, nts=None):
            """f8b_ds_store(v8) into exchange region (par, tensor, grp)."""
            xb = ((arith.ArithValue(par) * _i32(2) + _i32(tensor)) * _i32(2)
                  + grp) * _i32(XR_B // 4)
            if ms_xt_layout(tensor):
                assert DS8R % 16 == 0, "ms xt needs 16 B aligned rows"
                for nt in (range_constexpr(NT) if nts is None else nts):
                    n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
                    off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                           + l4 * _i32(4))
                    w4 = vector.bitcast(T.vec(4, T.i32), vector.shuffle(
                        vector.shuffle(v8[nt][0], v8[nt][1], [0, 1, 2, 3]),
                        vector.shuffle(v8[nt][2], v8[nt][3], [0, 1, 2, 3]),
                        list(range(8))))
                    vector.store(w4, kt8_lds, [_idx(off)], alignment=16)
                return
            for nt in (range_constexpr(NT) if nts is None else nts):
                n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
                for ms in range_constexpr(MSTEP):
                    off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                           + _i32(4 * ms) + l4)
                    vector.store(
                        vector.bitcast(T.vec(1, T.i32), v8[nt][ms]),
                        kt8_lds, [_idx(off)], alignment=4)

        def ms_x_cat(par, tensor, tile, nt):
            """_cat(x8[nt]) of tile `tile` read back from the exchange: the
            same 4 dwords (m = 16*ms + 4*l4 .. +3, ms = 0..3) at row n."""
            xb = ((arith.ArithValue(par) * _i32(2) + _i32(tensor)) * _i32(2)
                  + _i32(tile)) * _i32(XR_B // 4)
            n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
            if ms_xt_layout(tensor):
                off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                       + l4 * _i32(4))
                return vector.bitcast(T.vec(8, T.i16), vector.load_op(
                    T.vec(4, T.i32), kt8_lds, [_idx(off)]))
            p = []
            for ms in range_constexpr(MSTEP):
                off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                       + _i32(4 * ms) + l4)
                p.append(vector.bitcast(T.vec(2, T.i16), vector.load_op(
                    T.vec(1, T.i32), kt8_lds, [_idx(off)])))
            return vector.shuffle(
                vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                list(range(8)))

        def ms_bop(par, j):
            """_bop(4*grp + j) of pipe_pair, tiles of parity `par`."""
            v8i32_t = T.vec(8, T.i32)
            gofs = arith.ArithValue(grp) * _i32(2 * Q8BS)

            def _b8(tile, t, r):
                tb = ((arith.ArithValue(par) * _i32(4) + _i32(2 * tile + t))
                      * _i32(TILE_B))
                off = (tb + gofs
                       + _i32((j // 2) * Q8BS + 16 * (j % 2) + 32 * 32 * r)
                       + arith.ArithValue(t8_m) * _i32(32) + t8_k8)
                return vector.bitcast(T.vec(4, T.i16), tr8(ms_t8, off))

            def _w(t):
                h = [vector.shuffle(_b8(tile, t, 0), _b8(tile, t, 1),
                                    list(range(8))) for tile in (0, 1)]
                return vector.bitcast(
                    v8i32_t, vector.shuffle(h[0], h[1], list(range(16))))
            return _w(1), _w(0)  # (dO^T, q^T)

        ms_pre = {}

        def ms_x_ops(par):
            """out-gemm A operands (silu -> dv, ds -> dk) of parity `par`."""
            v8i32_t = T.vec(8, T.i32)

            def _wide(a, b):
                return vector.bitcast(v8i32_t,
                                      vector.shuffle(a, b, list(range(16))))
            a_dv = [_wide(ms_x_cat(par, 1, 0, nt), ms_x_cat(par, 1, 1, nt))
                    for nt in range_constexpr(NT)]
            a_dk = [_wide(ms_x_cat(par, 0, 0, nt), ms_x_cat(par, 0, 1, nt))
                    for nt in range_constexpr(NT)]
            return a_dv, a_dk

        def ms_e8sel(e8a, e8b):
            return arith.select(
                arith.cmpi(arith.CmpIPredicate.ult, l4, _i32(2)), e8a, e8b)

        def ms_e8tail(e8a, e8b):
            """Carried K-block bytes of a pair (ek: the selected one, twice)."""
            if MS_EK:
                sel = ms_e8sel(e8a, e8b)
                return [sel, sel]
            return [e8a, e8b]

        # ---- R15 iter73 gsplit (gsp; LAB, default off) ----------------------
        # Wave (grp, wave) owns tensor grp (0 = dV, 1 = dK) for all NJG = NSTEP
        # d tiles of the wave's 32 n rows: acc[nt*NJG + jj], carried as the
        # shipped (acc_dv, acc_dk) lists (acc_dv = nt 0, acc_dk = nt 1).
        NJG = NSTEP

        def ms_bop_g(par, jj):
            """gsp: the B operand of d tile jj (0..NJG-1) for this group's
            tensor -- dO^T on group 0, q^T on group 1.  Byte for byte what
            ms_bop(par, jj % NJ)[0 or 1] reads on wave group jj // NJ: its
            gofs + (j//2)*Q8BS + 16*(j%2) equals (jj//2)*Q8BS + 16*(jj%2)."""
            v8i32_t = T.vec(8, T.i32)
            tsel = _i32(1) - arith.ArithValue(grp)  # tile slot: 1 dO, 0 q

            def _b8(tile, r):
                tb = ((arith.ArithValue(par) * _i32(4) + _i32(2 * tile)
                       + tsel) * _i32(TILE_B))
                off = (tb
                       + _i32((jj // 2) * Q8BS + 16 * (jj % 2) + 32 * 32 * r)
                       + arith.ArithValue(t8_m) * _i32(32) + t8_k8)
                return vector.bitcast(T.vec(4, T.i16), tr8(ms_t8, off))
            h = [vector.shuffle(_b8(tile, 0), _b8(tile, 1), list(range(8)))
                 for tile in (0, 1)]
            return vector.bitcast(
                v8i32_t, vector.shuffle(h[0], h[1], list(range(16))))

        def ms_xa_g(par, own=None):
            """gsp: this group's out-gemm A operands a[nt] = [tile 0 | tile 1]
            of its tensor -- silu (tensor 1) on group 0, ds (tensor 0) on
            group 1: the a_dv / a_dk halves of ms_x_ops, the same 4 dwords
            (m = 16*ms + 4*l4 .. +3) per tile at row n.  Default: one read
            sequence for both groups at the region base (par, 1 - grp, tile)
            (uniform SGPR), silu in ds's plain byte(n, m) layout.  gsb: a
            uniform scalar branch on grp, silu keeping the xt b128 layout.
            own (gspr, gsb only): the group's own-tile half per nt (v8i16,
            the lane's cast registers of its tile, byte-identical to the
            readback); the other half is still read back."""
            v8i32_t = T.vec(8, T.i32)

            def _wide(a, b):
                return vector.bitcast(v8i32_t,
                                      vector.shuffle(a, b, list(range(16))))
            if MS_GSU:
                # gsp without gsb: both tensors in the plain byte(n, m)
                # layout; the region (par, tensor = 1 - grp, tile) base is
                # uniform, so both groups issue the same reads.
                tsel = _i32(1) - arith.ArithValue(grp)

                def _cat_u(tile, nt):
                    xb = (((arith.ArithValue(par) * _i32(2) + tsel) * _i32(2)
                           + _i32(tile)) * _i32(XR_B // 4))
                    n_loc = (wave * _i32(16 * NT) + _i32(16 * nt) + l16)
                    if MS_XTD:   # both tensors xt: one b128 per (tile, nt)
                        off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                               + l4 * _i32(4))
                        return vector.bitcast(T.vec(8, T.i16), vector.load_op(
                            T.vec(4, T.i32), kt8_lds, [_idx(off)]))
                    p = []
                    for ms in range_constexpr(MSTEP):
                        off = (xb + arith.ArithValue(n_loc) * _i32(DS8R // 4)
                               + _i32(4 * ms) + l4)
                        p.append(vector.bitcast(T.vec(2, T.i16),
                                                vector.load_op(
                                                    T.vec(1, T.i32), kt8_lds,
                                                    [_idx(off)])))
                    return vector.shuffle(
                        vector.shuffle(p[0], p[1], [0, 1, 2, 3]),
                        vector.shuffle(p[2], p[3], [0, 1, 2, 3]),
                        list(range(8)))
                assert own is None, "gspr needs gsb"
                return [_wide(_cat_u(0, nt), _cat_u(1, nt))
                        for nt in range_constexpr(NT)]
            br = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0)),
                          results_=[v8i32_t] * NT, has_else=True)
            with ir.InsertionPoint(br.then_block):
                a = [_wide(ms_x_cat(par, 1, 0, nt) if own is None else own[nt],
                           ms_x_cat(par, 1, 1, nt))
                     for nt in range_constexpr(NT)]
                scf.YieldOp([arith.unwrap(x) for x in a])
            with ir.InsertionPoint(br.else_block):
                a = [_wide(ms_x_cat(par, 0, 0, nt),
                           ms_x_cat(par, 0, 1, nt) if own is None else own[nt])
                     for nt in range_constexpr(NT)]
                scf.YieldOp([arith.unwrap(x) for x in a])
            return list(br.results)

        def ms_out_g(par, e8a, e8b, acc_dv, acc_dk, loop=False, own=None):
            """gsp out-gemm of one pair: every MFMA of this group's tensor,
            d tile jj = 0..NJG-1, nt = 0 then 1 (the two MFMAs of one jj
            share B).  Same (A, B, scale A, scale B, C) per accumulator as
            the shipped ms_out: dV mfma_k128_sc(a, b_o, acc, sc127, sc_kb),
            dK mfma_k128_sc(a, b_q, acc, sc_kb, sc127)."""
            acc = list(acc_dv) + list(acc_dk)
            sc_kb = e8a if MS_EK else ms_e8sel(e8a, e8b)
            g0 = arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0))
            sa = arith.select(g0, sc127, sc_kb)
            sb = arith.select(g0, sc_kb, sc127)
            a = ms_xa_g(par, own)
            la = (2 if MS_GSJ2 else 1) if (loop and MS_OBJ) else 0
            bq = {}
            if loop and MS_OB0:
                bq[0] = ms_pre.pop("ob0")
            for jj in range(min(la, NJG - 1) + 1):
                if jj not in bq:
                    bq[jj] = ms_bop_g(par, jj)
            for jj in range_constexpr(NJG):
                if jj + la < NJG and (jj + la) not in bq:
                    bq[jj + la] = ms_bop_g(par, jj + la)
                b = bq.pop(jj)
                nts = list(range(NT))
                if MS_OSN and jj % 2 == 1:
                    nts = nts[::-1]
                for nt in nts:
                    i = nt * NJG + jj
                    if MS_OSNB and (jj, nt) != (0, 0):
                        rocdl.sched_barrier(0x7F6)
                    acc[i] = mfma_k128_sc(a[nt], b, acc[i], sa, sb)
            na = len(acc_dv)
            return acc[:na], acc[na:]

        def ms_out(par, e8a, e8b, acc_dv, acc_dk, js=None, loop=False,
                   own=None):
            """K=128 out-gemm of one pair for this group's NJ d tiles: the
            mfma_k128_sc calls of pipe_pair._out, operand for operand."""
            if MS_GSP:
                assert js is None
                return ms_out_g(par, e8a, e8b, acc_dv, acc_dk, loop=loop,
                                own=own)
            v8i32_t = T.vec(8, T.i32)

            def _wide(a, b):
                return vector.bitcast(v8i32_t,
                                      vector.shuffle(a, b, list(range(16))))
            sc_kb = e8a if MS_EK else ms_e8sel(e8a, e8b)
            if loop and (MS_OBJ or MS_OB0) and js is None:
                # ob0/obj (loop body only): same MFMAs in the same (j, nt)
                # order; only the B-operand reads move ahead (B(j+1) before
                # the MFMAs of j with obj; B(0) issued before the sb fence
                # with ob0).
                a_dv, a_dk = ms_x_ops(par)
                new_dv = list(acc_dv)
                new_dk = list(acc_dk)
                la = 1 if MS_OBJ else 0
                bq = {}
                if MS_OB0:
                    bq[0] = ms_pre.pop("ob0")
                for j in range(min(la, NJ - 1) + 1):
                    if j not in bq:
                        bq[j] = ms_bop(par, j)
                for j in range_constexpr(NJ):
                    if j + la < NJ and (j + la) not in bq:
                        bq[j + la] = ms_bop(par, j + la)
                    b_o, b_q = bq.pop(j)
                    for nt in range_constexpr(NT):
                        i = nt * NJ + j
                        new_dv[i] = mfma_k128_sc(a_dv[nt], b_o, acc_dv[i],
                                                 sc127, sc_kb)
                        new_dk[i] = mfma_k128_sc(a_dk[nt], b_q, acc_dk[i],
                                                 sc_kb, sc127)
                return new_dv, new_dk
            if ms_pre.get("ax") is not None:
                a_dv, a_dk = ms_pre.pop("ax")
            else:
                a_dv, a_dk = ms_x_ops(par)
            new_dv = list(acc_dv)
            new_dk = list(acc_dk)
            if MS_NTM:
                for nt in range_constexpr(NT):
                    for j in range_constexpr(NJ):
                        b_o, b_q = ms_bop(par, j)
                        i = nt * NJ + j
                        new_dv[i] = mfma_k128_sc(a_dv[nt], b_o, acc_dv[i],
                                                 sc127, sc_kb)
                        new_dk[i] = mfma_k128_sc(a_dk[nt], b_q, acc_dk[i],
                                                 sc_kb, sc127)
                return new_dv, new_dk
            for j in (range_constexpr(NJ) if js is None else js):
                b_o, b_q = ms_bop(par, j)
                for nt in range_constexpr(NT):
                    i = nt * NJ + j
                    new_dv[i] = mfma_k128_sc(a_dv[nt], b_o, acc_dv[i],
                                             sc127, sc_kb)
                    new_dk[i] = mfma_k128_sc(a_dk[nt], b_q, acc_dk[i],
                                             sc_kb, sc127)
            return new_dv, new_dk

        def ms_score_act(mA, par, masked):
            """pipe_score + act + cast of THIS group's tile of the pair at mA,
            published to the parity-`par` exchange (ds and silu)."""
            ms_act(mA, par, masked, ms_score(par))

        def ms_score(par):
            tq = ((arith.ArithValue(par) * _i32(4) + arith.ArithValue(grp)
                   * _i32(2)) * _i32(TILE_DW))
            to = tq + _i32(TILE_DW)
            a_qk = [ms_a8(tq, ms) for ms in range_constexpr(MSTEP)]
            a_dp = [ms_a8(to, ms) for ms in range_constexpr(MSTEP)]
            if MS_LPOS in ("la2", "la4") and "ldA" in ms_pre:
                ms_pre["nAB"] = (ms_pre.pop("ldA")(), ms_pre.pop("ldB")())
                if MS_LPOS == "la4":
                    rocdl.sched_barrier(0)
            # valu tsc: the S MFMAs (not dP) carry the E8M0 scale 2^k.
            mfma_s = mfma_qk_k128_tsc if VALU_TSC else mfma_qk_k128
            qk, dp = [], []
            if MS_BST:
                # bst: B-stationary program order K0 x4, V0 x4, K1 x4, V1 x4
                # (the loop-invariant K/V operand repeats on srcB).  Each
                # accumulator is one C=0 MFMA, so any order is bitwise.
                qk = [[None] * MSTEP for _ in range(NT)]
                dp = [[None] * MSTEP for _ in range(NT)]
                for nt in range_constexpr(NT):
                    for ms in range_constexpr(MSTEP):
                        qk[nt][ms] = mfma_s(a_qk[ms], kf8[nt][0],
                                            zero_acc)
                    for ms in range_constexpr(MSTEP):
                        dp[nt][ms] = mfma_qk_k128(a_dp[ms], vf8[nt],
                                                  zero_acc)
                return qk, dp
            for nt in range_constexpr(NT):
                qk.append([])
                dp.append([])
                for ms in range_constexpr(MSTEP):
                    qk[nt].append(mfma_s(a_qk[ms], kf8[nt][0],
                                         zero_acc))
                    dp[nt].append(mfma_qk_k128(a_dp[ms], vf8[nt],
                                               zero_acc))
            return qk, dp

        def ms_act(mA, par, masked, qkdp):
            qk, dp = qkdp
            m_own = arith.ArithValue(mA) + arith.ArithValue(grp) * _i32(BM)
            _, _, sil, dsv, _ = act_tile(m_own, qk, dp, masked)
            s8, d8 = f8b_cast(sil, dsv)
            ms_x_store(d8, par, 0)
            ms_x_store(s8, par, 1)
            if MS_GSPR:
                ms_pre["own_new"] = [ms_own_sel(s8[nt], d8[nt])
                                     for nt in range_constexpr(NT)]

        def ms_own_sel(s8l, d8l):
            """gspr: this group's own-tile A half for one nt (v8i16): its
            silu cast (group 0, tile 0) or ds cast (group 1, tile 1), in the
            readback's dword order (ms_x_cat: m-subtile ms = 0..3)."""
            def _cat(x):
                return vector.shuffle(
                    vector.shuffle(x[0], x[1], [0, 1, 2, 3]),
                    vector.shuffle(x[2], x[3], [0, 1, 2, 3]),
                    list(range(8)))
            return arith.select(
                arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0)),
                _cat(s8l), _cat(d8l))

        def ms_ws(mA, par, a8=None, oob=None):
            """dqsplit_ws_store of this group's tile of the pair at mA."""
            if a8 is not None:
                dqsplit_ws_write(
                    arith.ArithValue(mA) + arith.ArithValue(grp) * _i32(BM),
                    a8, oob)
                return
            xb = ms_ws_xb(par)
            dqsplit_ws_store(
                arith.ArithValue(mA) + arith.ArithValue(grp) * _i32(BM), xb,
                oob)

        def ms_ws_xb(par):
            return (((arith.ArithValue(par) * _i32(2)) * _i32(2)
                     + arith.ArithValue(grp)) * _i32(XR_B))

        def ms_step(mA, par, pprev, e8p, acc_dv, acc_dk, first=False):
            """One barrier interval: ws + out-gemm of the previous pair (parity
            pprev, kblock bytes e8p) and score/act of the pair at mA."""
            if first:
                ms_score_act(mA, par, True)
                return acc_dv, acc_dk
            if not (MSPLIT == 1 and MS_MID and (MS_WSL or MS_WSE)) and not MS_WSB:
                ms_ws(arith.ArithValue(mA) - _i32(2 * BM), pprev)
            if MSPLIT == 1 and MS_MID:
                # out-gemm(k-1) emitted between the act of n tile 0 and n
                # tile 1 of score(k); tile 0 is cast + published first.
                ws_a8 = dqsplit_ws_read(ms_ws_xb(pprev)) if MS_WSS else None
                qk, dp = ms_score(par)
                if MS_LPOS == "la":
                    ms_pre["nAB"] = (ms_pre["ldA"](), ms_pre["ldB"]())
                if MS_LPOS == "ls":
                    ms_pre["nA"] = ms_pre["ldA"]()
                box = {}

                def _hook(nt, s32, d32):
                    if nt == 0 and MS_OB0:
                        # ob0: B(0) of out-gemm(k-1) ahead of the tile-0 cast
                        # and exchange store (before the sb fence); the tiles
                        # of parity pprev are stable from B1(k-1) to W(k+1).
                        ms_pre["ob0"] = (ms_bop_g(pprev, 0) if MS_GSP
                                         else ms_bop(pprev, 0))
                    s8 = [[pk4_from_f32(s32[ms], mu2_ssc,
                                        SAT_HI, rt=vf4_sop)
                           for ms in range_constexpr(MSTEP)]]
                    d8 = [[pk4_from_f32(d32[ms], mu2_dsc,
                                        SAT_BOTH)
                           for ms in range_constexpr(MSTEP)]]
                    v8d = [None] * NT
                    v8s = [None] * NT
                    v8d[nt], v8s[nt] = d8[0], s8[0]
                    ms_x_store(v8d, par, 0, nts=[nt])
                    ms_x_store(v8s, par, 1, nts=[nt])
                    if MS_GSPR:
                        ms_pre.setdefault("own_new", [None] * NT)[nt] = (
                            ms_own_sel(s8[0], d8[0]))
                    if nt == 0 and MS_OH:
                        h = NJ // 2
                        dv_, dk_ = ms_out(pprev, e8p[0], e8p[1], acc_dv,
                                          acc_dk, js=list(range(h)))
                        rocdl.sched_barrier(0)
                        box["acc"] = ms_out(pprev, e8p[0], e8p[1], dv_, dk_,
                                            js=list(range(h, NJ)))
                    elif nt == 0:
                        if MS_SB:
                            if MS_IA:
                                rocdl.iglp_opt(2)
                            rocdl.sched_barrier(0)
                        box["acc"] = ms_out(pprev, e8p[0], e8p[1], acc_dv,
                                            acc_dk, loop=True,
                                            own=ms_pre.pop("own_prev", None))
                    if nt == 0 and MS_WSL and MS_WSX:
                        box["a8"] = dqsplit_ws_read(ms_ws_xb(pprev))
                    elif nt == 0 and MS_WSL and not MS_WSB:
                        ms_ws(arith.ArithValue(mA) - _i32(2 * BM), pprev,
                              ws_a8)
                    if nt == 0 and MS_LPOS == "lo":
                        ms_pre["nAB"] = (ms_pre["ldA"](), ms_pre["ldB"]())
                    if nt == 0 and MS_LPOS == "ls":
                        ms_pre["nAB"] = (ms_pre.pop("nA"), ms_pre["ldB"]())
                    if nt == 0 and MS_B2E:
                        gpu.barrier()
                m_own = arith.ArithValue(mA) + arith.ArithValue(grp) * _i32(BM)
                act_tile(m_own, qk, dp, False, hook=_hook)
                if MS_LPOS == "le":
                    ms_pre["nAB"] = (ms_pre["ldA"](), ms_pre["ldB"]())
                if MS_WSE:
                    ms_ws(arith.ArithValue(mA) - _i32(2 * BM), pprev)
                if MS_WSX:
                    ms_ws(arith.ArithValue(mA) - _i32(2 * BM), pprev,
                          box["a8"])
                return box["acc"]
            if MSPLIT == 1 and MS_SO:
                qkdp = ms_score(par)
                acc_dv, acc_dk = ms_out(pprev, e8p[0], e8p[1], acc_dv, acc_dk)
                ms_act(mA, par, False, qkdp)
                return acc_dv, acc_dk
            if MSPLIT == 1 and MS_ANTI:
                na = len(acc_dv)
                for slot in range_constexpr(2):
                    do_out = arith.cmpi(arith.CmpIPredicate.eq, grp,
                                        _i32(slot))
                    br = scf.IfOp(do_out, results_=[T.f32x4] * (2 * na),
                                  has_else=True)
                    with ir.InsertionPoint(br.then_block):
                        dv_, dk_ = ms_out(pprev, e8p[0], e8p[1], acc_dv,
                                          acc_dk)
                        scf.YieldOp([arith.unwrap(x) for x in dv_ + dk_])
                    with ir.InsertionPoint(br.else_block):
                        ms_score_act(mA, par, False)
                        scf.YieldOp([arith.unwrap(x)
                                     for x in acc_dv + acc_dk])
                    res = list(br.results)
                    acc_dv, acc_dk = res[:na], res[na:]
                return acc_dv, acc_dk
            if MSPLIT == 1:
                acc_dv, acc_dk = ms_out(pprev, e8p[0], e8p[1], acc_dv, acc_dk)
                ms_score_act(mA, par, False)
                return acc_dv, acc_dk
            na = len(acc_dv)
            g0 = arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0))
            br = scf.IfOp(g0, results_=[T.f32x4] * (2 * na), has_else=True)
            with ir.InsertionPoint(br.then_block):
                dv_, dk_ = ms_out(pprev, e8p[0], e8p[1], acc_dv, acc_dk)
                ms_score_act(mA, par, False)
                scf.YieldOp([arith.unwrap(x) for x in dv_ + dk_])
            with ir.InsertionPoint(br.else_block):
                ms_score_act(mA, par, False)
                dv_, dk_ = ms_out(pprev, e8p[0], e8p[1], acc_dv, acc_dk)
                scf.YieldOp([arith.unwrap(x) for x in dv_ + dk_])
            res = list(br.results)
            return res[:na], res[na:]

        # ---- m loop ---------------------------------------------------------
        # The causal test m >= n is only non-trivial while the m tile [m0, m0+BM)
        # overlaps the block's n range [n0, n0+BN).  m0 = n0 + BM*it, so that is
        # the first BN/BM iterations and no others.  Peel exactly those, masked,
        # and run the loop from there with no mask at all -- the hand-written
        # assembly does the same thing in .Lpeel.  Peeling past the end of a short
        # sequence is harmless: those rows read back as zero and contribute zero.
        kt_frag = None
        if MS_NK:
            # dqsplit + msplit: kt_frag is never read (see MS_NK above).
            assert DQSPLIT and MSPLIT and F8B, "nk needs dqsplit + msplit (C8)"
        elif F8B:
            # K as e4m3, staged in the bf16p interleave but in BYTES: plane
            # p = 2*(d//32) + d%2 holds [BN rows][16 cols], col = (d%32)//2.
            # A thread's 8 consecutive d split into 4 even + 4 odd bytes, one
            # dword store per parity.
            KROWS = BN * D // (THREADS * 8)
            st_n = (tid // _i32(16)) * _i32(KROWS)
            w32 = st_d // _i32(32)
            col0 = (st_d % _i32(32)) // _i32(2)
            v8i8 = T.vec(8, i8_t)
            for r in range_constexpr(KROWS):
                n_row = st_n + _i32(r)
                n_safe = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, n0 + n_row, seq_len),
                    n0 + n_row, seq_last,
                )
                raw = vector.bitcast(v8i8, load_raw8(
                    rK, arith.ArithValue(n_safe) * _i32(RS_IN) + st_d))
                for par in range_constexpr(2):
                    half = vector.bitcast(T.vec(1, T.i32), vector.shuffle(
                        raw, raw, [par, 2 + par, 4 + par, 6 + par]))
                    off = (((arith.ArithValue(w32) * _i32(2) + _i32(par))
                            * _i32(BN) + n_row) * _i32(16) + col0) // _i32(4)
                    vector.store(half, kt8_lds, [_idx(off)], alignment=4)
            gpu.barrier()
            # B operand for d tile dt of wave w: column j <-> d = 32w + 2j + dt,
            # k byte b of lane group l4 <-> n = 32*l4 + b.
            kt_frag = []
            for dt in range_constexpr(DQD):
                parts = [tr8(kt8_lds,
                             ((wave * _i32(2) + _i32(dt)) * _i32(BN)
                              + t8_n + _i32(8 * r)) * _i32(16) + t8_k8)
                         for r in range_constexpr(4)]
                kt_frag.append(vector.shuffle(
                    vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                    vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                    list(range(8))))
            gpu.barrier()  # K is consumed; the same bytes become ds below
        elif fused:
            # Stage K once in the tr16 layout and pull dq's B operands into
            # registers.  They are loop-invariant, so this is paid once per
            # workgroup and the LDS is handed straight over to the ds exchange.
            KROWS = BN * D // (THREADS * 8)
            st_n = (tid // _i32(16)) * _i32(KROWS)
            for r in range_constexpr(KROWS):
                n_row = st_n + _i32(r)
                n_safe = arith.select(
                    arith.cmpi(arith.CmpIPredicate.slt, n0 + n_row, seq_len),
                    n0 + n_row, seq_last,
                )
                vec = load_qkv(
                    rK, arith.ArithValue(n_safe) * _i32(RS_IN) + st_d
                )
                if dq16 == "bf16p":
                    # Interleave d across the two 16-wide tiles a wave owns, so
                    # tile 2w holds the EVEN d of [32w, 32w+32) and tile 2w+1 the
                    # odd ones, both at column (d % 32) / 2.  The mfma then hands
                    # lane j the pair (32w + 2j, 32w + 2j + 1) in its two
                    # accumulators, and 16 lanes packing that pair span a whole
                    # 64 B line -- which is the unit the atomic roofline counts.
                    #
                    # A thread loads 8 consecutive d starting at a multiple of 8,
                    # so d // 32 is constant over them and the even and odd
                    # halves each land on 4 CONSECUTIVE columns: two 8-byte
                    # stores, not eight 2-byte ones.
                    w32 = st_d // _i32(32)
                    col0 = (st_d % _i32(32)) // _i32(2)
                    for par in range_constexpr(2):
                        half = vector.from_elements(v4bf, [
                            vector.extract(vec, static_position=[2 * e + par],
                                           dynamic_position=[])
                            for e in range_constexpr(4)
                        ])
                        off = ((arith.ArithValue(w32) * _i32(2) + _i32(par))
                               * _i32(BN) + n_row) * _i32(16) + col0
                        vector.store(half, kt_lds, [_idx(off)], alignment=8)
                    continue
                off = (arith.ArithValue(st_db) * _i32(BN) + n_row) * _i32(16) + st_din
                vector.store(vec, kt_lds, [_idx(off)], alignment=16)
            gpu.barrier()
            # Wave w owns d in [16*DQD*w, +16*DQD).
            dq_row0 = wave * _i32(DQD * BN)
            kt_frag = [[tr_pair(kt_lds, dq_row0 + _i32(dt * BN), t)
                        for t in range(DQK)] for dt in range(DQD)]
            gpu.barrier()  # K is consumed; the same bytes become ds below

        NACC = NT * (NSTEP // 2 if MSPLIT else NSTEP)
        n_iter = (seq_len - n0 + _i32(BM - 1)) // _i32(BM)
        acc = [arith.constant_vector(0.0, T.f32x4) for _ in range(2 * NACC)]
        pk_dv, pk_dk = acc[:NACC], acc[NACC:]
        pend = None  # MU2: the first m tile's half-operands, awaiting a partner
        assert not (MU2 and rev), "mu2 pairs m tiles in ascending order only"
        if rev:
            # Descend: every n_block of a (batch, head) starts at the SAME m,
            # the last tile of the sequence, instead of at its own diagonal.
            # Concurrent workgroups then walk the same dq_acc rows in step, so
            # their read-modify-writes collide in cache rather than each
            # streaming its own private stripe.  Accumulation is a sum, so the
            # order is free; the masked peel just moves to the end.
            # A sequence shorter than the peel makes the loop empty, and then
            # `staged` has to already be the first peel tile.
            last = arith.select(
                arith.cmpi(arith.CmpIPredicate.sgt, n_iter, _i32(NPEEL)),
                n_iter, _i32(NPEEL))
            staged = load_tiles(n0 + (arith.ArithValue(last) - _i32(1)) * _i32(BM))
        elif MSPLIT:
            if MS_PRIO:
                pb = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0)),
                              results_=[], has_else=False)
                with ir.InsertionPoint(pb.then_block):
                    rocdl.s_setprio(1)
                    scf.YieldOp([])
            # Peel = pair 0 (masked), parity 0: stage both tiles, score/act
            # this group's tile; its out-gemm runs in the next interval.
            stA = load_tiles(n0)
            stB = load_tiles(n0 + _i32(BM))
            ms_write_tile(stA, _i32(0), 0)
            ms_write_tile(stB, _i32(0), 1)
            gpu.barrier()
            nA = load_tiles(n0 + _i32(2 * BM))
            nB = load_tiles(n0 + _i32(3 * BM))
            ms_step(n0, _i32(0), None, None, pk_dv, pk_dk, first=True)
            if MS_B2EW:
                # Tile parity 1 is unread so far: W(1) needs no barrier.
                ms_write_tile(nA, _i32(1), 0)
                ms_write_tile(nB, _i32(1), 1)
            gpu.barrier()  # pair 0's silu/ds exchange is complete
            # Carried: next pair's staged loads + this pair's kblock bytes.
            staged = nA + nB + ms_e8tail(stA[E8S][0], stB[E8S][0])
            if MS_GSPR:
                staged = staged + ms_pre.pop("own_new")
        elif PIPE:
            # Peel = the first pair, masked.  Both tiles staged, one barrier.
            stA = load_tiles(n0)
            stB = load_tiles(n0 + _i32(BM))
            write_tiles(stA)
            write_tiles(stB, (q8b_lds, o8b_lds))
            gpu.barrier()
            nA = load_tiles(n0 + _i32(2 * BM))
            nB = None
            if PIPE == 1:
                nB = load_tiles(n0 + _i32(3 * BM))
            pk_dv, pk_dk, nb = pipe_pair(
                n0, pk_dv, pk_dk, True, stA[8:10], stB[8:10],
                prefetch_b=(None if PIPE == 1 else
                            (lambda: load_tiles(n0 + _i32(3 * BM)))))
            staged = nA + (nB if PIPE == 1 else nb)
        else:
            staged = load_tiles(n0)
            for p in range(NPEEL):
                # Redundant in the fused path: the previous sync (the K-consumed
                # barrier for p==0, the prior one_iter ds-exchange barrier for
                # p>0) already follows the last q_rm/o_rm read.  See the main loop.
                if not fused:
                    gpu.barrier()
                write_tiles(staged)
                gpu.barrier()
                e8_t = staged[8:10] if do_blk else None
                staged = load_tiles(n0 + _i32(BM * (p + 1)))
                pk_dv, pk_dk, pend = one_iter(n0 + _i32(BM * p), pk_dv, pk_dk,
                                              True, pend, e8=e8_t)

        lb = arith.constant(NPEEL, index=True)
        if MU2:
            # Two m tiles per body.  Round the trip count UP to an even number
            # of tiles past the peel: an extra tile reads q/dO past the
            # descriptor's num_records, gets zeros, and contributes exactly zero
            # to dv/dk/dq -- the same argument the existing one-tile prefetch
            # overrun already relies on.
            rem = arith.select(
                arith.cmpi(arith.CmpIPredicate.sgt, n_iter, _i32(NPEEL)),
                arith.ArithValue(n_iter) - _i32(NPEEL), _i32(0))
            ub_i32 = (_i32(NPEEL)
                      + ((arith.ArithValue(rem) + _i32(1)) // _i32(2))
                      * _i32(2))
            ub = arith.index_cast(T.index, arith.unwrap(ub_i32))
        else:
            ub = arith.index_cast(T.index, n_iter)
        step = arith.constant(2 if MU2 else 1, index=True)
        LC32 = bool(MSPLIT and MS_LC)
        if LC32:
            lb, ub, step = _i32(NPEEL), ub_i32, _i32(2)
        init = [arith.unwrap(x) for x in pk_dv + pk_dk + staged]
        loop = scf.ForOp(arith.unwrap(lb), arith.unwrap(ub), arith.unwrap(step), init)
        with ir.InsertionPoint(loop.body):
            it = arith.ArithValue(
                loop.induction_variable if LC32 else
                arith.index_cast(T.i32, loop.induction_variable))
            # Ascending induction either way; when reversed it indexes the tiles
            # from the top down, so the trip count and the loop bounds are
            # unchanged and only the address mapping flips.
            m0 = (n0 + (arith.ArithValue(n_iter) + _i32(NPEEL - 1) - it) * _i32(BM)
                  if rev else n0 + it * _i32(BM))
            acc_dv = list(loop.inner_iter_args[:NACC])
            acc_dk = list(loop.inner_iter_args[NACC : 2 * NACC])
            staged = list(loop.inner_iter_args[2 * NACC :])
            # The top-of-loop barrier only needs to keep write_tiles from
            # overwriting q_rm/o_rm before the PREVIOUS iteration's readers of
            # those tiles finish.  In the FUSED path the previous iteration's
            # ds-exchange barrier (one_iter, `if fused`) already fires AFTER the
            # last q_rm/o_rm read (all a_frag score reads and b_frag_tr out-gemm
            # reads precede it) and before write_tiles here, so this barrier is
            # redundant -- ds_lds (off_kt), the only LDS touched between them, is
            # a disjoint allocation from q_rm/o_rm.  Dropping it removes one of
            # three full-workgroup syncs per iteration on a latency-bound body.
            # The non-fused body has no such barrier, so it must keep this one.
            if not fused:
                gpu.barrier()
            if MSPLIT:
                # Pair k = it/2 at m0, parity k%2.  Its tiles' buffer was last
                # read by pair k-2's score (interval k-2) and out-gemm
                # (interval k-1): both before the previous closing barrier.
                NOWN = NT if MS_GSPR else 0
                NST = (len(staged) - 2 - NOWN) // 2
                e8p = staged[2 * NST:2 * NST + 2]
                if MS_GSPR:
                    ms_pre["own_prev"] = staged[2 * NST + 2:]
                stA, stB = staged[:NST], staged[NST:2 * NST]
                par = (it // _i32(2)) % _i32(2)
                pprev = _i32(1) - par
                if MS_WSB:
                    ms_ws(arith.ArithValue(m0) - _i32(2 * BM), pprev)
                if MS_XE:
                    ms_pre["ax"] = ms_x_ops(pprev)
                if not MS_B2EW:
                    ms_write_tile(stA, par, 0)
                    ms_write_tile(stB, par, 1)
                if MS_LB:
                    nA = load_tiles(m0 + _i32(2 * BM), MS_RS)
                    nB = load_tiles(m0 + _i32(3 * BM), MS_RS)
                gpu.barrier()
                if MS_LA and MS_RS2:
                    _rs2 = {}

                    def _ldA():
                        _rs2["p"] = load_pair_rs2(m0 + _i32(2 * BM))
                        return _rs2["p"][0]
                    ms_pre["ldA"] = _ldA
                    ms_pre["ldB"] = lambda: _rs2["p"][1]
                elif MS_LA:
                    ms_pre["ldA"] = lambda: load_tiles(m0 + _i32(2 * BM), MS_RS)
                    ms_pre["ldB"] = lambda: load_tiles(m0 + _i32(3 * BM), MS_RS)
                elif not MS_LB:
                    nA = load_tiles(m0 + _i32(2 * BM), MS_RS)
                    nB = load_tiles(m0 + _i32(3 * BM), MS_RS)
                acc_dv, acc_dk = ms_step(m0, par, pprev, e8p, acc_dv, acc_dk)
                if MS_LA:
                    nA, nB = ms_pre.pop("nAB")
                if MS_B2EW:
                    ms_write_tile(nA, pprev, 0)
                    ms_write_tile(nB, pprev, 1)
                # Pair k's exchange complete; pair k-1's readers done.
                if not MS_B2E:
                    gpu.barrier()
                nxt = nA + nB + ms_e8tail(stA[E8S][0], stB[E8S][0])
                if MS_GSPR:
                    nxt = nxt + ms_pre.pop("own_new")
            elif PIPE:
                NST = len(staged) // 2
                stA, stB = staged[:NST], staged[NST:]
                write_tiles(stA)
                write_tiles(stB, (q8b_lds, o8b_lds))
                gpu.barrier()
                nA = load_tiles(m0 + _i32(2 * BM))
                if PIPE == 1:
                    nB = load_tiles(m0 + _i32(3 * BM))
                acc_dv, acc_dk, nb = pipe_pair(
                    m0, acc_dv, acc_dk, False, stA[8:10], stB[8:10],
                    prefetch_b=(None if PIPE == 1 else
                                (lambda: load_tiles(m0 + _i32(3 * BM)))))
                nxt = nA + (nB if PIPE == 1 else nb)
            else:
                write_tiles(staged)
                gpu.barrier()
                # Issue the next tile's global loads before the math so the HBM
                # latency overlaps the mfma stream instead of stalling on it.
                nxt = load_tiles(m0 - _i32(BM) if rev else m0 + _i32(BM))
                if MU2:
                    # First half of the pair: score math + dq for tile m0, and the
                    # K=128 A/B halves parked in registers.  No out-gemm yet.
                    acc_dv, acc_dk, ph = one_iter(
                        m0, acc_dv, acc_dk, False, None,
                        e8=staged[8:10] if do_blk else None)
                    # Re-stage LDS for tile m0+BM.  The barrier pairing is exactly
                    # the one the un-unrolled body uses between consecutive
                    # iterations; the fused ds-exchange barrier inside one_iter
                    # already separates the previous readers of q_rm/o_rm.
                    write_tiles(nxt)
                    gpu.barrier()
                    e8_2 = nxt[8:10] if do_blk else None
                    nxt = load_tiles(m0 + _i32(2 * BM))
                    # Second half: same score math, then ONE K=128 out-gemm per
                    # (n-tile, d-tile) covering both m tiles.
                    acc_dv, acc_dk, _ = one_iter(m0 + _i32(BM), acc_dv, acc_dk,
                                                 False, ph, e8=e8_2)
                else:
                    acc_dv, acc_dk, _ = one_iter(m0, acc_dv, acc_dk, False)
            if MSPLIT and ms_tok:
                ms_emit_sched()
            else:
                emit_sched()
            scf.YieldOp([arith.unwrap(x) for x in acc_dv + acc_dk + nxt])

        res_dv = list(loop.results[:NACC])
        res_dk = list(loop.results[NACC : 2 * NACC])
        if MSPLIT:
            # Drain: ws + out-gemm of the last pair (the peel pair when the
            # loop ran zero times: ub_i32 == NPEEL -> tile 0, parity 0).
            fin = list(loop.results[2 * NACC:])
            if MS_MC:
                assert VALU_MSAT, "mc needs valu msat"
                rocdl.sched_barrier(0)
                llvm.InlineAsmOp(
                    None, [],
                    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n"
                    "s_nop 7\ns_nop 7", "", has_side_effects=True)
                rocdl.sched_barrier(0)
            if MS_B2E:
                gpu.barrier()  # last pair's exchange is complete
            last_t = arith.ArithValue(ub_i32) - _i32(2)
            plast = (last_t // _i32(2)) % _i32(2)
            mlast = n0 + last_t * _i32(BM)
            if MS_DZR:
                # dzr: as dz per ws ROW (lane): every row of the last pair
                # at/after seq_len is dropped, not only whole tiles.
                assert ws_l128t, "dzr is wired for the l128t ws layout"
                ms_ws(mlast, plast, oob="rows")
            elif MS_DZ:
                # dz: a tile starting at/after seq_len is padding; dq
                # never stores its rows, so its ws store is dead.  Push it
                # past num_records (nb*(nb+1)*8 KiB < 2^31 for nb <= 511) by
                # +2^31 (no 32-bit wrap) and the buffer unit drops it.
                ws_live = arith.cmpi(
                    arith.CmpIPredicate.slt,
                    arith.ArithValue(mlast) + arith.ArithValue(grp) * _i32(BM),
                    seq_len)
                ms_ws(mlast, plast, oob=arith.select(ws_live, _i32(0),
                                                     _i32(-(1 << 31))))
            else:
                ms_ws(mlast, plast)
            if MS_GSPR:
                res_dv, res_dk = ms_out(plast, fin[-2 - NT], fin[-1 - NT],
                                        res_dv, res_dk, own=fin[-NT:])
            else:
                res_dv, res_dk = ms_out(plast, fin[-2], fin[-1], res_dv,
                                        res_dk)
        if rev:
            # The masked tiles are the ones overlapping the n range, i.e. the
            # BOTTOM NPEEL of the m range, so descending reaches them last.
            staged = list(loop.results[2 * NACC :])
            for p in range(NPEEL - 1, -1, -1):
                if not fused:
                    gpu.barrier()
                write_tiles(staged)
                gpu.barrier()
                staged = load_tiles(n0 + _i32(BM * (p - 1)))
                res_dv, res_dk, _ = one_iter(n0 + _i32(BM * p), res_dv, res_dk,
                                             True)

        if VALU_MSAT and not (MSPLIT and MS_MC):
            # Clear MODE.FP16_OVFL before the bf16 dk/dv casts: with it set,
            # v_cvt_pk_bf16_f32 returns +-bf16_max instead of +-inf for
            # finite |x| >= 3.3961e38 (hw/modebf16.hip, 65536/2^32 patterns).
            llvm.InlineAsmOp(
                None, [],
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n"
                "s_nop 7\ns_nop 7", "", has_side_effects=True)
        if E8:
            (_e8_epilogue_gsp if MS_GSP else _e8_epilogue)(
                res_dv, res_dk, plast, e8, n0, wave, grp, l16, l4,
                tid, head, seq_start, seq_len, kt8_lds,
                dk_input_scale, value_input_scale,
                score_alpha_scale, inv_s_do, XR_B, vf4_undo=vf4_undo)
            return
        # ---- epilogue -------------------------------------------------------
        n_out_base = n0 + wave * _i32(16 * NT) + l4 * _i32(4)
        NDO = NSTEP // 2 if MSPLIT else NSTEP  # d tiles stored per wave
        if VALU_VF4:
            # uniform, from the prologue: 1/(mu2_ssc*sc*mu2_osc) * alpha*s
            dv_undo_vf4 = vf4_undo
        if MS_GSP:
            # gsp: wave (grp, wave) holds tensor grp for all NSTEP d tiles,
            # res[nt*NSTEP + jj] (res = res_dv + res_dk): group 0 stores dV,
            # group 1 dK (uniform branch), with the shipped per-element op
            # sequence below, to the shipped addresses (d = 16*jj + l16).
            res_g = list(res_dv) + list(res_dk)
            g_br = scf.IfOp(arith.cmpi(arith.CmpIPredicate.eq, grp, _i32(0)),
                            results_=[], has_else=True)
            for t_g, blk_g in ((0, g_br.then_block), (1, g_br.else_block)):
                with ir.InsertionPoint(blk_g):
                    for nt, jj in [(a, b) for a in range(NT)
                                   for b in range(NSTEP)]:
                        d_idx = l16 + _i32(16 * jj)
                        for i in range_constexpr(4):
                            n_idx = n_out_base + _i32(16 * nt) + _i32(i)
                            if VALU_NOBR:
                                st_if = None
                                ip = _NullCtx()
                            else:
                                ok = arith.cmpi(arith.CmpIPredicate.slt,
                                                n_idx, seq_len)
                                st_if = scf.IfOp(ok, results_=[],
                                                 has_else=False)
                                ip = ir.InsertionPoint(st_if.then_block)
                            with ip:
                                off = n_idx * _i32(RS_OUT) + d_idx
                                x_ = vector.extract(
                                    res_g[nt * NSTEP + jj],
                                    static_position=[i], dynamic_position=[])
                                if t_g == 0:
                                    dvv = x_
                                    if MU2 == "k128":
                                        dvv = arith.unwrap(
                                            arith.ArithValue(dvv)
                                            * (dv_undo_vf4 if VALU_VF4
                                               else _f32(1.0 / (mu2_ssc
                                                                * mu2_osc))))
                                        if do_dyn:
                                            dvv = arith.unwrap(
                                                arith.ArithValue(dvv)
                                                * inv_s_do)
                                    if VALU_VF2:
                                        dvv = arith.unwrap(
                                            arith.ArithValue(dvv)
                                            * score_alpha_scale)
                                    buffer_ops.buffer_store(
                                        arith.trunc_f(T.bf16, dvv), rDV, off)
                                else:
                                    dkv = arith.ArithValue(x_) * _f32(alpha)
                                    if MU2 == "k128":
                                        if do_dyn:
                                            dkv = (arith.ArithValue(dkv)
                                                   * inv_s_do)
                                        dkv = arith.ArithValue(dkv) * _f32(
                                            1.0 / (mu2_dsc * mu2_qsc))
                                    if input_fp8:
                                        dkv = (arith.ArithValue(dkv)
                                               * dk_input_scale)
                                    if VALU_VF2:
                                        dkv = (arith.ArithValue(dkv)
                                               * value_input_scale)
                                    buffer_ops.buffer_store(
                                        arith.trunc_f(T.bf16, dkv), rDK, off)
                                if st_if is not None:
                                    scf.YieldOp([])
                    scf.YieldOp([])
            return
        for nt, ds in [(a, b) for a in range(NT) for b in range(NDO)]:
            d_idx = l16 + _i32(16 * ds)
            if MSPLIT:
                d_idx = arith.ArithValue(d_idx) + grp * _i32(16 * NDO)
            for i in range_constexpr(4):
                n_idx = n_out_base + _i32(16 * nt) + _i32(i)
                if VALU_NOBR:
                    # Unconditional: the descriptor stops at row seq_len.
                    st_if = None
                    ip = _NullCtx()
                else:
                    ok = arith.cmpi(arith.CmpIPredicate.slt, n_idx, seq_len)
                    st_if = scf.IfOp(ok, results_=[], has_else=False)
                    ip = ir.InsertionPoint(st_if.then_block)
                with ip:
                    off = n_idx * _i32(RS_OUT) + d_idx
                    dvv = vector.extract(
                        res_dv[nt * NDO + ds], static_position=[i], dynamic_position=[]
                    )
                    dkv = arith.ArithValue(
                        vector.extract(
                            res_dk[nt * NDO + ds], static_position=[i], dynamic_position=[]
                        )
                    ) * _f32(alpha)
                    if MU2 == "k128":
                        # Undo the power-of-two pre-scales the fp8 casts folded
                        # in.  Both are exact, so this is a pure exponent shift.
                        # vf4: alpha*s rides in the same uniform factor.
                        dvv = arith.unwrap(
                            arith.ArithValue(dvv)
                            * (dv_undo_vf4 if VALU_VF4
                               else _f32(1.0 / (mu2_ssc * mu2_osc))))
                        if do_dyn:
                            dvv = arith.unwrap(
                                arith.ArithValue(dvv) * inv_s_do)
                            dkv = arith.ArithValue(dkv) * inv_s_do
                        dkv = arith.ArithValue(dkv) * _f32(
                            1.0 / (mu2_dsc * mu2_qsc))
                    if input_fp8:
                        dkv = arith.ArithValue(dkv) * dk_input_scale
                    if VALU_VF2:
                        dkv = arith.ArithValue(dkv) * value_input_scale
                        dvv = arith.unwrap(arith.ArithValue(dvv)
                                           * score_alpha_scale)
                    buffer_ops.buffer_store(arith.trunc_f(T.bf16, dvv), rDV, off)
                    buffer_ops.buffer_store(arith.trunc_f(T.bf16, dkv), rDK, off)
                    if st_if is not None:
                        scf.YieldOp([])

    # msplit's 512 threads: FlyDSL 0.2.4 needs known_block_size on the kernel
    # to launch more than 256 threads (_FLYC_KNOWN_BLOCK_SIZE).  Only then, so
    # the <= 256-thread builds and every FlyDSL 0.1.1 build keep the plain
    # decorator and their IR.
    if THREADS > 256 and _FLYC_KNOWN_BLOCK_SIZE:
        kernel_deco = flyc.kernel(known_block_size=[THREADS, 1, 1])
    else:
        kernel_deco = flyc.kernel

    # The build-time (Python) tests in the kernel / launch bodies below are
    # fx.const_expr(...): FlyDSL 0.2.4's AST rewriter turns every other `if`
    # into closures dispatched through scf_if_dispatch, and names first bound
    # inside a branch (batch, head, n_blk, grid) then stay local to the
    # closure.  FlyDSL 0.1.1 honours const_expr the same way (plain Python
    # if), so both versions trace the same ops.
    @kernel_deco
    def bwd_dkdv_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        DQ: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        INV_MAX_ARG: fx.Float32,
        QKV_SCALE_ARG: fx.Float32,
        NBATCH: fx.Int32,
        DO_SC: fx.Tensor,
    ):
        tid = _as_i32(fx.thread_idx.x)
        if fx.const_expr(VALU_MSAT):
            # MODE.FP16_OVFL = 1 for the whole wave: fp8 cvt saturates.
            llvm.InlineAsmOp(
                None, [],
                "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n"
                "s_nop 7\ns_nop 7", "", has_side_effects=True)
        if fx.const_expr(WAVES_HW != 4):
            # msplit: 512 threads.  Without an explicit range the ROCDL
            # translation stamps amdgpu-flat-work-group-size = 1,256 and a
            # 512-thread launch fails; the exact size also gives LLVM the
            # 2 waves/SIMD (256 register) budget.  (Under FlyDSL 0.2.4 the
            # kernel's known_block_size makes the lowering stamp the same
            # value; this keeps 0.1.1 builds working.)
            _set_flat_wg_size(THREADS)
        # Grid order.  In the dk/dv-only kernel n_block had to be the SLOWEST
        # dimension: under a causal mask a tile's cost grows with its n_block
        # index, so dispatching n_block fastest correlates cost with dispatch
        # position and leaves the CUs draining unevenly (that ordering was
        # worth -28%).  Fusing adds a competing pull -- every n_block of one
        # (batch, head) read-modify-writes the SAME dq_acc rows, so putting
        # them close in dispatch order is what keeps that traffic in L2 -- so
        # the choice is re-opened here rather than assumed.
        blk = [_as_i32(fx.block_idx.x), _as_i32(fx.block_idx.y),
               _as_i32(fx.block_idx.z)]
        # XKIND is a plain int decided at build time, so these are
        # fx.const_expr tests: flydsl's kernel decorator rewrites any other
        # `if` into device control flow (FlyDSL 0.1.1 only when the test calls
        # a non-helper function such as gorder.startswith(...); 0.2.4 always),
        # and the Python assignments inside stop escaping.
        if fx.const_expr(XKIND != 0):
            # Pin every workgroup of one batch to one XCD.  Workgroups are
            # handed to XCDs round-robin on the LINEAR id, id % NXCD, and the
            # linear id is x + gx*(y + gy*z); when gx is a multiple of NXCD the
            # y and z terms drop out and x % NXCD alone picks the XCD.  Putting
            # the batch index there means a batch's dq_acc rows are only ever
            # read-modify-written from one XCD's L2 instead of bouncing between
            # eight.  What shares the x axis decides what varies FASTEST within
            # an XCD, and so how small its live dq_acc footprint is: "xcd" puts
            # heads there (four heads of a batch in flight at once), "xcdn"
            # puts n_blocks there (one head of a batch at a time, a quarter of
            # the footprint).
            slot = blk[0] % _i32(NXCD)
            fast = blk[0] // _i32(NXCD)
            if fx.const_expr(XKIND == 1):
                head = fast
                n_blk = blk[1]
            else:
                head = blk[1]
                n_blk = fast
            batch = slot + _i32(NXCD) * blk[2]
        else:
            # gorder names what x, y, z carry, fastest axis first.
            batch, head, n_blk = (blk[gorder.index(c)] for c in "bhn")

        # Rounding the batch axis up to a multiple of NXCD dispatches
        # workgroups for batch indices past the end of the batch, and those
        # must read nothing and write nothing.  Two things are needed for that,
        # and the kernel previously had neither:
        #
        # num_records_bytes bounds the descriptor so an out-of-range index
        # returns 0 instead of whatever memory follows seq_offsets.  max_size
        # sets num_records to the maximum, which DISABLES the range check
        # entirely, and passing max_size=False is not enough on its own --
        # flydsl derives the extent from the memref type and gives up when the
        # shape is dynamic, which it always is here because batch varies.
        #
        # The explicit batch < NBATCH test then rejects the workgroup without
        # depending on that reasoning at all.  It is what actually stops the
        # fault: the loads themselves land in the same page and come back as
        # garbage rather than trapping, and it was USING that garbage as a
        # token offset into q/k/v that walked off the allocation.
        #
        # This only ever looked correct because every shape benchmarked before
        # now had a batch that was a multiple of NXCD, so no padded slot ever
        # existed.  At batch=4 it inflated every output by up to 1.9x; at
        # batch=1 it faulted.
        nbatch = arith.ArithValue(arith.unwrap(NBATCH))
        r_off = buffer_ops.create_buffer_resource(
            SEQ_OFF, max_size=False,
            num_records_bytes=arith.unwrap((nbatch + _i32(1)) * _i32(4)),
        )
        seq_start = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch, vec_width=1, dtype=T.i32)
        )
        seq_end = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch + _i32(1), vec_width=1, dtype=T.i32)
        )
        seq_len = seq_end - seq_start
        n0 = n_blk * _i32(BN)
        ws_pre_v = None
        if fx.const_expr(MS_WH):
            assert DQSPLIT, "wh needs dqsplit"
            r_hdr0 = buffer_ops.create_buffer_resource(
                DQ, max_size=False,
                num_records_bytes=arith.unwrap((nbatch + _i32(1)) * _i32(4)),
            )
            ws_pre_v = buffer_ops.buffer_load(r_hdr0, batch, vec_width=1,
                                              dtype=T.i32)

        guard = scf.IfOp(
            arith.andi(
                arith.cmpi(arith.CmpIPredicate.slt, n0, seq_len),
                arith.cmpi(arith.CmpIPredicate.slt, batch, nbatch),
            ),
            results_=[], has_else=False,
        )
        with ir.InsertionPoint(guard.then_block):
            emit_body(Q, K, V, DO, DK, DV, DQ,
                      arith.ArithValue(arith.unwrap(INV_MAX_ARG)),
                      arith.ArithValue(arith.unwrap(QKV_SCALE_ARG)),
                      tid, n0, head, seq_start, seq_len, DO_SC,
                      batch_idx=batch, nbatch=nbatch, ws_pre_v=ws_pre_v)
            scf.YieldOp([])

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        DQ: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        batch: fx.Int32,
        n_blocks: fx.Int32,
        inv_max: fx.Float32,
        qkv_scale: fx.Float32 = fx.Float32(1.0),
        do_scale: fx.Tensor = None,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        if fx.const_expr(gorder == "xcd"):
            grid = (NXCD * heads, n_blocks, (batch + NXCD - 1) // NXCD)
        elif fx.const_expr(gorder == "xcdn"):
            grid = (NXCD * n_blocks, heads, (batch + NXCD - 1) // NXCD)
        else:
            extent = {"b": batch, "h": heads, "n": n_blocks}
            grid = tuple(extent[c] for c in gorder)
        bwd_dkdv_kernel(
            Q,
            K,
            V,
            DO,
            DK,
            DV,
            DQ,
            SEQ_OFF,
            inv_max,
            qkv_scale,
            batch,
            do_scale,
        ).launch(
            grid=grid, block=(THREADS, 1, 1), stream=stream,
        )

    if E8:
        # R15 M-E1: the same kernel body with five more tensors.
        # Defined only when e8emit is set, so the default build above is
        # untouched (same kernel, same signature, same ISA).
        @kernel_deco
        def bwd_dkdv_kernel(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            DO: fx.Tensor,
            DK: fx.Tensor,
            DV: fx.Tensor,
            DQ: fx.Tensor,
            SEQ_OFF: fx.Tensor,
            INV_MAX_ARG: fx.Float32,
            QKV_SCALE_ARG: fx.Float32,
            NBATCH: fx.Int32,
            DO_SC: fx.Tensor,
            DZ8: fx.Tensor,
            E8INV: fx.Tensor,
            E8PFX: fx.Tensor,
            E8PAM: fx.Tensor,
            E8PCS: fx.Tensor,
        ):
            tid = _as_i32(fx.thread_idx.x)
            if fx.const_expr(VALU_MSAT):
                llvm.InlineAsmOp(
                    None, [],
                    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n"
                    "s_nop 7\ns_nop 7", "", has_side_effects=True)
            if fx.const_expr(WAVES_HW != 4):
                _set_flat_wg_size(THREADS)
            blk = [_as_i32(fx.block_idx.x), _as_i32(fx.block_idx.y),
                   _as_i32(fx.block_idx.z)]
            if fx.const_expr(XKIND != 0):
                slot = blk[0] % _i32(NXCD)
                fast = blk[0] // _i32(NXCD)
                if fx.const_expr(XKIND == 1):
                    head = fast
                    n_blk = blk[1]
                else:
                    head = blk[1]
                    n_blk = fast
                batch = slot + _i32(NXCD) * blk[2]
            else:
                batch, head, n_blk = (blk[gorder.index(c)] for c in "bhn")
            nbatch = arith.ArithValue(arith.unwrap(NBATCH))
            r_off = buffer_ops.create_buffer_resource(
                SEQ_OFF, max_size=False,
                num_records_bytes=arith.unwrap((nbatch + _i32(1)) * _i32(4)),
            )
            seq_start = arith.ArithValue(
                buffer_ops.buffer_load(r_off, batch, vec_width=1, dtype=T.i32)
            )
            seq_end = arith.ArithValue(
                buffer_ops.buffer_load(r_off, batch + _i32(1), vec_width=1,
                                       dtype=T.i32)
            )
            # The delayed inv_scale and this batch's partial-row prefix,
            # loaded in the prologue (wave-uniform).
            r_e8i = buffer_ops.create_buffer_resource(
                E8INV, max_size=False,
                num_records_bytes=arith.unwrap(_i32(4)))
            e8_inv = buffer_ops.buffer_load(r_e8i, _i32(0), vec_width=1,
                                            dtype=T.i32)
            r_e8p = buffer_ops.create_buffer_resource(
                E8PFX, max_size=False,
                num_records_bytes=arith.unwrap((nbatch + _i32(1))
                                               * _i32(4)))
            e8_pfx = buffer_ops.buffer_load(r_e8p, batch, vec_width=1,
                                            dtype=T.i32)
            e8_args = (DZ8, E8PAM, E8PCS,
                       arith.bitcast(T.f32, rocdl.readfirstlane(
                           T.i32, arith.unwrap(e8_inv))),
                       rocdl.readfirstlane(T.i32, arith.unwrap(e8_pfx)))
            seq_len = seq_end - seq_start
            n0 = n_blk * _i32(BN)
            ws_pre_v = None
            if fx.const_expr(MS_WH):
                assert DQSPLIT, "wh needs dqsplit"
                r_hdr0 = buffer_ops.create_buffer_resource(
                    DQ, max_size=False,
                    num_records_bytes=arith.unwrap((nbatch + _i32(1)) * _i32(4)),
                )
                ws_pre_v = buffer_ops.buffer_load(r_hdr0, batch, vec_width=1,
                                                  dtype=T.i32)

            guard = scf.IfOp(
                arith.andi(
                    arith.cmpi(arith.CmpIPredicate.slt, n0, seq_len),
                    arith.cmpi(arith.CmpIPredicate.slt, batch, nbatch),
                ),
                results_=[], has_else=False,
            )
            with ir.InsertionPoint(guard.then_block):
                emit_body(Q, K, V, DO, DK, DV, DQ,
                          arith.ArithValue(arith.unwrap(INV_MAX_ARG)),
                          arith.ArithValue(arith.unwrap(QKV_SCALE_ARG)),
                          tid, n0, head, seq_start, seq_len, DO_SC,
                          batch_idx=batch, nbatch=nbatch, ws_pre_v=ws_pre_v,
                          e8=e8_args)
                scf.YieldOp([])

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            DO: fx.Tensor,
            DK: fx.Tensor,
            DV: fx.Tensor,
            DQ: fx.Tensor,
            SEQ_OFF: fx.Tensor,
            batch: fx.Int32,
            n_blocks: fx.Int32,
            inv_max: fx.Float32,
            qkv_scale: fx.Float32 = fx.Float32(1.0),
            do_scale: fx.Tensor = None,
            stream: fx.Stream = fx.Stream(None),
            dz8: fx.Tensor = None,
            e8inv: fx.Tensor = None,
            e8pfx: fx.Tensor = None,
            e8pam: fx.Tensor = None,
            e8pcs: fx.Tensor = None,
        ):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            if fx.const_expr(gorder == "xcd"):
                grid = (NXCD * heads, n_blocks, (batch + NXCD - 1) // NXCD)
            elif fx.const_expr(gorder == "xcdn"):
                grid = (NXCD * n_blocks, heads, (batch + NXCD - 1) // NXCD)
            else:
                extent = {"b": batch, "h": heads, "n": n_blocks}
                grid = tuple(extent[c] for c in gorder)
            bwd_dkdv_kernel(
                Q, K, V, DO, DK, DV, DQ, SEQ_OFF, inv_max, qkv_scale, batch,
                do_scale, dz8, e8inv, e8pfx, e8pam, e8pcs,
            ).launch(
                grid=grid, block=(THREADS, 1, 1), stream=stream,
            )

    # flydsl's jit only knows how to marshal int, torch.Tensor and Stream from
    # plain Python; a bare float raises "No JitArgument registered".  Float32
    # itself satisfies the JitArgument protocol, so callers can keep passing an
    # ordinary float and we box it here rather than leaking fx types into the
    # integration API.
    def call(*args, **kw):
        if scale_in_cvt:
            _require_pow2_cvt_scale(
                kw["qkv_scale"] if "qkv_scale" in kw
                else (args[11] if len(args) > 11 else 1.0)
            )
        if VALU_TSC:
            # Every call, like the scale_in_cvt check: qkv_scale is a runtime
            # argument, and tsc is exact only for a snapped one.
            _qs = (kw["qkv_scale"] if "qkv_scale" in kw
                   else (args[11] if len(args) > 11 else 1.0))
            if tsc_exponent(alpha, _qs) is None:
                raise ValueError(
                    f"valu tsc needs a qkv_scale whose f32 c_t = "
                    f"f32(f32(f32(alpha)*f32(s*s))*f32(-log2e)) is exactly "
                    f"-2^k, k in [{TSC_K_MIN}, {TSC_K_MAX}] (alpha={alpha!r}, "
                    f"qkv_scale={_qs!r}); snap the scale or drop the token")
        if do_dyn:
            ds_t = kw.get("do_scale", args[12] if len(args) > 12 else None)
            assert ds_t is not None and str(ds_t.dtype) == "torch.float32" \
                and ds_t.numel() >= 2 and ds_t.is_cuda, (
                    "do_dyn kernel needs do_scale=[s_do, 1/s_do] fp32 cuda")
        elif do_blk:
            ds_t = kw.get("do_scale", args[12] if len(args) > 12 else None)
            do_t = kw.get("DO", args[3] if len(args) > 3 else None)
            assert ds_t is not None and str(ds_t.dtype) == "torch.uint8" \
                and ds_t.is_cuda and ds_t.is_contiguous() \
                and ds_t.numel() == do_t.shape[0] * heads, (
                    "do_blk kernel needs do_scale = E8 [T, H] uint8 cuda "
                    "(127 - group exponent per token)")
        elif "do_scale" not in kw and len(args) <= 12:
            # Unused by a static-scale kernel; any tensor fills the slot.
            kw["do_scale"] = args[7]
        if "stream" not in kw and len(args) < 14 and current_stream_launch():
            # Launch on the caller's current stream (see _current_stream); an
            # explicit stream, keyword or 14th positional, still wins.  With
            # $DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=0 no stream is passed: the
            # exact pre-5ae3127 launch arguments.
            kw["stream"] = _current_stream(args[0] if args else kw["Q"])
        if DQSPLIT:
            # The dq-split workspace (DQ slot) arrives as one flat bf16 view of
            # 10-19 GB.  FlyDSL 0.2.4 passes a torch.Tensor as a layout-dynamic
            # memref whose dynamic SHAPES it packs as int32 at every launch
            # (jit_argument._LayoutPlan), so a dim of 2^31 or more elements
            # raises struct.error.  The kernel reads DQ only through its base
            # pointer (_shifted_rsrc, explicit num_records), so launch a 2-D
            # view of the same bytes.  Every 1-D workspace gets it, so all
            # sizes share one jit cache key (0.2.4 keys on rank).
            if len(args) > 6:
                args = args[:6] + (_dq_ws_2d(args[6]),) + args[7:]
            elif "DQ" in kw:
                kw["DQ"] = _dq_ws_2d(kw["DQ"])
        args = tuple(fx.Float32(a) if isinstance(a, float) else a for a in args)
        if isinstance(kw.get("inv_max"), float):
            kw["inv_max"] = fx.Float32(kw["inv_max"])
        if isinstance(kw.get("qkv_scale"), float):
            kw["qkv_scale"] = fx.Float32(kw["qkv_scale"])
        # Same launch through a cached flyc.compile dispatcher under FlyDSL
        # 0.2.4 (ln_bwd.py _fast_call); the plain jit call under 0.1.1.
        return _fast_call(launch, *args, **kw)

    # Callers size the grid's n dimension, so they need the block's n extent.
    call.BN = BN
    call.launch = launch
    return call
