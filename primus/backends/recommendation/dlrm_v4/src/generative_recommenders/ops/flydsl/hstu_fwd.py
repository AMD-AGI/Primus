"""HSTU jagged-attention FORWARD written in FlyDSL for gfx950.

Two gemms with a pointwise between them, per (batch, head, m-block):

    qk  = alpha * q @ k.T                          (mfma, acc in f32)
    p   = where(m >= n, qk * sigmoid(qk) / N, 0)   (silu, causal INCLUDES diag)
    out = p @ v                                     (mfma, acc in f32)

There is NO softmax, no row max, no row sum, no rescale.  `out[m]` sums over
n <= m only, so an m-outer (q-outer) grid gives each workgroup exclusive
ownership of its output rows -- no atomics, no zero-fill.

Decomposition:
  * grid is XCD-pinned: (NXCD*heads, m_blocks, cdiv(batch, NXCD)), so each
    batch's K/V stays resident in one XCD's L2 instead of being re-fetched
    across all eight.  One workgroup owns BM output rows of one (batch, head);
    WAVES waves, wave w owns MT = BM/(16*WAVES) sixteen-row tiles starting at
    m0 + 16*MT*w.
  * Q for the owned rows is loaded once into registers (loop-invariant) and is
    the A-operand of the score gemm.  K and V are streamed over n and staged in
    LDS; K is the score gemm's B-operand, V is the out gemm's B-operand.
  * BOTH K AND V ARE B-OPERANDS, so they are broadcast across m: each fragment
    is read from LDS once and issued against all MT m-tiles.  That is why
    WAVES=2/MT=2 beats WAVES=4/MT=1 -- it halves the workgroup's K/V LDS reads
    without touching BM, and this kernel is LDS-throughput bound (measured:
    occupancy 3 wg/CU vs 2 is worth nothing, while a K-read bank conflict is
    worth 40%).
  * The score accumulator is TRANSPOSED -- mfma2(kf, qf) rather than
    mfma2(qf, kf) -- so the fragment it produces already carries n on the
    element axis, which is the layout the out gemm's A-operand needs.  V is read
    from a tr16 LDS layout with ds_read_b64_tr_b16 (transpose-on-read).
  * Every LDS tile is bank-conflict padded, and that is where most of this
    kernel's performance came from: k_lds stride D+16, p_lds stride BN+4,
    v_lds tr16 block stride BN+1.  Each pad was worth double digits; the sweep
    is exhausted (see ../../lineage/hstu_attention_forward.md).

Reads q/k/v in place at a caller-supplied row stride (`rs`), because production
hands attention COLUMN SLICES of one uvqk buffer rather than packed tensors.

This is the structural transpose of hstu_bwd_dkdv.py: it reuses that kernel's
mfma2 helper, _shifted_rsrc buffer descriptors and silu chain.
"""

import functools
import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, llvm, memref, scf
from flydsl.compiler.kernel_function import CompilationContext
try:
    # FlyDSL 0.2.4 (ROCm 10.2 UTD image) renamed fly_values to
    # extract_to_ir_values; same contract: the list of ir.Values of a DSL
    # value, [0] being the tensor's memref.
    from flydsl.compiler.protocol import extract_to_ir_values as fly_values
except ImportError:  # FlyDSL 0.1.1 (rocm/primus:v26.3)
    from flydsl.compiler.protocol import fly_values
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from generative_recommenders.ops.flydsl._compat import buffer_ops, rocdl_raw, vector  # FlyDSL 0.2 API on 0.2 / >= 0.3
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from generative_recommenders.ops.flydsl import current_stream_launch
# FlyDSL 0.2.4's per-call @flyc.jit overhead: see ln_bwd.py _fast_call.
from generative_recommenders.ops.flydsl.ln_bwd import _fast_call

LOG2E = 1.4426950408889634

# Shipping configuration in one place so bench and gate cannot drift.
#
# BN=32 (not 64): halves the K/V/p LDS tiles.  BN=64 re-measured at -17.6% on the
# production shape, so this is not merely an occupancy choice.
#
# waves=2 (not 4): with BM=64 this gives each wave MT=2 sixteen-row m-tiles, so a
# K or V fragment is read from LDS once and issued against both -- halving the
# workgroup's K/V LDS reads at identical BM, MFMA count and causal waste.  Worth
# +2.6% measured by repeated standalone runs and +4.7% by an in-process paired
# harness (identity control -0.05%, sign flips correctly when the arms swap);
# +2.6% is the figure banked, following the campaign's worse-of-two rule.
# waves=1 (MT=4) collapses (-69%): there is a register cliff between MT=2 and 4.
SHIPPED = {"BM": 64, "BN": 32, "sched": "iglp1", "waves": 2}

PTR_TY = "!llvm.ptr"
RSRC_TY = "!llvm.ptr<8>"
LDS_PTR_TY = "!llvm.ptr<3>"
GEP_DYNAMIC = -2147483648  # kDynamicIndex marker in llvm.getelementptr
CAT8 = [0, 1, 2, 3, 4, 5, 6, 7]  # concat two v4bf tr reads into a K=32 fragment
CDNA_BUFFER_FLAGS = (7 << 12) | (4 << 15)


def _lds_gep(mem, elem_off, elem_ty):
    """Raw `!llvm.ptr<3>` at a dynamic element offset into an LDS memref.

    rocdl.ds.read.tr16.b64 takes a raw LDS pointer, not the (memref, index) pair
    the vector ops accept, so the view is reduced to a pointer by hand.
    """
    ptr_ty = ir.Type.parse(LDS_PTR_TY)
    idx = memref.ExtractAlignedPointerAsIndexOp(mem).result
    base = llvm.IntToPtrOp(ptr_ty, arith.index_cast(T.i32, idx)).result
    return llvm.GEPOp(
        ptr_ty, base, [arith.unwrap(elem_off)], [GEP_DYNAMIC], elem_ty,
        llvm.GEPNoWrapFlags(0),
    ).result


def _i32(v):
    return arith.constant(int(v), type=T.i32)


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _i64(v):
    return arith.constant(int(v), type=T.i64)


def _to_i64(v):
    v = arith.unwrap(v)
    return v if v.type == T.i64 else arith.extsi(T.i64, v)


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


def _shifted_rsrc(tensor, elem_offset, elem_bytes, num_bytes=None):
    """Buffer resource whose base is advanced by elem_offset (computed in i64).

    `num_bytes`, when given, becomes the descriptor's num_records: a raw buffer
    load whose voffset reaches it returns zero instead of reading.  That turns
    "row past the end of this sequence" into an exact algebraic zero.

    `elem_offset` must ALREADY be i64 when it can exceed 2^31.  seq_start * RS is
    a whole-tensor element offset and DOES overflow i32 at production sizes: at
    batch 1024 and a uvqk row stride of 2048 it reaches 5.66e9 against i32's
    2.15e9, which faults on a wild address.  Only this base shift needs the wider
    type -- every voffset used against the returned descriptor is relative to it
    and bounded by one sequence.
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


@functools.lru_cache(maxsize=64)
def compile_hstu_fwd(
    heads: int,
    dim: int,
    alpha: float,
    BM: int = 64,
    BN: int = 32,
    sched: str = "",
    rs: int = 0,
    kpad: int = -1,
    lds_dummy: int = 0,
    waves: int = 2,
    # ---- true fp8 MFMA on the qk gemm -----------------------------------
    # Same argument as the backward's fp8_mfma (hstu_bwd_dkdv.py): in the FP8
    # arm q and k are ALREADY e4m3 in memory, and e4m3 -> bf16 is lossless (3
    # mantissa bits into 7), so the incumbent widens both operands and runs a
    # bf16 MFMA over values that are exactly fp8.  Substituting a native fp8
    # MFMA changes only the f32 accumulation order -- it is numerically free.
    # What it buys is NOT math rate:
    #   * k_lds halves, 128 B/row -> 64 B/row of payload;
    #   * "k128" (mfma_scale_f32_16x16x128_f8f6f4) collapses the score gemm's
    #     FOUR ds_read_b128 per (n-tile) into TWO, and its four mfma into one.
    # The K LDS read is this kernel's measured DOMINANT cost (the ablation in
    # the KPAD note above puts it at -17.6%, vs -8.4% for the V tr-reads), so
    # that read collapse is aimed straight at the top of the profile.
    fp8_mfma: str = "",
    # Feed V as e4m3 too.  Independent of fp8_mfma: it halves the v_lds tile
    # and the V global traffic even while the out gemm stays bf16, because the
    # tr16 staging then widens e4m3 -> bf16 on the way in.
    fp8_v: bool = False,
    # fp8 MFMA on the out gemm (p @ v).  Needs fp8_v, and unlike qk it is NOT
    # free: p is computed in f32 and has to be ROUNDED to e4m3.  p's
    # underflow is benign (0.009-0.017% flushed on real tensors, vs 96-99.9%
    # for the backward's ds/dO), so the cost is mantissa, not flush.
    # "" (default) keeps the bf16 P@V.  "k128" is the X4C19 iter46 FP8 P@V
    # with a K = 128 retile: a DECLARED PRECISION CHANGE (P f32 -> e4m3),
    # default off, ruling-gated, built on the FWD_PIPE config only (see the
    # pv=k128 knobs below and the pv=k128 section of emit_body).
    pv: str = "",
    # Bank-conflict padding, in BYTES, of the fp8 k_lds row.  Same role KPAD
    # plays for the bf16 tile; the bf16 sweep's answer does not carry over
    # because the row is half as wide, so this is swept separately.
    k8_pad: int = 16,
    # Bank-conflict padding, in BYTES, of the e4m3 V ring row that pv="k128"
    # reads with ds_read_b64_tr_b8 (unused when pv is off).
    v8_pad: int = 16,
    # ---- TRACK E2: silu-epilogue VALU reductions ------------------------
    # The rocprofv3 census says the kernel spends 57% of its SIMD cycles in
    # NON-MFMA VALU and only 28.6% in MFMA, at 12.0 VALU lane-ops per score
    # element (4 MUL, 2 TRANS, 2 INT32, 1.1 CVT, 1 ADD, 1.9 cmp/cndmask/mov).
    # Both knobs below delete lane-ops the counters name, and neither changes
    # what the MFMAs compute.
    #
    # `epi_fold`: the two constant MULs.  Reparametrise the silu on
    #   u = -log2(e) * score_scale * raw   (ONE mul, was two: score_scale then
    #   -LOG2E), so exp2's argument IS u; then silu = u*sig carries a constant
    #   factor -1/log2(e) which, together with inv_max, rides out on the
    #   epilogue's existing out_scale -- 128 f32 muls per output row instead of
    #   one per score element.  4 MUL/element -> 2.
    epi_fold: bool = False,
    # `mask_split`: the causal compare.  For an m-block at m0, every n-tile
    #   with n0+BN <= m0 is strictly below the diagonal, so m >= n holds for
    #   all of it and the mask is dead.  m0 is a multiple of BN (BM % BN == 0),
    #   so that is exactly the first m0/BN iterations.  Splitting the n loop
    #   there drops the n_idx add, the v_cmp and the v_cndmask -- 3 of 12
    #   lane-ops -- on each of those iterations and adds nothing to the masked
    #   ones (the split itself costs one more loop entry per m-block).  Like
    #   epi_fold this is an analytic VALU deletion, so the sign of the saving
    #   does not depend on the length distribution; the share of n-iterations
    #   that are fully below the diagonal sets only the SIZE of the benefit.
    #   That share is analytic too.  By the n-loop bounds in emit_body
    #   (n_iters = ceil(min(m0+BM, seq_len)/BN), head = the first m0/BN), a
    #   sequence has nm = ceil(seq_len/BM) live m-blocks, each visiting n-tiles
    #   up to and including its own diagonal block, and m-block j runs j*BM/BN
    #   head and BM/BN masked iterations.  BM/BN cancels, so the share is
    #   (nm*(nm-1)/2) / (nm*(nm+1)/2) = (nm-1)/(nm+1) for any BN dividing BM
    #   (a last m-block BN or more rows short of full has fewer masked
    #   iterations and a slightly higher share; nm = 1 gives 0).  At the
    #   configured maximum seq_len 4095 (yambda_5b.gin: 3*1362+9) that is
    #   63/65 = 96.9% for SHIPPED (BM=64, nm=64) and 31/33 = 93.9% for the
    #   BM=128 FWD_PIPE config (nm=32).  No dataset statistic is involved.
    mask_split: bool = False,
    # `epi_pk`: the two MULs and the ADD that survive epi_fold, at double rate.
    #   The re-profile after epi_fold+mask_split says the kernel is STILL
    #   VALU-bound and not MFMA-bound (VALUBusy 57.1% vs MfmaUtil 34.0%), and
    #   names the remainder exactly: per score element 2 TRANS (v_exp_f32 +
    #   v_rcp_f32, irreducible for silu), 2 MUL_F32 and 1 ADD_F32.  gfx950 has
    #   double-rate packed f32 -- verified at the ISA level, `llc -mcpu=gfx950`
    #   turns an <2 x float> fmul/fadd into v_pk_mul_f32/v_pk_add_f32 -- and the
    #   accumulator already holds 4 consecutive n per lane, so the pairs are
    #   free to form.  MUL 5.230e8 + ADD 2.497e8 = 7.73e8 of 2.016e9 VALU
    #   instructions halve, i.e. -19.2% of VALU issue.  The transcendentals stay
    #   scalar; only the surrounding arithmetic packs.  Bit-exact: same
    #   operations, same order, same f32 rounding, just two lanes per issue.
    epi_pk: bool = False,
    # ---- X4C19 iter36: K/V prefetch + direct P fragment ----------------------
    # `kv_prefetch` (fp8 q/k + fp8 V only): split stage_kv into load_kv and
    #   store_kv and carry the NEXT n-iteration's K/V global loads as scf.for
    #   iter_args, issued right after the RAW barrier so their latency overlaps
    #   this iteration's compute (also across the mask_split head->tail loop
    #   boundary).  The last iteration's prefetch (rows from n_end on; zero past
    #   the sequence end via num_records) is never stored.  Same bytes, same
    #   LDS layout, same math: bit-identical.
    kv_prefetch: bool = False,
    # `p_direct`: build the out-gemm A fragment straight from the bf16 silu
    #   values this lane computed (n-tile 2t ++ 2t+1, exactly the 8 values it
    #   used to store) instead of the p_lds store + read.  The read was a
    #   same-lane identity LLVM already forwarded; this deletes the dead
    #   ds_write and the p_lds allocation.  Bit-identical.
    p_direct: bool = False,
    # ---- X4C19 iter41 f4: causal-mask arithmetic of the masked n-iterations --
    # 0 = the shipped masked path (default; byte-identical kernel): per score
    #     element an f32 compare + select, after which LLVM unpacks/repacks the
    #     bf16 P pairs (v_perm/v_lshrrev) to apply the select per half.
    # 3 = (p_direct only) the silu is converted to bf16 unmasked and the causal
    #     zeros are applied to the packed bf16 pairs of the P fragment, one
    #     v_and_b32 per pair.  The per-lane keep-masks come from
    #     dmn = (l16 - 4*l4 + m_base) - n0 with two packed-int16 ops per
    #     distinct (16*(nt-mt), pair): half kept iff dmn >= 16*(nt-mt) + e,
    #     which is exactly m >= n.  Every score element is still computed and
    #     the predicate is a runtime per-lane value; bf16(+0.0) is 0x0000, so
    #     p is bit-identical to select-then-convert.  Masked loop 217 -> 179
    #     instructions per wave-iteration at the FWD_PIPE config.  (Values 1,
    #     2, 4 were other variants measured in Arbor x4c19 iter41 f4.)
    #     With pv="k128" (X4C19 iter47) the same predicate is applied to the
    #     packed e4m3 P8 dwords of the masked tail group instead: one v_and_b32
    #     per 4 keys, the byte keep-mask from two packed-int16 subtracts and
    #     one v_perm_b32 (sign-replicate selectors) per distinct 16*(nt-mt);
    #     e4m3(+0.0) is 0x00, so P8 is bit-identical to select-then-convert.
    mask_arith: int = 0,
    # ---- X4C19 iter46: pv="k128" (FP8 P@V, K = 128 retile) -----------------
    # DECLARED PRECISION CHANGE (P f32 -> e4m3), default off, ruling-gated.
    # Built on the FWD_PIPE config only (BN=32, BM=128, fp8_mfma="k128",
    # fp8_v, kv_prefetch, p_direct, mask_split, epi_fold, no epi_pk), with or
    # without mask_arith=3.  The knobs below take effect only with pv="k128";
    # with pv="" the build is the bf16-P@V kernel above, unchanged.
    # `pv_k`: P8 = e4m3(sat(P' * 2**pv_k)) where P' is the epi_fold silu;
    #   the cvt divides by its operand (2**-pv_k) and out_scale *= 2**-pv_k.
    # `pv_msat`: MODE.FP16_OVFL=1 for the main loop (saturating fp8 cvt),
    #   cleared before the bf16 epilogue.
    pv_k: int = 3,
    pv_msat: bool = True,
    # `pv_s` (X4C19 iter50): the static P scale.  0.0 (default) keeps the
    #   power-of-two scale 2**pv_k above (the iter46 kernel, same symbol and
    #   ISA).  s > 0 gives P8 = e4m3(sat(P' * s)) for a NON-power-of-two s:
    #   the silu chain's `rcp(ex + 1.0)` becomes `rcp(fma(ex, c, c))` with
    #   c = f32(2**pv_k / s), so x * rcp(c * (ex + 1)) = (s / 2**pv_k) * P'
    #   and the unchanged cvt (which divides by 2**-pv_k) yields
    #   e4m3(s * P'); out_scale *= f32(1/s) instead of 2**-pv_k.  The fma
    #   replaces the add (no extra hot-loop instruction; c lives in one SGPR,
    #   see pv_c).  Why: P' (the epi_fold silu, -log2(e) * silu) peaks over
    #   negative scores at the silu minimum (score -1.27846, P' = 0.40174),
    #   where silu is stationary, so P' piles up there.  With s = 8 that pile
    #   sits at 3.2139, 1.1% below the e4m3 grid point 3.25, and rounds UP
    #   systematically: a coherent row-sum error on rows whose keys cluster
    #   in the negative lobe.  Every power-of-two s keeps the same position
    #   inside the binade; a non-power-of-two s moves it (s = 10.04: 0.84%
    #   above the grid point 4.0).  Provenance of 10.04 (x4c19 iter67
    #   pvscale): located by the analytic route (format + kernel constants;
    #   mean-rounding-error zero s = 10.0379); not contradicted by a
    #   synthetic replication of the iter49 rule (non-discriminating on
    #   zero-mean V); replaces the real-dump emulation.  Accepted range
    #   [4, 32]: s * P' stays far
    #   below the e4m3 saturation (448) and |P'| >= 2**-12 stays out of the
    #   flush.  Same declared precision class as pv="k128" (P f32 -> e4m3).
    pv_s: float = 0.0,
    # `pv_split`: sched_barrier(0) between phase 3's score+silu and the P@V, so
    #   the scheduler cannot hoist the 32 tr8 B operands into the silu
    #   (VGPR: without it iglp0 hoisted 4 x 8 dwords there -> 172, which is
    #   past the 3-waves/SIMD limit).
    # `pf_late`: issue phase 3's next-group K/V prefetch after its silu (just
    #   before the P@V) instead of right after the RAW barrier.
    pv_split: bool = False,
    pf_late: bool = False,
    # `pv_tg`: tail guard -- in the (single, padded) tail group, phases whose
    #   first key is >= n_end skip score + silu (P8 = 0 via a uniform scf.if);
    #   their K/V staging still runs (zero rows), so the ring stays finite.
    #   Only padding work (keys >= seq_len) is skipped.
    pv_tg: bool = False,
    # `v8_swz` (X4C19 iter53 vswz; ported in iter70 as a LAB knob, default
    #   off): LDS layout of the pv="k128" V ring, bit 0 only.  Rows 16..31 of
    #   every ring slot sit 128 B further (slot = BN*V8S + 128 B).  At V8S =
    #   144 B (36 dwords) ring rows r and r+16 hit the SAME banks (36*16 = 0
    #   mod 64), and every 32-lane pass of the P@V tr8 B-operand read holds
    #   both (rows {0..7,16..23} / {8..15,24..31}): a 2-way bank conflict.
    #   +128 B = +32 dwords moves the upper half onto the complementary banks.
    #   The shift is additive and static per row half: the staging store
    #   offset gains a static +32 dwords for its second row pass (r = 1) and
    #   the tr8 read base gains t8_hi * 128 B once per lane, so the per-phase
    #   tr8 offsets stay instruction immediates.  Every value the MFMAs see is
    #   unchanged, so the output is bit-identical; the ring grows by 4 x 128 B
    #   of LDS.  Measured on the ship13 forward (iter53; hstu_fwd.py there ==
    #   this file modulo comments): LDS bank-conflict cycles -50.1% (PMC),
    #   faster in 9/9 lab cells, median -0.37% per call.  0 (default) is the
    #   shipped kernel byte for byte (same symbol, same IR).
    v8_swz: int = 0,
    # `rider` (X4C19 iter71; LAB knob, default 0): bit mask of bitwise
    #   pv="k128" epilogue diets; only 0 and 3 (R2a + R2b) are accepted.  The
    #   head loop's IR is not touched (CPU AOT: its ISA body is
    #   byte-identical).  Every bit keeps every stored value and address: the
    #   same out_scale v_mul_f32 and the same RNE bf16 convert per value, the
    #   same 16-bit store per (row, column).
    #   1 (R2a) per m-tile, a wave-uniform scalar branch on
    #     readfirstlane(m_base) + 16*(mt+1) <= seq_len selects stores with no
    #     per-row exec mask; an m-tile that crosses seq_len keeps the shipped
    #     per-row `m < seq_len` guards.  Drops the per-store
    #     s_and_saveexec / s_cbranch_execz / s_or_b64 on full m-tiles.
    #   2 (R2b) one byte-address base VGPR per m-tile,
    #     (m_base + 16*mt + 4*l4) * 2*H*D + 2*l16; the (e, dd) part
    #     2*H*D*e + 32*dd is a constant LLVM folds into the MUBUF 12-bit
    #     offset (at most 3*1024 + 7*32 = 3296), instead of a
    #     v_lshl_or_b32 per store.
    #   The iter71 lab bit 4 (R2c, inline asm) was removed in ship17 row
    #   (100) v2.  R3 (zero-init peel) is not here: every form tried changed
    #   the head loop's code (x4c19 iter71 rider RESULT.md).
    rider: int = 0,
    # `spf` (X4C19 iter71; LAB knob DLRM_HSTU_FP8_LAB_FWD_SPF=1, default
    #   off): score-scale power-of-two fold.  CALLER CONTRACT: qkv_scale is on
    #   the SPF grid, i.e. score_scale = fl(fl(fl(s*s)*alpha)*-LOG2E) is
    #   EXACTLY -2**j (ops/fp8_qkv_spf.py snaps the calibrated scale there and
    #   hstu_flydsl_mha arms spf only after checking it, j in [-108, 103]).
    #   The score gemm's MFMAs then take the runtime E8M0 A-scale byte
    #   127 + j = the exponent field of score_scale (B scale x1), so the
    #   accumulator is 2**j * raw, and x = -acc replaces x = raw * score_scale:
    #   the per-element v_mul_f32 is gone, exp2 and the silu multiply read the
    #   negation as VOP3 neg source modifiers.  P8 and out_scale are the
    #   snapped control's ("snap": same scale, spf=False) value for value when
    #   the scaled MFMA equals the unscaled one times 2**j (power-of-two
    #   scaling; the folded accumulator is a normal f32 in that j range).
    #   False (default) is the shipped kernel byte for byte.
    spf: bool = False,
    # `lut` (ship17 gin item (120) LUT-SiLU fwd, knob
    #   DLRM_HSTU_FP8_LUT_SILU_FWD; PRECISION class, X4C19 iter73 LUTSILU v1;
    #   default 0 = off = the shipped kernel byte for byte, same symbol): the
    #   per-element silu chain of the pv="k128" score phase (score-scale mul,
    #   exp2, fma, rcp, silu mul, fp8 cvt) becomes a gather of the final e4m3
    #   P8 byte from a per-call LDS table.  Only 13 (= index bits; table
    #   2**13 = 8192 B) is accepted.
    #   * Score gemm: the scaled MFMA with the runtime E8M0 A-scale byte =
    #     the exponent field of score_scale (spf's scaled score gemm above,
    #     ONE shared construction, built when spf or lut), B scale x1, so
    #     the accumulator is y = 2**j * raw.  lut alone needs NO snap:
    #     score_scale = -2**j * r, r in [1, 2), and the residual r lives in
    #     the table.
    #   * Index: h = v_cvt_pkrtz_f16_f32(y0, y1); (h >> 3) & 0x1FFF and
    #     h >> 19 (f16 sign | exp5 | top 7 mantissa bits).  MODE.FP16_OVFL is
    #     1 there (pv_msat): a finite |y| past the f16 range clamps to
    #     +-65504 (RTZ gives the same).
    #   * Gather: one i8 LDS load per element; byte e of the P8 dword <-> the
    #     accumulator element e (pk4_fp8's order); mask_arith=3's v_and on
    #     the packed dword is unchanged.  Each dword is built from two v2i16
    #     half-inserts {b0, b2}, {b1, b3} combined as lo | hi << 8: the form
    #     LLVM selects as ds_read_u8 + ds_read_u8_d16_hi only for a target
    #     that preserves the unused d16 bits (sramecc-; not this gfx950 /
    #     sramecc-any build, which gets 2 v_perm_b32 + 1 v_lshl_or_b32 per
    #     dword, 3 VALU, vs 4 for b0 | b1 << 8 | b2 << 16 | b3 << 24).
    #   * Table: new kernel argument LUT (uint8[8192], global, after
    #     SEQ_OFF), built on the host from the math function and the runtime
    #     f32 score_scale (ops/fp8_lut_silu.py build_table: entry i = the P8
    #     byte at the midpoint of f16 cell i, e4m3 RNE saturating to +-448;
    #     +inf cell -448, -inf cell +0, NaN cells 0x7F | sign).  It is copied
    #     to an LDS region appended after the existing ones by all THREADS
    #     threads (8192 / THREADS bytes each, dwordx4 loads, b128 LDS stores)
    #     before the head loop, so the first barrier of the first group
    #     (head or tail) orders it before the first gather.
    #   Requires pv="k128" (with its epi_fold) and mask_arith=3: under lut
    #   the per-element chain (and its causal select) runs 0 trips, so the
    #   packed P8 byte mask of mask_arith=3 is the masked tail group's only
    #   causal mask.  Composes with v8_swz (V ring layout) and rider
    #   (epilogue stores), which touch neither the score phase nor this
    #   region, and with spf (ship17 row (105)): under the
    #   SPF snap score_scale = -2**j exactly, so r = 1 and the table (built
    #   from that runtime score_scale) is the r = 1 table; the score MFMA is
    #   the same scaled form, and spf's per-element chain variant (x = -acc)
    #   is dropped with the rest of the chain.  Nothing else changes.
    #   Precision class: P8 bytes move at cell boundaries (x4c19 iter73
    #   DESIGN_LUTSILU_v1.md).  Caller contract (the launcher checks it,
    #   ops/fp8_lut_silu.py split_scale): score_scale is a normal finite
    #   negative f32 with j in [-108, 103], so 127 + j is a valid E8M0 byte
    #   and y stays a normal f32.
    lut: int = 0,
):
    """`kpad` overrides the k_lds row pad; `lds_dummy` wastes that many LDS bytes.

    Both exist to separate two effects the KPAD sweep confounded. Raising KPAD
    from 8 to 16 both fixes a K-read bank conflict AND costs 512 B/workgroup,
    which pushes 3x21.75 KB past the 64 KB the CU has and drops occupancy from 3
    workgroups to 2. The sweep preferred KPAD=16, so conflict-freedom is worth
    more than that occupancy -- but it never priced the occupancy on its own.
    `lds_dummy` does: it forces the same occupancy drop at an UNCHANGED conflict
    pattern, so KPAD=8+512 vs KPAD=8 isolates occupancy, and against KPAD=16 it
    isolates the conflict fix. Measured at the production shape: 3 wg/CU 6.192 ms
    vs 2 wg/CU 6.169 at KPAD=8, i.e. occupancy is worth NOTHING here, while
    KPAD=0 costs 40% (8.436). This kernel is LDS-throughput bound, not latency
    bound -- which is what makes `waves` below the interesting knob.

    `waves` sets waves per workgroup, and with BM fixed it sets MT = BM/(16*waves),
    the number of 16-row m-tiles ONE wave owns. It is the lever on LDS traffic:
    K and V are both B-operands, broadcast across m, so a wave that owns MT
    m-tiles reads each K/V fragment ONCE and issues MT mfmas against it. Halving
    `waves` therefore halves the workgroup's K/V LDS reads for identical MFMA
    work, at identical BM -- which is what distinguishes it from raising BM. BM
    sets the causal waste (the n loop runs to m0+BM, so a coarser BM overshoots
    the triangle), and that is why BM=128 measured neutral in the campaign: it
    bought this same amortization and paid it back in wasted tiles.
    """
    assert dim == 128, "kernel is specialised for D=128"
    assert BM % 64 == 0 and BN % 32 == 0
    # The packed path only implements the epi_fold parametrisation.
    assert not epi_pk or epi_fold, "epi_pk requires epi_fold"
    assert fp8_mfma in ("", "k32", "k128"), fp8_mfma
    assert pv in ("", "k128"), f"pv={pv!r}: only '' (bf16 P@V) and 'k128'"
    PV8 = pv == "k128"
    assert not (pv and not fp8_v), "pv needs an e4m3 V"
    if PV8:
        # BM == 128 exactly: the n loop below is m0/128 unmasked groups plus
        # ONE masked tail group of 128 keys, which covers the diagonal block
        # only when BM == 128 (BM=256 would drop keys m0+128..m0+255).
        assert (fp8_mfma == "k128" and kv_prefetch and p_direct and mask_split
                and epi_fold and not epi_pk and BN == 32 and BM == 128), (
            "pv=k128 is built on the FWD_PIPE config only")
    assert PV8 or not (pv_split or pf_late or pv_tg or pv_s), (
        "pv_split / pf_late / pv_tg / pv_s are pv='k128' knobs")
    assert v8_swz in (0, 1), f"v8_swz={v8_swz!r}: only bit 0 is implemented"
    assert PV8 or not v8_swz, "v8_swz is a pv='k128' knob"
    assert type(rider) is int and rider in (0, 3), (
        f"rider={rider!r}: 0 (off) or 3 (R2a + R2b)")
    assert PV8 or not rider, "rider is a pv='k128' knob"
    R2A, R2B = (bool(rider & b) for b in (1, 2))
    assert spf in (False, True), f"spf={spf!r}"
    assert PV8 or not spf, "spf is a pv='k128' knob"
    assert not spf or (epi_fold and fp8_mfma == "k128"), (
        "spf needs epi_fold (exp2's argument is the scaled score) and the "
        "k128 score gemm")
    # lut: an int, exactly 0 or 13 (True / 13.0 / other widths fail closed).
    assert type(lut) is int and lut in (0, 13), (
        f"lut={lut!r}: 0 (off) or 13 (13-bit P8 table)")
    assert not lut or (PV8 and epi_fold), "lut is a pv='k128' (epi_fold) knob"
    # lut skips the per-element select; mask_arith=3's packed P8 byte mask is
    # then the only causal mask (mask_arith=0 would leave n > m unmasked).
    assert not lut or mask_arith == 3, (
        "lut needs mask_arith=3 (the packed P8 byte mask is the lut path's "
        "only causal mask)")
    # pv_s: 0 = power-of-two 2**pv_k; otherwise a static scale in [4, 32]
    # (see the knob comment: saturation and flush edges).
    pv_s = float(pv_s)
    assert pv_s == 0.0 or 4.0 <= pv_s <= 32.0, f"pv_s={pv_s!r}"
    assert not kv_prefetch or (fp8_mfma and fp8_v), (
        "kv_prefetch is implemented for the fp8 q/k + fp8 V config only")
    assert mask_arith in (0, 3), mask_arith
    assert not mask_arith or p_direct, "mask_arith=3 needs p_direct"
    FP8_QK = fp8_mfma
    D = dim
    # k_lds is row-major [BN][D].  The score gemm's B read for lane l is
    # k[n = 16*nt + l16][d = 32*ks + 8*l4 .. +7] as a b128 (v8bf).  ABLATION
    # (collapsing the 8 K reads/iter to 1 reused read) shows these K reads are
    # this kernel's DOMINANT LDS cost: -17.6% at b256, vs V tr-reads -8.4% and
    # transcendentals ~0%.  So the K-read bank pattern is where the money is.  A
    # KPAD sweep over the alignment-preserving multiples of 8 (KSTRIDE must be
    # 8-elem == 16 B aligned or the b128 splits) at b256: pad=8 4.085 (old
    # leader), pad=16 3.87-4.04, pad=24 4.186, pad=32 4.223.  pad=16
    # (KSTRIDE=144, 288 B == 16 mod 32 banks) wins despite the +512 B pushing
    # 3 wg/CU -> 2 wg (66 KB > 64 KB for three): the K-read conflict reduction
    # beats the occupancy loss, matching the note that this kernel is no longer
    # latency-bound.  The earlier pad=8 "2-way" model was wrong; trust the sweep.
    KPAD = 16 if kpad < 0 else kpad
    assert KPAD % 8 == 0, "KSTRIDE must stay 16 B aligned or the b128 read splits"
    KSTRIDE = D + KPAD
    # fp8 k_lds: the SAME row-major [BN][row] tile, but the row is D BYTES wide
    # instead of D halfwords.  Padding is in bytes and has to keep the row 16 B
    # aligned, because "k128" reads it as two ds_read_b128.
    K8STRIDE = D + k8_pad
    assert K8STRIDE % 16 == 0, "fp8 k_lds row must stay 16 B aligned"
    K8_DW = K8STRIDE // 4
    # p_lds is per-wave [16 m_local][BN n].  With row stride exactly BN=32 elems
    # (64 B) two m_local rows 2 apart (=128 B, the 32-bank period) alias onto the
    # SAME banks: the out-gemm A read (row = l16, all 16 rows at once) is an 8-way
    # conflict on its 2 b64 reads/iter, and the p-store (rows l4*4+e, 4 apart =
    # 256 B = 2 periods) is a 4-way conflict on its 8 scalar writes/iter -- the
    # exact flaw the k_lds pad fixed.  Padding to 36 elems (72 B) puts rows k=1..4
    # apart at {72,16,88,32} B mod 128 (none 0), killing both.  PSTRIDE%4==0 keeps
    # the b64 A read 8 B aligned.  +512 B LDS -> 21.0 KB, still 3 wgs/CU (3 waves).
    PPAD = 4
    PSTRIDE = BN + PPAD
    # v_lds is tr16 [D/16 blocks][BN rows][16 cols]; the tr16 block stride
    # (ROWS in the (c/16)*ROWS+r formula) is BN, so the cooperative vec8 staging
    # store v_off = (v_blk*BN + n)*16 + v_in has byte bank =
    # 8*((v_blk*BN+n) mod 8)+{0,4} on gfx950's 64 banks.  BN=32 == 0 mod 8 drops
    # v_blk out entirely: all 8 d-blocks collide on the same banks for each
    # (n,half) -> 8-way store conflict, only 32 of 64 banks used.  Same flaw KPAD
    # fixed for k_lds, on the ONE tile never padded.  VROWS = BN+1 (== 1 mod 8)
    # makes 8*((v_blk+n) mod 8) spread all 8 blocks over distinct bank groups ->
    # conflict-free store.  Reads (ds_read_tr16, all lanes share dd) are
    # unaffected by block stride.  +256 B LDS; already 2 wg/CU so no cliff.
    VPAD = 1
    VROWS = BN + VPAD
    VLDS_HW = (D // 16) * VROWS * 16
    H = heads
    HD = H * D
    # Elements between consecutive tokens of q/k/v.  A packed [N, H, D] tensor
    # has H*D, but production never builds one: the UVQK gemm emits a single
    # [N, 2*H*hidden + 2*H*attn] buffer and hands attention COLUMN SLICES of it,
    # so one token's q sits `rs` elements from the next.  Passing that stride is
    # what lets the kernel read the slice in place; with rs unset the caller is
    # promising a packed tensor and gets the old math.
    #
    # This matters more here than it looks.  Packing q/k/v to contiguous costs
    # three full copies of the attention input per layer per step, and the
    # backward measured that class of wrapper tax at ~15 ms/step/rank for six
    # copies -- enough to eat this kernel's entire win.  The column offset needs
    # no parameter: a torch slice view carries it in its data_ptr, which is what
    # _shifted_rsrc starts from.
    #
    # OUT is NOT a slice of that buffer -- it is attention's own freshly
    # allocated output -- so it keeps the packed H*D stride and is addressed
    # separately below.
    RS = rs or HD
    # An 8-wide bf16 load is 16 B and must stay 16 B-aligned at every token, so
    # the row stride has to be a multiple of 8 elements.  Production's 2048 is.
    assert RS % 8 == 0, f"row stride {RS} must be a multiple of 8 bf16 elements"
    # Element size of each input, in bytes.  q/k follow fp8_mfma, v follows
    # fp8_v, and they are addressed independently because the two gemms consume
    # them independently: the qk operands are raw bytes fed to an fp8 MFMA while
    # V is still widened on the way into a bf16 tr16 tile.
    QK_BYTES = 1 if FP8_QK else 2
    V_BYTES = 1 if fp8_v else 2
    assert RS >= HD, f"row stride {RS} cannot be narrower than one H*D row"
    NXCD = 8                          # gfx950: 8 XCDs, each an independent L2
    WAVES = waves
    THREADS = WAVES * 64
    MT = BM // (16 * WAVES)          # 16-row m-tiles owned by one wave
    assert BM == 16 * WAVES * MT, f"BM {BM} != 16*{WAVES}*MT"
    assert MT >= 1
    # Rows of the K/V tile staged per cooperative pass. Each pass has one thread
    # per (row, 16-wide d block), so THREADS/16 rows move at a time; the count of
    # passes below is what keeps the whole BN-row tile covered as WAVES varies.
    ROWS_PER_PASS = THREADS // 16
    assert BN % ROWS_PER_PASS == 0, f"BN {BN} not divisible by {ROWS_PER_PASS}"
    KSTEP = D // 32                  # d-tiles of the score gemm (double-rate K=32)
    NTILE = BN // 16                 # 16-n tiles in one BN block (score B reads)
    NDSTEP = D // 16                 # 16-d tiles of the out accumulator
    NKSTEP = BN // 32                # 32-n contraction steps of the out gemm

    def emit_sched():
        if not sched:
            return
        if sched.startswith("iglp"):
            rocdl.iglp_opt(int(sched[4:]))
            return
        raise ValueError(f"unknown sched {sched!r}")

    arch = get_rocm_arch()
    allocator = SmemAllocator(
        None, arch=arch,
        global_sym_name=(f"hstu_fwd_smem_{BM}_{BN}_{KPAD}_{lds_dummy}_w{WAVES}"
                         f"_{FP8_QK or 'bf16'}_{k8_pad}_{int(fp8_v)}"
                         f"_{int(epi_fold)}{int(mask_split)}{int(epi_pk)}"
                         # Suffix only when a new knob is on, so the default
                         # build keeps the shipped symbol name. The literal
                         # "_ws0" keeps the name of the tested FWD_PIPE build.
                         + (f"_pf{int(kv_prefetch)}_pd{int(p_direct)}_ws0"
                            if (kv_prefetch or p_direct) else "")
                         + (f"_ma{int(mask_arith)}" if mask_arith else "")
                         + (f"_pv8k{pv_k}_s{int(pv_msat)}_vp{v8_pad}"
                            f"_x{int(pv_split)}{int(pf_late)}{int(pv_tg)}"
                            if PV8 else "")
                         # pv_s in 1e-6 units (the symbol stays [A-Za-z0-9_]);
                         # absent for pv_s == 0, so the iter46 name holds.
                         + (f"_ps{int(round(pv_s * 1e6))}" if pv_s else "")
                         # v8_swz: suffix only when on (shipped name holds).
                         + (f"_vz{int(v8_swz)}" if v8_swz else "")
                         # rider: suffix only when on (shipped name holds).
                         + (f"_rd{int(rider)}" if rider else "")
                         # spf: suffix only when on (shipped name holds).
                         + ("_spf1" if spf else "")
                         # lut: suffix only when on (shipped name holds); d1
                         # names the d16 combine form, the only one built.
                         + (f"_lut{int(lut)}d1" if lut else "")),
    )
    # pv=k128: V is a 4-slot ring of raw e4m3 rows [32 keys][128 + v8_pad B].
    V8S = D + v8_pad
    assert not PV8 or V8S % 8 == 0, "tr8 / b64 rows must stay 8 B aligned"
    # v8_swz bit 0: rows BN/2.. of a slot sit V8HI bytes further.  Designed
    # (and bank-modelled) for V8S = 144 B = 36 dwords, 16 staging rows per
    # pass (THREADS = 256: row >= 16 <=> pass r == 1) and tr8 t8_hi = row >> 4.
    V8HI = 128 if (v8_swz & 1) else 0
    if v8_swz:
        assert BN == 32 and V8S == 144 and WAVES * 64 == 256, (
            "v8_swz is designed for BN=32, v8_pad=16 (V8S=144), 4 waves")
    V8SLOT = BN * V8S + V8HI             # bytes per ring slot
    NPH = 4                              # BN=32 phases per K=128 group
    # k_lds[BN][D], v_lds[BN][D] (row-major), p_lds[WAVES][BN][16] (per-wave).
    off_k = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_k + (BN * K8STRIDE if FP8_QK else BN * KSTRIDE * 2)
    off_v = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_v + (NPH * V8SLOT if PV8 else VLDS_HW * 2)
    off_p = allocator._align(allocator.ptr, 16)
    # One region per (wave, m-tile) == BM/16 regions, so this is independent of
    # how the m-tiles are distributed over waves.
    allocator.ptr = off_p + (0 if p_direct else (BM // 16) * 16 * PSTRIDE * 2)
    # Never read or written; it exists only to consume the occupancy headroom.
    allocator.ptr = allocator._align(allocator.ptr + lds_dummy, 16)
    # lut: the 2**lut-byte P8 table, appended after every other region, so
    # the shipped offsets hold.  Base + index stays < 2**16: LLVM folds the
    # base into the ds_read_u8 instruction offset.
    LUT_B = (1 << lut) if lut else 0
    off_lut = allocator._align(allocator.ptr, 16)
    if lut:
        assert LUT_B % (THREADS * 16) == 0, "lut copy: dwordx4 per thread"
        assert off_lut + LUT_B <= 1 << 16, "lut: base + index must fit 16 bits"
        allocator.ptr = off_lut + LUT_B

    def emit_body(Q, K, V, OUT, inv_max, qkv_scale, tid, m0, head, seq_start,
                  seq_len, LUT=None):
        v1bf = T.vec(1, T.bf16)
        v4bf = T.vec(4, T.bf16)
        v8bf = T.vec(8, T.bf16)
        v2i32 = T.vec(2, T.i32)
        v4i32 = T.vec(4, T.i32)

        def mfma2(a, b, acc):
            return rocdl_raw.mfma_f32_16x16x32_bf16(
                T.f32x4, arith.unwrap(a), arith.unwrap(b), arith.unwrap(acc),
                0, 0, 0,
            ).result

        # ---- fp8 MFMA for the qk gemm ------------------------------------
        # Both take the SAME (lane, element) -> k mapping the bf16 16x16x32 op
        # uses, so the operands are the raw e4m3 bytes in exactly the order the
        # widened bf16 fragments carried them, and the operand SWAP the score
        # gemm relies on (A=k, B=q, transposed accumulator) is preserved.
        # No scale is folded into the instruction: qkv_scale**2 is applied once
        # per score in the silu epilogue, which keeps the dequant out of the
        # E8M0-only scale field.
        def mfma_qk_k32(a, b, acc):
            return rocdl_raw.mfma_f32_16x16x32_fp8_fp8(
                T.f32x4,
                llvm.BitcastOp(T.i64, arith.unwrap(a)).result,
                llvm.BitcastOp(T.i64, arith.unwrap(b)).result,
                arith.unwrap(acc), 0, 0, 0,
            ).result

        z_i32 = arith.unwrap(arith.constant(0, type=T.i32))

        def mfma_qk_k128(a, b, acc):
            # cbsz=0 / blgp=0 select e4m3 for both operands; scale=0 is E8M0
            # bias 127, multiplier 1.0, so the scale field is a no-op.
            return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4, arith.unwrap(a), arith.unwrap(b), arith.unwrap(acc),
                0, 0, 0, z_i32, 0, z_i32,
            ).result

        def widen8(packed):
            """8 packed e4m3 bytes (vec<2xi32>) -> vec<8xbf16>, exactly."""
            conv = []
            for dw in range_constexpr(2):
                src = vector.extract(packed, static_position=[dw],
                                     dynamic_position=[])
                for hi in (False, True):
                    conv.append(rocdl_raw.cvt_scalef32_pk_bf16_fp8(
                        T.vec(2, T.bf16), src, arith.unwrap(_f32(1.0)), hi))
            lo = vector.shuffle(conv[0], conv[1], [0, 1, 2, 3])
            hi = vector.shuffle(conv[2], conv[3], [0, 1, 2, 3])
            return vector.shuffle(lo, hi, CAT8)

        zero_acc = arith.constant_vector(0.0, T.f32x4)

        # q/k/v walk at RS (they may be column slices of uvqk); OUT is packed.
        # i64: seq_start * RS is a whole-tensor element offset and overflows i32
        # at production sizes -- see _shifted_rsrc.  Only these base shifts need
        # it; the per-row offsets below stay i32, bounded by one sequence.
        head_off = arith.ArithValue(_to_i64(head)) * _i64(D)
        tok_off = arith.ArithValue(_to_i64(seq_start)) * _i64(RS) + head_off
        out_off = (tok_off if RS == HD else
                   arith.ArithValue(_to_i64(seq_start)) * _i64(HD) + head_off)
        # Extent must stop at the LAST ADDRESSABLE ELEMENT, not at the start of
        # the row after the sequence.  The largest offset any load here forms is
        # (seq_len-1)*RS + D-1, so seq_len*RS over-reaches by RS-D elements --
        # invisible while RS == H*D (the slack lands inside the same tensor) but
        # a fault once RS is a uvqk row stride, because for the final sequence it
        # runs past the end of the whole buffer.  Cost a memory access fault at
        # batch 1024 while batch 64 passed on allocator slack.
        # seq_len >= 1 here: emit_body runs only under the m0 < seq_len guard.
        def _seq_bytes(eb):
            return (arith.ArithValue(seq_len) - _i32(1)) * _i32(RS * eb) \
                + _i32(D * eb)

        # Rows past the end of this sequence read back as zero: a zero k/q row
        # makes qk=0 -> silu(0)=0 -> p=0, contributing exactly nothing.
        rQ = _shifted_rsrc(Q, tok_off, QK_BYTES,
                           num_bytes=_seq_bytes(QK_BYTES))
        rK = _shifted_rsrc(K, tok_off, QK_BYTES,
                           num_bytes=_seq_bytes(QK_BYTES))
        rV = _shifted_rsrc(V, tok_off, V_BYTES, num_bytes=_seq_bytes(V_BYTES))
        rOUT = _shifted_rsrc(OUT, out_off, 2)

        base = allocator.get_base()
        if FP8_QK:
            # Typed i32 so every offset is a dword index: the byte tile is only
            # ever touched 8 or 16 bytes at a time, never element-wise.
            k_lds = SmemPtr(base, off_k, T.i32, shape=(BN * K8_DW,)).get()
        else:
            k_lds = SmemPtr(base, off_k, T.bf16, shape=(BN * KSTRIDE,)).get()
        if PV8:
            # Raw e4m3 ring, dword-typed for the b64 staging stores; the tr8
            # reads address it in bytes through _lds_gep.
            v_lds = SmemPtr(base, off_v, T.i32, shape=(NPH * V8SLOT // 4,)).get()
        else:
            v_lds = SmemPtr(base, off_v, T.bf16, shape=(VLDS_HW,)).get()
        p_lds = (None if p_direct else
                 SmemPtr(base, off_p, T.bf16, shape=((BM // 16) * 16 * PSTRIDE,)).get())
        lut_regs = []
        if lut:
            # lut: byte view (gathers) and dword view (prologue copy) of the
            # table.  The global reads are issued here, ahead of the Q and
            # first K/V loads, so the LDS stores before the head loop wait on
            # them with a nonzero vmcnt (VMEM returns in order).
            lut_lds8 = SmemPtr(base, off_lut, T.i8, shape=(LUT_B,)).get()
            lut_lds32 = SmemPtr(base, off_lut, T.i32, shape=(LUT_B // 4,)).get()
            r_lut = buffer_ops.create_buffer_resource(LUT, max_size=True)
            for c in range_constexpr(LUT_B // (THREADS * 16)):
                lut_regs.append(buffer_ops.buffer_load(
                    r_lut, (tid + _i32(c * THREADS)) * _i32(4), vec_width=4,
                    dtype=T.i32))

        lane = tid % _i32(64)
        wave = tid // _i32(64)
        l16 = lane % _i32(16)
        l4 = lane // _i32(16)
        l16q = l16 // _i32(4)                   # tr16: sub-group row selector
        l16r = (l16 % _i32(4)) * _i32(4)        # tr16: halfword offset s*4
        # Dequantization.  q and k are e4m3 with ONE per-tensor scale, so every
        # score carries qkv_scale**2 and it is applied once, in f32, on the way
        # into the silu -- not folded into the MFMA's E8M0-only scale field and
        # not applied per operand.  bf16 inputs pass qkv_scale = 1.0.
        score_scale = (arith.ArithValue(qkv_scale) * qkv_scale * _f32(alpha)
                       if FP8_QK else _f32(alpha))
        # V's scale rides on the accumulator instead, in the epilogue.
        out_scale = arith.ArithValue(qkv_scale) if fp8_v else None
        if epi_fold:
            # exp2's argument becomes the ONLY scaled form of the score, and
            # the leftover -1/log2(e) joins inv_max on the epilogue.
            score_scale = score_scale * _f32(-LOG2E)
            fold = arith.ArithValue(inv_max) * _f32(-1.0 / LOG2E)
            out_scale = fold if out_scale is None else out_scale * fold
        if PV8:
            # P8 carries P' * 2**pv_k (or P' * pv_s); undo it once per output
            # element.
            out_scale = out_scale * _f32(1.0 / pv_s if pv_s
                                         else 2.0 ** (-pv_k))
        mfma_qk_spf = None
        # The scaled score gemm, ONE construction shared by spf and lut.
        if spf or lut:
            # spf: the caller guarantees score_scale == -2**j exactly, so its
            # f32 exponent field IS the E8M0 encoding 127 + j.  lut:
            # score_scale = -2**j * r (r in [1, 2); r = 1 under the SPF snap),
            # the same field 127 + j (caller contract: normal, j in
            # [-108, 103]), so the accumulator is y = 2**j * raw (r is in the
            # table).  Uniform, once per workgroup; taken from THIS f32 value
            # (the one the snapped control multiplies by), never recomputed.
            ss_bits = llvm.BitcastOp(T.i32, arith.unwrap(score_scale)).result
            spf_byte = arith.andi(
                arith.shrui(ss_bits, arith.unwrap(_i32(23))),
                arith.unwrap(_i32(0xFF)))
            b127 = arith.unwrap(_i32(127))

            def mfma_qk_spf(a, b, acc):
                """Score gemm with the runtime A-scale 2**j; B scale x1 (a
                runtime 127, so LLVM keeps the scaled form)."""
                return rocdl_raw.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4, arith.unwrap(a), arith.unwrap(b),
                    arith.unwrap(acc), 0, 0, 0, arith.unwrap(spf_byte), 0,
                    b127,
                ).result
        V2F = T.vec(2, T.f32)
        # Packed-f32 constants, built once per kernel: a <2xf32> splat of the
        # (uniform) score scale and of 1.0.  LLVM keeps the splat in an SGPR
        # pair, so v_pk_mul_f32 reads it as a scalar operand.
        ss2 = (vector.from_elements(
            V2F, [arith.unwrap(score_scale), arith.unwrap(score_scale)])
            if epi_pk else None)
        one2 = arith.constant_vector(1.0, V2F) if epi_pk else None
        # This wave owns MT consecutive 16-row m-tiles: m_base + 16*mt.
        m_base = m0 + wave * _i32(16 * MT)
        p_wave = wave * _i32(MT * 16 * PSTRIDE)   # this wave's p_lds region base
        # mask_arith: m - n = dmn - 16*(nt-mt) - e for score element (mt, nt,
        # e) of this lane, with dmn = lane_m - n0.  lane_m is loop-invariant.
        lane_m = (arith.ArithValue(l16) - l4 * _i32(4) + m_base
                  if mask_arith else None)

        def tr_read(lds, row_base, inner):
            """One ds_read_b64_tr_b16 out of a tile in the tr16 layout
            halfword(r, c) = ((c/16)*ROWS + r)*16 + c%16, with ROWS = BN.
            `row_base` = (c/16)*BN selects the 16-wide c block.  Returns v4bf:
            lane l = 16*l4 + l16 gets r = 4*l4 + 16*inner + e (e=0..3) at
            c = 16*(row_base/BN) + l16 -- a transposed 4-wide slice.
            """
            if isinstance(row_base, int):
                row_base = _i32(row_base)
            off = (arith.ArithValue(row_base) + l4 * _i32(4) + _i32(16 * inner)
                   + l16q) * _i32(16) + l16r
            return rocdl_raw.ds_read_tr16_b64_(v4bf, _lds_gep(lds, off, T.bf16))

        def tr_pair(lds, row_base, inner2):
            """K=32 B fragment: two tr reads whose r values differ by 16.
            The mfma only needs A and B to agree on which logical k each
            (lane, elem) slot carries, not that k be contiguous -- so the two
            v4bf halves are concatenated as they come out."""
            return vector.shuffle(tr_read(lds, row_base, 2 * inner2),
                                  tr_read(lds, row_base, 2 * inner2 + 1), CAT8)

        def lds_scalar(lds, addr):
            v = vector.load_op(v1bf, lds, [_idx(addr)])
            return vector.extract(v, static_position=[0], dynamic_position=[])

        def lds_store_scalar(lds, addr, val_bf):
            v = vector.from_elements(v1bf, [val_bf])
            vector.store(v, lds, [_idx(addr)], alignment=2)

        # ---- Q fragments, A-operand of the score gemm, loop-invariant -------
        # lane l -> q[m = m_base + 16*mt + l16][d = 32*ks + l4*8 + 0..7]
        # MT sets of these are held live for the whole n loop; they are what the
        # shared K fragment is issued against MT times per read.
        qf = []
        for mt in range_constexpr(MT):
            row = arith.ArithValue(m_base + _i32(16 * mt) + l16)
            if FP8_QK == "k32":
                # Same slicing the bf16 op used, in bytes: 8 B at d = 8*l4+32*ks.
                # Halves Q's live register footprint (2 VGPR/fragment, not 4).
                qf.append([
                    buffer_ops.buffer_load(
                        rQ, (row * _i32(RS) + l4 * _i32(8)
                             + _i32(32 * ks)) // _i32(4),
                        vec_width=2, dtype=T.i32)
                    for ks in range_constexpr(KSTEP)
                ])
            elif FP8_QK == "k128":
                # K=128 wants lane l4 to hold d in [32*l4, 32*l4+32): one 32 B
                # run, which is two dwordx4 (a 256-bit buffer load is not
                # selectable on gfx950).  FOUR fragments collapse to one, and
                # the whole d axis is covered by a single mfma.
                b = (row * _i32(RS) + l4 * _i32(32)) // _i32(4)
                qf.append([vector.shuffle(
                    buffer_ops.buffer_load(rQ, b, vec_width=4, dtype=T.i32),
                    buffer_ops.buffer_load(rQ, b + _i32(4), vec_width=4,
                                           dtype=T.i32),
                    [0, 1, 2, 3, 4, 5, 6, 7])])
            else:
                qf.append([
                    buffer_ops.buffer_load(
                        rQ, row * _i32(RS) + l4 * _i32(8) + _i32(32 * ks),
                        vec_width=8, dtype=T.bf16)
                    for ks in range_constexpr(KSTEP)
                ])

        # ---- cooperative K/V staging into LDS -------------------------------
        # THREADS threads load BN*D bf16 as vec8: (BN*D/8)/THREADS vecs each.
        # One pass covers ROWS_PER_PASS = THREADS/16 rows, so the row stride
        # between passes is ROWS_PER_PASS, not a hardcoded 16 -- that only
        # coincided while THREADS was 256.
        KV_VECS = (BN * D) // (THREADS * 8)
        st_col16 = tid % _i32(16)               # 0..15
        st_row0 = tid // _i32(16)               # 0..ROWS_PER_PASS-1
        st_dcol = st_col16 * _i32(8)            # 0,8,..,120
        # V goes to LDS in the tr16 layout so the out-gemm reads it transposed
        # with ds_read_tr16_b64.  A vec8 over 8 consecutive d stays inside one
        # 16-wide d block, so it maps to 8 consecutive tr16 halfwords -> the
        # staging store is still ONE vec8 (no store-side transpose).
        v_blk = st_col16 // _i32(2)              # d/16 block for this vec8
        v_in = (st_col16 % _i32(2)) * _i32(8)    # d%16 offset (0 or 8)

        def stage_kv(n0):
            for r in range_constexpr(KV_VECS):
                n = st_row0 + _i32(ROWS_PER_PASS * r)   # local n in [0, BN)
                g_off = (arith.ArithValue(n0) + n) * _i32(RS) + st_dcol
                v_off = (v_blk * _i32(VROWS) + n) * _i32(16) + v_in
                if FP8_QK:
                    # Same 16-threads-per-row cooperative shape, 8 BYTES each
                    # instead of 8 halfwords: identical instruction count, half
                    # the global traffic and half the LDS written.
                    kv = buffer_ops.buffer_load(rK, g_off // _i32(4),
                                                vec_width=2, dtype=T.i32)
                    l_off = arith.ArithValue(n) * _i32(K8_DW) + st_dcol // _i32(4)
                    vector.store(kv, k_lds, [_idx(l_off)], alignment=8)
                else:
                    kv = buffer_ops.buffer_load(rK, g_off, vec_width=8,
                                                dtype=T.bf16)
                    l_off = n * _i32(KSTRIDE) + st_dcol
                    vector.store(kv, k_lds, [_idx(l_off)], alignment=16)
                if fp8_v:
                    # V's LDS tile stays bf16 (the out gemm is still bf16), so
                    # the e4m3 bytes are widened here, on the WRITE side, which
                    # is touched once per element -- the read side is touched
                    # BN/16 times more often.
                    vv = widen8(buffer_ops.buffer_load(
                        rV, g_off // _i32(4), vec_width=2, dtype=T.i32))
                else:
                    vv = buffer_ops.buffer_load(rV, g_off, vec_width=8,
                                                dtype=T.bf16)
                vector.store(vv, v_lds, [_idx(v_off)], alignment=16)

        # kv_prefetch: stage_kv split in two (fp8 q/k + fp8 V only), so the
        # global loads of iteration i+1 can be issued during iteration i and
        # carried as loop iter_args.  Same addresses, same LDS stores.
        def load_kv(n0):
            regs = []
            for r in range_constexpr(KV_VECS):
                n = st_row0 + _i32(ROWS_PER_PASS * r)
                g_off = (arith.ArithValue(n0) + n) * _i32(RS) + st_dcol
                regs.append(buffer_ops.buffer_load(rK, g_off // _i32(4),
                                                   vec_width=2, dtype=T.i32))
                regs.append(buffer_ops.buffer_load(rV, g_off // _i32(4),
                                                   vec_width=2, dtype=T.i32))
            return regs

        def store_kv(regs):
            for r in range_constexpr(KV_VECS):
                n = st_row0 + _i32(ROWS_PER_PASS * r)
                v_off = (v_blk * _i32(VROWS) + n) * _i32(16) + v_in
                l_off = arith.ArithValue(n) * _i32(K8_DW) + st_dcol // _i32(4)
                vector.store(regs[2 * r], k_lds, [_idx(l_off)], alignment=8)
                vector.store(widen8(regs[2 * r + 1]), v_lds, [_idx(v_off)],
                             alignment=16)

        # ---- pv=k128: raw e4m3 V into ring slot `ph` (static), no widen ----
        def store_kv8(regs, ph):
            for r in range_constexpr(KV_VECS):
                n = st_row0 + _i32(ROWS_PER_PASS * r)
                l_off = arith.ArithValue(n) * _i32(K8_DW) + st_dcol // _i32(4)
                vector.store(regs[2 * r], k_lds, [_idx(l_off)], alignment=8)
                # v8_swz bit 0: row n >= 16 <=> pass r == 1 (static +V8HI B).
                v8_off = (arith.ArithValue(n) * _i32(V8S // 4)
                          + st_dcol // _i32(4)
                          + _i32(ph * V8SLOT // 4 + (V8HI // 4) * r))
                vector.store(regs[2 * r + 1], v_lds, [_idx(v8_off)],
                             alignment=8)

        def one_iter(n0, acc, masked=True):
            # Raise wave issue-priority for the whole MFMA-heavy compute region.
            # Workgroups co-reside with INDEPENDENT barriers,
            # so at any instant one wave is usually in this compute while its
            # neighbour is in the memory-heavy K/V staging below.  s_setprio(1)
            # here (paired with s_setprio(0) before the yield) lets the compute
            # wave win the shared issue slot and keep the single MFMA pipe fed,
            # instead of being interleaved with the neighbour's LDS/global
            # traffic.  Purely an arbitration hint -- semantically a no-op, and
            # a lever neither LLVM's SSA scheduler nor iglp_opt expresses.
            rocdl.s_setprio(1)
            # mask_arith: dmn once per masked iteration; element (mt, nt, e) is
            # kept iff dmn >= 16*(nt-mt) + e  (== m_idx >= n_idx, exactly).
            dmn = (arith.ArithValue(lane_m) - n0
                  if (masked and mask_arith) else None)
            pair_mask = None
            if masked and mask_arith:
                from flydsl._mlir.dialects import arith as _ma
                v2i16 = T.vec(2, T.i16)
                ddc = dmn
                if not mask_split:
                    # mask_split bounds |dmn| by BM+15 in the masked loop;
                    # without it, clamp so the int16 math below cannot wrap.
                    # The clamp range strictly contains every compare constant
                    # 16*(nt-mt)+e, so no predicate changes.
                    ddc = arith.ArithValue(_ma.maxsi(arith.unwrap(ddc),
                                                     arith.unwrap(_i32(-16 * MT))))
                    ddc = arith.ArithValue(_ma.minsi(arith.unwrap(ddc),
                                                     arith.unwrap(_i32(16 * NTILE + 4))))
                d16 = _ma.trunci(T.i16, arith.unwrap(ddc))
                dd2 = vector.from_elements(v2i16, [d16, d16])
                sh15 = arith.constant_vector(15, v2i16)
                pm_cache = {}

                def pair_mask(off, pp):
                    """i32 keep-mask of the bf16 pair (e = 2pp, 2pp+1): 0xFFFF
                    per half iff dmn >= off + e, from ((off+e-1) - dmn) >> 15."""
                    if (off, pp) not in pm_cache:
                        cv = vector.from_elements(v2i16, [
                            arith.unwrap(arith.constant(off + 2 * pp - 1, type=T.i16)),
                            arith.unwrap(arith.constant(off + 2 * pp, type=T.i16))])
                        y = _ma.subi(cv, dd2)
                        pm_cache[(off, pp)] = llvm.BitcastOp(
                            T.i32, _ma.shrsi(y, sh15)).result
                    return pm_cache[(off, pp)]
            # ---- score: qk.T[n][m] = (q @ k.T).T  (per wave, its 16 m x BN n) --
            # OPERANDS SWAPPED vs the natural mfma2(qf, kf): kf and qf have
            # IDENTICAL fragment structure (lane l -> 16-index=l16, k=l4*8+32ks),
            # so putting kf in the A slot and qf in the B slot computes the same
            # scores but with the accumulator TRANSPOSED: output row = n-space,
            # col = m-space.  This is NOT the eliminate-the-round-trip transpose
            # (gen 12, -20.51%, VGPR 78->88 from keeping p live): p is still
            # round-tripped through LDS and freed, so the accumulator is still 4
            # f32/lane and VGPR is unchanged.  The payoff is purely the p-STORE:
            # a lane now holds 4 CONSECUTIVE n (l4*4+0..3) for one m=l16, so the
            # 8 scalar ds_write_u16/iter collapse to 2 ds_write_b64/iter, while
            # the a_p read below is byte-identical (p_lds is the same [m][n]
            # row-major layout, just written transposed).
            # K IS READ ONCE AND ISSUED MT TIMES. The K fragment depends only on
            # (nt, ks), not on m, so hoisting the read above the mt loop is what
            # makes MT>1 cheaper than the equivalent BM increase: the workgroup's
            # K LDS reads scale with WAVES, while its MFMA work scales with BM.
            qk = [[None] * NTILE for _ in range_constexpr(MT)]
            for nt in range_constexpr(NTILE):
                accs = [zero_acc for _ in range_constexpr(MT)]
                if FP8_QK == "k128":
                    # ONE contiguous 32 B run per lane, read as TWO b128 --
                    # against FOUR b128 for the bf16 tile.  This is the k128
                    # lesson: the read INSTRUCTION count halves and the bytes
                    # quarter, on the read the ablation named dominant.
                    kb = (_i32(16 * nt) + l16) * _i32(K8_DW) + l4 * _i32(8)
                    kf = vector.shuffle(
                        vector.load_op(v4i32, k_lds, [_idx(kb)]),
                        vector.load_op(v4i32, k_lds,
                                       [_idx(arith.ArithValue(kb) + _i32(4))]),
                        [0, 1, 2, 3, 4, 5, 6, 7])
                    for mt in range_constexpr(MT):
                        accs[mt] = mfma_qk_k128(kf, qf[mt][0], accs[mt])
                else:
                    for ks in range_constexpr(KSTEP):
                        if FP8_QK == "k32":
                            koff = ((_i32(16 * nt) + l16) * _i32(K8_DW)
                                    + l4 * _i32(2) + _i32(8 * ks))
                            kf = vector.load_op(v2i32, k_lds, [_idx(koff)])
                        else:
                            koff = ((_i32(16 * nt) + l16) * _i32(KSTRIDE)
                                    + l4 * _i32(8) + _i32(32 * ks))
                            kf = vector.load_op(v8bf, k_lds, [_idx(koff)])
                        for mt in range_constexpr(MT):
                            accs[mt] = (mfma_qk_k32 if FP8_QK else mfma2)(
                                kf, qf[mt][ks], accs[mt])
                for mt in range_constexpr(MT):
                    qk[mt][nt] = accs[mt]

            # ---- silu + causal mask -> p, stored [m][n] into p_lds ----------
            # transposed accumulator: lane l elem e -> n = 16*nt + l4*4 + e (4
            # consecutive n), m = l16 (fixed).  Storing p[m=l16][n] into the
            # [m][n] row-major p_lds puts the 4 e's at contiguous LDS offsets ->
            # ONE ds_write_b64 per nt (2/iter).  The out-gemm a_p read (below)
            # reads p_lds[m=l16][n = 32*t + l4*4 ..] unchanged.
            # p_direct keeps the bf16 values in registers instead (p_bf).
            p_bf = [[[None] * 4 for _ in range_constexpr(NTILE)]
                    for _ in range_constexpr(MT)]
            for mt in range_constexpr(MT):
                m_idx = m_base + _i32(16 * mt) + l16
                p_tile = p_wave + _i32(mt * 16 * PSTRIDE)
                for nt in range_constexpr(NTILE):
                    silus = [None] * 4
                    if epi_pk:
                        # The accumulator already holds 4 CONSECUTIVE n per
                        # lane, so pairs form for free: one v_pk_mul_f32 for
                        # x = raw*score_scale, one v_pk_add_f32 for ex+1, one
                        # v_pk_mul_f32 for x*sig, over two elements each.  Only
                        # exp2/rcp stay scalar -- there is no packed
                        # transcendental on gfx950.
                        for e0 in range_constexpr(2):
                            acc4 = qk[mt][nt]
                            x2 = arith.mulf(
                                vector.shuffle(acc4, acc4,
                                               [2 * e0, 2 * e0 + 1]),
                                ss2)
                            exs = [
                                rocdl.exp2(T.f32, arith.unwrap(vector.extract(
                                    x2, static_position=[j],
                                    dynamic_position=[])))
                                for j in range_constexpr(2)
                            ]
                            d2 = arith.addf(
                                vector.from_elements(V2F, exs), one2)
                            sigs = [
                                rocdl.rcp(T.f32, arith.unwrap(vector.extract(
                                    d2, static_position=[j],
                                    dynamic_position=[])))
                                for j in range_constexpr(2)
                            ]
                            s2 = arith.mulf(
                                x2, vector.from_elements(V2F, sigs))
                            for j in range_constexpr(2):
                                silus[2 * e0 + j] = arith.ArithValue(
                                    vector.extract(s2, static_position=[j],
                                                   dynamic_position=[]))
                    for e in range_constexpr(4):
                        if epi_pk:
                            silu = silus[e]
                        else:
                            raw = arith.ArithValue(
                                vector.extract(qk[mt][nt], static_position=[e],
                                               dynamic_position=[]))
                            x = raw * score_scale
                            if epi_fold:
                                # score_scale already carries -log2(e): x IS the
                                # exp2 argument, and x*sig differs from the silu
                                # only by the constant now living in out_scale.
                                ex = rocdl.exp2(T.f32, arith.unwrap(x))
                                sig = arith.ArithValue(rocdl.rcp(
                                    T.f32,
                                    arith.unwrap(arith.ArithValue(ex) + _f32(1.0))))
                                silu = x * sig
                            else:
                                ex = rocdl.exp2(
                                    T.f32, arith.unwrap(x * _f32(-LOG2E)))
                                sig = arith.ArithValue(rocdl.rcp(
                                    T.f32,
                                    arith.unwrap(arith.ArithValue(ex) + _f32(1.0))))
                                silu = arith.ArithValue(x * sig) * inv_max
                        # mask_arith=3: applied to the packed bf16 P below.
                        if masked and not mask_arith:
                            n_idx = (arith.ArithValue(n0) + _i32(16 * nt)
                                     + l4 * _i32(4) + _i32(e))
                            ok = arith.cmpi(arith.CmpIPredicate.sge, arith.unwrap(m_idx),
                                            arith.unwrap(n_idx))
                            silu = arith.ArithValue(arith.select(
                                arith.unwrap(ok), arith.unwrap(silu),
                                arith.unwrap(_f32(0.0))))
                        poff = (p_tile + l16 * _i32(PSTRIDE) + _i32(16 * nt)
                                + l4 * _i32(4) + _i32(e))
                        if p_direct:
                            p_bf[mt][nt][e] = arith.trunc_f(T.bf16, silu)
                        else:
                            lds_store_scalar(p_lds, poff,
                                             arith.trunc_f(T.bf16, silu))

            # p_lds is per-(wave, m-tile) (p_wave + mt separates the regions),
            # so the store->read exchange below is entirely INTRA-wave.  A
            # cross-wave gpu.barrier() is not needed for it; the same-buffer LDS
            # RAW is ordered by a compiler lgkmcnt wait.  v_lds (shared, read in
            # the out gemm) was staged behind the barriers at the loop top and is
            # not touched here, so dropping this barrier is safe.

            # ---- out: out[m][d] += p @ v -----------------------------------
            # V's B-fragment is read transposed straight out of the tr16 tile
            # with ds_read_tr16_b64 (2 b64 tr reads per K=32 step), replacing the
            # old 64 strided scalar loads/iter.  The tr layout hands each lane
            # group g the n set {4g..4g+3, 4g+16..4g+19}, so p's A-fragment is
            # read in that SAME k-labelling: two contiguous v4bf (n=4l4+e and
            # n=4l4+16+e) concatenated, invariant across the D-tiles.
            a_p = [[None] * NKSTEP for _ in range_constexpr(MT)]
            for mt in range_constexpr(MT):
                p_tile = p_wave + _i32(mt * 16 * PSTRIDE)
                for t in range_constexpr(NKSTEP):
                    if p_direct:
                        # lo = n-tile 2t (n = 32t + 4*l4 + e), hi = n-tile 2t+1
                        # (n + 16): exactly the 8 values this lane computed.
                        a_p[mt][t] = vector.from_elements(
                            v8bf, [arith.unwrap(x) for x in
                                   p_bf[mt][2 * t] + p_bf[mt][2 * t + 1]])
                        if pair_mask is not None:
                            # dword k of the fragment = bf16 elements 2k, 2k+1
                            # = (n-tile 2t+k//2, e = 2*(k%2), +1).
                            km = vector.from_elements(v4i32, [
                                pair_mask(16 * (2 * t + k // 2 - mt), k % 2)
                                for k in range(4)])
                            a_p[mt][t] = vector.bitcast(v8bf, arith.andi(
                                vector.bitcast(v4i32, a_p[mt][t]), km))
                        continue
                    base = p_tile + l16 * _i32(PSTRIDE) + _i32(32 * t) + l4 * _i32(4)
                    lo = vector.load_op(v4bf, p_lds, [_idx(base)])
                    hi = vector.load_op(v4bf, p_lds, [_idx(base + _i32(16))])
                    a_p[mt][t] = vector.shuffle(lo, hi, CAT8)
            # V IS READ ONCE AND ISSUED MT TIMES, for the same reason as K: the V
            # fragment depends on (dd, t) only. With the two loops ordered
            # (dd, t) outer and mt inner, the workgroup's V tr-reads scale with
            # WAVES while the MFMA work scales with BM.
            new_acc = [[acc[mt][dd] for dd in range_constexpr(NDSTEP)]
                       for mt in range_constexpr(MT)]
            for dd in range_constexpr(NDSTEP):
                for t in range_constexpr(NKSTEP):
                    b_v = tr_pair(v_lds, dd * VROWS, t)
                    for mt in range_constexpr(MT):
                        new_acc[mt][dd] = mfma2(a_p[mt][t], b_v, new_acc[mt][dd])
            emit_sched()
            # Drop priority so the next iteration's barrier + K/V staging (memory
            # phase) runs at low priority, yielding issue to a neighbour wave
            # that is in its own compute region.
            rocdl.s_setprio(0)
            return new_acc

        # ==== pv=k128 (X4C19 iter46): FP8 P@V over 4 BN=32 phases ==========
        # Declared precision change (P f32 -> e4m3), default off.  The n loop
        # is unrolled by 4: one group = 4 BN=32 phases of score + silu, whose P
        # stays in registers as e4m3, then ONE K=128 P@V
        # (v_mfma_scale_f32_16x16x128_f8f6f4) per (m-tile, 16-d tile).
        # A = P8 (registers): lane l, m = l16; byte 8*ph + 4*nt + e <-> key
        #     32*ph + 16*nt + 4*l4 + e, exactly the silu values this lane made.
        # B = V8 (LDS ring, tr8): source lane j=l16 of group l4 addresses row
        #     key(e=j//2) at column 16*dd + 8*(j%2); receiver j gets
        #     V[key(e)][16*dd + j] as byte e (hstu_bwd_dkdv.py tr8 note), so
        #     its bytes 0..3 / 4..7 are the n-tile-0 / n-tile-1 keys of the
        #     lane's group -- the same labelling as A.
        if PV8:
            i8_t = ir.IntegerType.get_signless(8)
            v8i32 = T.vec(8, T.i32)
            v2i16 = T.vec(2, T.i16)
            _tr8_op = getattr(rocdl_raw, "ds_read_tr8_b64_", None)

            def _res(x):
                return x.result if hasattr(x, "result") else x

            def tr8(byte_off):
                ptr = _lds_gep(v_lds, byte_off, i8_t)
                if _tr8_op is not None:
                    return _res(_tr8_op(v2i32, ptr))
                addr = llvm.PtrToIntOp(T.i32, ptr).result
                return llvm.inline_asm(
                    v2i32, [addr], "ds_read_b64_tr_b8 $0, $1", "=v,v,~{memory}",
                    has_side_effects=True)

            # Per-lane key offset (within a phase) of n-tile nt's 4 keys.
            key_nt = [_i32(16 * nt) + l4 * _i32(4) for nt in range(NTILE)]
            # Score K-read row of lane l16 for n-tile nt (A-row i <-> key).
            k_row = [_i32(16 * nt) + l16 for nt in range(NTILE)]
            # tr8 source row/column for this lane (see the B note above).
            t8_e = l16 // _i32(2)
            t8_hi = t8_e // _i32(4)
            t8_row = t8_hi * _i32(16) + l4 * _i32(4) + t8_e % _i32(4)
            t8_base = t8_row * _i32(V8S) + (l16 % _i32(2)) * _i32(8)
            if V8HI:
                # v8_swz bit 0: rows 16..31 (t8_hi = 1) sit V8HI B further.
                t8_base = t8_base + t8_hi * _i32(V8HI)
            pv_sc = arith.unwrap(_f32(2.0 ** (-pv_k)))
            # pv_s: the silu denominator constant c = f32(2**pv_k / pv_s)
            # (None for the power-of-two scale: den = ex + 1.0 as before).
            # c is materialised ONCE into an SGPR by an opaque s_mov_b32
            # (inline asm "=s", outside the n loop), so fma(ex, c, c) selects
            # VOP3 `v_fma_f32 V, V, S, S`: one VGPR read, no literal, pv8's
            # schedule and VGPR count.  A plain constant c lowers to
            # `v_fmamk_f32 V, V, K, Vc` with c ALSO held in a VGPR (+4 VGPR;
            # +0.68% per call vs pv8, this form +0.23%: X4C19 iter50 fmavar).
            # Same f32 c, same single-rounded fma: bit-identical.  (A whole
            # inline-asm v_fma is NOT safe: LLVM does not pad gfx950's
            # TRANS -> VALU forwarding hazard in front of inline asm.)
            pv_c = None
            # lut: the table carries pv_s (no chain element needs c).
            if pv_s and not lut:
                pv_c = llvm.inline_asm(
                    T.f32, [], "s_mov_b32 $0, 0x%08x" % struct.unpack(
                        "<I", struct.pack("<f", 2.0 ** pv_k / pv_s))[0],
                    "=s")

            def pk4_fp8(vals):
                """4 f32 -> one v2i16 (dword) of 4 e4m3, element order;
                dst = e4m3(src / 2**-pv_k) (the cvt divides)."""
                acc8 = llvm.UndefOp(v2i16).result
                for half in range_constexpr(2):
                    acc8 = _res(rocdl_raw.cvt_scalef32_pk_fp8_f32(
                        v2i16, acc8, arith.unwrap(vals[2 * half]),
                        arith.unwrap(vals[2 * half + 1]), pv_sc, half))
                return acc8

            # ---- lut (LUT-SiLU, see the lut parameter note) -----------------
            def lut_p8(acc4):
                """f32x4 accumulator y (= 2**j * raw) -> one v2i16 P8 dword:
                byte e = T[idx(f16_rtz(y_e))], pk4_fp8's element order."""
                from flydsl._mlir.dialects import arith as _ma
                # cvt_pkrtz, the 13-bit indices, the 4 i8 LDS gathers (bytes
                # in element order).
                v2f16 = T.vec(2, T.f16)
                ys = [arith.unwrap(vector.extract(acc4, static_position=[e],
                                                  dynamic_position=[]))
                      for e in range(4)]
                idx = []
                for p in range(2):
                    h = llvm.BitcastOp(T.i32, _res(rocdl_raw.cvt_pkrtz(
                        v2f16, ys[2 * p], ys[2 * p + 1]))).result
                    # lo half: bits [15 : 16-lut]; hi half: bits [31 : 32-lut]
                    idx.append(_ma.andi(
                        _ma.shrui(h, arith.unwrap(_i32(16 - lut))),
                        arith.unwrap(_i32(LUT_B - 1))))
                    idx.append(_ma.shrui(h, arith.unwrap(_i32(32 - lut))))
                bs = [vector.extract(
                    vector.load_op(T.vec(1, i8_t), lut_lds8, [_idx(i)]),
                    static_position=[0], dynamic_position=[]) for i in idx]

                # The d16 combine: {b0, b2} and {b1, b3} as v2i16
                # (zero-extended bytes in each half), dword = lo | hi << 8.
                def pair(a, b):
                    return llvm.BitcastOp(T.i32, vector.from_elements(
                        v2i16, [_ma.extui(T.i16, arith.unwrap(a)),
                                _ma.extui(T.i16, arith.unwrap(b))])).result
                w = _ma.ori(pair(bs[0], bs[2]), _ma.shli(
                    pair(bs[1], bs[3]), arith.unwrap(_i32(8))))
                return llvm.BitcastOp(v2i16, w).result

            # mask_arith=3 on P8 (X4C19 iter47): in a masked phase the causal
            # zeros go onto the packed e4m3 dword of (mt, nt) -- byte e <-> key
            # n0 + 16*nt + 4*l4 + e -- as ONE v_and_b32, instead of a compare
            # + select per f32 silu before the cvt.  Byte e is kept iff
            # dmn >= 16*(nt-mt) + e, dmn = lane_m - n0 (== m_idx >= n_idx, the
            # same identity as one_iter's pair masks).  Per distinct
            # off = 16*(nt-mt): y0 = (off-1, off) - dmn and y1 = (off+1, off+2)
            # - dmn as packed int16, whose four sign bits are the keep bits of
            # e = 0..3, gathered into whole bytes by one v_perm_b32 with the
            # sign-replicate selectors 8..11 (byte = 0xFF iff bit 15 / 31 of
            # S1 / S0).  The tail group has mask_split's bound |dmn| <= BM+15,
            # so the int16 math cannot wrap.  e4m3(+0.0) is 0x00 and each
            # byte converts independently, so P8 is bit-identical to
            # select-then-convert for every input, non-finite included.
            PV_MA = mask_arith == 3
            SEL_SIGN4 = 0x0B0A0908  # bytes <- sign(S1.lo), (S1.hi), (S0.lo), (S0.hi)

            def p8_byte_masks(n0):
                """off -> i32 keep-mask of the P8 dword at this phase's n0."""
                from flydsl._mlir.dialects import arith as _ma
                d16 = _ma.trunci(T.i16, arith.unwrap(
                    arith.ArithValue(lane_m) - n0))
                dd2 = vector.from_elements(v2i16, [d16, d16])
                sel = arith.unwrap(_i32(SEL_SIGN4))
                cache = {}

                def mask(off):
                    if off not in cache:
                        ys = []
                        for c0 in (off - 1, off + 1):
                            cv = vector.from_elements(v2i16, [
                                arith.unwrap(arith.constant(c0, type=T.i16)),
                                arith.unwrap(arith.constant(c0 + 1, type=T.i16))])
                            ys.append(llvm.BitcastOp(
                                T.i32, _ma.subi(cv, dd2)).result)
                        # v_perm_b32 D = perm(S0 = y1 (e2|e3), S1 = y0 (e0|e1))
                        cache[off] = llvm.call_intrinsic(
                            T.i32, "llvm.amdgcn.perm", [ys[1], ys[0], sel],
                            [], [])
                    return cache[off]
                return mask

            def one_phase(n0, masked):
                """Score + silu of one BN=32 phase -> P8 pieces [mt][nt]."""
                rocdl.s_setprio(1)
                bmask = p8_byte_masks(n0) if (masked and PV_MA) else None
                qk = [[None] * NTILE for _ in range_constexpr(MT)]
                for nt in range_constexpr(NTILE):
                    kb = arith.ArithValue(k_row[nt]) * _i32(K8_DW) + l4 * _i32(8)
                    kf = vector.shuffle(
                        vector.load_op(v4i32, k_lds, [_idx(kb)]),
                        vector.load_op(v4i32, k_lds,
                                       [_idx(arith.ArithValue(kb) + _i32(4))]),
                        [0, 1, 2, 3, 4, 5, 6, 7])
                    for mt in range_constexpr(MT):
                        qk[mt][nt] = (mfma_qk_spf if (spf or lut)
                                      else mfma_qk_k128)(
                            kf, qf[mt][0], zero_acc)
                p8 = [[None] * NTILE for _ in range_constexpr(MT)]
                for mt in range_constexpr(MT):
                    m_idx = m_base + _i32(16 * mt) + l16
                    for nt in range_constexpr(NTILE):
                        vals = []
                        # lut: no per-element chain (0 trips); lut_p8 below.
                        for e in range_constexpr(0 if lut else 4):
                            raw = arith.ArithValue(
                                vector.extract(qk[mt][nt], static_position=[e],
                                               dynamic_position=[]))
                            if spf:
                                # raw already carries 2**j = -score_scale:
                                # x = -raw, the value raw * score_scale has in
                                # the snapped control; the negation folds into
                                # exp2's and the silu multiply's VOP3 source
                                # modifiers (no instruction).
                                x = arith.ArithValue(
                                    arith.negf(arith.unwrap(raw)))
                            else:
                                x = raw * score_scale
                            ex = rocdl.exp2(T.f32, arith.unwrap(x))
                            if pv_c is not None:
                                # pv_s: c * (ex + 1) as ONE fma, so P8 =
                                # e4m3(pv_s * P') through the same cvt.
                                den = llvm.intr_fma(arith.unwrap(ex), pv_c,
                                                    pv_c)
                            else:
                                den = arith.unwrap(
                                    arith.ArithValue(ex) + _f32(1.0))
                            sig = arith.ArithValue(rocdl.rcp(T.f32, den))
                            silu = x * sig
                            # mask_arith=3: applied to the packed P8 below.
                            if masked and not PV_MA:
                                n_idx = (arith.ArithValue(n0) + key_nt[nt]
                                         + _i32(e))
                                ok = arith.cmpi(arith.CmpIPredicate.sge,
                                                arith.unwrap(m_idx),
                                                arith.unwrap(n_idx))
                                silu = arith.ArithValue(arith.select(
                                    arith.unwrap(ok), arith.unwrap(silu),
                                    arith.unwrap(_f32(0.0))))
                            vals.append(silu)
                        p8[mt][nt] = (lut_p8(qk[mt][nt]) if lut
                                      else pk4_fp8(vals))
                        if bmask is not None:
                            p8[mt][nt] = arith.andi(p8[mt][nt], llvm.BitcastOp(
                                v2i16, bmask(16 * (nt - mt))).result)
                return p8

            def close_phase():
                emit_sched()
                rocdl.s_setprio(0)

            def pv_group(acc, p8s):
                """K=128 P@V of one 4-phase group: 32 tr8 + 16 f8f6f4 / wave."""
                a8 = []
                for mt in range_constexpr(MT):
                    pcs = [p8s[ph][mt][nt] for ph in range(NPH)
                           for nt in range(NTILE)]
                    q4 = [vector.shuffle(pcs[2 * i], pcs[2 * i + 1], [0, 1, 2, 3])
                          for i in range(4)]
                    h8 = [vector.shuffle(q4[2 * i], q4[2 * i + 1], list(range(8)))
                          for i in range(2)]
                    a8.append(vector.bitcast(
                        v8i32, vector.shuffle(h8[0], h8[1], list(range(16)))))
                new_acc = [[acc[mt][dd] for dd in range_constexpr(NDSTEP)]
                           for mt in range_constexpr(MT)]
                for dd in range_constexpr(NDSTEP):
                    parts = [tr8(arith.ArithValue(t8_base)
                                 + _i32(ph * V8SLOT + 16 * dd))
                             for ph in range(NPH)]
                    b8 = vector.shuffle(
                        vector.shuffle(parts[0], parts[1], [0, 1, 2, 3]),
                        vector.shuffle(parts[2], parts[3], [0, 1, 2, 3]),
                        list(range(8)))
                    for mt in range_constexpr(MT):
                        new_acc[mt][dd] = mfma_qk_k128(a8[mt], b8,
                                                       new_acc[mt][dd])
                return new_acc

            def group(n0g, acc, regs, masked, last=False):
                """4 phases (barrier, stage, barrier, prefetch, score+silu)
                then one K=128 P@V.  Returns (acc, next phase's K/V regs)."""
                p8s = []
                for ph in range_constexpr(NPH):
                    gpu.barrier()
                    store_kv8(regs, ph)
                    gpu.barrier()
                    n0 = arith.ArithValue(n0g) + _i32(BN * ph)
                    late = pf_late and ph == NPH - 1
                    if last and ph == NPH - 1:
                        regs = []
                    elif not late:
                        regs = load_kv(n0 + _i32(BN))
                    if pv_tg and last and ph > 0:
                        # uniform branch: this padded phase has no key < n_end
                        live = arith.cmpi(arith.CmpIPredicate.slt,
                                          arith.unwrap(n0), arith.unwrap(n_end))
                        rt = [v2i16] * (MT * NTILE)
                        br = scf.IfOp(live, results_=rt, has_else=True)
                        with ir.InsertionPoint(br.then_block):
                            pp = one_phase(n0, masked)
                            scf.YieldOp([arith.unwrap(pp[mt][nt]) for mt in range(MT)
                                         for nt in range(NTILE)])
                        with ir.InsertionPoint(br.else_block):
                            z = arith.unwrap(arith.constant_vector(0, v2i16))
                            scf.YieldOp([z] * (MT * NTILE))
                        res_ = list(br.results)
                        p8s.append([[res_[mt * NTILE + nt] for nt in range(NTILE)]
                                    for mt in range(MT)])
                    else:
                        p8s.append(one_phase(n0, masked))
                    if ph != NPH - 1:
                        close_phase()
                if pf_late and not last:
                    regs = load_kv(arith.ArithValue(n0g) + _i32(BN * NPH))
                if pv_split:
                    emit_sched()
                    rocdl.sched_barrier(0)
                acc = pv_group(acc, p8s)
                close_phase()
                return acc, regs

        # ---- n loop ---------------------------------------------------------
        # out[m] sums n <= m, m up to m0+BM-1, so n in [0, min(m0+BM, seq_len)).
        n_end = arith.ArithValue(
            arith.select(
                arith.cmpi(arith.CmpIPredicate.slt, arith.unwrap(m0 + _i32(BM)),
                           arith.unwrap(seq_len)),
                arith.unwrap(m0 + _i32(BM)), arith.unwrap(seq_len)))
        n_iters = (n_end + _i32(BN - 1)) // _i32(BN)

        step = arith.constant(1, index=True)
        # The loop carries MT accumulator sets, flattened as mt-major.
        NACC = MT * NDSTEP

        def n_loop(lo, hi, init, masked):
            lp = scf.ForOp(arith.unwrap(arith.index_cast(T.index, arith.unwrap(lo))),
                           arith.unwrap(arith.index_cast(T.index, arith.unwrap(hi))),
                           arith.unwrap(step), init)
            with ir.InsertionPoint(lp.body):
                it = arith.ArithValue(arith.index_cast(T.i32, lp.induction_variable))
                n0 = it * _i32(BN)
                flat = list(lp.inner_iter_args[:NACC])
                acc = [flat[mt * NDSTEP:(mt + 1) * NDSTEP] for mt in range(MT)]
                gpu.barrier()
                if kv_prefetch:
                    # K/V of THIS iteration were loaded one iteration earlier.
                    store_kv(list(lp.inner_iter_args[NACC:]))
                else:
                    stage_kv(n0)
                gpu.barrier()
                # kv_prefetch: issue the NEXT iteration's loads now; the last
                # iteration's prefetch (rows >= n_end) is never stored.
                nxt = load_kv(n0 + _i32(BN)) if kv_prefetch else []
                acc = one_iter(n0, acc, masked=masked)
                scf.YieldOp([arith.unwrap(x) for row in acc for x in row]
                            + [arith.unwrap(x) for x in nxt])
            return lp

        init = [arith.unwrap(zero_acc) for _ in range(NACC)]
        if kv_prefetch:
            init = init + [arith.unwrap(x) for x in load_kv(_i32(0))]
        if PV8:
            if lut:
                # lut: table global -> LDS (b128 per dwordx4 read).  No new
                # barrier: the first barrier of the first group (head loop
                # trip 0, or the tail group when m0 = 0) precedes the first
                # gather on every path, and the table is never rewritten.
                for c in range_constexpr(len(lut_regs)):
                    vector.store(lut_regs[c], lut_lds32,
                                 [_idx((tid + _i32(c * THREADS)) * _i32(4))],
                                 alignment=16)
            if pv_msat:
                # MODE.FP16_OVFL = 1: v_cvt_scalef32_pk_fp8_f32 saturates to
                # +-448 instead of NaN (backward msat, hstu_bwd_dkdv.py).
                llvm.InlineAsmOp(
                    None, [],
                    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 1\n"
                    "s_nop 7\ns_nop 7", "", has_side_effects=True)
            # Head: m0/128 whole groups strictly below the diagonal (BM=128
            # makes m0/BN a multiple of 4).  Tail: ONE masked group starting
            # at n0 = m0, padded to 4 phases; padding rows are >= n_end =
            # seq_len and read back as zero through num_records.
            n_groups = arith.ArithValue(m0) // _i32(BN * NPH)
            lp = scf.ForOp(
                arith.unwrap(arith.index_cast(T.index, arith.unwrap(_i32(0)))),
                arith.unwrap(arith.index_cast(T.index, arith.unwrap(n_groups))),
                arith.unwrap(step), init)
            with ir.InsertionPoint(lp.body):
                gi = arith.ArithValue(arith.index_cast(T.i32, lp.induction_variable))
                flat = list(lp.inner_iter_args[:NACC])
                acc = [flat[mt * NDSTEP:(mt + 1) * NDSTEP] for mt in range(MT)]
                acc, nxt = group(gi * _i32(BN * NPH), acc,
                                 list(lp.inner_iter_args[NACC:]), masked=False)
                scf.YieldOp([arith.unwrap(x) for row in acc for x in row]
                            + [arith.unwrap(x) for x in nxt])
            flat = list(lp.results[:NACC])
            acc = [flat[mt * NDSTEP:(mt + 1) * NDSTEP] for mt in range(MT)]
            res, _ = group(m0, acc, list(lp.results[NACC:]), masked=True,
                           last=True)
            if pv_msat:
                llvm.InlineAsmOp(
                    None, [],
                    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 23, 1), 0\n"
                    "s_nop 7\ns_nop 7", "", has_side_effects=True)
        elif mask_split:
            # n0 + BN <= m0  <=>  it < m0/BN, and m0 is a multiple of BN, so the
            # first m0/BN iterations are strictly below the diagonal: every
            # (m, n) they touch has n <= m0-1 < m0 <= m.  The mask is dead there.
            safe = arith.ArithValue(m0) // _i32(BN)
            head = n_loop(_i32(0), safe, init, masked=False)
            # kv_prefetch: the head loop's prefetched K/V carry into the tail.
            init = list(head.results if kv_prefetch else head.results[:NACC])
            loop = n_loop(safe, n_iters, init, masked=True)
        else:
            loop = n_loop(_i32(0), n_iters, init, masked=True)

        if not PV8:
            flat_res = list(loop.results[:NACC])
            res = [flat_res[mt * NDSTEP:(mt + 1) * NDSTEP] for mt in range(MT)]

        def rider_epilogue(res):
            """rider R2a / R2b epilogue (see the `rider` parameter):
            the same 64 bf16 values per lane, each from the same out_scale
            v_mul_f32 and RNE convert, to the same addresses."""
            ROWB = HD * 2                    # bytes per output row
            # R2b: the (e, dd) part of the address is a MUBUF imm (12 bits).
            assert ROWB * 3 + 32 * (NDSTEP - 1) < 4096, ROWB
            # R2a: m_base is wave-uniform (wave = tid // 64); readfirstlane
            # makes the full-tile test a scalar compare and branch.
            m_base_s = (arith.ArithValue(rocdl.readfirstlane(
                T.i32, arith.unwrap(m_base))) if R2A else None)
            for mt in range_constexpr(MT):
                r0 = m_base + _i32(16 * mt) + l4 * _i32(4)    # row of e = 0
                vb = (arith.ArithValue(r0) * _i32(ROWB) + l16 * _i32(2)
                      if R2B else None)

                def scaled(dd, e):
                    val = vector.extract(res[mt][dd], static_position=[e],
                                         dynamic_position=[])
                    if out_scale is not None:
                        val = arith.unwrap(arith.ArithValue(val) * out_scale)
                    return val

                def guard(e, emit):
                    """emit() under the shipped per-row guard m < seq_len."""
                    ok = arith.cmpi(arith.CmpIPredicate.slt,
                                    arith.unwrap(r0 + _i32(e)),
                                    arith.unwrap(seq_len))
                    st_if = scf.IfOp(ok, results_=[], has_else=False)
                    with ir.InsertionPoint(st_if.then_block):
                        emit()
                        scf.YieldOp([])

                def store(dd, e, guarded, get_bf):
                    """Row e, column 16*dd + l16 <- get_bf() (16 bits)."""
                    def emit():
                        bf = get_bf()
                        if R2B:
                            buffer_ops.buffer_store(
                                bf, rOUT, vb + _i32(ROWB * e + 32 * dd),
                                offset_is_bytes=True)
                        else:
                            off = (arith.ArithValue(r0 + _i32(e)) * _i32(HD)
                                   + _i32(16 * dd) + l16)
                            buffer_ops.buffer_store(bf, rOUT, off)
                    if guarded:
                        guard(e, emit)
                    else:
                        emit()

                def stores(guarded):
                    for dd in range_constexpr(NDSTEP):
                        for e in range_constexpr(4):
                            store(dd, e, guarded,
                                  lambda dd=dd, e=e: arith.trunc_f(
                                      T.bf16, scaled(dd, e)))

                if not R2A:
                    stores(True)
                    continue
                # R2a: all 16 rows of this m-tile are < seq_len.
                full = arith.cmpi(
                    arith.CmpIPredicate.sle,
                    arith.unwrap(m_base_s + _i32(16 * (mt + 1))),
                    arith.unwrap(seq_len))
                fb = scf.IfOp(full, results_=[], has_else=True)
                with ir.InsertionPoint(fb.then_block):
                    stores(False)
                    scf.YieldOp([])
                with ir.InsertionPoint(fb.else_block):
                    stores(True)
                    scf.YieldOp([])

        # ---- epilogue: store out[MT*16 m][128 d] for this wave --------------
        # lane l elem e -> m = m_base + 16*mt + l4*4 + e, d = 16*dd + l16
        if rider:
            rider_epilogue(res)
            return
        for mt in range_constexpr(MT):
            for dd in range_constexpr(NDSTEP):
                d_idx = _i32(16 * dd) + l16
                for e in range_constexpr(4):
                    m_idx = m_base + _i32(16 * mt) + l4 * _i32(4) + _i32(e)
                    ok = arith.cmpi(arith.CmpIPredicate.slt, arith.unwrap(m_idx),
                                    arith.unwrap(seq_len))
                    st_if = scf.IfOp(ok, results_=[], has_else=False)
                    with ir.InsertionPoint(st_if.then_block):
                        off = arith.ArithValue(m_idx) * _i32(HD) + d_idx
                        val = vector.extract(res[mt][dd], static_position=[e],
                                            dynamic_position=[])
                        if out_scale is not None:
                            val = arith.unwrap(
                                arith.ArithValue(val) * out_scale)
                        buffer_ops.buffer_store(arith.trunc_f(T.bf16, val), rOUT, off)
                        scf.YieldOp([])

    @flyc.kernel
    def fwd_kernel(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        OUT: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        INV_MAX_ARG: fx.Float32,
        QKV_SCALE_ARG: fx.Float32,
        NBATCH: fx.Int32,
    ):
        tid = _as_i32(fx.thread_idx.x)
        # XCD-pinned dispatch: grid.x = NXCD*heads carries (slot, head) with slot
        # fastest, so the linear id's XCD selector (id % NXCD, since NXCD*heads is
        # a multiple of NXCD) is exactly `slot`.  batch = slot + NXCD*blk_z, so
        # every workgroup of one batch (all heads, all m-blocks) shares slot and
        # thus one XCD's L2.  One sequence's K+V (~5.5MB for 4 heads) is fetched
        # once and reused across its m-blocks, instead of being pulled into all 8
        # L2s when the default order scatters m-blocks by m_blk % NXCD.
        blk0 = _as_i32(fx.block_idx.x)      # in [0, NXCD*heads)
        m_blk = _as_i32(fx.block_idx.y)     # m-tile, sizes the grid
        blk_z = _as_i32(fx.block_idx.z)     # in [0, cdiv(batch, NXCD))
        slot = blk0 % _i32(NXCD)
        head = blk0 // _i32(NXCD)
        batch = slot + _i32(NXCD) * blk_z

        r_off = buffer_ops.create_buffer_resource(SEQ_OFF, max_size=True)
        seq_start = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch, vec_width=1, dtype=T.i32))
        seq_end = arith.ArithValue(
            buffer_ops.buffer_load(r_off, batch + _i32(1), vec_width=1, dtype=T.i32))
        seq_len = seq_end - seq_start
        m0 = m_blk * _i32(BM)

        guard = scf.IfOp(
            arith.cmpi(arith.CmpIPredicate.slt, arith.unwrap(m0), arith.unwrap(seq_len)),
            results_=[], has_else=False,
        )
        with ir.InsertionPoint(guard.then_block):
            emit_body(Q, K, V, OUT,
                      arith.ArithValue(arith.unwrap(INV_MAX_ARG)),
                      arith.ArithValue(arith.unwrap(QKV_SCALE_ARG)),
                      tid, m0, head, seq_start, seq_len)
            scf.YieldOp([])

    @flyc.jit
    def launch(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        OUT: fx.Tensor,
        SEQ_OFF: fx.Tensor,
        batch: fx.Int32,
        m_blocks: fx.Int32,
        inv_max: fx.Float32,
        qkv_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        # XCD-pinned dispatch (see emit_body): grid.x = NXCD*heads packs
        # (slot, head), grid.z rounds batch up to a multiple of NXCD.  Padded
        # slots (batch >= NBATCH) read seq_off[batch] and [batch + 1] through
        # the max_size resource, which has NO range check (num_records
        # 0xFFFFFFFF), so SEQ_OFF must hold 8*cdiv(NBATCH, 8) + 1 entries: the
        # launcher pads it with the total token count when NBATCH % 8 != 0
        # (hstu_flydsl_mha._fwd_seq_offsets_i32), which gives every padded
        # slot seq_len 0, dropped by the m0<seq_len guard.  Without that
        # padding a padded slot runs on the stale bytes after the tensor.
        fwd_kernel(Q, K, V, OUT, SEQ_OFF, inv_max, qkv_scale, batch).launch(
            grid=(NXCD * heads, m_blocks, (batch + NXCD - 1) // NXCD),
            block=(THREADS, 1, 1), stream=stream,
        )

    if lut:
        # lut: the same entry and launch with one more kernel argument, LUT
        # (uint8[2**lut], contiguous, on Q's device) after SEQ_OFF.  Defined
        # only when lut is on, under the same names, so the symbol stays
        # fwd_kernel_0 and lut = 0 keeps the shipped definitions above
        # untouched.
        @flyc.kernel
        def fwd_kernel(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            OUT: fx.Tensor,
            SEQ_OFF: fx.Tensor,
            LUT: fx.Tensor,
            INV_MAX_ARG: fx.Float32,
            QKV_SCALE_ARG: fx.Float32,
            NBATCH: fx.Int32,
        ):
            # Same XCD-pinned decode and m0 guard as the shipped entry.
            tid = _as_i32(fx.thread_idx.x)
            blk0 = _as_i32(fx.block_idx.x)
            m_blk = _as_i32(fx.block_idx.y)
            blk_z = _as_i32(fx.block_idx.z)
            slot = blk0 % _i32(NXCD)
            head = blk0 // _i32(NXCD)
            batch = slot + _i32(NXCD) * blk_z

            r_off = buffer_ops.create_buffer_resource(SEQ_OFF, max_size=True)
            seq_start = arith.ArithValue(
                buffer_ops.buffer_load(r_off, batch, vec_width=1, dtype=T.i32))
            seq_end = arith.ArithValue(
                buffer_ops.buffer_load(r_off, batch + _i32(1), vec_width=1, dtype=T.i32))
            seq_len = seq_end - seq_start
            m0 = m_blk * _i32(BM)

            guard = scf.IfOp(
                arith.cmpi(arith.CmpIPredicate.slt, arith.unwrap(m0), arith.unwrap(seq_len)),
                results_=[], has_else=False,
            )
            with ir.InsertionPoint(guard.then_block):
                emit_body(Q, K, V, OUT,
                          arith.ArithValue(arith.unwrap(INV_MAX_ARG)),
                          arith.ArithValue(arith.unwrap(QKV_SCALE_ARG)),
                          tid, m0, head, seq_start, seq_len, LUT=LUT)
                scf.YieldOp([])

        @flyc.jit
        def launch(
            Q: fx.Tensor,
            K: fx.Tensor,
            V: fx.Tensor,
            OUT: fx.Tensor,
            SEQ_OFF: fx.Tensor,
            LUT: fx.Tensor,
            batch: fx.Int32,
            m_blocks: fx.Int32,
            inv_max: fx.Float32,
            qkv_scale: fx.Float32,
            stream: fx.Stream = fx.Stream(None),
        ):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            fwd_kernel(Q, K, V, OUT, SEQ_OFF, LUT, inv_max, qkv_scale,
                       batch).launch(
                grid=(NXCD * heads, m_blocks, (batch + NXCD - 1) // NXCD),
                block=(THREADS, 1, 1), stream=stream,
            )

    # lut: LUT is one more positional argument (after SEQ_OFF).
    NPOS_LUT = 1 if lut else 0

    def call(*args, **kw):
        # qkv_scale is optional at the call site: bf16 callers (production's
        # hstu_flydsl_mha) pass 8 positional args and mean 1.0.  With lut on
        # every count below is one higher (LUT after SEQ_OFF).
        if len(args) == 8 + NPOS_LUT and "qkv_scale" not in kw:
            args = args + (1.0,)
        if ("stream" not in kw and len(args) < 10 + NPOS_LUT
                and current_stream_launch()):
            # Launch on the caller's current PyTorch stream, not launch()'s
            # default null stream, which does not order against non-blocking
            # side streams.  Same handle on the default stream; an explicit
            # stream, keyword or 10th positional, still wins.  With
            # $DLRM_HSTU_FP8_FLYDSL_CURRENT_STREAM=0 no stream is passed: the
            # exact pre-5ae3127 launch arguments.
            import torch

            q = args[0] if args else kw["Q"]
            kw["stream"] = fx.Stream(torch.cuda.current_stream(q.device))
        args = tuple(fx.Float32(a) if isinstance(a, float) else a for a in args)
        if isinstance(kw.get("inv_max"), float):
            kw["inv_max"] = fx.Float32(kw["inv_max"])
        # Same launch through a cached flyc.compile dispatcher under FlyDSL
        # 0.2.4 (ln_bwd.py _fast_call); the plain jit call under 0.1.1.
        return _fast_call(launch, *args, **kw)

    call.BM = BM
    call.BN = BN
    call.launch = launch
    # 13 when the LUT-SiLU table argument is expected, 0 when off.
    call.lut = lut
    return call
