# Grouped FlyDSL weighted LayerNorm backward for the three LayerNorm(512) dx-sum
# backwards of the preprocessor tail, which share one incoming gradient (Arbor
# P7 milestone M1b on PR #7 v2.1; caller: ops/preproc_grouped_tail.py, E-M2).
"""ln_bwd_grouped3: three PR #7 LayerNorm(512) dx-sum backwards that share dy.

The grouped preprocessor tail (E-M1, ops/preproc_grouped_tail.py) computes
O = LN_x(h_x) (+) (LN_a(h_a) (+) LN_c(h_c)), so the three last-pair LayerNorm
backwards receive the same dy [N, 512] bf16. With PR #7 on (gin item (29),
DLRM_HSTU_FP8_WLN_BWD_FLYDSL) each of them is one ops/flydsl/ln_bwd.py
ln_bwd(..., sum_dx=True) call (fp32 weights): main kernel
ln_bwd_flydsl_d512_dxsum_x16_c3 plus a finalize, 3 KB/row each.

The main kernel here (ln_bwd_flydsl_grp3_d512_dxsum_x{R}_c{C}) loads each dy
row slot once and, for each problem p, runs ln_bwd.py's exact per-row
arithmetic of the dx-sum variant (FMA-folded form, tri_order False,
build_ln_bwd compute_slot) with p's x, mean, rstd and weight; it stores dx_p
and keeps p's own three accumulators (dW, dB, dx sum). 7 KB/row for the three.

BITWISE CONTRACT: every output (dx_p bf16; dW_p, dB_p in the weight dtype and
the dx sum fp32) equals the corresponding output of ln_bwd(dy, x_p, w_p, b_p,
mean_p, rstd_p, sum_dx=True) with ln_bwd.py's production options, because
  * the persistent grid G = grid_for(M, ..., WAVES, tile_mult), the rows per
    block-iteration RPS, the row -> block mapping and the per-lane
    accumulation order (iteration-major, then slot u) are ln_bwd.py's, with
    its dx-sum cfg DEFAULT_CFG[(512, False, False, True)];
  * the v2.1 options are ln_bwd.py's own, read through
    ln_bwd.kernel_options(512, False, False, True) (and therefore the same
    kill switches, DLRM_HSTU_FP8_WLN_BWD_FLYDSL_XCD_REMAP / _STREAM_CACHE):
    the XCD-aware block order REMAP = R maps physical block b to the same
    logical block lb as ln_bwd.py, and lb replaces b in both the row mapping
    and the partial row, so logical block lb sums exactly ln_bwd's block-lb
    rows; the cache policy (load / store aux bits) changes no value;
  * so each partial row PART[3p + o, lb, :] equals ln_bwd's PART[o, lb, :] of
    problem p, and the grouped finalize (build_finalize_grouped) runs
    ln_bwd.py build_finalize's arithmetic per (8 columns, output).

COMPUTATION (MLPerf closed): no arithmetic is shared between the problems.
The three dB sums are mathematically equal (dB = column sum of the shared
dy), and each problem still accumulates and reduces its own; only the dy
LOAD is shared (memory fusion).

Partials: one fp32 PART [3*NPROB, G, D] (row q = 3*p + o). The block's LDS
reduction reuses one NACC*SLOTS*D buffer per problem, with barriers between.
"""

import functools
from typing import Dict, List, Optional, Sequence, Tuple

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu
from generative_recommenders.ops.flydsl._compat import buffer_ops, vector  # FlyDSL 0.2 API on 0.2 / >= 0.3
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from generative_recommenders.ops.flydsl import ln_bwd as _v2
from generative_recommenders.ops.flydsl.ln_bwd import (
    FIN_THREADS, _as_i32, _elems, _f32, _i32, _idx, _rsrc, _shifted_rsrc, _u, _v8f, _xor_sum,
)

NPROB = 3
D_GROUPED = 512
# ln_bwd.py's variant key of the three calls this replaces: (D, swish, residual, dx sum)
_V2_KEY = (D_GROUPED, False, False, True)
# ln_bwd.py's d512 dx-sum cfg: (waves, unroll, tile_mult, dpp). G, RPS and the row
# mapping decide the partial order, so the grouped kernel must use exactly this.
GROUPED_CFG = _v2.DEFAULT_CFG[_V2_KEY]


def grouped_kernel_options(cache=None, remap=None, store_cache=None) -> Tuple[int, int, int]:
    """(remap run length, load cache policy, store cache policy): ln_bwd.py's
    kernel_options for the d512 dx-sum variant, so the grouped kernel follows
    PR #7 v2.1's production options and kill switches."""
    return _v2.kernel_options(*_V2_KEY, cache=cache, remap=remap, store_cache=store_cache)


@functools.lru_cache(maxsize=None)
def build_ln_bwd_grouped(D: int = 512, WAVES: int = 4, UNROLL: int = 4, CACHE: int = 0,
                         DPP: bool = True, REMAP: int = 0, STORE_CACHE: int = -1):
    """Main kernel for NPROB=3 dx-sum problems (fp32 weights, no swish, no residual).
    CACHE / STORE_CACHE / REMAP as ln_bwd.py build_ln_bwd (the loads' and the dx
    stores' cache policy, -1 = CACHE; the XCD block-order run length, 0 = off)."""
    assert D in (256, 512), D
    assert not isinstance(REMAP, bool) and REMAP >= 0, REMAP
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
    # ln_bwd.py's naming, with grp3 for the grouping
    name = (f"ln_bwd_flydsl_grp{NPROB}_d{D}_dxsum" + (f"_x{REMAP}" if REMAP else "")
            + (f"_c{CACHE}" if CACHE else "") + (f"_s{SCACHE}" if SCACHE != CACHE else ""))
    allocator = SmemAllocator(None, arch=get_rocm_arch(),
                              global_sym_name=f"{name}_w{WAVES}_u{UNROLL}_p{int(DPP)}_smem")
    off_red = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_red + NACC * SLOTS * D * 4

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

    def emit(DXS, DY, XS, WS, MEANS, RSTDS, PART, M_ARG, ITERS_ARG, G_ARG):
        tid = arith.ArithValue(_as_i32(fx.thread_idx.x))
        blk = arith.ArithValue(_as_i32(fx.block_idx.x))
        M = arith.ArithValue(_u(M_ARG))
        ITERS = arith.ArithValue(_u(ITERS_ARG))
        G = arith.ArithValue(_u(G_ARG))
        if REMAP:
            # ln_bwd.py's XCD order: the logical block, used for the rows AND
            # the PART row; identity unless G % (NXCD*REMAP) == 0
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
        rDX = [_rsrc(DX, rbytes) for DX in DXS]
        rMEAN = [_rsrc(MN, sbytes) for MN in MEANS]
        rRSTD = [_rsrc(RS, sbytes) for RS in RSTDS]

        def load_param(P):
            r = buffer_ops.create_buffer_resource(P, max_size=True)
            lo = buffer_ops.buffer_load(r, col, vec_width=4, dtype=T.f32)
            hi = buffer_ops.buffer_load(r, col + _i32(4), vec_width=4, dtype=T.f32)
            return vector.shuffle(_u(lo), _u(hi), list(range(8)))

        ws = [load_param(W) for W in WS]

        def row_sum(s):
            return _xor_sum(s, lane, LPR, DPP)

        def load_slot(r):
            # raw loads only (dy once, then each problem's x, mean, rstd)
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

        def compute(ld, p, acc, wv):
            # ln_bwd.py build_ln_bwd compute_slot, dx-sum variant (tri_order
            # False, no swish, no residual), op for op, on problem p's own
            # accumulators
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
            buffer_ops.buffer_store(arith.truncf(T.vec(8, T.bf16), _u(dx)), rDX[p], ld["off"],
                                    cache_modifier=SCACHE)
            return [acc_dw, acc_db, vadd(acc[2], dx)]

        zero = arith.constant_vector(0.0, _v8f())
        nflat = NACC * NPROB    # per problem [dW, dB, dXS]: nine separate accumulators
        loop = scf.ForOp(_idx(_i32(0)), _idx(ITERS), _u(arith.constant(1, index=True)),
                         [_u(zero)] * nflat)
        with ir.InsertionPoint(loop.body):
            flat = list(loop.inner_iter_args[:nflat])
            acc = [flat[NACC * p:NACC * p + NACC] for p in range(NPROB)]
            it = arith.ArithValue(arith.index_cast(T.i32, loop.induction_variable))
            base = (it * G + blk) * _i32(RPS) + wave * _i32(RPW) + sub
            slots = [load_slot(base + _i32(u * RPI)) for u in range(UNROLL)]
            # slot-major, then problem: each problem's own order over the slots
            # u is ln_bwd's
            for ld in slots:
                for p in range(NPROB):
                    acc[p] = compute(ld, p, acc[p], ws[p])
            scf.YieldOp([_u(v) for p in range(NPROB) for v in acc[p]])
        res = list(loop.results[:nflat])
        res_acc = [res[NACC * p:NACC * p + NACC] for p in range(NPROB)]

        red = SmemPtr(allocator.get_base(), off_red, T.f32, shape=(NACC * SLOTS * D,)).get()
        slot = wave * _i32(RPW) + sub
        rP = buffer_ops.create_buffer_resource(PART, max_size=True)
        c0 = tid * _i32(VEC)
        for p in range(NPROB):
            if p:
                gpu.barrier()       # the previous problem's LDS reads are done
            for o in range(NACC):
                vector.store(_u(res_acc[p][o]), red,
                             [_idx(_i32(o * SLOTS * D) + slot * _i32(D) + col)], alignment=16)
            gpu.barrier()
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
             PART: fx.Tensor, M_ARG: fx.Int32, ITERS_ARG: fx.Int32, G_ARG: fx.Int32):
        emit([DX0, DX1, DX2], DY, [X0, X1, X2], [W0, W1, W2], [MEAN0, MEAN1, MEAN2],
             [RSTD0, RSTD1, RSTD2], PART, M_ARG, ITERS_ARG, G_ARG)

    @flyc.jit
    def launch(DX0: fx.Tensor, DX1: fx.Tensor, DX2: fx.Tensor, DY: fx.Tensor,
               X0: fx.Tensor, X1: fx.Tensor, X2: fx.Tensor,
               W0: fx.Tensor, W1: fx.Tensor, W2: fx.Tensor,
               MEAN0: fx.Tensor, MEAN1: fx.Tensor, MEAN2: fx.Tensor,
               RSTD0: fx.Tensor, RSTD1: fx.Tensor, RSTD2: fx.Tensor,
               PART: fx.Tensor, M: fx.Int32, ITERS: fx.Int32, G: fx.Int32,
               stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(DX0, DX1, DX2, DY, X0, X1, X2, W0, W1, W2, MEAN0, MEAN1, MEAN2,
             RSTD0, RSTD1, RSTD2, PART, M, ITERS, G).launch(
            grid=(G, 1, 1), block=(THREADS, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.rows_per_iter = RPS
    return launch


@functools.lru_cache(maxsize=None)
def build_finalize_grouped(D: int = 512):
    """PART[q, 0:G, D] fp32 -> OUT_q[D] fp32 for q = 3*p + o (o: dW, dB, dx sum),
    q in 0..8, one launch. Per block: ln_bwd.py build_finalize's arithmetic
    exactly (fp32 weights), so each OUT_q equals that finalize's output."""
    NQ = 3 * NPROB
    NCG = D // 8
    WAVES = FIN_THREADS // 64
    name = f"ln_bwd_flydsl_fing{NQ}_d{D}_wf32"
    allocator = SmemAllocator(None, arch=get_rocm_arch(), global_sym_name=f"{name}_smem")
    off_red = allocator._align(allocator.ptr, 16)
    allocator.ptr = off_red + WAVES * 8 * 4

    def emit(PART, OUTS, G_ARG):
        tid = arith.ArithValue(_as_i32(fx.thread_idx.x))
        bid = arith.ArithValue(_as_i32(fx.block_idx.x))
        G = arith.ArithValue(_u(G_ARG))
        lane = tid % _i32(64)
        wave = tid // _i32(64)
        cg = bid % _i32(NCG)
        q = bid // _i32(NCG)
        col = cg * _i32(8)
        q64 = arith.ArithValue(arith.extsi(T.i64, _u(q)))
        g64 = arith.ArithValue(arith.extsi(T.i64, _u(G)))
        rP = _shifted_rsrc(PART, q64 * g64 * arith.constant(D, type=T.i64), 4, G * _i32(D * 4))
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
        for qi, OUT in enumerate(OUTS):
            is_q = arith.andi(first8, arith.cmpi(arith.CmpIPredicate.eq, _u(q), _u(_i32(qi))))
            rO = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=D * 4)
            buffer_ops.buffer_store(_u(tot), rO, col + j, mask=is_q)

    @flyc.kernel(name=name)
    def kern(PART: fx.Tensor, O0: fx.Tensor, O1: fx.Tensor, O2: fx.Tensor, O3: fx.Tensor,
             O4: fx.Tensor, O5: fx.Tensor, O6: fx.Tensor, O7: fx.Tensor, O8: fx.Tensor,
             G_ARG: fx.Int32):
        emit(PART, [O0, O1, O2, O3, O4, O5, O6, O7, O8], G_ARG)

    @flyc.jit
    def launch(PART: fx.Tensor, O0: fx.Tensor, O1: fx.Tensor, O2: fx.Tensor, O3: fx.Tensor,
               O4: fx.Tensor, O5: fx.Tensor, O6: fx.Tensor, O7: fx.Tensor, O8: fx.Tensor,
               G: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        kern(PART, O0, O1, O2, O3, O4, O5, O6, O7, O8, G).launch(
            grid=(NCG * NQ, 1, 1), block=(FIN_THREADS, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch


def grouped_reason(dy: torch.Tensor, xs: Sequence[torch.Tensor],
                   weights: Sequence[torch.Tensor], means: Sequence[torch.Tensor],
                   rstds: Sequence[torch.Tensor]) -> Optional[str]:
    """None when ln_bwd_grouped3 covers these inputs, else the (stable, N-free)
    reason. Covered: exactly NPROB problems; dy and every x bf16, contiguous,
    [N, 512], one device; fp32 contiguous (512,) weights and N-element fp32
    mean/rstd; N under ln_bwd.py's 4 GiB and padded-row (offsets_fit) bounds
    (each of the seven [N, 512] streams has its own buffer resource)."""
    if not (len(xs) == len(weights) == len(means) == len(rstds) == NPROB):
        return f"{len(xs)} problems (the grouped kernel takes {NPROB})"
    if dy.dim() != 2 or dy.shape[1] != D_GROUPED or dy.dtype != torch.bfloat16 \
            or not dy.is_contiguous() or not dy.is_cuda:
        return "dy is not a contiguous CUDA bf16 [N, 512]"
    M = dy.shape[0]
    for p, (x, w, mn, rs) in enumerate(zip(xs, weights, means, rstds)):
        if x.shape != dy.shape or x.dtype != torch.bfloat16 or not x.is_contiguous() \
                or x.device != dy.device:
            return f"x_{p} is not a contiguous bf16 tensor shaped and placed as dy"
        if w.dtype != torch.float32 or tuple(w.shape) != (D_GROUPED,) or not w.is_contiguous() \
                or w.device != dy.device:
            return f"weight_{p} is not a contiguous fp32 (512,) tensor on dy's device"
        for s in (mn, rs):
            if s.dtype != torch.float32 or s.numel() != M or not s.is_contiguous() \
                    or s.device != dy.device:
                return f"mean/rstd_{p} is not a contiguous fp32 N-vector on dy's device"
    if M == 0:
        return None
    if M * D_GROUPED * 2 >= 2 ** 32:
        return "tensor >= 4 GiB"
    if not _v2.offsets_fit(M, D_GROUPED, dy.device, False, False, True, GROUPED_CFG):
        return "padded rows past 32-bit offsets"
    return None


def ln_bwd_grouped3(dy, xs, weights, biases, means, rstds, *, cache=None, remap=None,
                    store_cache=None, fin="grouped", stream=None, out=None,
                    state: Optional[Dict[str, bool]] = None
                    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """[(dx_p, dw_p, db_p, dxsum_p) for p in 0..2], bitwise equal to the three
    ln_bwd(dy, xs[p], weights[p], biases[p], means[p], rstds[p], sum_dx=True)
    calls with the same cache/remap/store_cache options (default: ln_bwd.py's
    production options). biases are not read (no swish). The caller checks
    grouped_reason() first.
    fin: "grouped" (one finalize launch, production) or "v2" (three ln_bwd.py
    build_finalize launches on PART views; a test cross-check).
    `out` may hold preallocated "dx" (3 tensors; xs itself for the in-place
    donation: each lane reads its x before it writes dx to the same address),
    "fin" (9 fp32 [D] tensors: dw0, db0, s0, dw1, ...) and "part" (flat fp32,
    at least 9*G*D). `state`, if given, gets "main_launched" = True once the
    main kernel was enqueued (after that point an in-place call has
    overwritten xs)."""
    out = out or {}
    dy = dy.detach()
    xs = [x.detach() for x in xs]
    weights = [w.detach() for w in weights]
    means = [m.detach() for m in means]
    rstds = [r.detach() for r in rstds]
    M, D = dy.shape
    waves, unroll, tile_mult, *rest = GROUPED_CFG
    dpp = bool(rest[0]) if rest else False
    remap, cache, store_cache = grouped_kernel_options(cache, remap, store_cache)
    G = _v2.grid_for(M, dy.device, waves, tile_mult)
    dxs = list(out["dx"]) if "dx" in out else [torch.empty_like(x) for x in xs]
    fouts = list(out["fin"]) if "fin" in out else [
        torch.empty((D,), dtype=torch.float32, device=dy.device) for _ in range(3 * NPROB)]
    if M == 0:
        for t in fouts:
            t.zero_()
    else:
        if stream is None:
            stream = torch.cuda.current_stream(dy.device)
        part = (out["part"][: 3 * NPROB * G * D].view(3 * NPROB, G, D) if "part" in out
                else torch.empty((3 * NPROB, G, D), dtype=torch.float32, device=dy.device))
        main = build_ln_bwd_grouped(D, waves, unroll, cache, dpp, remap, store_cache)
        fin_kernel = build_finalize_grouped(D) if fin == "grouped" else None
        iters = (M + G * main.rows_per_iter - 1) // (G * main.rows_per_iter)
        # Every launch here goes through _v2._fast_call (ln_bwd.py module top):
        # under FlyDSL 0.2.4 the plain @flyc.jit call re-derives its cache key
        # and re-snapshots the launcher's globals on every call. Host time per
        # ln_bwd_grouped3 call (cluster A MI355X, median): 217 us through the plain
        # 0.2.4 jit, 45 us through _fast_call, 61 us under FlyDSL 0.1.1.
        # FlyDSL 0.1.1 and COMPILE_ONLY=1 keep the plain jit call. Same
        # kernels, same arguments (stream positional); outputs bit-identical.
        _v2._fast_call(main, dxs[0], dxs[1], dxs[2], dy, xs[0], xs[1], xs[2], weights[0],
                       weights[1], weights[2], means[0], means[1], means[2], rstds[0],
                       rstds[1], rstds[2], part, M, iters, G, fx.Stream(stream))
        if state is not None:
            state["main_launched"] = True
        if fin_kernel is not None:
            _v2._fast_call(fin_kernel, part, *fouts, G, fx.Stream(stream))
        else:
            f2 = _v2.build_finalize(D, 3, True)
            for p in range(NPROB):
                _v2._fast_call(f2, part[3 * p:3 * p + 3], fouts[3 * p], fouts[3 * p + 1],
                               fouts[3 * p + 2], G, fx.Stream(stream))
    return [(dxs[p], fouts[3 * p], fouts[3 * p + 1], fouts[3 * p + 2]) for p in range(NPROB)]


def grouped_kernel_name() -> str:
    """The production main kernel's name (for the arming line and traces)."""
    remap, cache, store_cache = grouped_kernel_options()
    waves, unroll, _tile_mult, *rest = GROUPED_CFG
    return build_ln_bwd_grouped(D_GROUPED, waves, unroll, cache, bool(rest[0]) if rest else False,
                                remap, store_cache).kernel_name


def prebuild_grouped(device=None, M: int = 1024) -> None:
    """Startup prebuild (dlrm_v4/train/blas_warmup.py): JIT-compile, load and
    launch once the grouped main kernel and the grouped finalize on synthetic
    zero/one tensors. M, ITERS and G are runtime arguments and the jit cache
    key holds no shape, so a small M builds the production specialization."""
    device = device or torch.device("cuda", torch.cuda.current_device())
    dy = torch.zeros((M, D_GROUPED), dtype=torch.bfloat16, device=device)
    xs = [torch.zeros_like(dy) for _ in range(NPROB)]
    ws = [torch.ones((D_GROUPED,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    bs = [torch.zeros_like(w) for w in ws]
    ms = [torch.zeros((M,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    rs = [torch.ones((M,), dtype=torch.float32, device=device) for _ in range(NPROB)]
    ln_bwd_grouped3(dy, xs, ws, bs, ms, rs)
