###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax-M3 indexer block scores without the S x S score matrix.

The indexer scores every query against every causal key (``q . k``, unscaled,
fp32) and max-pools each ``block_size``-key block. Done as a matmul followed
by a mask and an ``amax``, that is a ``[b, n_index, S, S]`` fp32 tensor -- and
``amax``'s backward keeps it alive until the backward pass (16 GB at 32k).

This kernel folds the max-pool into the matmul: it walks each query's causal
key blocks, scores a 16-key tile at a time on the MFMAs and keeps a running
per-block max, so only ``[b, n_index, S, n_blocks]`` is ever written. It also
writes which key won each block, so the backward can send the gradient of a
block score straight to that one key, which is all ``amax``'s gradient does.

Work split: the index heads share one key projection (MQA), so a work-group
takes 64 queries x all index heads, stages each 128-key block of K in LDS
once, and its 4 waves -- 16 queries each, the MFMA's 16 columns -- score it
against every head. S^T = K Q^T puts keys on the lane's 4 rows and the query
on its column, so the max over a block is 4 in-lane values, 8 tiles, then a
cross-lane reduction over the 4 lane groups, carrying the key index along.

Layouts: q ``[S, B, n_index, D]`` and k ``[S, B, D]`` bf16 (the indexer's own
``[sq, b, heads, dim]`` after norm and rope); outputs ``[B, n_index, S,
n_blocks]``: fp32 block max (-inf where no key is visible) and int32 argmax
key (-1 there).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd import (
    _tr16,
)

WAVES = 4
THREADS = 64 * WAVES
QPW = 16  # queries per wave == MFMA columns
QPG = WAVES * QPW  # queries per work-group
TKEYS = 16  # keys per MFMA tile
# Elements per thread in the row-sum backward kernels: a 2-wide bf16 access,
# so any even index dim works (a 1-wide vector does not lower).
_ROW_EPT = 2


def build_index_block_max(num_heads: int, index_dim: int, block_size: int):
    elem = fx.BFloat16
    H, DI = num_heads, index_dim
    assert DI % 32 == 0 and block_size % TKEYS == 0
    KSI = DI // 32  # MFMA K-steps
    KT = block_size // TKEYS  # key tiles per block
    STR = DI + 16  # LDS row stride (elements)
    CPL = block_size * DI // THREADS  # elements each thread stages per block
    assert CPL % 8 == 0 and DI % CPL == 0
    LPR = DI // CPL  # staging lanes per key row

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_index_block_max_smem")
    k_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(k_off + block_size * STR * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        BMAX: fx.Tensor,
        ARG: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
        NB: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_k = SmemPtr(allocator.get_base(), k_off, elem.ir_type, shape=(block_size * STR,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(64)
        lane = tid % fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        qg = fx.Index(gpu.block_idx.x)
        b = fx.Index(gpu.block_idx.y)

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(H * DI * 2))
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(DI * 2))
        )
        out_bytes = _raw(Bn * fx.Index(H) * Sn * NBn * fx.Index(4))
        bmax_rsrc = buffer_ops.create_buffer_resource(BMAX, max_size=False, num_records_bytes=out_bytes)
        arg_rsrc = buffer_ops.create_buffer_resource(ARG, max_size=False, num_records_bytes=out_bytes)

        c_neg_inf = fx.Float32(float("-inf"))
        pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

        # ---- this lane's query (MFMA column lo), all heads' q as B operands ----
        t = qg * fx.Index(QPG) + wave * fx.Index(QPW) + lo
        t_i32 = fx.Int32(t)
        q_packs = [
            [
                buffer_ops.buffer_load(
                    q_rsrc,
                    ((t * Bn + b) * fx.Index(H) + fx.Index(h)) * fx.Index(DI)
                    + fx.Index(ks * 32)
                    + grp * fx.Index(8),
                    vec_width=8,
                    dtype=elem,
                )
                for ks in range_constexpr(KSI)
            ]
            for h in range_constexpr(H)
        ]

        # every block that holds a key some query of this work-group can see
        last_q = fx.Int32(qg * fx.Index(QPG) + fx.Index(QPG - 1))
        last_q = fx.Int32(
            ArithValue(last_q < fx.Int32(S)).select(_raw(last_q), _raw(fx.Int32(S) - fx.Int32(1)))
        )
        n_kb = fx.Index(last_q // fx.Int32(block_size) + fx.Int32(1))

        st_row = tid // fx.Index(LPR)
        st_col = (tid % fx.Index(LPR)) * fx.Index(CPL)
        write_ok = ArithValue(
            arith.AndIOp(_raw(ArithValue(grp == fx.Index(0))), _raw(ArithValue(t_i32 < fx.Int32(S)))).result
        )

        def swap(x_i32, op):
            sw = op(pair_ty, x_i32, x_i32, False, True)
            return llvm.extractvalue(fx.Int32.ir_type, sw, [0]), llvm.extractvalue(fx.Int32.ir_type, sw, [1])

        def as_i32(x):
            return _raw(ArithValue(_raw(x)).bitcast(fx.Int32.ir_type))

        def as_f32(x):
            return fx.Float32(_raw(ArithValue(x).bitcast(fx.Float32.ir_type)))

        def pick(v1, a1, v2, a2):
            # larger score wins; on a tie the earlier key, like torch.argmax
            better = ArithValue(
                arith.OrIOp(
                    _raw(ArithValue(v2 > v1)),
                    _raw(
                        ArithValue(arith.AndIOp(_raw(ArithValue(v2 == v1)), _raw(ArithValue(a2 < a1))).result)
                    ),
                ).result
            )
            return fx.Float32(better.select(_raw(v2), _raw(v1))), fx.Int32(better.select(_raw(a2), _raw(a1)))

        def crossgrp_argmax(v, a):
            for op in (rocdl.permlane16_swap, rocdl.permlane32_swap):
                va, vb = swap(as_i32(v), op)
                aa, ab = swap(_raw(a), op)
                v, a = pick(as_f32(va), fx.Int32(aa), as_f32(vb), fx.Int32(ab))
            return v, a

        for kb, _ in range(fx.Index(0), n_kb, fx.Index(1), init=[]):
            key0 = kb * fx.Index(block_size)

            # ---- stage the block's keys in LDS, shared by the 4 waves ----
            gpu.barrier()  # WAR: every wave is done with the previous block
            src = ((key0 + st_row) * Bn + b) * fx.Index(DI) + st_col
            for c in range_constexpr(CPL // 8):
                val = buffer_ops.buffer_load(k_rsrc, src + fx.Index(c * 8), vec_width=8, dtype=elem)
                Vec(val).store(lds_k, [st_row * fx.Index(STR) + st_col + fx.Index(c * 8)])
            gpu.barrier()

            key0_i32 = fx.Int32(key0)
            for h in range_constexpr(H):
                mx = c_neg_inf
                am = fx.Int32(-1)
                for kt in range_constexpr(KT):
                    acc = Vec.filled(4, 0.0, fx.Float32)
                    for ks in range_constexpr(KSI):
                        a_k = Vec.load(
                            v8,
                            lds_k,
                            [
                                (fx.Index(kt * TKEYS) + lo) * fx.Index(STR)
                                + fx.Index(ks * 32)
                                + grp * fx.Index(8)
                            ],
                        )
                        acc = rocdl.mfma_f32_16x16x32_bf16(v4f, [_raw(a_k), q_packs[h][ks], acc])
                    # lane holds S[key = kt*16 + grp*4 + i, query = t]
                    for i in range_constexpr(4):
                        key = key0_i32 + fx.Int32(grp) * fx.Int32(4) + fx.Int32(kt * TKEYS + i)
                        ok = ArithValue(
                            arith.AndIOp(
                                _raw(ArithValue(key <= t_i32)), _raw(ArithValue(key < fx.Int32(S)))
                            ).result
                        )
                        v = fx.Float32(ok.select(_raw(fx.Float32(_raw(Vec(acc)[i]))), _raw(c_neg_inf)))
                        # strictly greater: within a lane the earlier key already holds a tie
                        gt = ArithValue(v > mx)
                        mx = fx.Float32(gt.select(_raw(v), _raw(mx)))
                        am = fx.Int32(gt.select(_raw(key), _raw(am)))
                mx, am = crossgrp_argmax(mx, am)
                out = ((b * fx.Index(H) + fx.Index(h)) * Sn + t) * NBn + kb
                buffer_ops.buffer_store(
                    mx, bmax_rsrc, out * fx.Index(4), mask=_raw(write_ok), offset_is_bytes=True
                )
                buffer_ops.buffer_store(
                    am, arg_rsrc, out * fx.Index(4), mask=_raw(write_ok), offset_is_bytes=True
                )
            yield []

    @flyc.jit
    def launch(Q, K, BMAX, ARG, S, B, NB, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, BMAX, ARG, S, B, NB).launch(
            grid=((fx.Index(S) + fx.Index(QPG - 1)) // fx.Index(QPG), fx.Index(B), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_index_dk(num_heads: int, index_dim: int, topk: int, block_size: int):
    """dK for the fused selection, pass 1: one work-group per chunk of the block plan.

    A block score's gradient ``g`` lands on the key that won the block, so over
    one KV block ``dK[block keys] = A^T Q`` with ``A[slot, key] = g`` where
    ``key`` won that slot's block, 0 elsewhere -- the attention's ``dS^T Q``
    with a one-hot ``dS``. The work therefore follows the attention's own
    :class:`BlockPlan`: a chunk is a run of the table entries that picked one
    (batch, index head, block), and no sort by winning key is needed.

    8 waves, each owning 16 of the block's keys, take 64 entries per step --
    all their loads in flight together -- as four 16-entry MFMA sub-steps: the
    entries' index-q rows are staged in LDS and read back transposed as the A
    operand of ``dK^T += Q^T A``, and each lane builds its column of ``A``
    from the entries' gradients and winning keys. ``g`` is fp32; it goes into
    the bf16 MFMA as a high and a low half, 16 mantissa bits in all, so its
    rounding stays far below that of the bf16 dK the pass returns. Each chunk
    writes an fp32 ``[block_size, D]`` partial;
    :func:`build_index_dk_reduce` sums them in order, so the result is
    deterministic.

    ``ENT`` holds flat ``[B, n_index, S, topk]`` entry ids; ``GS`` and ``KEYS``
    are indexed by them (``GS`` is 0 on slots that carry no gradient).
    """
    elem = fx.BFloat16
    H, DI = num_heads, index_dim
    KW = block_size // TKEYS  # waves, one per 16 keys
    NTH = 64 * KW
    TK = 16  # entries per MFMA: its K
    NSUB = 4  # MFMA sub-steps per step: every load of 64 entries is in flight at once
    TT = TK * NSUB
    DTI = DI // 16
    STR = DI + 16  # LDS row stride (elements)
    VEC = 8  # elements per staging load
    NP = TT * DI // (NTH * VEC)  # staging loads per thread per step
    assert DI % 16 == 0 and block_size % TKEYS == 0 and TT * DI % (NTH * VEC) == 0 and DI % VEC == 0

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_index_dk_smem")
    q_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(q_off + TT * STR * 2, 16)

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def k_fn(
        ENT: fx.Tensor,
        CHUNKS: fx.Tensor,
        GS: fx.Tensor,
        KEYS: fx.Tensor,
        Q: fx.Tensor,
        WS: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
        NB: fx.Int32,
        NCH: fx.Int32,
    ):
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(TT * STR,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(64)
        lane = tid % fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        chunk = fx.Index(gpu.block_idx.x)

        n_entries = Bn * fx.Index(H) * Sn * fx.Index(topk)
        entry_bytes = _raw(n_entries * fx.Index(4))
        ent_rsrc = buffer_ops.create_buffer_resource(ENT, max_size=False, num_records_bytes=entry_bytes)
        gs_rsrc = buffer_ops.create_buffer_resource(GS, max_size=False, num_records_bytes=entry_bytes)
        key_rsrc = buffer_ops.create_buffer_resource(KEYS, max_size=False, num_records_bytes=entry_bytes)
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(H * DI * 2))
        )
        ch_rsrc = buffer_ops.create_buffer_resource(
            CHUNKS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(3 * 4))
        )
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(block_size * DI * 4))
        )

        def load_i32(rsrc, idx):
            return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=fx.Int32))

        # ---- this chunk: (plan row, first entry, end entry); row -> (batch, index head, block) ----
        row = fx.Index(load_i32(ch_rsrc, chunk * fx.Index(3)))
        e_begin = load_i32(ch_rsrc, chunk * fx.Index(3) + fx.Index(1))
        e_end = load_i32(ch_rsrc, chunk * fx.Index(3) + fx.Index(2))
        bh = row // NBn
        blk = row % NBn
        b = bh // fx.Index(H)
        h = bh % fx.Index(H)
        my_key = blk * fx.Index(block_size) + wave * fx.Index(TKEYS) + lo  # this lane's column of A
        my_key_i32 = fx.Int32(my_key)
        c_zero = fx.Float32(0.0)

        st_flat = [(fx.Index(p * NTH) + tid) * fx.Index(VEC) for p in range_constexpr(NP)]
        st_rows = [f // fx.Index(DI) for f in st_flat]
        st_cols = [f % fx.Index(DI) for f in st_flat]
        # transposed read of a 16-row tile: lane gets Q^T[d = lo, entry = grp*4 .. grp*4+3]
        q_tr = fx.Int64(
            ((grp * fx.Index(4) + lo // fx.Index(4)) * fx.Index(STR) + (lo % fx.Index(4)) * fx.Index(4))
            * fx.Index(2)
            + fx.Index(q_off)
        )

        def entry_at(r):  # clamped into the chunk, so every load stays in bounds
            r = ArithValue(r < e_end).select(_raw(r), _raw(e_end - fx.Int32(1)))
            return fx.Index(load_i32(ent_rsrc, fx.Index(fx.Int32(r))))

        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = [c_zero_v4 for _ in range_constexpr(DTI)]
        results = init
        for r0, it in range(fx.Index(e_begin), fx.Index(e_end), fx.Index(TT), init=init):
            r0_i32 = fx.Int32(r0)
            # ---- every load of the step first: the 64 index-q rows, and this lane's
            # 16 (gradient, winning key) pairs -- entries sub*16 + grp*4 + i ----
            q_st = []
            for p in range_constexpr(NP):
                t_st = (entry_at(r0_i32 + fx.Int32(st_rows[p])) // fx.Index(topk)) % Sn
                q_st.append(
                    buffer_ops.buffer_load(
                        q_rsrc,
                        ((t_st * Bn + b) * fx.Index(H) + h) * fx.Index(DI) + st_cols[p],
                        vec_width=VEC,
                        dtype=elem,
                    )
                )
            meta = []
            for sub in range_constexpr(NSUB):
                for i in range_constexpr(4):
                    r = r0_i32 + fx.Int32(grp * fx.Index(4) + fx.Index(sub * TK + i))
                    e = entry_at(r)
                    meta.append(
                        (
                            r,
                            fx.Float32(buffer_ops.buffer_load(gs_rsrc, e, vec_width=1, dtype=fx.Float32)),
                            load_i32(key_rsrc, e),
                        )
                    )
            gpu.barrier()  # WAR: every wave is done with the previous step's tile
            for p in range_constexpr(NP):
                Vec(q_st[p]).store(lds_q, [st_rows[p] * fx.Index(STR) + st_cols[p]])
            gpu.barrier()
            acc = [it[dt] for dt in range_constexpr(DTI)]
            for sub in range_constexpr(NSUB):
                # this lane's column of A for the sub-step, as a high and a low bf16 half
                hi_v, lo_v = [], []
                for i in range_constexpr(4):
                    r, gv, kv = meta[sub * 4 + i]
                    hit = ArithValue(
                        arith.AndIOp(_raw(ArithValue(r < e_end)), _raw(ArithValue(kv == my_key_i32))).result
                    )
                    val = fx.Float32(hit.select(_raw(gv), _raw(c_zero)))
                    hb = fx.BFloat16(_raw(val))
                    hi_v.append(hb)
                    lo_v.append(
                        fx.BFloat16(_raw(val - fx.Float32(arith.ExtFOp(fx.Float32.ir_type, _raw(hb)).result)))
                    )
                a_hi = _raw(Vec.from_elements(hi_v, elem).bitcast(fx.Int16))
                a_lo = _raw(Vec.from_elements(lo_v, elem).bitcast(fx.Int16))
                for dt in range_constexpr(DTI):
                    q_t = _tr16(v4, q_tr + fx.Int64((sub * TK * STR + dt * 16) * 2))
                    acc[dt] = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [q_t, a_hi, acc[dt]])
                    acc[dt] = rocdl.mfma_f32_16x16x16bf16_1k(v4f, [q_t, a_lo, acc[dt]])
            results = yield acc

        # fp32 partial [chunk][key in block][d]: lane holds dK^T[d = dt*16 + grp*4 + i, key = wave*16 + lo]
        ws_row = (chunk * fx.Index(block_size) + wave * fx.Index(TKEYS) + lo) * fx.Index(DI)
        for dt in range_constexpr(DTI):
            off = ws_row + fx.Index(dt * 16) + grp * fx.Index(4)
            buffer_ops.buffer_store(results[dt], ws_rsrc, off * fx.Index(4), offset_is_bytes=True)

    @flyc.jit
    def launch(ENT, CHUNKS, GS, KEYS, Q, WS, S, B, NB, NCH, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(ENT, CHUNKS, GS, KEYS, Q, WS, S, B, NB, NCH).launch(
            grid=(fx.Index(NCH), 1, 1), block=(NTH, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_index_dk_reduce(num_heads: int, index_dim: int, block_size: int):
    """dK for the fused selection, pass 2: one work-group per (batch, KV block,
    1024-element slice of the block's dK).

    The index keys are shared by every index head (MQA), so a block's dK sums
    the chunk partials of all its heads' plan rows, head by head and chunk by
    chunk -- a fixed order. Every plan row has at least one chunk, so a block
    nobody picked still gets its zeros written. The slices keep the GPU busy
    at short sequences, where there are only a few dozen blocks.
    """
    H, DI = num_heads, index_dim
    NTH = 256
    NE = block_size * DI  # elements per partial
    PER = 1  # v4 groups per thread
    SLICE = NTH * 4 * PER
    SPLIT = NE // SLICE
    assert NE % SLICE == 0

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def k_fn(
        WS: fx.Tensor, CHPTR: fx.Tensor, DK: fx.Tensor, S: fx.Int32, B: fx.Int32, NB: fx.Int32, NCH: fx.Int32
    ):
        tid = fx.Index(gpu.thread_idx.x)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        wg = fx.Index(gpu.block_idx.x)
        bb = wg // fx.Index(SPLIT)  # b * NB + block
        part = wg % fx.Index(SPLIT)
        b = bb // NBn
        blk = bb % NBn
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(NE * 4))
        )
        cp_rsrc = buffer_ops.create_buffer_resource(
            CHPTR,
            max_size=False,
            num_records_bytes=_raw((Bn * fx.Index(H) * NBn + fx.Index(1)) * fx.Index(4)),
        )
        dk_rsrc = buffer_ops.create_buffer_resource(
            DK, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(DI * 2))
        )
        elems = [
            part * fx.Index(SLICE) + tid * fx.Index(4) + fx.Index(j * NTH * 4) for j in range_constexpr(PER)
        ]
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        acc = [c_zero_v4 for _ in range_constexpr(PER)]
        for h in range_constexpr(H):
            row = (b * fx.Index(H) + fx.Index(h)) * NBn + blk
            c_begin = fx.Index(fx.Int32(buffer_ops.buffer_load(cp_rsrc, row, vec_width=1, dtype=fx.Int32)))
            c_end = fx.Index(
                fx.Int32(buffer_ops.buffer_load(cp_rsrc, row + fx.Index(1), vec_width=1, dtype=fx.Int32))
            )
            # plus a counter: a loop carrying one value hands it back unwrapped
            init = acc + [fx.Int32(0)]
            results = init
            for c, it in range(c_begin, c_end, fx.Index(2), init=init):
                # two chunks per step, so their loads overlap; still summed in chunk order
                has2 = ArithValue(c + fx.Index(1) < c_end)
                new = []
                for j in range_constexpr(PER):
                    w0 = buffer_ops.buffer_load(
                        ws_rsrc, c * fx.Index(NE) + elems[j], vec_width=4, dtype=fx.Float32
                    )
                    w1 = buffer_ops.buffer_load(
                        ws_rsrc, (c + fx.Index(1)) * fx.Index(NE) + elems[j], vec_width=4, dtype=fx.Float32
                    )
                    s = Vec(it[j]) + Vec(w0)
                    new.append(_raw(has2.select(_raw(s + Vec(w1)), _raw(s))))
                results = yield new + [fx.Int32(it[PER]) + fx.Int32(1)]
            acc = [results[j] for j in range_constexpr(PER)]

        for j in range_constexpr(PER):
            key = blk * fx.Index(block_size) + elems[j] // fx.Index(DI)
            off = (key * Bn + b) * fx.Index(DI) + elems[j] % fx.Index(DI)
            ov = Vec(acc[j])
            pk0 = rocdl.cvt_pk_bf16_f32(_raw(ov[0]), _raw(ov[1]))
            pk1 = rocdl.cvt_pk_bf16_f32(_raw(ov[2]), _raw(ov[3]))
            # keys past S fall off the end of the buffer
            buffer_ops.buffer_store(
                _raw(Vec.from_elements([fx.Int32(_raw(pk0)), fx.Int32(_raw(pk1))], fx.Int32)),
                dk_rsrc,
                off * fx.Index(2),
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(WS, CHPTR, DK, S, B, NB, NCH, stream):
        k_fn(WS, CHPTR, DK, S, B, NB, NCH).launch(
            grid=(fx.Index(B) * fx.Index(NB) * fx.Index(SPLIT), 1, 1), block=(NTH, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_index_dq(num_heads: int, index_dim: int, topk: int):
    """dQ for the fused selection: one wave per (batch, head, token) sums its slots.

    ``dq[t, h] = sum_s g[h, t, s] * k[key[h, t, s]]`` over the token's selected
    slots, straight from the key rows -- no ``[.., topk, D]`` gather in memory.
    """
    elem = fx.BFloat16
    H, DI = num_heads, index_dim
    EPT = _ROW_EPT
    THR = DI // EPT
    assert DI % EPT == 0

    @flyc.kernel(known_block_size=[THR, 1, 1])
    def k_fn(G: fx.Tensor, KEYS: fx.Tensor, K: fx.Tensor, DQ: fx.Tensor, S: fx.Int32, B: fx.Int32):
        tid = fx.Index(gpu.thread_idx.x)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        row = fx.Index(gpu.block_idx.x)  # (b * H + h) * S + t
        t = row % Sn
        h = (row // Sn) % fx.Index(H)
        b = row // (Sn * fx.Index(H))
        n_entries = Bn * fx.Index(H) * Sn * fx.Index(topk)
        g_rsrc = buffer_ops.create_buffer_resource(
            G, max_size=False, num_records_bytes=_raw(n_entries * fx.Index(4))
        )
        key_rsrc = buffer_ops.create_buffer_resource(
            KEYS, max_size=False, num_records_bytes=_raw(n_entries * fx.Index(4))
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(DI * 2))
        )
        dq_rsrc = buffer_ops.create_buffer_resource(
            DQ, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(H * DI * 2))
        )
        col = tid * fx.Index(EPT)
        vf = Vec.make_type(EPT, fx.Float32)
        acc = [fx.Float32(0.0) for _ in range_constexpr(EPT)]
        for s in range_constexpr(topk):
            e = row * fx.Index(topk) + fx.Index(s)
            g = fx.Float32(buffer_ops.buffer_load(g_rsrc, e, vec_width=1, dtype=fx.Float32))
            key = fx.Index(fx.Int32(buffer_ops.buffer_load(key_rsrc, e, vec_width=1, dtype=fx.Int32)))
            kv = buffer_ops.buffer_load(
                k_rsrc, (key * Bn + b) * fx.Index(DI) + col, vec_width=EPT, dtype=elem
            )
            kf = Vec(arith.ExtFOp(vf, _raw(kv)).result)
            acc = [acc[i] + g * fx.Float32(_raw(kf[i])) for i in range_constexpr(EPT)]
        packed = [fx.BFloat16(_raw(acc[i])) for i in range_constexpr(EPT)]
        buffer_ops.buffer_store(
            _raw(Vec.from_elements(packed, elem)),
            dq_rsrc,
            (((t * Bn + b) * fx.Index(H) + h) * fx.Index(DI) + col) * fx.Index(2),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(G, KEYS, K, DQ, S, B, stream):
        k_fn(G, KEYS, K, DQ, S, B).launch(
            grid=(fx.Index(B) * fx.Index(H) * fx.Index(S), 1, 1), block=(THR, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CACHE: dict = {}
_DK_CACHE: dict = {}
_DQ_CACHE: dict = {}


def index_dq(g, keys, k, num_heads):
    """dQ of the index queries; see :func:`build_index_dq`.

    Args:
        g: ``[B, n_index, S, topk]`` fp32; keys: same shape, int32 winning keys (>= 0).
        k: ``[S, B, D]`` bf16 index keys.

    Returns:
        ``dq`` ``[S, B, n_index, D]`` bf16.
    """
    S, B, DI = k.shape
    topk = g.shape[-1]
    dq = torch.empty((S, B, num_heads, DI), dtype=torch.bfloat16, device=k.device)
    args = (
        g.contiguous(),
        keys.contiguous(),
        k.contiguous(),
        dq,
        int(S),
        int(B),
        torch.cuda.current_stream(),
    )
    key = (num_heads, DI, topk)
    fn = _DQ_CACHE.get(key)
    if fn is None:
        fn = build_index_dq(num_heads, DI, topk).compile(*args)
        _DQ_CACHE[key] = fn
    fn(*args)
    return dq


def index_dk(plan, g, keys, q, block_size=128):
    """Deterministic dK of the index keys; see :func:`build_index_dk`.

    Args:
        plan: the selection's :class:`BlockPlan` (``msa_token_bwd``'s, shared).
        g: ``[B, n_index, S, topk]`` fp32 gradients of the selected block
            scores, 0 on slots that carry none.
        keys: same shape, int32 key that won each slot's block.
        q: ``[S, B, n_index, D]`` bf16 index queries.

    Returns:
        ``dk`` ``[S, B, D]`` bf16.
    """
    S, B, H, DI = q.shape
    topk = g.shape[-1]
    dk = torch.empty((S, B, DI), dtype=torch.bfloat16, device=q.device)
    ws = torch.empty((plan.n_chunks, block_size, DI), dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream()
    nb, nch = int(plan.n_blocks), int(plan.n_chunks)
    dk_args = (
        plan.entries,
        plan.chunks,
        g.contiguous(),
        keys.contiguous(),
        q.contiguous(),
        ws,
        int(S),
        int(B),
        nb,
        nch,
        stream,
    )
    red_args = (ws, plan.chunk_ptr, dk, int(S), int(B), nb, nch, stream)
    key = (H, DI, topk, block_size)
    fns = _DK_CACHE.get(key)
    if fns is None:
        fns = (
            build_index_dk(H, DI, topk, block_size).compile(*dk_args),
            build_index_dk_reduce(H, DI, block_size).compile(*red_args),
        )
        _DK_CACHE[key] = fns
    fns[0](*dk_args)
    fns[1](*red_args)
    return dk


def index_block_max(q: torch.Tensor, k: torch.Tensor, block_size: int = 128):
    """Per-block max of ``q . k`` over each query's causal keys.

    Args:
        q: ``[S, B, n_index, D]`` bf16; k: ``[S, B, D]`` (or ``[S, B, 1, D]``) bf16.

    Returns:
        ``block_max`` fp32 and ``argmax`` int32, both ``[B, n_index, S, n_blocks]``;
        -inf / -1 on blocks with no visible key.
    """
    S, B, H, DI = q.shape
    k = k.reshape(S, B, DI)
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    q, k = q.contiguous(), k.contiguous()
    n_blocks = -(-S // block_size)
    block_max = torch.full((B, H, S, n_blocks), float("-inf"), dtype=torch.float32, device=q.device)
    argmax = torch.full((B, H, S, n_blocks), -1, dtype=torch.int32, device=q.device)
    args = (q, k, block_max, argmax, int(S), int(B), int(n_blocks), torch.cuda.current_stream())
    key = (H, DI, block_size)
    fn = _CACHE.get(key)
    if fn is None:
        fn = build_index_block_max(H, DI, block_size).compile(*args)
        _CACHE[key] = fn
    fn(*args)
    return block_max, argmax
