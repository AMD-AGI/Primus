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

WAVES = 4
THREADS = 64 * WAVES
QPW = 16  # queries per wave == MFMA columns
QPG = WAVES * QPW  # queries per work-group
TKEYS = 16  # keys per MFMA tile
# Elements per thread in the row-sum backward kernels: a 2-wide bf16 access,
# so any even index dim works (a 1-wide vector does not lower).
_ROW_EPT = 2
_DK_CHUNK = 256  # key-sorted entries per work-group in the dK pass
_DK_UNROLL = 8  # entries whose loads are in flight together


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


def _and(a, b):
    return ArithValue(arith.AndIOp(_raw(a), _raw(b)).result)


def _flag(cond):
    """i1 -> Int32 0/1, so flags can be negated (``== 0``) and carried by loops."""
    return fx.Int32(ArithValue(cond).select(_raw(fx.Int32(1)), _raw(fx.Int32(0))))


def build_index_dk_chunks(num_heads: int, index_dim: int, topk: int, chunk: int = _DK_CHUNK):
    """dK, pass 1: each work-group sums one fixed-size chunk of the key-sorted entries.

    Entry ``e`` is a flat index into ``[B, n_index, S, topk]`` (a selected slot).
    ``ORDER`` lists the live entries sorted by the key that won their block,
    ``SKEY`` holds that key (``b * S + key``) per sorted position, and
    ``BOUNDS`` is the CSR over keys; dead entries carry the sentinel key
    ``B * S`` and sort past ``BOUNDS[B * S]``, so no work-group reads them.
    ``dk[j] = sum g[e] * q[token(e), head(e)]`` over key ``j``'s run.

    Splitting the sorted array, not the keys, keeps the work even when a few
    keys win most blocks -- an attention-sink-like direction shared by q and k
    hands a handful of keys thousands of entries each. A run that lies inside
    the chunk is written straight to ``DK``; one that crosses a chunk edge
    leaves an fp32 partial in ``WS[chunk, slot]`` -- slot 0 for the chunk's
    first run, 1 for its last -- which pass 2 sums in chunk order.
    """
    elem = fx.BFloat16
    H, DI = num_heads, index_dim
    EPT = _ROW_EPT
    THR = DI // EPT  # one thread per EPT-element slice of a row
    U = _DK_UNROLL
    assert DI % EPT == 0 and chunk % U == 0

    @flyc.kernel(known_block_size=[THR, 1, 1])
    def k_fn(
        ORDER: fx.Tensor,
        SKEY: fx.Tensor,
        BOUNDS: fx.Tensor,
        G: fx.Tensor,
        Q: fx.Tensor,
        DK: fx.Tensor,
        WS: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
    ):
        tid = fx.Index(gpu.thread_idx.x)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        c = fx.Index(gpu.block_idx.x)
        n_entries = Bn * fx.Index(H) * Sn * fx.Index(topk)
        n_keys = Bn * Sn
        n_chunks = (n_entries + fx.Index(chunk - 1)) // fx.Index(chunk)
        entry_bytes = _raw(n_entries * fx.Index(4))
        ord_rsrc = buffer_ops.create_buffer_resource(ORDER, max_size=False, num_records_bytes=entry_bytes)
        skey_rsrc = buffer_ops.create_buffer_resource(SKEY, max_size=False, num_records_bytes=entry_bytes)
        g_rsrc = buffer_ops.create_buffer_resource(G, max_size=False, num_records_bytes=entry_bytes)
        bnd_rsrc = buffer_ops.create_buffer_resource(
            BOUNDS, max_size=False, num_records_bytes=_raw((n_keys + fx.Index(1)) * fx.Index(4))
        )
        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(H * DI * 2))
        )
        dk_rsrc = buffer_ops.create_buffer_resource(
            DK, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(DI * 2))
        )
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(n_chunks * fx.Index(2 * DI * 4))
        )
        col = tid * fx.Index(EPT)
        vf = Vec.make_type(EPT, fx.Float32)
        zero = fx.Int32(0)

        def load_i32(rsrc, idx):
            return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=fx.Int32))

        def at(x):  # Int32 position -> Index, clamped at 0
            return fx.Index(fx.Int32(ArithValue(x > zero).select(_raw(x), _raw(zero))))

        def store_dk(key_flat, vals, mask):
            kf = fx.Index(key_flat)
            off = ((kf % Sn) * Bn + kf // Sn) * fx.Index(DI) + col
            packed = [fx.BFloat16(_raw(v)) for v in vals]
            buffer_ops.buffer_store(
                _raw(Vec.from_elements(packed, elem)),
                dk_rsrc,
                off * fx.Index(2),
                mask=_raw(mask),
                offset_is_bytes=True,
            )

        def store_ws(slot, vals, mask):
            off = (c * fx.Index(2) + slot) * fx.Index(DI) + col
            buffer_ops.buffer_store(
                _raw(Vec.from_elements(vals, fx.Float32)),
                ws_rsrc,
                off * fx.Index(4),
                mask=_raw(mask),
                offset_is_bytes=True,
            )

        live = load_i32(bnd_rsrc, n_keys)
        r0 = fx.Int32(c * fx.Index(chunk))
        r1 = r0 + fx.Int32(chunk)
        r1 = fx.Int32(ArithValue(r1 < live).select(_raw(r1), _raw(live)))
        has_any = ArithValue(r0 < r1)
        first_key = load_i32(skey_rsrc, fx.Index(r0))
        # does the chunk's first run begin in an earlier chunk / its last run go on past it?
        cross_l = _flag(
            _and(ArithValue(r0 > zero), ArithValue(load_i32(skey_rsrc, at(r0 - fx.Int32(1))) == first_key))
        )
        last_key = load_i32(skey_rsrc, at(r1 - fx.Int32(1)))
        cross_r = _flag(_and(ArithValue(r1 < live), ArithValue(load_i32(skey_rsrc, at(r1)) == last_key)))

        # carry: the running sum, the run's key, and whether it is the chunk's first run
        init = [fx.Float32(0.0) for _ in range_constexpr(EPT)] + [first_key, fx.Int32(1)]
        results = init
        for r, it in range(fx.Index(r0), fx.Index(r1), fx.Index(U), init=init):
            acc = [fx.Float32(it[i]) for i in range_constexpr(EPT)]
            cur = fx.Int32(it[EPT])
            first = fx.Int32(it[EPT + 1])
            # issue all U entries' loads before using any of them
            idx = []
            for u in range_constexpr(U):
                ru = r + fx.Index(u)
                idx.append(
                    (ArithValue(fx.Int32(ru) < r1), load_i32(skey_rsrc, ru), fx.Index(load_i32(ord_rsrc, ru)))
                )
            rows = []
            for ok, ku, e in idx:
                tok = (e // fx.Index(topk)) % Sn
                h = (e // fx.Index(topk) // Sn) % fx.Index(H)
                b = e // (fx.Index(topk * H) * Sn)
                g = fx.Float32(buffer_ops.buffer_load(g_rsrc, e, vec_width=1, dtype=fx.Float32))
                qv = buffer_ops.buffer_load(
                    q_rsrc, ((tok * Bn + b) * fx.Index(H) + h) * fx.Index(DI) + col, vec_width=EPT, dtype=elem
                )
                rows.append((ok, ku, g, qv))
            for ok, ku, g, qv in rows:
                ku = fx.Int32(ok.select(_raw(ku), _raw(cur)))
                g = fx.Float32(ok.select(_raw(g), _raw(fx.Float32(0.0))))
                qf = Vec(arith.ExtFOp(vf, _raw(qv)).result)
                changed = ArithValue(ku != cur)
                # the run that just ended is a partial only if it came in from the previous chunk
                partial = first * cross_l
                store_ws(fx.Index(0), acc, _and(changed, ArithValue(partial == fx.Int32(1))))
                store_dk(cur, acc, _and(changed, ArithValue(partial == zero)))
                prod = [g * fx.Float32(_raw(qf[i])) for i in range_constexpr(EPT)]
                acc = [
                    fx.Float32(changed.select(_raw(prod[i]), _raw(acc[i] + prod[i])))
                    for i in range_constexpr(EPT)
                ]
                first = fx.Int32(changed.select(_raw(zero), _raw(first)))
                cur = ku
            results = yield acc + [cur, first]

        acc = [fx.Float32(results[i]) for i in range_constexpr(EPT)]
        cur = fx.Int32(results[EPT])
        first = fx.Int32(results[EPT + 1])
        either = _flag(ArithValue(cross_l + cross_r > zero))
        crossing = fx.Int32(ArithValue(first == fx.Int32(1)).select(_raw(either), _raw(cross_r)))
        store_ws(fx.Index(fx.Int32(1) - first), acc, _and(has_any, ArithValue(crossing == fx.Int32(1))))
        store_dk(cur, acc, _and(has_any, ArithValue(crossing == zero)))

    @flyc.jit
    def launch(ORDER, SKEY, BOUNDS, G, Q, DK, WS, S, B, stream):
        n_entries = fx.Index(B) * fx.Index(H) * fx.Index(S) * fx.Index(topk)
        k_fn(ORDER, SKEY, BOUNDS, G, Q, DK, WS, S, B).launch(
            grid=((n_entries + fx.Index(chunk - 1)) // fx.Index(chunk), 1, 1),
            block=(THR, 1, 1),
            stream=stream,
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_index_dk_fix(num_heads: int, index_dim: int, topk: int, chunk: int = _DK_CHUNK):
    """dK, pass 2: one work-group per key finishes a run that crossed chunks.

    It sums the run's partials in chunk order -- its first chunk's (slot 0 if
    the run starts that chunk, else slot 1), then slot 0 of every later chunk
    it reaches -- and writes zeros for a key no entry picked. Runs inside one
    chunk were written by pass 1 and are left alone.
    """
    elem = fx.BFloat16
    H, DI = num_heads, index_dim
    EPT = _ROW_EPT
    THR = DI // EPT
    U = 4
    assert DI % EPT == 0

    @flyc.kernel(known_block_size=[THR, 1, 1])
    def k_fn(BOUNDS: fx.Tensor, WS: fx.Tensor, DK: fx.Tensor, S: fx.Int32, B: fx.Int32):
        tid = fx.Index(gpu.thread_idx.x)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        j = fx.Index(gpu.block_idx.x)  # b * S + key
        n_entries = Bn * fx.Index(H) * Sn * fx.Index(topk)
        n_chunks = (n_entries + fx.Index(chunk - 1)) // fx.Index(chunk)
        bnd_rsrc = buffer_ops.create_buffer_resource(
            BOUNDS, max_size=False, num_records_bytes=_raw((Bn * Sn + fx.Index(1)) * fx.Index(4))
        )
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(n_chunks * fx.Index(2 * DI * 4))
        )
        dk_rsrc = buffer_ops.create_buffer_resource(
            DK, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(DI * 2))
        )
        col = tid * fx.Index(EPT)
        zero = fx.Int32(0)
        c_zero = fx.Float32(0.0)

        def load_ws(cc, slot):
            v = buffer_ops.buffer_load(
                ws_rsrc, (cc * fx.Index(2) + slot) * fx.Index(DI) + col, vec_width=EPT, dtype=fx.Float32
            )
            return [fx.Float32(_raw(Vec(v)[i])) for i in range_constexpr(EPT)]

        lo = fx.Int32(buffer_ops.buffer_load(bnd_rsrc, j, vec_width=1, dtype=fx.Int32))
        hi = fx.Int32(buffer_ops.buffer_load(bnd_rsrc, j + fx.Index(1), vec_width=1, dtype=fx.Int32))
        empty = ArithValue(lo == hi)
        last = fx.Int32(empty.select(_raw(lo), _raw(hi - fx.Int32(1))))
        c0 = lo // fx.Int32(chunk)
        c1 = last // fx.Int32(chunk)
        multi = _flag(ArithValue(c1 > c0))
        slot0 = fx.Index(_flag(ArithValue(lo != c0 * fx.Int32(chunk))))

        init = load_ws(fx.Index(c0), slot0) + [zero]
        results = init
        for cc, it in range(fx.Index(c0 + fx.Int32(1)), fx.Index(c1 + fx.Int32(1)), fx.Index(U), init=init):
            vals = [
                (ArithValue(fx.Int32(cc + fx.Index(u)) <= c1), load_ws(cc + fx.Index(u), fx.Index(0)))
                for u in range_constexpr(U)
            ]
            acc = [fx.Float32(it[i]) for i in range_constexpr(EPT)]
            for ok, v in vals:
                acc = [acc[i] + fx.Float32(ok.select(_raw(v[i]), _raw(c_zero))) for i in range_constexpr(EPT)]
            results = yield acc + [fx.Int32(it[EPT]) + fx.Int32(1)]

        packed = [
            fx.BFloat16(_raw(fx.Float32(empty.select(_raw(c_zero), _raw(fx.Float32(results[i]))))))
            for i in range_constexpr(EPT)
        ]
        write = ArithValue(_flag(empty) + multi > zero)
        buffer_ops.buffer_store(
            _raw(Vec.from_elements(packed, elem)),
            dk_rsrc,
            (((j % Sn) * Bn + j // Sn) * fx.Index(DI) + col) * fx.Index(2),
            mask=_raw(write),
            offset_is_bytes=True,
        )

    @flyc.jit
    def launch(BOUNDS, WS, DK, S, B, stream):
        k_fn(BOUNDS, WS, DK, S, B).launch(
            grid=(fx.Index(B) * fx.Index(S), 1, 1), block=(THR, 1, 1), stream=stream
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


def index_dk(order, sorted_keys, bounds, g, q, topk):
    """Deterministic dK of the index keys; see :func:`build_index_dk_chunks`.

    Args:
        order: ``[B * n_index * S * topk]`` int32 entries, stably sorted by key.
        sorted_keys: the same length, int32 ``b * S + key`` per sorted position
            (``B * S`` for dead entries, which sort last).
        bounds: ``[B * S + 1]`` int32 CSR over keys.
        g: ``[B, n_index, S, topk]`` fp32 gradients of the selected block scores.
        q: ``[S, B, n_index, D]`` bf16 index queries.

    Returns:
        ``dk`` ``[S, B, D]`` bf16.
    """
    S, B, H, DI = q.shape
    dk = torch.empty((S, B, DI), dtype=torch.bfloat16, device=q.device)
    ws = torch.empty((-(-order.numel() // _DK_CHUNK), 2, DI), dtype=torch.float32, device=q.device)
    stream = torch.cuda.current_stream()
    chunk_args = (order, sorted_keys, bounds, g.contiguous(), q.contiguous(), dk, ws, int(S), int(B), stream)
    fix_args = (bounds, ws, dk, int(S), int(B), stream)
    key = (H, DI, topk)
    fns = _DK_CACHE.get(key)
    if fns is None:
        fns = (
            build_index_dk_chunks(H, DI, topk).compile(*chunk_args),
            build_index_dk_fix(H, DI, topk).compile(*fix_args),
        )
        _DK_CACHE[key] = fns
    fns[0](*chunk_args)
    fns[1](*fix_args)
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
