###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention backward: dQ, dK, dV for ``msa_token_fwd``.

With ``P = exp(scale * Q K^T - lse)`` over each token's selected keys, and
``delta = rowsum(dO * O)``:

    dS = P * (dO V^T - delta)
    dQ = scale * dS K        dK = scale * dS^T Q        dV = P^T dO

Two kernels, both deterministic (no atomics):

``dq``: one single-wave work-group per (query token, GQA group), walking the
token's selection exactly like the forward. S^T = K Q^T and dP^T = V dO^T take
K and V rows as the A operand; dQ^T += K^T dS^T reads K back transposed with
``ds_read_tr16_b64``, and dS^T already sits in the B-operand layout that MFMA
wants -- the same trick the forward plays with P. It also computes ``delta``
from the dO and O rows it holds and writes it out for ``dkdv``.

``dkdv``: one work-group per chunk of the tokens that selected one (batch,
GQA group, KV block), 8 waves, each holding 16 of the block's keys (K and V in
registers as B operands). The token lists are an inverted, CSR-shaped copy of
the block table, built on the host by one stable sort, so tokens arrive in
ascending order. Each token's 16 query heads of Q and dO are staged in LDS and
shared by the 8 waves. S = Q K^T and dP = dO V^T put heads on MFMA rows;
dV^T += dO^T P and dK^T += Q^T dS read Q and dO back transposed.

The lists are chunked because they are badly skewed -- early blocks are
visible to, and picked by, far more tokens than late ones (max/mean 2-4x on
random selections), and at short sequences there are fewer blocks than CUs --
so one work-group per block leaves most of the GPU idle behind the longest
list. Each chunk writes fp32 partials; ``dkdv_reduce`` sums a block's chunks in
order, keeping the result deterministic.

Masking is the forward's: ``key <= t`` per element, which covers the diagonal
block and a partial last block, and -1 slots contribute nothing.

The host side never synchronizes with the device: every tensor size, grid and
workspace follows from the input shapes alone, not from the selection.
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

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
    HPW,
    KS,
    STRIDE,
    TK,
    D,
)

_LOG2E = 1.4426950408889634
DT = D // 16


def _crossgrp_sum(x):
    """Sum over the 4 lane groups (lanes lo, lo+16, lo+32, lo+48) with permlane swaps."""
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

    def swap_add(v, op):
        v_i32 = _raw(ArithValue(_raw(v)).bitcast(fx.Int32.ir_type))
        sw = op(pair_ty, v_i32, v_i32, False, True)
        a = fx.Float32(
            _raw(ArithValue(llvm.extractvalue(fx.Int32.ir_type, sw, [0])).bitcast(fx.Float32.ir_type))
        )
        c = fx.Float32(
            _raw(ArithValue(llvm.extractvalue(fx.Int32.ir_type, sw, [1])).bitcast(fx.Float32.ir_type))
        )
        return fx.Float32(arith.AddFOp(_raw(a), _raw(c)).result)

    return swap_add(swap_add(x, rocdl.permlane16_swap), rocdl.permlane32_swap)


def _tr16(v4, byte_addr):
    ptr = buffer_ops.create_llvm_ptr(_raw(byte_addr), address_space=3)
    return _raw(Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16))


def _tr16_base(lo, grp, lds_byte_off):
    """Per-lane byte address of the transposed read of a 16-row LDS tile."""
    return fx.Int64(
        ((grp * fx.Index(4) + lo // fx.Index(4)) * fx.Index(STRIDE) + (lo % fx.Index(4)) * fx.Index(4))
        * fx.Index(2)
        + fx.Index(lds_byte_off)
    )


def _store_bf16x4(rsrc, elem_offset, v4f_val, scale_v):
    ov = Vec(v4f_val) * scale_v
    pk0 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))
    pk1 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))
    buffer_ops.buffer_store(
        _raw(Vec.from_elements([fx.Int32(_raw(pk0)), fx.Int32(_raw(pk1))], fx.Int32)),
        rsrc,
        elem_offset * fx.Index(2),
        offset_is_bytes=True,
    )


# ============================================================================
# dQ
# ============================================================================


def build_dq(num_kv_heads: int, topk: int, block_size: int, scale: float):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    THREADS = 64
    SPB = block_size // TK
    LPR = THREADS // TK  # loader lanes per key row
    CPL = D // LPR
    NV8 = CPL // 8
    BUF = TK * STRIDE
    NBUF = 2

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dq_smem")
    k_off = allocator._align(allocator.ptr, 16)
    v_off = allocator._align(k_off + NBUF * BUF * 2, 16)
    allocator.ptr = allocator._align(v_off + NBUF * BUF * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        TBL: fx.Tensor,
        DQ: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        v8f = Vec.make_type(8, fx.Float32)
        lds_k = SmemPtr(allocator.get_base(), k_off, elem.ir_type, shape=(NBUF * BUF,)).get()
        lds_v = SmemPtr(allocator.get_base(), v_off, elem.ir_type, shape=(NBUF * BUF,)).get()

        lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        tok = fx.Index(gpu.block_idx.x)
        bg = fx.Index(gpu.block_idx.y)
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)

        q_bytes = _raw(Sn * Bn * fx.Index(Hq * D * 2))
        kv_bytes = _raw(Sn * Bn * fx.Index(Hkv * D * 2))
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=q_bytes)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=q_bytes)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=q_bytes)
        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=False, num_records_bytes=q_bytes)
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=kv_bytes)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=kv_bytes)
        row_bytes = _raw(Sn * Bn * fx.Index(Hq * 4))
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=False, num_records_bytes=row_bytes)
        dl_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=False, num_records_bytes=row_bytes)
        t_rsrc = buffer_ops.create_buffer_resource(
            TBL, max_size=False, num_records_bytes=_raw(Bn * fx.Index(Hkv) * Sn * fx.Index(topk * 4))
        )

        c_sl = fx.Float32(scale * _LOG2E)
        c_log2e = fx.Float32(_LOG2E)
        c_zero = fx.Float32(0.0)

        # ---- this lane's head (= lo): Q, dO as B operands; lse, delta as scalars ----
        head = g * fx.Index(HPW) + lo
        row = (tok * Bn + b) * fx.Index(Hq) + head
        q_packs = [
            buffer_ops.buffer_load(
                q_rsrc, row * fx.Index(D) + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]
        do_packs = [
            buffer_ops.buffer_load(
                do_rsrc, row * fx.Index(D) + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]
        lse2 = fx.Float32(buffer_ops.buffer_load(lse_rsrc, row, vec_width=1, dtype=fx.Float32)) * c_log2e

        # delta = rowsum(dO * O) for head lo: each lane dots its 32 elements, then
        # the 4 lane groups sum. Written out for the dK/dV kernel, which runs next.
        prod = Vec.filled(8, 0.0, fx.Float32)
        for ks in range_constexpr(KS):
            o_pack = buffer_ops.buffer_load(
                o_rsrc, row * fx.Index(D) + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            prod = prod + Vec(arith.ExtFOp(v8f, _raw(do_packs[ks])).result) * Vec(
                arith.ExtFOp(v8f, _raw(o_pack)).result
            )
        part = fx.Float32(_raw(Vec(prod)[0]))
        for e in range_constexpr(1, 8):
            part = part + fx.Float32(_raw(Vec(prod)[e]))
        delta = _crossgrp_sum(part)
        buffer_ops.buffer_store(
            delta, dl_rsrc, row * fx.Index(4), mask=_raw(ArithValue(grp == fx.Index(0))), offset_is_bytes=True
        )

        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = fx.Int32(ArithValue(n_vis < fx.Int32(topk)).select(_raw(n_vis), _raw(fx.Int32(topk))))
        n_steps = fx.Index(n_valid) * fx.Index(SPB)
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)

        def kv_offset(key):
            return ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)

        ld_row = lane // fx.Index(LPR)
        ld_col = (lane % fx.Index(LPR)) * fx.Index(CPL)
        ld_lds = ld_row * fx.Index(STRIDE) + ld_col

        def load_blk(step):
            return fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + step // fx.Index(SPB), vec_width=1, dtype=fx.Int32)
            )

        def step_kv0(blk, step):
            safe = fx.Int32(ArithValue(blk >= fx.Int32(0)).select(_raw(blk), _raw(fx.Int32(0))))
            return safe * fx.Int32(block_size) + fx.Int32(step % fx.Index(SPB)) * fx.Int32(TK)

        def load_rows(kv0):
            src = kv_offset(fx.Index(kv0) + ld_row) + ld_col
            return [
                buffer_ops.buffer_load(k_rsrc, src + fx.Index(c * 8), vec_width=8, dtype=elem)
                for c in range_constexpr(NV8)
            ] + [
                buffer_ops.buffer_load(v_rsrc, src + fx.Index(c * 8), vec_width=8, dtype=elem)
                for c in range_constexpr(NV8)
            ]

        k_tr = _tr16_base(lo, grp, k_off)
        NR = 2 * NV8
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        blk0 = load_blk(fx.Index(0))
        init = [c_zero_v4 for _ in range_constexpr(DT)] + load_rows(step_kv0(blk0, fx.Index(0))) + [blk0]
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            dq_acc = [it[dt] for dt in range_constexpr(DT)]
            rows = [it[DT + c] for c in range_constexpr(NR)]
            blk = it[DT + NR]
            jn = j + fx.Index(1)
            jn = fx.Index(ArithValue(jn < n_steps).select(_raw(jn), _raw(j)))
            blk_n = load_blk(jn)
            carry_next = load_rows(step_kv0(blk_n, jn)) + [blk_n]

            blk_ok = ArithValue(blk >= fx.Int32(0))
            kv0 = step_kv0(blk, j)
            buf = (j % fx.Index(NBUF)) * fx.Index(BUF)
            for c in range_constexpr(NV8):
                Vec(rows[c]).store(lds_k, [buf + ld_lds + fx.Index(c * 8)])
                Vec(rows[NV8 + c]).store(lds_v, [buf + ld_lds + fx.Index(c * 8)])

            # S^T = K Q^T, dP^T = V dO^T: lane holds [key = grp*4 + i, head = lo]
            s_acc = Vec.filled(4, 0.0, fx.Float32)
            dp_acc = Vec.filled(4, 0.0, fx.Float32)
            for ks in range_constexpr(KS):
                a_off = buf + lo * fx.Index(STRIDE) + fx.Index(ks * 32) + grp * fx.Index(8)
                s_acc = rocdl.mfma_f32_16x16x32_bf16(
                    v4f, [_raw(Vec.load(v8, lds_k, [a_off])), q_packs[ks], s_acc]
                )
                dp_acc = rocdl.mfma_f32_16x16x32_bf16(
                    v4f, [_raw(Vec.load(v8, lds_v, [a_off])), do_packs[ks], dp_acc]
                )

            key0 = kv0 + fx.Int32(grp) * fx.Int32(4)
            ds = []
            for i in range_constexpr(4):
                ok = ArithValue(
                    arith.AndIOp(_raw(blk_ok), _raw(ArithValue(key0 + fx.Int32(i) <= tok_i32))).result
                )
                s = fx.Float32(_raw(Vec(s_acc)[i]))
                p = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(s * c_sl - lse2)))
                p = fx.Float32(ok.select(_raw(p), _raw(c_zero)))
                ds.append(p * (fx.Float32(_raw(Vec(dp_acc)[i])) - delta))

            # dQ^T += K^T dS^T: A = K^T (transposed LDS read), B = dS^T
            dsB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in ds], elem).bitcast(fx.Int16))
            k_tr_b = k_tr + fx.Int64(buf * fx.Index(2))
            new_dq = [
                rocdl.mfma_f32_16x16x16bf16_1k(v4f, [_tr16(v4, k_tr_b + fx.Int64(dt * 32)), dsB, dq_acc[dt]])
                for dt in range_constexpr(DT)
            ]
            results = yield new_dq + carry_next

        # lane holds dQ[head = lo, d = dt*16 + grp*4 + i]
        scale_v = Vec.from_elements([fx.Float32(scale)] * 4, fx.Float32)
        for dt in range_constexpr(DT):
            _store_bf16x4(
                dq_rsrc, row * fx.Index(D) + fx.Index(dt * 16) + grp * fx.Index(4), results[dt], scale_v
            )

    @flyc.jit
    def launch(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B).launch(
            grid=(fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


# ============================================================================
# dK, dV
# ============================================================================


def build_dkdv(num_kv_heads: int, block_size: int, scale: float):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    WAVES = block_size // TK
    THREADS = 64 * WAVES
    # Q/dO staging: 16 rows x D, 4 elements per thread each
    ST_CPL = HPW * D // THREADS
    ST_LPR = D // ST_CPL
    assert HPW * D % THREADS == 0 and ST_CPL in (2, 4, 8)
    BUF = HPW * STRIDE

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dkdv_smem")
    q_off = allocator._align(allocator.ptr, 16)
    do_off = allocator._align(q_off + BUF * 2, 16)
    allocator.ptr = allocator._align(do_off + BUF * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        CHUNKS: fx.Tensor,
        TOK: fx.Tensor,
        WS: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
        NB: fx.Int32,
        NTOK: fx.Int32,
        NCH: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(BUF,)).get()
        lds_do = SmemPtr(allocator.get_base(), do_off, elem.ir_type, shape=(BUF,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(64)
        lane = tid % fx.Index(64)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        chunk = fx.Index(gpu.block_idx.x)

        q_bytes = _raw(Sn * Bn * fx.Index(Hq * D * 2))
        kv_bytes = _raw(Sn * Bn * fx.Index(Hkv * D * 2))
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=q_bytes)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=q_bytes)
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=False, num_records_bytes=kv_bytes)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=False, num_records_bytes=kv_bytes)
        row_bytes = _raw(Sn * Bn * fx.Index(Hq * 4))
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=False, num_records_bytes=row_bytes)
        dl_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=False, num_records_bytes=row_bytes)
        ch_rsrc = buffer_ops.create_buffer_resource(
            CHUNKS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(3 * 4))
        )
        tok_rsrc = buffer_ops.create_buffer_resource(
            TOK, max_size=False, num_records_bytes=_raw(fx.Index(NTOK) * fx.Index(4))
        )
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(2 * block_size * D * 4))
        )

        # ---- this chunk: (csr row, first token, end token); row -> (batch, KV head, block) ----
        csr = fx.Index(
            fx.Int32(buffer_ops.buffer_load(ch_rsrc, chunk * fx.Index(3), vec_width=1, dtype=fx.Int32))
        )
        t_begin = fx.Int32(
            buffer_ops.buffer_load(ch_rsrc, chunk * fx.Index(3) + fx.Index(1), vec_width=1, dtype=fx.Int32)
        )
        t_end = fx.Int32(
            buffer_ops.buffer_load(ch_rsrc, chunk * fx.Index(3) + fx.Index(2), vec_width=1, dtype=fx.Int32)
        )
        n_tok = fx.Index(t_end - t_begin)
        bg = csr // NBn
        blk = csr % NBn
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)

        c_sl = fx.Float32(scale * _LOG2E)
        c_log2e = fx.Float32(_LOG2E)
        c_zero = fx.Float32(0.0)

        # ---- this wave's 16 keys, K and V as B operands: lane holds key kv0 + lo ----
        kv0 = blk * fx.Index(block_size) + wave * fx.Index(TK)
        key = kv0 + lo
        key_i32 = fx.Int32(key)
        kv_row = ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)
        k_ops = [
            buffer_ops.buffer_load(
                k_rsrc, kv_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]
        v_ops = [
            buffer_ops.buffer_load(
                v_rsrc, kv_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]

        st_row = tid // fx.Index(ST_LPR)
        st_col = (tid % fx.Index(ST_LPR)) * fx.Index(ST_CPL)
        st_lds = st_row * fx.Index(STRIDE) + st_col
        q_tr = _tr16_base(lo, grp, q_off)
        do_tr = _tr16_base(lo, grp, do_off)

        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = [c_zero_v4 for _ in range_constexpr(2 * DT)]
        results = init
        for i, it in range(fx.Index(0), n_tok, fx.Index(1), init=init):
            dk_acc = [it[dt] for dt in range_constexpr(DT)]
            dv_acc = [it[DT + dt] for dt in range_constexpr(DT)]
            t = fx.Int32(buffer_ops.buffer_load(tok_rsrc, fx.Index(t_begin) + i, vec_width=1, dtype=fx.Int32))
            head0 = (fx.Index(t) * Bn + b) * fx.Index(Hq) + g * fx.Index(HPW)

            # ---- stage the token's Q and dO heads in LDS, shared by all waves ----
            st_src = (head0 + st_row) * fx.Index(D) + st_col
            q_st = buffer_ops.buffer_load(q_rsrc, st_src, vec_width=ST_CPL, dtype=elem)
            do_st = buffer_ops.buffer_load(do_rsrc, st_src, vec_width=ST_CPL, dtype=elem)
            # lse, delta for heads grp*4 + i (the rows this lane holds)
            lse4 = buffer_ops.buffer_load(lse_rsrc, head0 + grp * fx.Index(4), vec_width=4, dtype=fx.Float32)
            dl4 = buffer_ops.buffer_load(dl_rsrc, head0 + grp * fx.Index(4), vec_width=4, dtype=fx.Float32)
            gpu.barrier()  # WAR: every wave is done with the previous token's tiles
            Vec(q_st).store(lds_q, [st_lds])
            Vec(do_st).store(lds_do, [st_lds])
            gpu.barrier()

            # S = Q K^T, dP = dO V^T: lane holds [head = grp*4 + i, key = lo]
            s_acc = Vec.filled(4, 0.0, fx.Float32)
            dp_acc = Vec.filled(4, 0.0, fx.Float32)
            for ks in range_constexpr(KS):
                a_off = lo * fx.Index(STRIDE) + fx.Index(ks * 32) + grp * fx.Index(8)
                s_acc = rocdl.mfma_f32_16x16x32_bf16(
                    v4f, [_raw(Vec.load(v8, lds_q, [a_off])), k_ops[ks], s_acc]
                )
                dp_acc = rocdl.mfma_f32_16x16x32_bf16(
                    v4f, [_raw(Vec.load(v8, lds_do, [a_off])), v_ops[ks], dp_acc]
                )

            ok = ArithValue(key_i32 <= t)
            p = []
            ds = []
            for r in range_constexpr(4):
                s = fx.Float32(_raw(Vec(s_acc)[r]))
                pv = fx.Float32(
                    rocdl.exp2(fx.Float32.ir_type, _raw(s * c_sl - fx.Float32(_raw(Vec(lse4)[r])) * c_log2e))
                )
                pv = fx.Float32(ok.select(_raw(pv), _raw(c_zero)))
                p.append(pv)
                ds.append(pv * (fx.Float32(_raw(Vec(dp_acc)[r])) - fx.Float32(_raw(Vec(dl4)[r]))))

            # dV^T += dO^T P, dK^T += Q^T dS: A = dO^T / Q^T (transposed reads), B = P / dS
            pB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in p], elem).bitcast(fx.Int16))
            dsB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in ds], elem).bitcast(fx.Int16))
            new_dk = []
            new_dv = []
            for dt in range_constexpr(DT):
                new_dv.append(
                    rocdl.mfma_f32_16x16x16bf16_1k(
                        v4f, [_tr16(v4, do_tr + fx.Int64(dt * 32)), pB, dv_acc[dt]]
                    )
                )
                new_dk.append(
                    rocdl.mfma_f32_16x16x16bf16_1k(
                        v4f, [_tr16(v4, q_tr + fx.Int64(dt * 32)), dsB, dk_acc[dt]]
                    )
                )
            results = yield new_dk + new_dv

        # fp32 partials, chunk-major [chunk][dK | dV][key in block][d]:
        # lane holds dK^T / dV^T[d = dt*16 + grp*4 + i, key = wave*16 + lo]
        scale_v = Vec.from_elements([fx.Float32(scale)] * 4, fx.Float32)
        ws_row = (chunk * fx.Index(2 * block_size) + wave * fx.Index(TK) + lo) * fx.Index(D)
        for dt in range_constexpr(DT):
            off = ws_row + fx.Index(dt * 16) + grp * fx.Index(4)
            buffer_ops.buffer_store(
                _raw(Vec(results[dt]) * scale_v), ws_rsrc, off * fx.Index(4), offset_is_bytes=True
            )
            buffer_ops.buffer_store(
                results[DT + dt],
                ws_rsrc,
                (off + fx.Index(block_size * D)) * fx.Index(4),
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(Q, K, V, DO, LSE, DELTA, CHUNKS, TOK, WS, S, B, NB, NTOK, NCH, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, DO, LSE, DELTA, CHUNKS, TOK, WS, S, B, NB, NTOK, NCH).launch(
            grid=(fx.Index(NCH), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_dkdv_reduce(num_kv_heads: int, block_size: int):
    """Sum each (batch, KV head, block)'s chunk partials in chunk order -> bf16 dK, dV."""
    fx.BFloat16
    Hkv = num_kv_heads
    THREADS = 256
    NE = block_size * D  # elements per partial
    PER = NE // (THREADS * 4)  # v4 groups per thread
    assert NE % (THREADS * 4) == 0

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        WS: fx.Tensor,
        CHPTR: fx.Tensor,
        DK: fx.Tensor,
        DV: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
        NB: fx.Int32,
        NCH: fx.Int32,
    ):
        tid = fx.Index(gpu.thread_idx.x)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        row = fx.Index(gpu.block_idx.x)
        bg = row // NBn
        blk = row % NBn
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)

        kv_bytes = _raw(Sn * Bn * fx.Index(Hkv * D * 2))
        dk_rsrc = buffer_ops.create_buffer_resource(DK, max_size=False, num_records_bytes=kv_bytes)
        dv_rsrc = buffer_ops.create_buffer_resource(DV, max_size=False, num_records_bytes=kv_bytes)
        ws_rsrc = buffer_ops.create_buffer_resource(
            WS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(2 * NE * 4))
        )
        cp_rsrc = buffer_ops.create_buffer_resource(
            CHPTR,
            max_size=False,
            num_records_bytes=_raw((Bn * fx.Index(Hkv) * NBn + fx.Index(1)) * fx.Index(4)),
        )
        c_begin = fx.Index(fx.Int32(buffer_ops.buffer_load(cp_rsrc, row, vec_width=1, dtype=fx.Int32)))
        c_end = fx.Index(
            fx.Int32(buffer_ops.buffer_load(cp_rsrc, row + fx.Index(1), vec_width=1, dtype=fx.Int32))
        )

        elems = [tid * fx.Index(4) + fx.Index(j * THREADS * 4) for j in range_constexpr(PER)]
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = [c_zero_v4 for _ in range_constexpr(2 * PER)]
        results = init
        for c, it in range(c_begin, c_end, fx.Index(1), init=init):
            base = c * fx.Index(2 * NE)
            new = []
            for j in range_constexpr(PER):
                new.append(
                    Vec(it[j])
                    + Vec(buffer_ops.buffer_load(ws_rsrc, base + elems[j], vec_width=4, dtype=fx.Float32))
                )
            for j in range_constexpr(PER):
                new.append(
                    Vec(it[PER + j])
                    + Vec(
                        buffer_ops.buffer_load(
                            ws_rsrc, base + fx.Index(NE) + elems[j], vec_width=4, dtype=fx.Float32
                        )
                    )
                )
            results = yield [_raw(x) for x in new]

        one_v = Vec.from_elements([fx.Float32(1.0)] * 4, fx.Float32)
        for j in range_constexpr(PER):
            key = blk * fx.Index(block_size) + elems[j] // fx.Index(D)
            off = ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D) + elems[j] % fx.Index(D)
            _store_bf16x4(dk_rsrc, off, results[j], one_v)  # keys past S fall off the buffer
            _store_bf16x4(dv_rsrc, off, results[PER + j], one_v)

    @flyc.jit
    def launch(WS, CHPTR, DK, DV, S, B, NB, NCH, stream):
        k_fn(WS, CHPTR, DK, DV, S, B, NB, NCH).launch(
            grid=(fx.Index(B) * fx.Index(Hkv) * fx.Index(NB), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


# ============================================================================
# host
# ============================================================================


def build_inverted_table(block_table: torch.Tensor, n_blocks: int):
    """CSR of the block table's transpose: for each (batch, KV head, block), the
    tokens that selected it, ascending.

    Returns ``(ptr, tokens)``: ``ptr`` is ``[B * Hkv * n_blocks + 1]`` int32 and
    ``tokens[ptr[i]:ptr[i + 1]]`` are the tokens of row
    ``i = (b * Hkv + g) * n_blocks + block``. One stable sort, the same approach
    as Primus-Turbo's sparse-MLA backward.

    Sync-free: every table entry is sorted, -1 slots under a key past the last
    row, so ``tokens`` always has ``B * Hkv * S * topk`` entries and only its
    first ``ptr[-1]`` are real. No size ever depends on the table's contents.
    """
    B, Hkv, S, topk = block_table.shape
    device = block_table.device
    n_rows = B * Hkv * n_blocks
    rows = torch.arange(B * Hkv, device=device, dtype=torch.int32).view(B, Hkv, 1, 1) * n_blocks
    keys = torch.where(block_table >= 0, rows + block_table, n_rows).flatten()
    toks = (
        torch.arange(S, device=device, dtype=torch.int32).view(1, 1, S, 1).expand(B, Hkv, S, topk).flatten()
    )
    keys_sorted, order = torch.sort(keys, stable=True)
    tokens = toks[order].contiguous()
    bounds = torch.arange(n_rows + 1, device=device, dtype=torch.int32)
    ptr = torch.searchsorted(keys_sorted, bounds).to(torch.int32)
    return ptr, tokens


def plan_chunks(ptr: torch.Tensor, max_tokens: int, target_chunks: int = 2048, min_chunk: int = 64):
    """Split every CSR row into equal-sized token chunks, about ``target_chunks`` in all.

    Every row gets at least one chunk, so rows no token selected still write zero
    partials. Sync-free: the chunk size comes from ``max_tokens`` (the table's
    capacity, not its live count) and the chunk array is sized for the worst case,
    ``n_rows + ceil(max_tokens / size)``; chunks past the live ones are empty
    (``begin == end``) and no row's ``chunk_ptr`` range reaches them.

    Returns ``(chunks, chunk_ptr, n_chunks)``: ``chunks`` is ``[n_chunks, 3]``
    int32 ``(row, begin, end)`` in row order, ``chunk_ptr`` ``[n_rows + 1]`` int32.
    """
    device = ptr.device
    n_rows = ptr.numel() - 1
    size = max(min_chunk, -(-max_tokens // target_chunks))
    counts = (ptr[1:] - ptr[:-1]).long()
    per_row = ((counts + size - 1) // size).clamp_min(1)
    chunk_ptr = torch.zeros(n_rows + 1, dtype=torch.long, device=device)
    chunk_ptr[1:] = torch.cumsum(per_row, 0)

    n_chunks = n_rows + -(-max_tokens // size)
    c = torch.arange(n_chunks, device=device)
    live = c < chunk_ptr[-1]
    rows = torch.searchsorted(chunk_ptr[1:], c, right=True).clamp_max(n_rows - 1)
    begin = ptr[rows].long() + (c - chunk_ptr[rows]) * size
    end = torch.minimum(begin + size, ptr[rows + 1].long())
    zero = torch.zeros_like(c)
    chunks = torch.stack(
        [torch.where(live, rows, zero), torch.where(live, begin, zero), torch.where(live, end, zero)], dim=1
    )
    return chunks.to(torch.int32).contiguous(), chunk_ptr.to(torch.int32), n_chunks


_DQ_CACHE: dict = {}
_DKDV_CACHE: dict = {}
_RED_CACHE: dict = {}


def msa_token_bwd(dout, q, k, v, out, lse, block_table, softmax_scale=None, block_size=128):
    """Gradients of ``msa_token_fwd``.

    Args:
        dout: ``[S, B, Hq, 128]`` bf16, the gradient of ``out``.
        q, k, v, block_table, softmax_scale, block_size: as passed to the forward.
        out, lse: the forward's outputs.

    Returns:
        ``dq`` ``[S, B, Hq, 128]``, ``dk`` and ``dv`` ``[S, B, Hkv, 128]``, bf16.
    """
    S, B, Hq, Dq = q.shape
    Hkv = k.shape[2]
    topk = block_table.shape[-1]
    assert Dq == D and Hq == HPW * Hkv and block_table.shape == (B, Hkv, S, topk)
    if softmax_scale is None:
        softmax_scale = D**-0.5
    dout, q, k, v, block_table = (t.contiguous() for t in (dout, q, k, v, block_table))
    lse = lse.contiguous()
    n_blocks = -(-S // block_size)

    out = out.contiguous()
    ptr, tokens = build_inverted_table(block_table, n_blocks)

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)  # written by the dq kernel
    stream = torch.cuda.current_stream()

    dq_args = (q, k, v, dout, out, lse, delta, block_table, dq, int(S), int(B), stream)
    dq_key = (Hkv, topk, block_size, float(softmax_scale))
    fn = _DQ_CACHE.get(dq_key)
    if fn is None:
        fn = build_dq(Hkv, topk, block_size, float(softmax_scale)).compile(*dq_args)
        _DQ_CACHE[dq_key] = fn
    fn(*dq_args)

    chunks, chunk_ptr, n_chunks = plan_chunks(ptr, tokens.numel())
    ws = torch.empty((n_chunks, 2, block_size, D), dtype=torch.float32, device=q.device)
    kv_args = (
        q,
        k,
        v,
        dout,
        lse,
        delta,
        chunks,
        tokens,
        ws,
        int(S),
        int(B),
        int(n_blocks),
        int(tokens.numel()),
        int(n_chunks),
        stream,
    )
    kv_key = (Hkv, block_size, float(softmax_scale))
    fn = _DKDV_CACHE.get(kv_key)
    if fn is None:
        fn = build_dkdv(Hkv, block_size, float(softmax_scale)).compile(*kv_args)
        _DKDV_CACHE[kv_key] = fn
    fn(*kv_args)

    red_args = (ws, chunk_ptr, dk, dv, int(S), int(B), int(n_blocks), int(n_chunks), stream)
    red_key = (Hkv, block_size)
    fn = _RED_CACHE.get(red_key)
    if fn is None:
        fn = build_dkdv_reduce(Hkv, block_size).compile(*red_args)
        _RED_CACHE[red_key] = fn
    fn(*red_args)
    return dq, dk, dv
