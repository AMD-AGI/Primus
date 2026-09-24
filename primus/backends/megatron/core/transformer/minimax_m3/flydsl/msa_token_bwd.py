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
the block table (:class:`BlockPlan`, which the indexer's backward shares),
holding table entries in ascending token order. Four tokens at a time, their
16 query heads of Q and dO (and lse, delta) are staged in LDS and shared by
the 8 waves. S = Q K^T and dP = dO V^T put heads on MFMA rows; dV^T += dO^T P
and dK^T += Q^T dS read Q and dO back transposed. The staging is software
pipelined -- a step's rows are loaded while the previous step computes, its
tokens one step earlier still -- because at this register count one
work-group fills a CU, and nothing else would hide the load latency.

The inverted table itself is a counting sort (:func:`build_inverted_table`):
the keys are block ids, so two passes that count in LDS replace a general
sort.

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

from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as _memref
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


def build_dkdv(num_kv_heads: int, block_size: int, scale: float, topk: int, tokens_per_step: int = 4):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    WAVES = block_size // TK
    THREADS = 64 * WAVES
    TPS = tokens_per_step
    ROWS = TPS * HPW  # staged rows per step: TPS tokens x 16 heads
    VEC = 8
    NP = ROWS * D // (THREADS * VEC)  # staging loads per thread per tensor per step
    assert ROWS * D % (THREADS * VEC) == 0 and ROWS <= THREADS
    BUF = ROWS * STRIDE

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dkdv_smem")
    q_off = allocator._align(allocator.ptr, 16)
    do_off = allocator._align(q_off + BUF * 2, 16)
    lse_off = allocator._align(do_off + BUF * 2, 16)
    dl_off = allocator._align(lse_off + THREADS * 4, 16)
    allocator.ptr = allocator._align(dl_off + THREADS * 4, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DELTA: fx.Tensor,
        CHUNKS: fx.Tensor,
        ENT: fx.Tensor,
        WS: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
        NB: fx.Int32,
        NENT: fx.Int32,
        NCH: fx.Int32,
    ):
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(BUF,)).get()
        lds_do = SmemPtr(allocator.get_base(), do_off, elem.ir_type, shape=(BUF,)).get()
        lds_lse = SmemPtr(allocator.get_base(), lse_off, fx.Float32.ir_type, shape=(THREADS,)).get()
        lds_dl = SmemPtr(allocator.get_base(), dl_off, fx.Float32.ir_type, shape=(THREADS,)).get()

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
        ent_rsrc = buffer_ops.create_buffer_resource(
            ENT, max_size=False, num_records_bytes=_raw(fx.Index(NENT) * fx.Index(4))
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

        # staging: TPS tokens x 16 heads of Q and dO, VEC elements per load
        st_flat = [(fx.Index(p * THREADS) + tid) * fx.Index(VEC) for p in range_constexpr(NP)]
        st_rows = [f // fx.Index(D) for f in st_flat]
        st_cols = [f % fx.Index(D) for f in st_flat]
        # lse / delta: thread r < ROWS stages row r (token r // 16, head r % 16); the
        # rest repeat those rows into slots nobody reads
        ld_row = tid % fx.Index(ROWS)
        q_tr = _tr16_base(lo, grp, q_off)
        do_tr = _tr16_base(lo, grp, do_off)

        def pick(vals, j):  # vals[j] for a per-thread j < TPS
            out = vals[0]
            for jj in range_constexpr(TPS - 1):
                out = fx.Int32(ArithValue(j == fx.Index(jj + 1)).select(_raw(vals[jj + 1]), _raw(out)))
            return out

        def load_tokens(r0):
            """The tokens of the step starting at entry r0; past the chunk's end
            they repeat its last one (masked out by the caller)."""
            tok = []
            for j in range_constexpr(TPS):
                r = r0 + fx.Int32(j)
                r = fx.Int32(ArithValue(r < t_end).select(_raw(r), _raw(t_end - fx.Int32(1))))
                e = fx.Index(
                    fx.Int32(buffer_ops.buffer_load(ent_rsrc, fx.Index(r), vec_width=1, dtype=fx.Int32))
                )
                tok.append(fx.Int32((e // fx.Index(topk)) % Sn))
            return tok

        def head_row(t_i32, head):
            return (fx.Index(t_i32) * Bn + b) * fx.Index(Hq) + g * fx.Index(HPW) + head

        def load_rows(tok):
            """This thread's share of the step's Q, dO rows and lse, delta values."""
            q_st, do_st = [], []
            for p in range_constexpr(NP):
                src = (
                    head_row(pick(tok, st_rows[p] // fx.Index(HPW)), st_rows[p] % fx.Index(HPW)) * fx.Index(D)
                    + st_cols[p]
                )
                q_st.append(buffer_ops.buffer_load(q_rsrc, src, vec_width=VEC, dtype=elem))
                do_st.append(buffer_ops.buffer_load(do_rsrc, src, vec_width=VEC, dtype=elem))
            lrow = head_row(pick(tok, ld_row // fx.Index(HPW)), ld_row % fx.Index(HPW))
            lse_st = buffer_ops.buffer_load(lse_rsrc, lrow, vec_width=1, dtype=fx.Float32)
            dl_st = buffer_ops.buffer_load(dl_rsrc, lrow, vec_width=1, dtype=fx.Float32)
            return [_raw(x) for x in q_st + do_st + [lse_st, dl_st]]

        # Software pipeline: a step's rows are loaded during the previous step's
        # compute, and its tokens one step before that, so neither load's
        # latency -- nor the entry -> token -> row dependency -- is exposed.
        NR = 2 * NP + 2  # carried row values
        tok0 = load_tokens(t_begin)
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = (
            [c_zero_v4 for _ in range_constexpr(2 * DT)]
            + load_rows(tok0)
            + tok0
            + load_tokens(t_begin + fx.Int32(TPS))
        )
        results = init
        for i, it in range(fx.Index(0), n_tok, fx.Index(TPS), init=init):
            dk_acc = [it[dt] for dt in range_constexpr(DT)]
            dv_acc = [it[DT + dt] for dt in range_constexpr(DT)]
            rows = [it[2 * DT + x] for x in range_constexpr(NR)]
            tok = [fx.Int32(it[2 * DT + NR + j]) for j in range_constexpr(TPS)]
            tok_next = [fx.Int32(it[2 * DT + NR + TPS + j]) for j in range_constexpr(TPS)]
            r0 = t_begin + fx.Int32(i)
            valid = [ArithValue(r0 + fx.Int32(j) < t_end) for j in range_constexpr(TPS)]

            gpu.barrier()  # WAR: every wave is done with the previous step's tiles
            for p in range_constexpr(NP):
                lds_at = st_rows[p] * fx.Index(STRIDE) + st_cols[p]
                Vec(rows[p]).store(lds_q, [lds_at])
                Vec(rows[NP + p]).store(lds_do, [lds_at])
            _memref.store(rows[2 * NP], lds_lse, [_raw(tid)])
            _memref.store(rows[2 * NP + 1], lds_dl, [_raw(tid)])
            gpu.barrier()
            # in flight while this step computes
            rows_next = load_rows(tok_next)
            tok_after = load_tokens(r0 + fx.Int32(2 * TPS))

            for j in range_constexpr(TPS):
                # S = Q K^T, dP = dO V^T: lane holds [head = grp*4 + i, key = lo]
                s_acc = Vec.filled(4, 0.0, fx.Float32)
                dp_acc = Vec.filled(4, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    a_off = (
                        (fx.Index(j * HPW) + lo) * fx.Index(STRIDE) + fx.Index(ks * 32) + grp * fx.Index(8)
                    )
                    s_acc = rocdl.mfma_f32_16x16x32_bf16(
                        v4f, [_raw(Vec.load(v8, lds_q, [a_off])), k_ops[ks], s_acc]
                    )
                    dp_acc = rocdl.mfma_f32_16x16x32_bf16(
                        v4f, [_raw(Vec.load(v8, lds_do, [a_off])), v_ops[ks], dp_acc]
                    )
                lse4 = Vec.load(v4f, lds_lse, [fx.Index(j * HPW) + grp * fx.Index(4)])
                dl4 = Vec.load(v4f, lds_dl, [fx.Index(j * HPW) + grp * fx.Index(4)])

                ok = ArithValue(arith.AndIOp(_raw(ArithValue(key_i32 <= tok[j])), _raw(valid[j])).result)
                p = []
                ds = []
                for r in range_constexpr(4):
                    s = fx.Float32(_raw(Vec(s_acc)[r]))
                    pv = fx.Float32(
                        rocdl.exp2(
                            fx.Float32.ir_type, _raw(s * c_sl - fx.Float32(_raw(Vec(lse4)[r])) * c_log2e)
                        )
                    )
                    pv = fx.Float32(ok.select(_raw(pv), _raw(c_zero)))
                    p.append(pv)
                    ds.append(pv * (fx.Float32(_raw(Vec(dp_acc)[r])) - fx.Float32(_raw(Vec(dl4)[r]))))

                # dV^T += dO^T P, dK^T += Q^T dS: A = dO^T / Q^T (transposed reads), B = P / dS
                pB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in p], elem).bitcast(fx.Int16))
                dsB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in ds], elem).bitcast(fx.Int16))
                tile = fx.Int64(j * HPW * STRIDE * 2)
                for dt in range_constexpr(DT):
                    dv_acc[dt] = rocdl.mfma_f32_16x16x16bf16_1k(
                        v4f, [_tr16(v4, do_tr + tile + fx.Int64(dt * 32)), pB, dv_acc[dt]]
                    )
                    dk_acc[dt] = rocdl.mfma_f32_16x16x16bf16_1k(
                        v4f, [_tr16(v4, q_tr + tile + fx.Int64(dt * 32)), dsB, dk_acc[dt]]
                    )
            results = (
                yield dk_acc + dv_acc + rows_next + [_raw(t) for t in tok_next] + [_raw(t) for t in tok_after]
            )

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
    def launch(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH).launch(
            grid=(fx.Index(NCH), 1, 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_dkdv_reduce(num_kv_heads: int, block_size: int):
    """Sum each (batch, KV head, block)'s chunk partials in chunk order -> bf16 dK, dV.

    One work-group per 1024-element slice of a block's dK and dV, so short
    sequences -- a few dozen blocks -- still fill the GPU.
    """
    Hkv = num_kv_heads
    THREADS = 256
    NE = block_size * D  # elements per partial
    PER = 1  # v4 groups per thread, per tensor
    SLICE = THREADS * 4 * PER
    SPLIT = NE // SLICE
    assert NE % SLICE == 0

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
        wg = fx.Index(gpu.block_idx.x)
        row = wg // fx.Index(SPLIT)
        part = wg % fx.Index(SPLIT)
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

        elems = [
            part * fx.Index(SLICE) + tid * fx.Index(4) + fx.Index(j * THREADS * 4)
            for j in range_constexpr(PER)
        ]
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = [c_zero_v4 for _ in range_constexpr(2 * PER)]
        results = init
        for c, it in range(c_begin, c_end, fx.Index(2), init=init):
            # two chunks per step, so their loads overlap; still summed in chunk order
            has2 = ArithValue(c + fx.Index(1) < c_end)
            new = []
            for half in range_constexpr(2):
                for j in range_constexpr(PER):
                    at = fx.Index(half * NE) + elems[j]
                    w0 = buffer_ops.buffer_load(
                        ws_rsrc, c * fx.Index(2 * NE) + at, vec_width=4, dtype=fx.Float32
                    )
                    w1 = buffer_ops.buffer_load(
                        ws_rsrc, (c + fx.Index(1)) * fx.Index(2 * NE) + at, vec_width=4, dtype=fx.Float32
                    )
                    s = Vec(it[half * PER + j]) + Vec(w0)
                    new.append(has2.select(_raw(s + Vec(w1)), _raw(s)))
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
            grid=(fx.Index(B) * fx.Index(Hkv) * fx.Index(NB) * fx.Index(SPLIT), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


# ============================================================================
# inverted block table
# ============================================================================

_PLACE_SUB = 128  # tokens whose table rows are staged in LDS at a time


def build_inverted_pass(topk: int, nb_cap: int, place: bool):
    """One pass of the counting sort: a single-wave work-group per (batch x KV
    head, token chunk) walks the chunk's tokens in order with one LDS counter
    per block. A token's slots name distinct blocks, so its lanes never share
    a counter.

    ``place=False`` counts: the counters start at 0 and end up in
    ``RUNS[(bg * NB + block) * NC + chunk]``. ``place=True`` places: ``RUNS``
    holds where each run starts (the exclusive cumsum of the counts), and each
    entry is written at its counter -- in token order, so every row ascends by
    token, the same result as a stable sort, deterministically.
    """
    NTH = 64
    SUB = _PLACE_SUB
    PER = SUB * topk // NTH  # table entries each lane stages per tile
    assert topk <= NTH and SUB * topk % NTH == 0
    # the kernel body is traced into device control flow, so the two modes
    # differ by constants rather than by Python branches
    PLACE = 1 if place else 0

    allocator = SmemAllocator(
        None, arch=get_hip_arch(), global_sym_name=f"msa_inverted_{'place' if place else 'count'}_smem"
    )
    tab_off = allocator._align(allocator.ptr, 16)
    cnt_off = allocator._align(tab_off + SUB * topk * 4, 16)
    allocator.ptr = allocator._align(cnt_off + (nb_cap + 1) * 4, 16)

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def k_fn(
        TBL: fx.Tensor,
        RUNS: fx.Tensor,
        ENT: fx.Tensor,
        S: fx.Int32,
        BG: fx.Int32,
        NB: fx.Int32,
        NC: fx.Int32,
        TC: fx.Int32,
    ):
        i32 = fx.Int32.ir_type
        lds_tab = SmemPtr(allocator.get_base(), tab_off, i32, shape=(SUB * topk,)).get()
        lds_cnt = SmemPtr(allocator.get_base(), cnt_off, i32, shape=(nb_cap + 1,)).get()

        def lds_load(mem, idx):
            return fx.Int32(_memref.load(mem, [_raw(fx.Index(idx))]))

        def lds_store(mem, idx, val):
            _memref.store(_raw(fx.Int32(val)), mem, [_raw(fx.Index(idx))])

        lane = fx.Index(gpu.thread_idx.x)
        lane_i32 = fx.Int32(lane)
        wg = fx.Index(gpu.block_idx.x)
        NCn = fx.Index(NC)
        NBn = fx.Index(NB)
        bg = wg // NCn
        c = wg % NCn
        n_entries = fx.Index(BG) * fx.Index(S) * fx.Index(topk)
        tbl_rsrc = buffer_ops.create_buffer_resource(
            TBL, max_size=False, num_records_bytes=_raw(n_entries * fx.Index(4))
        )
        ent_rsrc = buffer_ops.create_buffer_resource(
            ENT, max_size=False, num_records_bytes=_raw(n_entries * fx.Index(4))
        )
        runs_rsrc = buffer_ops.create_buffer_resource(
            RUNS, max_size=False, num_records_bytes=_raw(fx.Index(BG) * NBn * NCn * fx.Index(4))
        )
        dummy = [fx.Int32(0), fx.Int32(0)]  # loops carry a list; one value would come back unwrapped

        # ---- one counter per block: 0, or this chunk's offset in the block's row ----
        for n, it in range(lane, NBn, fx.Index(NTH), init=dummy):
            start = fx.Int32(
                buffer_ops.buffer_load(runs_rsrc, (bg * NBn + n) * NCn + c, vec_width=1, dtype=fx.Int32)
            )
            lds_store(lds_cnt, n, start * fx.Int32(PLACE))
            yield [it[0], it[1]]
        do_place = ArithValue(fx.Int32(PLACE) == fx.Int32(1))

        c_neg1 = fx.Int32(-1)
        active = ArithValue(lane_i32 < fx.Int32(topk))
        my_slot = fx.Int32(active.select(_raw(lane_i32), _raw(fx.Int32(topk - 1))))
        t0 = fx.Int32(c) * TC
        t_end = t0 + TC
        t_end = fx.Int32(ArithValue(t_end < S).select(_raw(t_end), _raw(S)))
        for sub, it in range(fx.Index(t0), fx.Index(t_end), fx.Index(SUB), init=dummy):
            sub_i32 = fx.Int32(sub)
            gpu.barrier()  # WAR on the staged tile; the first pass also orders the counter setup
            for j in range_constexpr(PER):
                le = fx.Index(j * NTH) + lane
                tok = sub_i32 + fx.Int32(le // fx.Index(topk))
                ok = ArithValue(tok < t_end)
                v = fx.Int32(
                    buffer_ops.buffer_load(
                        tbl_rsrc, (bg * fx.Index(S) + sub) * fx.Index(topk) + le, vec_width=1, dtype=fx.Int32
                    )
                )
                lds_store(lds_tab, le, fx.Int32(ok.select(_raw(v), _raw(c_neg1))))
            gpu.barrier()
            n_tok = t_end - sub_i32
            n_tok = fx.Int32(ArithValue(n_tok < fx.Int32(SUB)).select(_raw(n_tok), _raw(fx.Int32(SUB))))
            for i, it2 in range(fx.Index(0), fx.Index(n_tok), fx.Index(1), init=dummy):
                blk = lds_load(lds_tab, i * fx.Index(topk) + fx.Index(my_slot))
                valid = ArithValue(arith.AndIOp(_raw(active), _raw(ArithValue(blk >= fx.Int32(0)))).result)
                addr = fx.Int32(valid.select(_raw(blk), _raw(fx.Int32(nb_cap))))
                pos = lds_load(lds_cnt, addr)
                lds_store(lds_cnt, addr, pos + fx.Int32(1))
                e = ((bg * fx.Index(S) + sub + i) * fx.Index(topk)) + fx.Index(my_slot)
                buffer_ops.buffer_store(
                    fx.Int32(e),
                    ent_rsrc,
                    fx.Index(pos) * fx.Index(4),
                    mask=_raw(ArithValue(arith.AndIOp(_raw(valid), _raw(do_place)).result)),
                    offset_is_bytes=True,
                )
                yield [it2[0], it2[1]]
            yield [it[0], it[1]]

        # counting pass: publish the counters (the placing pass loops zero times)
        gpu.barrier()
        for n, it in range(lane, NBn * fx.Index(1 - PLACE), fx.Index(NTH), init=dummy):
            buffer_ops.buffer_store(
                lds_load(lds_cnt, n),
                runs_rsrc,
                ((bg * NBn + n) * NCn + c) * fx.Index(4),
                offset_is_bytes=True,
            )
            yield [it[0], it[1]]

    @flyc.jit
    def launch(TBL, RUNS, ENT, S, BG, NB, NC, TC, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(TBL, RUNS, ENT, S, BG, NB, NC, TC).launch(
            grid=(fx.Index(BG) * fx.Index(NC), 1, 1), block=(NTH, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_PLACE_CACHE: dict = {}
_PLACE_MAX_BLOCKS = 16384  # LDS holds one counter per block


def build_inverted_table(block_table: torch.Tensor, n_blocks: int):
    """CSR of the block table's transpose: for each (batch, KV head, block), the
    table entries that selected it, in ascending token order.

    Returns ``(ptr, entries)``: ``ptr`` is ``[B * Hkv * n_blocks + 1]`` int32 and
    ``entries[ptr[i]:ptr[i + 1]]`` are the flat ``[B, Hkv, S, topk]`` indices of
    the slots that picked row ``i = (b * Hkv + g) * n_blocks + block`` -- the
    token is ``entry // topk % S``, and the slot is what the indexer's backward
    reads its per-slot gradient from.

    A counting sort rather than a general one: the keys are block ids, few and
    small. Tokens are cut into chunks; :func:`build_inverted_pass` counts each
    chunk's run in every row, a cumsum turns the counts into run offsets, and
    a second pass writes each chunk's entries in token order. Both passes
    count in LDS, so there are no global atomics and nothing to sort.

    Sync-free: ``entries`` always has ``B * Hkv * S * topk`` elements, of which
    the first ``ptr[-1]`` are real; -1 slots are never placed. No size ever
    depends on the table's contents.
    """
    B, Hkv, S, topk = block_table.shape
    device = block_table.device
    BG = B * Hkv
    n_rows = BG * n_blocks
    table = block_table.to(torch.int32).contiguous()
    if n_blocks > _PLACE_MAX_BLOCKS or topk > 64:
        return _inverted_table_by_sort(table, n_blocks)

    # chunks of 128 tokens, grown so there are at most 256 per row: the run
    # counts stay O(rows * 256) however long the sequence
    tc = _PLACE_SUB * -(-n_blocks // 256)
    n_chunks = -(-S // tc)
    nb_cap = max(64, 1 << (n_blocks - 1).bit_length())
    stream = torch.cuda.current_stream()
    runs = torch.empty(n_rows * n_chunks, dtype=torch.int32, device=device)
    entries = torch.empty(BG * S * topk, dtype=torch.int32, device=device)
    fns = _PLACE_CACHE.get((topk, nb_cap))
    args = (table, runs, entries, int(S), int(BG), int(n_blocks), int(n_chunks), int(tc), stream)
    if fns is None:
        fns = tuple(build_inverted_pass(topk, nb_cap, place).compile(*args) for place in (False, True))
        _PLACE_CACHE[(topk, nb_cap)] = fns

    fns[0](*args)  # runs <- counts
    ends = torch.cumsum(runs, 0, dtype=torch.int32)
    offsets = ends - runs
    ptr = torch.cat([offsets.view(n_rows, n_chunks)[:, 0], ends[-1:]])
    fns[1](table, offsets, entries, int(S), int(BG), int(n_blocks), int(n_chunks), int(tc), stream)
    return ptr, entries


def _inverted_table_by_sort(table: torch.Tensor, n_blocks: int):
    """:func:`build_inverted_table` by one stable sort: for more blocks than
    fit a counter each in LDS."""
    B, Hkv, S, topk = table.shape
    n_rows = B * Hkv * n_blocks
    rows = torch.arange(B * Hkv, device=table.device, dtype=torch.int32).view(B, Hkv, 1, 1) * n_blocks
    keys = torch.where(table >= 0, rows + table, n_rows).flatten()
    keys_sorted, order = torch.sort(keys, stable=True)
    bounds = torch.arange(n_rows + 1, device=table.device, dtype=torch.int32)
    ptr = torch.searchsorted(keys_sorted, bounds, out_int32=True)
    return ptr, order.to(torch.int32)


# ============================================================================
# host
# ============================================================================


_SEARCH_STEPS = 24  # binary-search depth: up to 2**24 CSR rows


def build_chunk_writer():
    """One thread per chunk: find its CSR row by binary search over
    ``CHPTR`` and write ``(row, begin, end)``, or zeros past the live chunks."""
    NTH = 256

    @flyc.kernel(known_block_size=[NTH, 1, 1])
    def k_fn(
        PTR: fx.Tensor, CHPTR: fx.Tensor, CHUNKS: fx.Tensor, NROWS: fx.Int32, NCH: fx.Int32, SIZE: fx.Int32
    ):
        c = fx.Int32(fx.Index(gpu.block_idx.x) * fx.Index(NTH) + fx.Index(gpu.thread_idx.x))
        rows_bytes = _raw((fx.Index(NROWS) + fx.Index(1)) * fx.Index(4))
        ptr_rsrc = buffer_ops.create_buffer_resource(PTR, max_size=False, num_records_bytes=rows_bytes)
        cp_rsrc = buffer_ops.create_buffer_resource(CHPTR, max_size=False, num_records_bytes=rows_bytes)
        ch_rsrc = buffer_ops.create_buffer_resource(
            CHUNKS, max_size=False, num_records_bytes=_raw(fx.Index(NCH) * fx.Index(3 * 4))
        )

        def load(rsrc, i):
            return fx.Int32(buffer_ops.buffer_load(rsrc, fx.Index(i), vec_width=1, dtype=fx.Int32))

        # last row whose first chunk is <= c
        lo = fx.Int32(0)
        hi = NROWS
        for _ in range_constexpr(_SEARCH_STEPS):
            mid = (lo + hi) // fx.Int32(2)
            go_right = ArithValue(load(cp_rsrc, mid) <= c)
            lo = fx.Int32(go_right.select(_raw(mid), _raw(lo)))
            hi = fx.Int32(go_right.select(_raw(hi), _raw(mid)))
        row = lo
        live = ArithValue(c < load(cp_rsrc, NROWS))
        begin = load(ptr_rsrc, row) + (c - load(cp_rsrc, row)) * SIZE
        end = begin + SIZE
        row_end = load(ptr_rsrc, row + fx.Int32(1))
        end = fx.Int32(ArithValue(end < row_end).select(_raw(end), _raw(row_end)))
        zero = fx.Int32(0)
        vals = [fx.Int32(live.select(_raw(x), _raw(zero))) for x in (row, begin, end)]
        in_range = ArithValue(c < NCH)
        for i, x in enumerate(vals):
            buffer_ops.buffer_store(
                x,
                ch_rsrc,
                (fx.Index(c) * fx.Index(3) + fx.Index(i)) * fx.Index(4),
                mask=_raw(in_range),
                offset_is_bytes=True,
            )

    @flyc.jit
    def launch(PTR, CHPTR, CHUNKS, NROWS, NCH, SIZE, stream):
        k_fn(PTR, CHPTR, CHUNKS, NROWS, NCH, SIZE).launch(
            grid=((fx.Index(NCH) + fx.Index(NTH - 1)) // fx.Index(NTH), 1, 1),
            block=(NTH, 1, 1),
            stream=stream,
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CHUNK_WRITER: list = []


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
    n_rows = ptr.numel() - 1
    assert n_rows < 2**_SEARCH_STEPS
    size = max(min_chunk, -(-max_tokens // target_chunks))
    per_row = torch.div(ptr[1:] - ptr[:-1] + (size - 1), size, rounding_mode="floor").clamp_min_(1)
    chunk_ptr = torch.cat([torch.zeros_like(ptr[:1]), torch.cumsum(per_row, 0, dtype=torch.int32)])

    n_chunks = n_rows + -(-max_tokens // size)
    chunks = torch.empty((n_chunks, 3), dtype=torch.int32, device=ptr.device)
    args = (ptr, chunk_ptr, chunks, int(n_rows), int(n_chunks), int(size), torch.cuda.current_stream())
    if not _CHUNK_WRITER:
        _CHUNK_WRITER.append(build_chunk_writer().compile(*args))
    _CHUNK_WRITER[0](*args)
    return chunks, chunk_ptr, n_chunks


class BlockPlan(NamedTuple):
    """The block table inverted and chunked, shared by every backward that walks
    it block by block: the attention's dK/dV and the indexer's dK.

    ``ptr``/``entries`` are :func:`build_inverted_table`'s CSR, ``chunks``,
    ``chunk_ptr`` and ``n_chunks`` :func:`plan_chunks`' split of it.
    """

    ptr: torch.Tensor
    entries: torch.Tensor
    chunks: torch.Tensor
    chunk_ptr: torch.Tensor
    n_chunks: int
    n_blocks: int


def build_block_plan(block_table: torch.Tensor, block_size: int = 128) -> BlockPlan:
    """:class:`BlockPlan` for a ``[B, Hkv, S, topk]`` block table (-1 padded)."""
    n_blocks = -(-block_table.shape[2] // block_size)
    ptr, entries = build_inverted_table(block_table, n_blocks)
    chunks, chunk_ptr, n_chunks = plan_chunks(ptr, entries.numel())
    return BlockPlan(ptr, entries, chunks, chunk_ptr, n_chunks, n_blocks)


_DQ_CACHE: dict = {}
_DKDV_CACHE: dict = {}
_RED_CACHE: dict = {}


def msa_token_bwd(dout, q, k, v, out, lse, block_table, softmax_scale=None, block_size=128, plan=None):
    """Gradients of ``msa_token_fwd``.

    Args:
        dout: ``[S, B, Hq, 128]`` bf16, the gradient of ``out``.
        q, k, v, block_table, softmax_scale, block_size: as passed to the forward.
        out, lse: the forward's outputs.
        plan: ``block_table``'s :class:`BlockPlan`, if the caller already built
            it; built here otherwise.

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
    if plan is None:
        plan = build_block_plan(block_table, block_size)
    n_blocks = plan.n_blocks
    assert n_blocks == -(-S // block_size)

    out = out.contiguous()

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

    n_chunks = plan.n_chunks
    ws = torch.empty((n_chunks, 2, block_size, D), dtype=torch.float32, device=q.device)
    kv_args = (
        q,
        k,
        v,
        dout,
        lse,
        delta,
        plan.chunks,
        plan.entries,
        ws,
        int(S),
        int(B),
        int(n_blocks),
        int(plan.entries.numel()),
        int(n_chunks),
        stream,
    )
    kv_key = (Hkv, block_size, float(softmax_scale), topk)
    fn = _DKDV_CACHE.get(kv_key)
    if fn is None:
        fn = build_dkdv(Hkv, block_size, float(softmax_scale), topk).compile(*kv_args)
        _DKDV_CACHE[kv_key] = fn
    fn(*kv_args)

    red_args = (ws, plan.chunk_ptr, dk, dv, int(S), int(B), int(n_blocks), int(n_chunks), stream)
    red_key = (Hkv, block_size)
    fn = _RED_CACHE.get(red_key)
    if fn is None:
        fn = build_dkdv_reduce(Hkv, block_size).compile(*red_args)
        _RED_CACHE[red_key] = fn
    fn(*red_args)
    return dq, dk, dv
