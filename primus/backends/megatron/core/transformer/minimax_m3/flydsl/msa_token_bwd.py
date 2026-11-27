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

from typing import NamedTuple, Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as _memref
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from primus_turbo.flydsl.utils import buffer_ops

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
    HPW,
    KS,
    MAX_BUFFER_BYTES,
    STRIDE,
    TK,
    D,
    check_buffer_bytes,
    kernel_name,
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
    RUN = 8 * LPR  # elements the LPR lanes of a row load per chunk, contiguously
    BUF = TK * STRIDE
    # One wave per work-group: its LDS accesses execute in program order, so a second
    # buffer would buy no overlap and only double the LDS footprint that caps occupancy.
    NBUF = 1

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dq_smem")
    k_off = allocator._align(allocator.ptr, 16)
    v_off = allocator._align(k_off + NBUF * BUF * 2, 16)
    allocator.ptr = allocator._align(v_off + NBUF * BUF * 2, 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1], name=kernel_name("msa_bwd_dq"))
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

        # lane -> (row = lane // LPR, chunk c at column (lane % LPR)*8 + c*RUN): for each
        # chunk the LPR lanes of a row read one contiguous run of the 256-byte row
        ld_row = lane // fx.Index(LPR)
        ld_col = (lane % fx.Index(LPR)) * fx.Index(8)
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
                buffer_ops.buffer_load(k_rsrc, src + fx.Index(c * RUN), vec_width=8, dtype=elem)
                for c in range_constexpr(NV8)
            ] + [
                buffer_ops.buffer_load(v_rsrc, src + fx.Index(c * RUN), vec_width=8, dtype=elem)
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
                Vec(rows[c]).store(lds_k, [buf + ld_lds + fx.Index(c * RUN)])
                Vec(rows[NV8 + c]).store(lds_v, [buf + ld_lds + fx.Index(c * RUN)])

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

    @flyc.kernel(
        known_block_size=[THREADS, 1, 1], name=kernel_name("msa_bwd_dkdv", "" if TPS == 4 else f"step{TPS}")
    )
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

    @flyc.kernel(known_block_size=[THREADS, 1, 1], name=kernel_name("msa_bwd_dkdv_reduce"))
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

    @flyc.kernel(
        known_block_size=[NTH, 1, 1], name=kernel_name("msa_plan_inverted", "place" if place else "count")
    )
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

    @flyc.kernel(known_block_size=[NTH, 1, 1], name=kernel_name("msa_plan_chunk_rows"))
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


# msa_bounded_plan_chunks: entries per chunk = ceil(capacity / target), kept between the two bounds,
# and never more than BOUNDED_CHUNK_COUNT chunks (plus one per row)
BOUNDED_CHUNK_TARGET = 4096
BOUNDED_CHUNK_MIN = 256
BOUNDED_CHUNK_MAX = 384
BOUNDED_CHUNK_COUNT = 8192
# FP32 dK and dV partials of one chunk (block 128, D 128): the largest per-chunk workspace of the plan's kernels
BOUNDED_CHUNK_WS_BYTES = 2 * 128 * 128 * 4


def plan_chunk_size(
    max_tokens: int, target_chunks: int = 2048, min_chunk: int = 64, bounded: bool = False
) -> int:
    """Entries per chunk of a :class:`BlockPlan` whose table holds ``max_tokens`` entries
    (``B * Hkv * S * topk``, its capacity).

    The reference rule aims at ``target_chunks`` chunks in all: ``max(min_chunk, ceil(max_tokens /
    target_chunks))``, 128 entries at S 4096 and 1024 at S 32768 (B 1, 4 KV heads, top-k 16).

    ``bounded`` (``msa_bounded_plan_chunks``): ``ceil(max_tokens / BOUNDED_CHUNK_TARGET)`` kept
    within ``[BOUNDED_CHUNK_MIN, BOUNDED_CHUNK_MAX]``. A chunk is one workgroup of the dK/dV and
    index dK kernels, and its size has two costs. Short chunks pay the workgroup's fixed part (the
    K/V prologue, the pipeline fill, 131 KB of FP32 partials and their reduce) too often; long
    chunks make a tail, because a workgroup takes its steps one after another and the kernel
    cannot end before its longest chunk does while the rows of a causal table are very uneven.
    Past ``BOUNDED_CHUNK_MAX * BOUNDED_CHUNK_COUNT`` entries the size grows again,
    ``ceil(max_tokens / BOUNDED_CHUNK_COUNT)``, so a plan never has more than that many chunks plus
    one per row and the kernels' FP32 partial workspace (131 KB per chunk for dK / dV) stays
    bounded as the reference rule's does. The constants do not depend on the shape, so the rule
    holds for any S, B and top-k."""
    if bounded:
        size = min(BOUNDED_CHUNK_MAX, max(BOUNDED_CHUNK_MIN, -(-max_tokens // BOUNDED_CHUNK_TARGET)))
        return max(size, -(-max_tokens // BOUNDED_CHUNK_COUNT))
    return max(min_chunk, -(-max_tokens // target_chunks))


def plan_chunk_size_of(
    n_rows: int, max_tokens: int, target_chunks: int = 2048, min_chunk: int = 64, bounded: bool = False
) -> int:
    """The chunk size :func:`plan_chunks` uses for a table of ``n_rows`` rows: :func:`plan_chunk_size`,
    except that a bounded plan whose worst-case chunk count would take the dK / dV partials past
    the kernels' 32-bit buffer limit falls back to the reference rule. An integer function of the
    shape, decided before any work; every builder of a :class:`BlockPlan` that can see such row
    counts (``plan_chunks``, ``select_plan``'s wide plan) takes its size from here."""
    size = plan_chunk_size(max_tokens, target_chunks, min_chunk, bounded)
    if bounded and (n_rows + -(-max_tokens // size)) * BOUNDED_CHUNK_WS_BYTES >= MAX_BUFFER_BYTES:
        # B * Hkv * n_blocks rows of 8192 and more: the reference rule
        size = plan_chunk_size(max_tokens, target_chunks, min_chunk)
    return size


def plan_chunks(
    ptr: torch.Tensor, max_tokens: int, target_chunks: int = 2048, min_chunk: int = 64, bounded: bool = False
):
    """Split every CSR row into equal-sized token chunks, about ``target_chunks`` in all
    (``bounded``: the ``msa_bounded_plan_chunks`` size, see :func:`plan_chunk_size`).

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
    size = plan_chunk_size_of(n_rows, max_tokens, target_chunks, min_chunk, bounded)
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
    # the int32 block table itself, when the plan's builder already made it (select_plan)
    table: Optional[torch.Tensor] = None


def build_block_plan(
    block_table: torch.Tensor, block_size: int = 128, bounded_chunks: bool = False
) -> BlockPlan:
    """:class:`BlockPlan` for a ``[B, Hkv, S, topk]`` block table (-1 padded).

    ``bounded_chunks`` (``msa_bounded_plan_chunks``): the chunk size of
    :func:`plan_chunk_size` with ``bounded``. The entries and their order are the same; the FP32
    partials of dK / dV and of the index dK are summed over other groups of them."""
    n_blocks = -(-block_table.shape[2] // block_size)
    ptr, entries = build_inverted_table(block_table, n_blocks)
    chunks, chunk_ptr, n_chunks = plan_chunks(ptr, entries.numel(), bounded=bounded_chunks)
    return BlockPlan(ptr, entries, chunks, chunk_ptr, n_chunks, n_blocks)


_DQ_CACHE: dict = {}
_DKDV_CACHE: dict = {}
_RED_CACHE: dict = {}

# msa_dq_from_ds: the first-call verdict per shape, the tolerance of that comparison (relative
# Frobenius error of dq / dk / dv between the two paths; they agree to ~1e-4 when both are right)
# and the largest dS buffer the path may allocate
_DS_OK: dict = {}
_DS_CHECK_TOL = 1.0e-3
_DS_MAX_BYTES = 16 * 2**30
# msa_ds_chunk_store: the dK/dV kernel addresses a (batch, KV head) slice of the dS buffer with 32-bit
# byte offsets through one descriptor; larger slices keep one descriptor per record
_DS_SLICE_MAX_BYTES = 2**31


def _ds_eligible(B, Hkv, S, topk, block_size, block_table, walk_slots) -> bool:
    """Shapes ``dq_from_ds`` takes: the dS buffer (4 KB per table entry at block size 128) fits
    ``_DS_MAX_BYTES``, and the walk's slot table, when there is one, is the int32 table's twin."""
    if B * Hkv * S * topk * block_size * HPW * 2 > _DS_MAX_BYTES:
        return False
    if walk_slots is not None and (
        walk_slots.shape != block_table.shape
        or walk_slots.dtype != torch.int32
        or not walk_slots.is_contiguous()
    ):
        return False
    return True


_DS_NOT_TAKEN_LOGGED: set = set()


def _log_ds_not_taken(B, Hkv, S, topk, block_size, walk_slots) -> None:
    """Once per shape: why ``_ds_eligible`` keeps the dQ kernel that recomputes S and dS."""
    key = (S, B, Hkv, topk, block_size, walk_slots is not None)
    if key in _DS_NOT_TAKEN_LOGGED:
        return
    _DS_NOT_TAKEN_LOGGED.add(key)
    from primus.core.utils.module_utils import log_rank_0

    size = B * Hkv * S * topk * block_size * HPW * 2
    if size > _DS_MAX_BYTES:
        why = f"its dS buffer would take {size / 2**30:.1f} GiB (limit {_DS_MAX_BYTES / 2**30:.0f} GiB)"
    else:
        why = "the walk's slot table is not the int32 table's twin"
    log_rank_0(f"[msa_dq_from_ds {key}] not taken: {why}; the dQ kernel recomputes S and dS")


# msa_dkdv_two_token_steps: the gfx1250 dK/dV kernel's 4-token step needs 8 waves of 16 keys
# (sparse_block_size 128); the 2-token layout is offered for exactly that kernel
_TWO_TOKEN_STEP_BLOCK = 128


def dkdv_two_token_steps_eligible(block_size: int, w32: bool) -> bool:
    """``msa_dkdv_two_token_steps``: whether the dK/dV kernel with two tokens per step may stand
    in for the one with four. A rule on the kernel family and the block size alone (the layout
    is the same at every S, B and selection; a longer table only means more chunks of the same
    kind), decided before any buffer or kernel is made."""
    return bool(w32) and int(block_size) == _TWO_TOKEN_STEP_BLOCK


def _ds_first_call(key, args, kwargs, walk_slots):
    """Both paths on the first call's tensors: use ``dq_from_ds`` for this shape only if two of
    its runs give the same bits and its ``dq``, ``dk`` and ``dv`` are finite and within
    ``_DS_CHECK_TOL`` of the reference kernels'. Returns the gradients of the path that is kept.
    Nothing is decided while the reference gradients are all zero (nothing to compare). A GPU
    fault is raised, any other error keeps the reference kernels."""
    from primus.backends.megatron.core.fusions import bitwise_check
    from primus.core.utils.module_utils import log_rank_0, warning_rank_all

    want = msa_token_bwd(*args, **kwargs)
    norms = [w.double().norm() for w in want]
    if not any(bool(n) for n in norms):
        return want
    ok, why, got = False, "", None
    try:
        got = msa_token_bwd(*args, **kwargs, dq_from_ds="force", walk_slots=walk_slots)
        again = msa_token_bwd(*args, **kwargs, dq_from_ds="force", walk_slots=walk_slots)
        same = all(bitwise_check.same_bits(g, a) for g, a in zip(got, again))
        errs = [
            ((g.double() - w.double()).norm() / n.clamp_min(1e-30)).item()
            for g, w, n in zip(got, want, norms)
        ]
        finite = all(bool(torch.isfinite(g.float()).all()) for g in got)
        ok = same and finite and all(e <= _DS_CHECK_TOL for e in errs)
        why = "relative difference dq / dk / dv " + " / ".join(f"{e:.2e}" for e in errs)
        why += f", two runs identical: {same}"
    except Exception as e:  # noqa: BLE001 - any failure but a GPU fault keeps the reference kernels
        if bitwise_check._device_error(e):
            raise
        why = f"{type(e).__name__}: {e}"
    _DS_OK[key] = ok
    if ok:
        log_rank_0(f"[msa_dq_from_ds {key}] agrees with the reference dQ kernel ({why}); in use")
        return got
    warning_rank_all(f"[msa_dq_from_ds {key}] not used ({why}); keeping the reference dQ kernel")
    return want


def msa_token_bwd(
    dout,
    q,
    k,
    v,
    out,
    lse,
    block_table,
    softmax_scale=None,
    block_size=128,
    plan=None,
    shared_tiles=False,
    step_mask_skip=False,
    dq_from_ds=False,
    walk_slots=None,
    ds_chunk_store=False,
    dq_early_issue=False,
    dkdv_two_token_steps=False,
):
    """Gradients of ``msa_token_fwd``.

    Args:
        dout: ``[S, B, Hq, 128]`` bf16, the gradient of ``out``.
        q, k, v, block_table, softmax_scale, block_size: as passed to the forward.
        out, lse: the forward's outputs.
        plan: ``block_table``'s :class:`BlockPlan`, if the caller already built
            it; built here otherwise.
        shared_tiles: gfx1250 only (``msa_shared_tile_staging``): the dQ kernel runs adjacent
            tokens in one workgroup with one staging of each K/V tile they share. The first
            call for a shape runs both layouts on that call's tensors and compares every bit of
            ``dq`` and ``delta`` (``bitwise_check``); on a mismatch the run keeps one token per
            workgroup.

        step_mask_skip: gfx1250 only (``msa_lean_masks``): the dQ kernel's fully visible steps
            skip the per-key causal compare and select (``build_dq_w32``). Checked like
            ``shared_tiles``: the first call for a shape compares every bit of ``dq`` and ``delta``
            with the kernel without it and keeps that kernel on a mismatch.

        dq_from_ds: gfx1250 only (``msa_dq_from_ds``, ``msa_bwd_store_ds``): the dK/dV kernel also
            writes its bf16 ``dS`` operand per (token, selected block) and a slim dQ kernel
            (``build_dq_ds_w32``) forms ``dQ = scale * dS K`` from it, instead of the dQ kernel
            staging V and recomputing ``S`` and ``dS`` itself; ``delta`` comes from one Triton
            launch. Same terms, the ``dS`` of the dK/dV kernel instead of the dQ kernel's own (they
            differ in the last bit of ``S``) and ``delta`` summed in another order. Deterministic:
            every ``dS`` record is written by exactly one workgroup. The first call for a shape
            runs both paths on that call's tensors and keeps the reference kernels unless two runs
            of ``dq_from_ds`` give the same bits and its ``dq``, ``dk`` and ``dv`` agree within
            ``_DS_CHECK_TOL`` (and are finite). Shapes whose ``dS`` buffer would exceed
            ``_DS_MAX_BYTES`` keep the reference kernels (logged once per shape).
        ds_chunk_store: with ``dq_from_ds`` (``msa_ds_chunk_store``): the same ``dS`` bytes and the same
            arithmetic: the dK/dV kernel stores ``dS`` through one buffer descriptor per
            workgroup instead of one per record (``build_dkdv_w32``), only while a (batch, KV
            head) slice of the buffer is at most ``_DS_SLICE_MAX_BYTES``; larger shapes keep the
            per-record stores. Bit-identical to ``dq_from_ds`` without it (unit-tested); the
            first-call comparison of ``dq_from_ds`` runs with it.
        dq_early_issue: with ``dq_from_ds`` (``msa_dq_early_issue``): the slim dQ kernel issues the
            next step's copies before it waits for the current step's (``build_dq_ds_w32``,
            ``early_issue``; lockstep layout only). Same tiles and arithmetic: the first call for a
            shape compares every bit of ``dq`` with the kernel without it, on that call's ``dS``
            (``bitwise_check``), and keeps that kernel on a mismatch.
        walk_slots: with ``dq_from_ds`` and a permuted ``block_table`` (``msa_aligned_block_walk``),
            the int32 ``[B, Hkv, S, topk]`` slot of each walk position in the table ``plan`` was
            built from.
        dkdv_two_token_steps: gfx1250 only (``msa_dkdv_two_token_steps``): the dK/dV kernel takes
            two tokens per step instead of four (``build_dkdv_w32``, ``tokens_per_step``). Same
            entries, accumulators and order; the first call for a shape compares every bit of
            the fp32 chunk partials (and of dS with ``dq_from_ds``) with the four-token kernel
            and keeps that one on a mismatch.

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
        if walk_slots is not None:
            raise ValueError("walk_slots needs the plan of the table the walk permutes")
        plan = build_block_plan(block_table, block_size)
    n_blocks = plan.n_blocks
    assert n_blocks == -(-S // block_size)

    w32 = "gfx1250" in get_hip_arch()
    # msa_dq_from_ds: eligibility first (gfx1250, a dS buffer that fits), then one comparison of
    # the two paths per shape; "force" is that comparison's own call
    use_ds = False
    ds_ok = bool(dq_from_ds) and w32 and _ds_eligible(B, Hkv, S, topk, block_size, block_table, walk_slots)
    if dq_from_ds and w32 and not ds_ok:
        _log_ds_not_taken(B, Hkv, S, topk, block_size, walk_slots)
    if ds_ok:
        if dq_from_ds == "force":
            use_ds = True
        else:
            args = (dout, q, k, v, out, lse, block_table, softmax_scale, block_size)
            kwargs = dict(plan=plan, shared_tiles=shared_tiles, step_mask_skip=step_mask_skip)
            if ds_chunk_store:
                kwargs["ds_chunk_store"] = True  # the comparison runs the variant that will be used
            if dq_early_issue:
                kwargs["dq_early_issue"] = True
            if dkdv_two_token_steps:
                kwargs["dkdv_two_token_steps"] = True
            key = (
                S,
                B,
                Hkv,
                topk,
                block_size,
                float(softmax_scale),
                walk_slots is not None,
                bool(ds_chunk_store),
                bool(dq_early_issue),
                bool(dkdv_two_token_steps),
            )
            decided = _DS_OK.get(key)
            if decided is None:
                return _ds_first_call(key, args, kwargs, walk_slots)
            use_ds = decided

    out = out.contiguous()

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    n_chunks = plan.n_chunks
    ws = torch.empty((n_chunks, 2, block_size, D), dtype=torch.float32, device=q.device)
    check_buffer_bytes(q=q, k=k, v=v, block_table=block_table, entries=plan.entries, workspace=ws)
    stream = torch.cuda.current_stream()

    def run_dq(tpw, skip=False):
        if tpw > 1 and (S % tpw or block_size % tpw):
            # the lockstep layout's barrier count is workgroup-uniform only for these shapes
            raise ValueError(f"msa_shared_tile_staging dQ: S={S} block_size={block_size}")
        dq = torch.empty_like(q)
        delta = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)  # written by the dq kernel
        dq_args = (q, k, v, dout, out, lse, delta, block_table, dq, int(S), int(B), stream)
        dq_key = (Hkv, topk, block_size, float(softmax_scale), tpw)
        if skip:
            dq_key = ("step_mask_skip",) + dq_key  # tokens_per_wg stays the last element
        fn = _DQ_CACHE.get(dq_key)
        if fn is None:
            if w32:
                from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd_w32 import (
                    build_dq_w32,
                )

                built = build_dq_w32(
                    Hkv,
                    topk,
                    block_size,
                    float(softmax_scale),
                    tokens_per_wg=tpw,
                    **({"step_mask_skip": True} if skip else {}),
                )
            else:
                built = build_dq(Hkv, topk, block_size, float(softmax_scale))
            fn = built.compile(*dq_args)
            _DQ_CACHE[dq_key] = fn
        fn(*dq_args)
        return dq, delta

    # eligibility first: the shared layout is the gfx1250 kernel's and needs an S / block size it divides
    if use_ds:
        # delta first (the dQ kernel no longer makes it); dQ runs after dK/dV
        from primus.backends.megatron.core.transformer.minimax_m3.flydsl.row_dot import (
            row_dot,
        )

        delta = row_dot(dout, out)
        dq = None
    else:
        tpw = 1
        if shared_tiles and w32:
            from primus.backends.megatron.core.fusions import bitwise_check
            from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd_w32 import (
                dq_tokens_per_wg,
            )

            tpw = dq_tokens_per_wg(int(S), block_size)
            if tpw > 1 and not bitwise_check.passed(
                "msa_shared_tile_staging dQ",
                lambda: run_dq(tpw),
                lambda: run_dq(1),
                # everything the kernel is built from or lays its grid out by
                key=(S, B, Hkv, topk, block_size, float(softmax_scale), tpw),
            ):
                tpw = 1
        skip = False
        if step_mask_skip and w32:
            from primus.backends.megatron.core.fusions import bitwise_check

            # msa_lean_masks: decided once per shape, against the same layout without it
            skip = bitwise_check.passed(
                "msa_lean_masks dQ",
                lambda: run_dq(tpw, True),
                lambda: run_dq(tpw, False),
                key=(S, B, Hkv, topk, block_size, float(softmax_scale), tpw),
            )
        dq, delta = run_dq(tpw, skip)

    def run_kv(skip, ws_out, ds_out=None, two=False):
        emit = ds_out is not None
        kv_args = (q, k, v, dout, lse, delta, plan.chunks, plan.entries, ws_out)
        if w32:
            # build_dkdv_w32 takes a DS operand (written only with emit_ds); build_dkdv has none
            kv_args += (ds_out if emit else torch.empty(16, dtype=torch.bfloat16, device=q.device),)
        kv_args += (int(S), int(B), int(n_blocks), int(plan.entries.numel()), int(n_chunks), stream)
        kv_key = (Hkv, block_size, float(softmax_scale), topk)
        if skip:
            kv_key = ("step_mask_skip",) + kv_key
        if emit:
            kv_key = ("emit_ds",) + kv_key
        # msa_ds_chunk_store: one dS descriptor per workgroup needs a (batch, KV head) slice of at most 2 GiB
        chunk_store = emit and bool(ds_chunk_store) and S * topk * block_size * HPW * 2 <= _DS_SLICE_MAX_BYTES
        if chunk_store:
            kv_key = ("ds_chunk_store",) + kv_key
        if two:
            kv_key = ("two_token_steps",) + kv_key
        fn = _DKDV_CACHE.get(kv_key)
        if fn is None:
            if w32:
                from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_bwd_w32 import (
                    build_dkdv_w32,
                )

                built = build_dkdv_w32(
                    Hkv,
                    block_size,
                    float(softmax_scale),
                    topk,
                    **({"step_mask_skip": True} if skip else {}),
                    **({"emit_ds": True} if emit else {}),
                    **({"ds_chunk_store": True} if chunk_store else {}),
                    **({"tokens_per_step": 2} if two else {}),
                )
            else:
                built = build_dkdv(Hkv, block_size, float(softmax_scale), topk)
            fn = built.compile(*kv_args)
            _DKDV_CACHE[kv_key] = fn
        fn(*kv_args)
        return (ws_out, ds_out) if emit else (ws_out,)

    skip = False
    if step_mask_skip and w32:
        from primus.backends.megatron.core.fusions import bitwise_check

        # the dK/dV kernel's own check: its fp32 chunk partials, bit for bit
        skip = bitwise_check.passed(
            "msa_lean_masks dK/dV",
            lambda: run_kv(True, torch.empty_like(ws)),
            lambda: run_kv(False, torch.empty_like(ws)),
            key=(S, B, Hkv, topk, block_size, float(softmax_scale), int(n_chunks)),
        )
    ds = None
    if use_ds:
        from primus.backends.megatron.core.transformer.minimax_m3.flydsl import (
            msa_token_bwd_w32 as _w32,
        )

        # one [block_size keys, 16 heads] bf16 record per table entry; dK/dV writes the live ones
        ds = torch.empty((B * Hkv * S * topk, block_size, HPW), dtype=torch.bfloat16, device=q.device)
    two = False
    # eligibility first (kernel family, block size), then one comparison per shape and kernel variant
    if dkdv_two_token_steps and dkdv_two_token_steps_eligible(block_size, w32):
        from primus.backends.megatron.core.fusions import bitwise_check

        def run_kv_probe(two_tokens):
            # the variant this call is about to use (lean masks, dS stores), on this call's tensors:
            # the fp32 chunk partials and, with msa_dq_from_ds, every dS record (zeroed first: the
            # kernels write the live records only). The four-token probe writes into ws and ds
            # themselves, which the launch below rewrites, so the check holds one extra copy of
            # each; compared as integers (every bit, without same_bits' NaN masks and copies).
            ws_probe = torch.empty_like(ws) if two_tokens else ws
            ds_probe = None
            if use_ds:
                ds_probe = torch.empty_like(ds) if two_tokens else ds
                ds_probe.zero_()
            got = run_kv(skip, ws_probe, ds_probe, two=two_tokens)
            return [got[0].view(torch.int32)] + ([got[1].view(torch.int16)] if use_ds else [])

        two = bitwise_check.passed(
            "msa_dkdv_two_token_steps dK/dV",
            lambda: run_kv_probe(True),
            lambda: run_kv_probe(False),
            key=(
                S,
                B,
                Hkv,
                topk,
                block_size,
                float(softmax_scale),
                int(n_chunks),
                skip,
                use_ds,
                bool(ds_chunk_store),
            ),
        )
    run_kv(skip, ws, ds, two=two)
    if use_ds:
        tpw2 = _w32.dq_ds_tokens_per_wg(int(S), block_size)
        if tpw2 > 1 and (S % tpw2 or block_size % tpw2):
            # the lockstep layout's barrier count is workgroup-uniform only for these shapes
            raise ValueError(f"msa_dq_from_ds dQ: S={S} block_size={block_size}")
        slots = walk_slots if walk_slots is not None else block_table  # unused without slots

        def run_dq_ds(early):
            dq_out = torch.empty_like(q)
            ds_args = (k, ds, block_table, slots, dq_out, int(S), int(B), stream)
            ds_key = ("dq_from_ds", walk_slots is not None, Hkv, topk, block_size, float(softmax_scale), tpw2)
            if early:
                ds_key = ("early_issue",) + ds_key  # tokens_per_wg stays the last element
            fn = _DQ_CACHE.get(ds_key)
            if fn is None:
                fn = _w32.build_dq_ds_w32(
                    Hkv,
                    topk,
                    block_size,
                    float(softmax_scale),
                    tokens_per_wg=tpw2,
                    with_slots=walk_slots is not None,
                    **({"early_issue": True} if early else {}),
                ).compile(*ds_args)
                _DQ_CACHE[ds_key] = fn
            fn(*ds_args)
            return (dq_out,)

        early = False
        if dq_early_issue and tpw2 > 1:
            from primus.backends.megatron.core.fusions import bitwise_check

            # msa_dq_early_issue: decided once per shape, on this call's dS, against the kernel without it
            early = bitwise_check.passed(
                "msa_dq_early_issue dQ",
                lambda: run_dq_ds(True),
                lambda: run_dq_ds(False),
                key=(S, B, Hkv, topk, block_size, float(softmax_scale), walk_slots is not None, tpw2),
            )
        (dq,) = run_dq_ds(early)

    red_args = (ws, plan.chunk_ptr, dk, dv, int(S), int(B), int(n_blocks), int(n_chunks), stream)
    red_key = (Hkv, block_size)
    fn = _RED_CACHE.get(red_key)
    if fn is None:
        fn = build_dkdv_reduce(Hkv, block_size).compile(*red_args)
        _RED_CACHE[red_key] = fn
    fn(*red_args)
    return dq, dk, dv
