###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 MSA backward: the dQ and dK/dV kernels of ``msa_token_bwd`` on wave32 WMMA.

dQ is one wave32 per token. dK/dV is one wave per 16 keys of a block; the two
lane halves split K on QK, and each holds 8 of the 16 heads. The dS/P
contraction is the forward PV schedule (K=16 in the low half of a K=32 WMMA,
transposed operand gathered from LDS). dK/dV partials use the gfx950 layout, a
dense ``[chunk, 2, block, D]`` fp32 buffer, so the same reduce kernel sums them.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
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
    STRIDE,
    TK,
    D,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_w32 import (
    cat_bf16_16,
    crossgrp_sum,
    pack_low16_bf16,
    wmma_f32_16x16x32_bf16,
)

_LOG2E = 1.4426950408889634
DT = D // 16
WAVE = 32


def _andb(a, b):
    return ArithValue(arith.AndIOp(_raw(a), _raw(b)).result)


def _sum_elems(vec, n):
    total = fx.Float32(_raw(Vec(vec)[0]))
    for i in range_constexpr(1, n):
        total = total + fx.Float32(_raw(Vec(vec)[i]))
    return total


def _store_bf16x8(rsrc, elem_offset, v8f, scale):
    scale_v = Vec.from_elements([scale] * 8, fx.Float32)
    ov = Vec(v8f) * scale_v
    pk = [rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2 * i]), _raw(Vec(ov)[2 * i + 1])) for i in range_constexpr(4)]
    buffer_ops.buffer_store(
        _raw(Vec.from_elements([fx.Int32(_raw(w)) for w in pk], fx.Int32)),
        rsrc,
        elem_offset * fx.Index(2),
        offset_is_bytes=True,
    )


def _store_f32x8(rsrc, elem_offset, v8f):
    values = Vec(v8f)
    for half in range_constexpr(2):
        chunk = Vec.from_elements([values[half * 4 + i] for i in range_constexpr(4)], fx.Float32)
        buffer_ops.buffer_store(
            _raw(chunk),
            rsrc,
            (elem_offset + fx.Index(half * 4)) * fx.Index(4),
            offset_is_bytes=True,
        )


def build_dq_w32(num_kv_heads: int, topk: int, block_size: int, scale: float):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    assert block_size % TK == 0
    SPB = block_size // TK
    # 32 lanes cover 16 key rows, two lanes per row, 8 bf16 at a time.
    LPR = WAVE // TK
    RUN = 8 * LPR
    NV8 = D // RUN

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dq_w32_smem")
    k_off = allocator._align(allocator.ptr, 16)
    v_off = allocator._align(k_off + TK * STRIDE * 2, 16)
    allocator.ptr = allocator._align(v_off + TK * STRIDE * 2, 16)

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
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
        v1 = Vec.make_type(1, elem)
        lds_k = SmemPtr(allocator.get_base(), k_off, elem.ir_type, shape=(TK * STRIDE,)).get()
        lds_v = SmemPtr(allocator.get_base(), v_off, elem.ir_type, shape=(TK * STRIDE,)).get()

        lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)
        grp0 = ArithValue(half == fx.Index(0))
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
        c_bf0 = fx.BFloat16(0.0)

        head = g * fx.Index(HPW) + lo
        row = (tok * Bn + b) * fx.Index(Hq) + head
        q_packs, do_packs = [], []
        for ks in range_constexpr(KS):
            base = row * fx.Index(D) + fx.Index(ks * 32) + half * fx.Index(16)
            q_packs.append(
                cat_bf16_16(
                    buffer_ops.buffer_load(q_rsrc, base, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(q_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
                )
            )
            do_packs.append(
                cat_bf16_16(
                    buffer_ops.buffer_load(do_rsrc, base, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(do_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
                )
            )
        lse2 = fx.Float32(buffer_ops.buffer_load(lse_rsrc, row, vec_width=1, dtype=fx.Float32)) * c_log2e

        # delta = rowsum(dO * O). Each lane owns 16 of every 32-d slice; the two
        # halves cover the row.
        v8f = Vec.make_type(8, fx.Float32)
        part = fx.Float32(0.0)
        for ks in range_constexpr(KS):
            base = row * fx.Index(D) + fx.Index(ks * 32) + half * fx.Index(16)
            o_pack = cat_bf16_16(
                buffer_ops.buffer_load(o_rsrc, base, vec_width=8, dtype=elem),
                buffer_ops.buffer_load(o_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
            )
            d0 = Vec.from_elements([Vec(do_packs[ks])[i] for i in range_constexpr(8)], elem)
            d1 = Vec.from_elements([Vec(do_packs[ks])[i] for i in range_constexpr(8, 16)], elem)
            o0 = Vec.from_elements([Vec(o_pack)[i] for i in range_constexpr(8)], elem)
            o1 = Vec.from_elements([Vec(o_pack)[i] for i in range_constexpr(8, 16)], elem)
            p0 = Vec(arith.ExtFOp(v8f, _raw(d0)).result) * Vec(arith.ExtFOp(v8f, _raw(o0)).result)
            p1 = Vec(arith.ExtFOp(v8f, _raw(d1)).result) * Vec(arith.ExtFOp(v8f, _raw(o1)).result)
            part = part + _sum_elems(p0, 8) + _sum_elems(p1, 8)
        delta = crossgrp_sum(part)
        buffer_ops.buffer_store(delta, dl_rsrc, row * fx.Index(4), mask=_raw(grp0), offset_is_bytes=True)

        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = fx.Int32(ArithValue(n_vis < fx.Int32(topk)).select(_raw(n_vis), _raw(fx.Int32(topk))))
        n_steps = fx.Index(n_valid) * fx.Index(SPB)
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)

        def kv_offset(key):
            return ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)

        ld_row = lane // fx.Index(LPR)
        ld_col = (lane % fx.Index(LPR)) * fx.Index(8)

        def stage(kv0):
            for c in range_constexpr(NV8):
                col = ld_col + fx.Index(c * RUN)
                src = kv_offset(fx.Index(kv0) + ld_row) + col
                at = ld_row * fx.Index(STRIDE) + col
                Vec(buffer_ops.buffer_load(k_rsrc, src, vec_width=8, dtype=elem)).store(lds_k, [at])
                Vec(buffer_ops.buffer_load(v_rsrc, src, vec_width=8, dtype=elem)).store(lds_v, [at])

        def lds16(mem, row_e):
            return cat_bf16_16(Vec.load(v8, mem, [row_e]), Vec.load(v8, mem, [row_e + fx.Index(8)]))

        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        init = [c_zero_v8 for _ in range_constexpr(DT)]
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            dq_acc = [it[dt] for dt in range_constexpr(DT)]
            blk = fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + j // fx.Index(SPB), vec_width=1, dtype=fx.Int32)
            )
            blk_ok = ArithValue(blk >= fx.Int32(0))
            safe = fx.Int32(blk_ok.select(_raw(blk), _raw(fx.Int32(0))))
            kv0 = safe * fx.Int32(block_size) + fx.Int32(j % fx.Index(SPB)) * fx.Int32(TK)
            stage(kv0)
            gpu.barrier()

            s_acc = Vec.filled(8, 0.0, fx.Float32)
            dp_acc = Vec.filled(8, 0.0, fx.Float32)
            for ks in range_constexpr(KS):
                a_row = lo * fx.Index(STRIDE) + fx.Index(ks * 32) + half * fx.Index(16)
                s_acc = wmma_f32_16x16x32_bf16(lds16(lds_k, a_row), q_packs[ks], s_acc)
                dp_acc = wmma_f32_16x16x32_bf16(lds16(lds_v, a_row), do_packs[ks], dp_acc)

            # lane holds keys (half*8 + i) of head lo
            ds = []
            key0 = kv0 + fx.Int32(half) * fx.Int32(8)
            for i in range_constexpr(8):
                ok = _andb(blk_ok, ArithValue(key0 + fx.Int32(i) <= tok_i32))
                score = fx.Float32(_raw(Vec(s_acc)[i]))
                prob = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(score * c_sl - lse2)))
                prob = fx.Float32(ok.select(_raw(prob), _raw(c_zero)))
                dp = fx.Float32(_raw(Vec(dp_acc)[i]))
                ds.append(prob * (dp - delta))
            dsB = pack_low16_bf16(ds, grp0)

            new_dq = []
            for dt in range_constexpr(DT):
                d_col = fx.Index(dt * 16) + lo
                a_elems = []
                for i in range_constexpr(16):
                    ld = fx.BFloat16(_raw(Vec.load(v1, lds_k, [fx.Index(i) * fx.Index(STRIDE) + d_col])[0]))
                    a_elems.append(fx.BFloat16(grp0.select(_raw(ld), _raw(c_bf0))))
                new_dq.append(wmma_f32_16x16x32_bf16(Vec.from_elements(a_elems, elem), dsB, dq_acc[dt]))
            results = yield new_dq

        for dt in range_constexpr(DT):
            _store_bf16x8(
                dq_rsrc,
                row * fx.Index(D) + fx.Index(dt * 16) + half * fx.Index(8),
                results[dt],
                fx.Float32(scale),
            )

    @flyc.jit
    def launch(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B).launch(
            grid=(fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1), block=(WAVE, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_dkdv_w32(num_kv_heads: int, block_size: int, scale: float, topk: int, tokens_per_step: int = 4):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    waves = block_size // TK
    threads = WAVE * waves
    tps = tokens_per_step
    rows = tps * HPW
    vec = 8
    np = rows * D // (threads * vec)
    assert rows * D % (threads * vec) == 0 and rows <= threads
    buf = rows * STRIDE

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_bwd_dkdv_w32_smem")
    q_off = allocator._align(allocator.ptr, 16)
    do_off = allocator._align(q_off + buf * 2, 16)
    lse_off = allocator._align(do_off + buf * 2, 16)
    dl_off = allocator._align(lse_off + threads * 4, 16)
    allocator.ptr = allocator._align(dl_off + threads * 4, 16)

    @flyc.kernel(known_block_size=[threads, 1, 1])
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
        v1 = Vec.make_type(1, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_q = SmemPtr(allocator.get_base(), q_off, elem.ir_type, shape=(buf,)).get()
        lds_do = SmemPtr(allocator.get_base(), do_off, elem.ir_type, shape=(buf,)).get()
        lds_lse = SmemPtr(allocator.get_base(), lse_off, fx.Float32.ir_type, shape=(threads,)).get()
        lds_dl = SmemPtr(allocator.get_base(), dl_off, fx.Float32.ir_type, shape=(threads,)).get()

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(WAVE)
        lane = tid % fx.Index(WAVE)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)
        grp0 = ArithValue(half == fx.Index(0))
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
        c_bf0 = fx.BFloat16(0.0)

        # this wave's 16 keys. Both halves share the key; they split its K.
        kv0 = blk * fx.Index(block_size) + wave * fx.Index(TK)
        key = kv0 + lo
        key_i32 = fx.Int32(key)
        kv_row = ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)
        k_ops, v_ops = [], []
        for ks in range_constexpr(KS):
            base = kv_row + fx.Index(ks * 32) + half * fx.Index(16)
            k_ops.append(
                cat_bf16_16(
                    buffer_ops.buffer_load(k_rsrc, base, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(k_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
                )
            )
            v_ops.append(
                cat_bf16_16(
                    buffer_ops.buffer_load(v_rsrc, base, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(v_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
                )
            )

        st_flat = [(fx.Index(p * threads) + tid) * fx.Index(vec) for p in range_constexpr(np)]
        st_rows = [f // fx.Index(D) for f in st_flat]
        st_cols = [f % fx.Index(D) for f in st_flat]
        ld_row = tid % fx.Index(rows)

        def pick(vals, j):
            out = vals[0]
            for jj in range_constexpr(tps - 1):
                out = fx.Int32(ArithValue(j == fx.Index(jj + 1)).select(_raw(vals[jj + 1]), _raw(out)))
            return out

        def load_tokens(r0):
            tok = []
            for j in range_constexpr(tps):
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
            q_st, do_st = [], []
            for p in range_constexpr(np):
                src = (
                    head_row(pick(tok, st_rows[p] // fx.Index(HPW)), st_rows[p] % fx.Index(HPW)) * fx.Index(D)
                    + st_cols[p]
                )
                q_st.append(buffer_ops.buffer_load(q_rsrc, src, vec_width=vec, dtype=elem))
                do_st.append(buffer_ops.buffer_load(do_rsrc, src, vec_width=vec, dtype=elem))
            lrow = head_row(pick(tok, ld_row // fx.Index(HPW)), ld_row % fx.Index(HPW))
            lse_st = fx.Float32(buffer_ops.buffer_load(lse_rsrc, lrow, vec_width=1, dtype=fx.Float32))
            dl_st = fx.Float32(buffer_ops.buffer_load(dl_rsrc, lrow, vec_width=1, dtype=fx.Float32))
            return q_st, do_st, lse_st, dl_st

        def lds16(mem, row_e):
            return cat_bf16_16(Vec.load(v8, mem, [row_e]), Vec.load(v8, mem, [row_e + fx.Index(8)]))

        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        init = [c_zero_v8 for _ in range_constexpr(2 * DT)]
        results = init
        for i, it in range(fx.Index(0), n_tok, fx.Index(tps), init=init):
            dk_acc = [it[dt] for dt in range_constexpr(DT)]
            dv_acc = [it[DT + dt] for dt in range_constexpr(DT)]
            r0 = t_begin + fx.Int32(i)
            tok = load_tokens(r0)
            valid = [ArithValue(r0 + fx.Int32(j) < t_end) for j in range_constexpr(tps)]
            q_st, do_st, lse_st, dl_st = load_rows(tok)

            gpu.barrier()
            for p in range_constexpr(np):
                at = st_rows[p] * fx.Index(STRIDE) + st_cols[p]
                Vec(q_st[p]).store(lds_q, [at])
                Vec(do_st[p]).store(lds_do, [at])
            _memref.store(_raw(lse_st), lds_lse, [_raw(tid)])
            _memref.store(_raw(dl_st), lds_dl, [_raw(tid)])
            gpu.barrier()

            for j in range_constexpr(tps):
                s_acc = Vec.filled(8, 0.0, fx.Float32)
                dp_acc = Vec.filled(8, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    a_row = (
                        (fx.Index(j * HPW) + lo) * fx.Index(STRIDE) + fx.Index(ks * 32) + half * fx.Index(16)
                    )
                    s_acc = wmma_f32_16x16x32_bf16(lds16(lds_q, a_row), k_ops[ks], s_acc)
                    dp_acc = wmma_f32_16x16x32_bf16(lds16(lds_do, a_row), v_ops[ks], dp_acc)
                lse_base = fx.Index(j * HPW) + half * fx.Index(8)
                lse0 = Vec.load(v4f, lds_lse, [lse_base])
                lse1 = Vec.load(v4f, lds_lse, [lse_base + fx.Index(4)])
                dl0 = Vec.load(v4f, lds_dl, [lse_base])
                dl1 = Vec.load(v4f, lds_dl, [lse_base + fx.Index(4)])
                lse8 = [lse0[0], lse0[1], lse0[2], lse0[3], lse1[0], lse1[1], lse1[2], lse1[3]]
                dl8 = [dl0[0], dl0[1], dl0[2], dl0[3], dl1[0], dl1[1], dl1[2], dl1[3]]

                ok = _andb(ArithValue(key_i32 <= tok[j]), valid[j])
                probs, dss = [], []
                for r in range_constexpr(8):
                    score = fx.Float32(_raw(Vec(s_acc)[r]))
                    lse_r = fx.Float32(_raw(lse8[r]))
                    prob = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(score * c_sl - lse_r * c_log2e)))
                    prob = fx.Float32(ok.select(_raw(prob), _raw(c_zero)))
                    probs.append(prob)
                    dss.append(prob * (fx.Float32(_raw(Vec(dp_acc)[r])) - fx.Float32(_raw(dl8[r]))))
                pB = pack_low16_bf16(probs, grp0)
                dsB = pack_low16_bf16(dss, grp0)

                new_dk, new_dv = [], []
                for dt in range_constexpr(DT):
                    d_col = fx.Index(dt * 16) + lo
                    q_elems, do_elems = [], []
                    for r in range_constexpr(16):
                        lrow = fx.Index(j * HPW + r) * fx.Index(STRIDE) + d_col
                        qv = fx.BFloat16(_raw(Vec.load(v1, lds_q, [lrow])[0]))
                        dv = fx.BFloat16(_raw(Vec.load(v1, lds_do, [lrow])[0]))
                        q_elems.append(fx.BFloat16(grp0.select(_raw(qv), _raw(c_bf0))))
                        do_elems.append(fx.BFloat16(grp0.select(_raw(dv), _raw(c_bf0))))
                    new_dv.append(wmma_f32_16x16x32_bf16(Vec.from_elements(do_elems, elem), pB, dv_acc[dt]))
                    new_dk.append(wmma_f32_16x16x32_bf16(Vec.from_elements(q_elems, elem), dsB, dk_acc[dt]))
                dk_acc, dv_acc = new_dk, new_dv
            results = yield dk_acc + dv_acc

        # lane holds d[key = wave*16 + lo, d = dt*16 + half*8 + i]
        ws_row = (chunk * fx.Index(2 * block_size) + wave * fx.Index(TK) + lo) * fx.Index(D)
        scale_v = Vec.from_elements([fx.Float32(scale)] * 8, fx.Float32)
        for dt in range_constexpr(DT):
            off = ws_row + fx.Index(dt * 16) + half * fx.Index(8)
            _store_f32x8(ws_rsrc, off, Vec(results[dt]) * scale_v)
            _store_f32x8(ws_rsrc, off + fx.Index(block_size * D), results[DT + dt])

    @flyc.jit
    def launch(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH).launch(
            grid=(fx.Index(NCH), 1, 1), block=(threads, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch
