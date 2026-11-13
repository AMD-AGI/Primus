###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 MSA backward: the dQ and dK/dV kernels of ``msa_token_bwd`` on wave32 WMMA.

dQ is one wave32 per token. Each step takes 32 keys of a block as two 16-key
tiles, whose dS fill the K=32 of the dS.K WMMA, one tile per lane half. The
step's K and V tiles are copied to a per-wave 2-stage LDS ring by the Tensor
Data Mover; S and dP read K / V rows from it, and dQ reads K^T with
``ds_load_tr16_b128`` (lane row ``half*16 + lane%8`` then +8, column
``((lane//8)%2)*8``: each lane gets keys ``half*16 + 0..15`` of one column, the
K order of a transposed copy). The grid is the forward's "lpt" order.

dK/dV is one wave per 16 keys of a block; the two lane halves split K on QK,
and each holds 8 of the 16 heads. In the dS/P contractions the lane halves take
one token each, so two tokens fill K=32. Each step's 4 tokens x 16 head rows of
Q and dO are copied to a 2-stage LDS ring by the Tensor Data Mover (each of the
8 waves issues one 16-row tile; tokens come from scalar loads two steps ahead,
and a table entry maps to its token by a subtraction, not a modulo); S / dP read
the rows, dV / dK read them transposed with ``ds_load_tr16_b128`` (same lane
addressing as dQ). Compiled with ``waves_per_eu=4``: 256 VGPRs, no
``s_set_vgpr_msb``. Each wave's K / V fragments (16 keys) are parked in a per-wave LDS
tile once and read back by S / dP (same operand values): 64 fewer live VGPRs, scratch
584 -> 24 B, 0.61 -> 0.51 ms per layer (bitwise). dK/dV partials use the gfx950 layout, a dense
``[chunk, 2, block, D]`` fp32 buffer, so the same reduce kernel sums them.

Both kernels are bitwise identical to the register-staged ones they replace; per
layer at S=4096 on real tables dQ 0.570 -> 0.488 ms, dK/dV 0.770 -> 0.615 ms.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr import arith, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
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
    pack_pair_bf16,
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


def build_dq_w32(num_kv_heads: int, topk: int, block_size: int, scale: float, nbuf: int = 2):
    NBUF = int(nbuf)  # LDS ring stages; 3 measured slower (0.745 vs 0.488 ms: occupancy)
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    KPS = 2 * TK  # keys per step
    assert block_size % KPS == 0
    SPB = block_size // KPS
    ROWE = STRIDE  # padded LDS row (elements), as the staged copy of the reference kernel
    T_BYTES = KPS * ROWE * 2
    STAGE = 2 * T_BYTES  # K then V
    LDS_BYTES = NBUF * STAGE

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
        v8_ty = ir.VectorType.get([8], elem.ir_type)
        smem = fx.SharedAllocator().allocate(LDS_BYTES)
        lds_base = fx.Int32(fx.ptrtoint(smem.peek().ptr))

        lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)
        grp0 = ArithValue(half == fx.Index(0))
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        # Same "lpt" order as the forward: heaviest (latest) tokens first, and consecutive
        # workgroups cycle through (b, g), so each XCD's L2 holds one KV group's K/V.
        raw = fx.Index(gpu.block_idx.x)
        bgn = Bn * fx.Index(Hkv)
        tok = Sn - fx.Index(1) - raw // bgn
        bg = raw % bgn
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)

        q_bytes = _raw(Sn * Bn * fx.Index(Hq * D * 2))
        kv_bytes = _raw(Sn * Bn * fx.Index(Hkv * D * 2))
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=False, num_records_bytes=q_bytes)
        do_rsrc = buffer_ops.create_buffer_resource(DO, max_size=False, num_records_bytes=q_bytes)
        o_rsrc = buffer_ops.create_buffer_resource(O, max_size=False, num_records_bytes=q_bytes)
        dq_rsrc = buffer_ops.create_buffer_resource(DQ, max_size=False, num_records_bytes=q_bytes)
        row_bytes = _raw(Sn * Bn * fx.Index(Hq * 4))
        lse_rsrc = buffer_ops.create_buffer_resource(LSE, max_size=False, num_records_bytes=row_bytes)
        dl_rsrc = buffer_ops.create_buffer_resource(DELTA, max_size=False, num_records_bytes=row_bytes)
        t_rsrc = buffer_ops.create_buffer_resource(
            TBL, max_size=False, num_records_bytes=_raw(Bn * fx.Index(Hkv) * Sn * fx.Index(topk * 4))
        )

        c_sl = fx.Float32(scale * _LOG2E)
        c_log2e = fx.Float32(_LOG2E)
        c_zero = fx.Float32(0.0)

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

        ptr3 = ir.Type.parse("!llvm.ptr<3>")

        def lds_ptr(addr):
            return llvm.inttoptr(ptr3, _raw(fx.Int32(addr)))

        def lds_b128(addr):
            return Vec(llvm.load(v8_ty, lds_ptr(addr), alignment=16))

        def lds_tr16(addr):
            r = _rocdl.ds_load_tr16_b128(v8_ty, lds_ptr(addr))
            return Vec(r if isinstance(r, ir.Value) else r.result)

        lds_ty = fx.PointerType.get(elem_ty=elem.ir_type, address_space=fx.AddressSpace.Shared, alignment=16)
        row_stride = Bn * fx.Index(Hkv) * fx.Index(D)
        head_off = (b * fx.Index(Hkv) + g) * fx.Index(D)

        def tdm_tile(T_, kv0, lds_addr):
            off = fx.Int64(fx.Int32(kv0)) * fx.Int64(fx.Int32(row_stride)) + fx.Int64(fx.Int32(head_off))
            gbase = fx.add_offset(fx.get_iter(T_), off)
            g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((KPS, D), (D, 1))))
            atom = fx.rocdl.make_tdm_atom(
                g_view, [fx.Int32(KPS), None], strides=[fx.Int32(row_stride), None], num_warps=1,
                pad_interval=D, pad_amount=ROWE - D,
            )
            l_view = fx.Tensor(fx.make_view(fx.inttoptr(lds_ty, lds_addr), fx.make_layout((KPS, D), (ROWE, 1))))
            fx.copy_atom_call(atom, g_view, l_view)

        def stage_base(jj):
            return lds_base + fx.Int32(jj % fx.Index(NBUF)) * fx.Int32(STAGE)

        def issue(jj, kv0):
            sb = stage_base(jj)
            tdm_tile(K, kv0, sb)
            tdm_tile(V, kv0, sb + fx.Int32(T_BYTES))

        def step_blk(jj):
            return fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + jj // fx.Index(SPB), vec_width=1, is_scalar=True)
            )

        def step_kv0(blk, jj):
            safe = fx.Int32(ArithValue(blk >= fx.Int32(0)).select(_raw(blk), _raw(fx.Int32(0))))
            return safe * fx.Int32(block_size) + fx.Int32(jj % fx.Index(SPB)) * fx.Int32(KPS)

        # A operands: K / V rows (key lo, d chunk half*16) for S and dP; K^T for dQ by
        # ds_load_tr16_b128 at lane row half*16 + lane%8, column ((lane//8)%2)*8 -- the same
        # K order as the transposed LDS copy of the reference kernel.
        r_lane = fx.Int32(lo) * fx.Int32(ROWE * 2) + fx.Int32(half) * fx.Int32(32)
        t_lane = (fx.Int32(half) * fx.Int32(16) + fx.Int32(lane % fx.Index(8))) * fx.Int32(ROWE * 2) + fx.Int32(
            (lane // fx.Index(8)) % fx.Index(2)
        ) * fx.Int32(16)

        def sel_idx(c, a_, b_):
            return fx.Index(fx.Int32(ArithValue(c).select(_raw(fx.Int32(a_)), _raw(fx.Int32(b_)))))

        for pj in range_constexpr(NBUF - 1):
            jj = sel_idx(fx.Index(pj) < n_steps, fx.Index(pj), fx.Index(0))
            issue(fx.Index(pj), step_kv0(step_blk(jj), jj))

        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        init = [c_zero_v8 for _ in range_constexpr(DT)]
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            dq_acc = [it[dt] for dt in range_constexpr(DT)]
            blk = step_blk(j)
            blk_ok = ArithValue(blk >= fx.Int32(0))
            kv0 = step_kv0(blk, j)
            tdm_ops.tensor_wait(2 * (NBUF - 2))
            j_f = j + fx.Index(NBUF - 1)
            j_c = sel_idx(j_f < n_steps, j_f, j)
            issue(j_f, step_kv0(step_blk(j_c), j_c))
            sb = stage_base(j)
            kb = sb + r_lane
            vb = sb + fx.Int32(T_BYTES) + r_lane
            ktb = sb + t_lane

            ds_tiles = []
            for t in range_constexpr(KPS // TK):
                s_acc = Vec.filled(8, 0.0, fx.Float32)
                dp_acc = Vec.filled(8, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    o_ = fx.Int32((t * TK * ROWE + ks * 32) * 2)
                    a_k = cat_bf16_16(lds_b128(kb + o_), lds_b128(kb + o_ + fx.Int32(16)))
                    a_v = cat_bf16_16(lds_b128(vb + o_), lds_b128(vb + o_ + fx.Int32(16)))
                    s_acc = wmma_f32_16x16x32_bf16(a_k, q_packs[ks], s_acc)
                    dp_acc = wmma_f32_16x16x32_bf16(a_v, do_packs[ks], dp_acc)
                ds = []
                key0 = kv0 + fx.Int32(t * TK) + fx.Int32(half) * fx.Int32(8)
                for i in range_constexpr(8):
                    ok = _andb(blk_ok, ArithValue(key0 + fx.Int32(i) <= tok_i32))
                    score = fx.Float32(_raw(Vec(s_acc)[i]))
                    prob = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw(score * c_sl - lse2)))
                    prob = fx.Float32(ok.select(_raw(prob), _raw(c_zero)))
                    dp = fx.Float32(_raw(Vec(dp_acc)[i]))
                    ds.append(prob * (dp - delta))
                ds_tiles.append(ds)
            dsB = pack_pair_bf16(ds_tiles[0], ds_tiles[1])

            new_dq = []
            for dt in range_constexpr(DT):
                ta = ktb + fx.Int32(dt * 16 * 2)
                a_kt = cat_bf16_16(lds_tr16(ta), lds_tr16(ta + fx.Int32(8 * ROWE * 2)))
                new_dq.append(wmma_f32_16x16x32_bf16(a_kt, dsB, dq_acc[dt]))
            results = yield new_dq
        tdm_ops.tensor_wait(0)

        for dt in range_constexpr(DT):
            _store_bf16x8(
                dq_rsrc,
                row * fx.Index(D) + fx.Index(dt * 16) + half * fx.Index(8),
                results[dt],
                fx.Float32(scale),
            )

    @flyc.jit
    def launch(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B, stream):
        k_fn(Q, K, V, DO, O, LSE, DELTA, TBL, DQ, S, B).launch(
            grid=(fx.Index(S) * fx.Index(B) * fx.Index(Hkv), 1, 1), block=(WAVE, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


def build_dkdv_w32(num_kv_heads: int, block_size: int, scale: float, topk: int, tokens_per_step: int = 4):
    """dK/dV with the Q / dO rows of each step staged by the Tensor Data Mover.

    Same arithmetic, in the same order, as the register-staged kernel: S / dP read Q / dO rows
    from LDS (row-major, STRIDE), dV / dK read them transposed with ``ds_load_tr16_b128``
    (lane row half*16 + lane%8, column ((lane//8)%2)*8: each lane gets the same 16 token-head
    rows of one d column that the transposed copy held). Each of the 8 waves issues one 16-row
    TDM tile per step (token w//2, Q or dO by w%2) into a 2-stage ring; tokens come from
    scalar loads two steps ahead, and entries map to tokens by a subtraction."""
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    waves = block_size // TK
    threads = WAVE * waves
    tps = tokens_per_step
    rows = tps * HPW
    assert rows <= threads and tps % 2 == 0 and 2 * tps == waves
    NBUF = 2
    T_BYTES = rows * STRIDE * 2
    STAGE = 2 * T_BYTES  # Q rows then dO rows
    LSE_OFF = NBUF * STAGE
    KV_OFF = LSE_OFF + 2 * threads * 4
    KV_W = 2 * TK * STRIDE * 2  # per wave: K tile then V tile, 16 keys x STRIDE bf16
    LDS_BYTES = KV_OFF + waves * KV_W

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
        v8_ty = ir.VectorType.get([8], elem.ir_type)
        v4f_ty = ir.VectorType.get([4], fx.Float32.ir_type)
        smem = fx.SharedAllocator().allocate(LDS_BYTES)
        lds_base = fx.Int32(fx.ptrtoint(smem.peek().ptr))
        ptr3 = ir.Type.parse("!llvm.ptr<3>")

        def lds_ptr(addr):
            return llvm.inttoptr(ptr3, _raw(fx.Int32(addr)))

        def lds_b128(addr):
            return Vec(llvm.load(v8_ty, lds_ptr(addr), alignment=16))

        def lds_f32x4(addr):
            return Vec(llvm.load(v4f_ty, lds_ptr(addr), alignment=16))

        def lds_st_f32(addr, val):
            llvm.store(_raw(fx.Float32(val)), lds_ptr(addr), alignment=4)

        def lds_tr16(addr):
            r = _rocdl.ds_load_tr16_b128(v8_ty, lds_ptr(addr))
            return Vec(r if isinstance(r, ir.Value) else r.result)

        def uniform(x):
            return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, _raw(fx.Int32(x))))

        tid = fx.Index(gpu.thread_idx.x)
        wave = tid // fx.Index(WAVE)
        lane = tid % fx.Index(WAVE)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)
        Sn = fx.Index(S)
        Bn = fx.Index(B)
        NBn = fx.Index(NB)
        chunk = fx.Index(gpu.block_idx.x)

        kv_bytes = _raw(Sn * Bn * fx.Index(Hkv * D * 2))
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

        def sload(rsrc, idx):
            return fx.Int32(buffer_ops.buffer_load(rsrc, idx, vec_width=1, dtype=fx.Int32, is_scalar=True))

        csr = fx.Index(sload(ch_rsrc, chunk * fx.Index(3)))
        t_begin = sload(ch_rsrc, chunk * fx.Index(3) + fx.Index(1))
        t_end = sload(ch_rsrc, chunk * fx.Index(3) + fx.Index(2))
        n_tok = fx.Index(t_end - t_begin)
        bg = csr // NBn
        blk = csr % NBn
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)
        tok_base = fx.Int32(bg * Sn)

        c_sl = fx.Float32(scale * _LOG2E)
        c_log2e = fx.Float32(_LOG2E)
        c_zero = fx.Float32(0.0)

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

        # K / V fragments of this wave's 16 keys go to a per-wave LDS tile (row lo, columns ks*32 + half*16 ..+16)
        # and are read back in the step loop: same operand values, 64 fewer live VGPRs.
        kv_lds = lds_base + fx.Int32(KV_OFF) + fx.Int32(wave) * fx.Int32(KV_W)
        kv_lane = kv_lds + fx.Int32(lo) * fx.Int32(STRIDE * 2) + fx.Int32(half) * fx.Int32(32)
        for ks in range_constexpr(KS):
            for which, ops in ((0, k_ops), (1, v_ops)):
                addr = kv_lane + fx.Int32(which * TK * STRIDE * 2 + ks * 64)
                llvm.store(_raw(ops[ks]), lds_ptr(addr), alignment=16)
        k_ops = v_ops = None

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
                e = sload(ent_rsrc, fx.Index(r))
                tok.append(fx.Int32(e // fx.Int32(topk)) - tok_base)
            return tok

        def head_row(t_i32, head):
            return (fx.Index(t_i32) * Bn + b) * fx.Index(Hq) + g * fx.Index(HPW) + head

        def load_lse(tok):
            lrow = head_row(pick(tok, ld_row // fx.Index(HPW)), ld_row % fx.Index(HPW))
            lse_st = fx.Float32(buffer_ops.buffer_load(lse_rsrc, lrow, vec_width=1, dtype=fx.Float32))
            dl_st = fx.Float32(buffer_ops.buffer_load(dl_rsrc, lrow, vec_width=1, dtype=fx.Float32))
            return lse_st, dl_st

        # TDM: wave w stages token w//2's 16 head rows of Q (w even) or dO (w odd)
        lds_ty = fx.PointerType.get(elem_ty=elem.ir_type, address_space=fx.AddressSpace.Shared, alignment=16)
        w_tok = wave // fx.Index(2)
        w_do = ArithValue(wave % fx.Index(2) == fx.Index(1))
        q_iter = fx.get_iter(Q)
        do_delta = (fx.Int64(fx.ptrtoint(fx.get_iter(DO))) - fx.Int64(fx.ptrtoint(q_iter))) // fx.Int64(2)
        src_shift = fx.Int64(ArithValue(w_do).select(_raw(do_delta), _raw(fx.Int64(0))))
        dst_shift = fx.Int32(ArithValue(w_do).select(_raw(fx.Int32(T_BYTES)), _raw(fx.Int32(0)))) + fx.Int32(
            w_tok
        ) * fx.Int32(HPW * STRIDE * 2)

        def stage_base(i_step):
            return lds_base + fx.Int32(i_step % fx.Index(2)) * fx.Int32(STAGE)

        def issue(i_step, tok):
            t_w = uniform(pick(tok, w_tok))
            off = fx.Int64(fx.Int32(head_row(t_w, fx.Index(0)) * fx.Index(D))) + src_shift
            gbase = fx.add_offset(q_iter, off)
            g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((HPW, D), (D, 1))))
            atom = fx.rocdl.make_tdm_atom(
                g_view, [fx.Int32(HPW), None], strides=[fx.Int32(D), None], num_warps=1,
                pad_interval=D, pad_amount=STRIDE - D,
            )
            dst = uniform(stage_base(i_step) + dst_shift)
            l_view = fx.Tensor(fx.make_view(fx.inttoptr(lds_ty, dst), fx.make_layout((HPW, D), (STRIDE, 1))))
            fx.copy_atom_call(atom, g_view, l_view)

        # per-lane LDS byte offsets
        a_lane = fx.Int32(lo) * fx.Int32(STRIDE * 2) + fx.Int32(half) * fx.Int32(32)
        t_lane = (fx.Int32(half) * fx.Int32(16) + fx.Int32(lane % fx.Index(8))) * fx.Int32(STRIDE * 2) + fx.Int32(
            (lane // fx.Index(8)) % fx.Index(2)
        ) * fx.Int32(16)
        lse_lds = lds_base + fx.Int32(LSE_OFF)
        dl_lds = lse_lds + fx.Int32(threads * 4)

        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        NA = 2 * DT
        tok_0 = load_tokens(t_begin)
        issue(fx.Index(0), tok_0)
        lse_0, dl_0 = load_lse(tok_0)
        init = [c_zero_v8 for _ in range_constexpr(NA)] + tok_0 + [lse_0, dl_0] + load_tokens(t_begin + fx.Int32(tps))
        results = init
        for i, it in range(fx.Index(0), n_tok, fx.Index(tps), init=init):
            dk_acc = [it[dt] for dt in range_constexpr(DT)]
            dv_acc = [it[DT + dt] for dt in range_constexpr(DT)]
            r0 = t_begin + fx.Int32(i)
            tok = [it[NA + j] for j in range_constexpr(tps)]
            lse_st = it[NA + tps]
            dl_st = it[NA + tps + 1]
            tok_n = [it[NA + tps + 2 + j] for j in range_constexpr(tps)]
            valid = [ArithValue(r0 + fx.Int32(j) < t_end) for j in range_constexpr(tps)]
            istep = i // fx.Index(tps)

            gpu.barrier()  # WAR: the previous step's lse / delta reads are done
            lds_st_f32(lse_lds + fx.Int32(tid) * fx.Int32(4), lse_st)
            lds_st_f32(dl_lds + fx.Int32(tid) * fx.Int32(4), dl_st)
            tdm_ops.tensor_wait(0)  # this wave's tile of the step
            gpu.barrier()  # every wave's tile and the lse / delta stores are visible
            # the next step's tiles go to the other stage, read by the previous step (done: the barrier)
            issue(istep + fx.Index(1), tok_n)
            lse_n, dl_n = load_lse(tok_n)
            tok_nn = load_tokens(r0 + fx.Int32(2 * tps))
            sb = stage_base(istep)
            qa = sb + a_lane
            da = sb + fx.Int32(T_BYTES) + a_lane

            def probs_and_ds(j):
                s_acc = Vec.filled(8, 0.0, fx.Float32)
                dp_acc = Vec.filled(8, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    o_ = fx.Int32((j * HPW * STRIDE + ks * 32) * 2)
                    ka = kv_lane + fx.Int32(ks * 64)
                    va = ka + fx.Int32(TK * STRIDE * 2)
                    s_acc = wmma_f32_16x16x32_bf16(
                        cat_bf16_16(lds_b128(qa + o_), lds_b128(qa + o_ + fx.Int32(16))),
                        cat_bf16_16(lds_b128(ka), lds_b128(ka + fx.Int32(16))), s_acc
                    )
                    dp_acc = wmma_f32_16x16x32_bf16(
                        cat_bf16_16(lds_b128(da + o_), lds_b128(da + o_ + fx.Int32(16))),
                        cat_bf16_16(lds_b128(va), lds_b128(va + fx.Int32(16))), dp_acc
                    )
                lb = fx.Int32((j * HPW) * 4) + fx.Int32(half) * fx.Int32(32)
                lse0 = lds_f32x4(lse_lds + lb)
                lse1 = lds_f32x4(lse_lds + lb + fx.Int32(16))
                dl0 = lds_f32x4(dl_lds + lb)
                dl1 = lds_f32x4(dl_lds + lb + fx.Int32(16))
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
                return probs, dss

            for pr in range_constexpr(tps // 2):
                probs0, dss0 = probs_and_ds(2 * pr)
                probs1, dss1 = probs_and_ds(2 * pr + 1)
                pB = pack_pair_bf16(probs0, probs1)
                dsB = pack_pair_bf16(dss0, dss1)
                new_dk, new_dv = [], []
                for dt in range_constexpr(DT):
                    o_ = fx.Int32((2 * pr * HPW * STRIDE + dt * 16) * 2)
                    at_q = sb + t_lane + o_
                    at_d = sb + fx.Int32(T_BYTES) + t_lane + o_
                    a_dot = cat_bf16_16(lds_tr16(at_d), lds_tr16(at_d + fx.Int32(8 * STRIDE * 2)))
                    a_qt = cat_bf16_16(lds_tr16(at_q), lds_tr16(at_q + fx.Int32(8 * STRIDE * 2)))
                    new_dv.append(wmma_f32_16x16x32_bf16(a_dot, pB, dv_acc[dt]))
                    new_dk.append(wmma_f32_16x16x32_bf16(a_qt, dsB, dk_acc[dt]))
                dk_acc, dv_acc = new_dk, new_dv
            results = yield dk_acc + dv_acc + tok_n + [lse_n, dl_n] + tok_nn
        tdm_ops.tensor_wait(0)

        ws_row = (chunk * fx.Index(2 * block_size) + wave * fx.Index(TK) + lo) * fx.Index(D)
        scale_v = Vec.from_elements([fx.Float32(scale)] * 8, fx.Float32)
        for dt in range_constexpr(DT):
            off = ws_row + fx.Index(dt * 16) + half * fx.Index(8)
            _store_f32x8(ws_rsrc, off, Vec(results[dt]) * scale_v)
            _store_f32x8(ws_rsrc, off + fx.Index(block_size * D), results[DT + dt])

    @flyc.jit
    def launch(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH, stream):
        k_fn(Q, K, V, DO, LSE, DELTA, CHUNKS, ENT, WS, S, B, NB, NENT, NCH).launch(
            grid=(fx.Index(NCH), 1, 1), block=(threads, 1, 1), stream=stream
        )

    def _compile(*a):
        from flydsl.compiler.kernel_function import CompilationContext as _CC
        with _CC.compile_hints({'waves_per_eu': 4}):
            return flyc.compile(launch, *a)

    launch.compile = _compile
    return launch
