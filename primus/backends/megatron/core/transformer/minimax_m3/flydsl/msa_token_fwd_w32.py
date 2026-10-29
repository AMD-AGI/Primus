###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 MSA forward: one wave32 per query token, WMMA 16x16x32 bf16.

Same contract as ``msa_token_fwd``, on the wave32 layout documented in ``msa_w32``:

* QK is ``S[key, head] = K[key] · Q[head]``. A is the key row, B is the query
  row read as a column. Each lane then holds 8 keys of one head.
* The 16 keys of a step reduce with one ``permlane16`` (low 8 + high 8).
* PV is ``O[head, d] += P[head, key] * V[key, d]``, issued as one K=32 WMMA
  whose upper 16 K are zero. ``P`` is gathered into B's column layout with the
  same permlane, and ``V`` is gathered transposed out of LDS.
* The epilogue stores the 8 contiguous ``d`` values each lane holds.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from primus_turbo.flydsl.utils import buffer_ops

from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_token_fwd import (
    HPW,
    MAX_TOPK,
    NXCD,
    D,
    check_buffer_bytes,
)
from primus.backends.megatron.core.transformer.minimax_m3.flydsl.msa_w32 import (
    cat_bf16_16,
    crossgrp_max,
    crossgrp_sum,
    pack_low16_bf16,
    wmma_f32_16x16x32_bf16,
)

_LOG2E = 1.4426950408889634
WAVE = 32
TK = 16
KS = D // 32
DT = D // 16
STRIDE = D + 16


def build_fwd_wave32(
    num_kv_heads: int,
    topk: int,
    block_size: int,
    scale: float,
    remap: str = "lpt",
    emit_slot_lse: bool = False,
):
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    assert block_size % TK == 0
    assert remap in ("none", "block", "lpt")
    SPB = block_size // TK
    # 32 lanes, 8 bf16 each: two key rows per instruction, eight instructions per step.
    LPR = D // 8
    RPP = WAVE // LPR
    assert TK % RPP == 0
    NV8 = TK // RPP

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_token_fwd_w32_smem")
    v_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = allocator._align(v_off + TK * STRIDE * 2, 16)

    @flyc.kernel(known_block_size=[WAVE, 1, 1])
    def k_fn(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        TBL: fx.Tensor,
        O: fx.Tensor,
        LSE: fx.Tensor,
        SLSE: fx.Tensor,
        S: fx.Int32,
        B: fx.Int32,
    ):
        v1 = Vec.make_type(1, elem)
        lds_v = SmemPtr(allocator.get_base(), v_off, elem.ir_type, shape=(TK * STRIDE,)).get()

        def sel_i32(cond, a, b):
            return fx.Int32(ArithValue(cond).select(_raw(a), _raw(b)))

        def andb(a, b):
            return ArithValue(arith.AndIOp(_raw(a), _raw(b)).result)

        lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)  # 0: K 0..15, 1: K 16..31 of the WMMA operand

        Sn = fx.Index(S)
        Bn = fx.Index(B)
        raw = fx.Index(gpu.block_idx.x)
        if const_expr(remap == "lpt"):
            bgn = Bn * fx.Index(Hkv)
            ti = raw // bgn
            tok = Sn - fx.Index(1) - ti
            bg = raw % bgn
        elif const_expr(remap == "block"):
            w = raw // fx.Index(NXCD)
            qblk = raw % fx.Index(NXCD) + fx.Index(NXCD) * (w // fx.Index(block_size))
            tok = qblk * fx.Index(block_size) + w % fx.Index(block_size)
            bg = fx.Index(gpu.block_idx.y)
        else:
            tok = raw
            bg = fx.Index(gpu.block_idx.y)
        b = bg // fx.Index(Hkv)
        g = bg % fx.Index(Hkv)

        q_rsrc = buffer_ops.create_buffer_resource(
            Q, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hq * D * 2))
        )
        k_rsrc = buffer_ops.create_buffer_resource(
            K, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hkv * D * 2))
        )
        v_rsrc = buffer_ops.create_buffer_resource(
            V, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hkv * D * 2))
        )
        t_rsrc = buffer_ops.create_buffer_resource(
            TBL, max_size=False, num_records_bytes=_raw(Bn * fx.Index(Hkv) * Sn * fx.Index(topk * 4))
        )
        o_rsrc = buffer_ops.create_buffer_resource(
            O, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hq * D * 2))
        )
        lse_rsrc = buffer_ops.create_buffer_resource(
            LSE, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hq * 4))
        )
        if const_expr(emit_slot_lse):
            slse_rsrc = buffer_ops.create_buffer_resource(
                SLSE, max_size=False, num_records_bytes=_raw(Sn * Bn * fx.Index(Hq * topk * 4))
            )

        c_scale = fx.Float32(scale)
        c_sl = fx.Float32(scale * _LOG2E)
        c_neg_inf = fx.Float32(float("-inf"))
        c_big_neg = fx.Float32(-3.0e38)
        c_zero = fx.Float32(0.0)
        c_i0 = fx.Int32(0)
        c_bf0 = fx.BFloat16(0.0)

        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = sel_i32(ArithValue(n_vis < fx.Int32(topk)), n_vis, fx.Int32(topk))
        n_steps = fx.Index(n_valid) * fx.Index(SPB)
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)

        # B operand of QK: Q[head, d], 16 contiguous d per K-half.
        head = g * fx.Index(HPW) + lo
        q_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)
        q_packs = []
        for ks in range_constexpr(KS):
            base = q_row + fx.Index(ks * 32) + half * fx.Index(16)
            q_packs.append(
                cat_bf16_16(
                    buffer_ops.buffer_load(q_rsrc, base, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(q_rsrc, base + fx.Index(8), vec_width=8, dtype=elem),
                )
            )

        def kv_offset(key):
            return ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)

        ld_row = lane // fx.Index(LPR)
        ld_col = (lane % fx.Index(LPR)) * fx.Index(8)

        def load_v(kv0):
            return [
                buffer_ops.buffer_load(
                    v_rsrc,
                    kv_offset(fx.Index(kv0) + ld_row + fx.Index(c * RPP)) + ld_col,
                    vec_width=8,
                    dtype=elem,
                )
                for c in range_constexpr(NV8)
            ]

        grp0 = ArithValue(half == fx.Index(0))
        head_row = (tok * Bn + b) * fx.Index(Hq) + head
        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        init = [c_big_neg, c_zero] + [c_zero_v8 for _ in range_constexpr(DT)]
        if const_expr(emit_slot_lse):
            init = init + [c_zero]
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            m_cur = it[0]
            l_cur = it[1]
            o_cur = [it[2 + dt] for dt in range_constexpr(DT)]
            sub = j % fx.Index(SPB)
            blk = fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + j // fx.Index(SPB), vec_width=1, dtype=fx.Int32)
            )
            blk_ok = ArithValue(blk >= c_i0)
            safe = sel_i32(blk_ok, blk, c_i0)
            kv0 = safe * fx.Int32(block_size) + fx.Int32(sub) * fx.Int32(TK)

            rows = load_v(kv0)
            for c in range_constexpr(NV8):
                Vec(rows[c]).store(lds_v, [(ld_row + fx.Index(c * RPP)) * fx.Index(STRIDE) + ld_col])

            # S[key, head] = K[key] · Q[head]. A row = lane%16 = key in the tile.
            acc = Vec.filled(8, 0.0, fx.Float32)
            for ks in range_constexpr(KS):
                kbase = kv_offset(fx.Index(kv0) + lo) + fx.Index(ks * 32) + half * fx.Index(16)
                a_k = cat_bf16_16(
                    buffer_ops.buffer_load(k_rsrc, kbase, vec_width=8, dtype=elem),
                    buffer_ops.buffer_load(k_rsrc, kbase + fx.Index(8), vec_width=8, dtype=elem),
                )
                acc = wmma_f32_16x16x32_bf16(a_k, q_packs[ks], acc)
            # lane holds keys (half*8 + i) of head lo
            scores = []
            key0 = kv0 + fx.Int32(half) * fx.Int32(8)
            for i in range_constexpr(8):
                ok = andb(blk_ok, ArithValue(key0 + fx.Int32(i) <= tok_i32))
                raw_s = fx.Float32(_raw(Vec(acc)[i]))
                scores.append(fx.Float32(ok.select(_raw(raw_s), _raw(c_neg_inf))))

            lmax = scores[0]
            for x in scores[1:]:
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(x)).result)
            m_new = fx.Float32(arith.MaxNumFOp(_raw(m_cur), _raw(crossgrp_max(lmax))).result)
            alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_cur - m_new) * c_sl)))
            p = [fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((x - m_new) * c_sl))) for x in scores]
            lsum = p[0]
            for x in p[1:]:
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(x)).result)
            l_new = fx.Float32(alpha * l_cur + lsum)

            # B column = head lo, K 0..15 from the two halves. K 16..31 stays zero,
            # so only the low lane group contributes this step's 16 keys.
            pB = pack_low16_bf16(p, grp0)

            alpha_v = Vec.from_elements([alpha] * 8, fx.Float32)
            new_o = []
            for dt in range_constexpr(DT):
                a_elems = []
                d_col = fx.Index(dt * 16) + lo
                for i in range_constexpr(16):
                    ld = fx.BFloat16(_raw(Vec.load(v1, lds_v, [fx.Index(i) * fx.Index(STRIDE) + d_col])[0]))
                    a_elems.append(fx.BFloat16(grp0.select(_raw(ld), _raw(c_bf0))))
                a_v = Vec.from_elements(a_elems, elem)
                new_o.append(wmma_f32_16x16x32_bf16(a_v, pB, Vec(o_cur[dt]) * Vec(alpha_v)))

            out = [m_new, l_new] + new_o
            if const_expr(emit_slot_lse):
                reset = ArithValue(sub == fx.Index(0))
                ls_prev = fx.Float32(reset.select(_raw(c_zero), _raw(it[2 + DT])))
                ls_new = fx.Float32(alpha * ls_prev + lsum)
                last = andb(ArithValue(sub == fx.Index(SPB - 1)), grp0)
                slot = j // fx.Index(SPB)
                slot_lse = fmath.log(crossgrp_sum(ls_new)) + fx.Float32(m_new * c_scale)
                buffer_ops.buffer_store(
                    slot_lse,
                    slse_rsrc,
                    (head_row * fx.Index(topk) + slot) * fx.Index(4),
                    mask=_raw(andb(last, ArithValue(fx.Int32(slot) < n_valid))),
                    offset_is_bytes=True,
                )
                out = out + [ls_new]
            results = yield out

        m_run = results[0]
        l_sum = crossgrp_sum(results[1])
        o_acc = [results[2 + dt] for dt in range_constexpr(DT)]
        inv = fx.Float32(rocdl.rcp(fx.Float32.ir_type, _raw(l_sum)))
        inv_v = Vec.from_elements([inv] * 8, fx.Float32)
        o_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)
        # lane holds O[head = lo, d = dt*16 + half*8 + i]
        for dt in range_constexpr(DT):
            ov = Vec(o_acc[dt]) * inv_v
            pk = [
                rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2 * i]), _raw(Vec(ov)[2 * i + 1]))
                for i in range_constexpr(4)
            ]
            dbase = fx.Index(dt * 16) + half * fx.Index(8)
            buffer_ops.buffer_store(
                _raw(Vec.from_elements([fx.Int32(_raw(w)) for w in pk], fx.Int32)),
                o_rsrc,
                (o_row + dbase) * fx.Index(2),
                offset_is_bytes=True,
            )
        lse_val = fx.Float32(m_run * c_scale) + fmath.log(l_sum)
        buffer_ops.buffer_store(
            lse_val,
            lse_rsrc,
            ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(4),
            mask=_raw(grp0),
            offset_is_bytes=True,
        )
        if const_expr(emit_slot_lse):
            for s in range_constexpr(topk):
                buffer_ops.buffer_store(
                    c_neg_inf,
                    slse_rsrc,
                    (head_row * fx.Index(topk) + fx.Index(s)) * fx.Index(4),
                    mask=_raw(andb(grp0, ArithValue(fx.Int32(s) >= n_valid))),
                    offset_is_bytes=True,
                )

    def _grid(S, B):
        if remap == "lpt":
            return (fx.Index(S) * fx.Index(B) * fx.Index(Hkv), 1, 1)
        return (fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1)

    @flyc.jit
    def launch(Q, K, V, TBL, O, LSE, SLSE, S, B, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, TBL, O, LSE, SLSE, S, B).launch(grid=_grid(S, B), block=(WAVE, 1, 1), stream=stream)

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CACHE: dict = {}


def msa_token_fwd_wave32(
    q, k, v, block_table, softmax_scale=None, block_size=128, return_slot_lse=False, **config
):
    """Wave32 MSA forward. Same tensors as ``msa_token_fwd``; ``waves`` is ignored."""
    S, B, Hq, Dq = q.shape
    Hkv = k.shape[2]
    topk = block_table.shape[-1]
    assert topk <= MAX_TOPK, f"topk={topk}: the MSA forward supports at most {MAX_TOPK} blocks per token"
    assert Dq == D and k.shape[-1] == D and v.shape[-1] == D, f"head_dim must be {D}"
    assert Hq == HPW * Hkv, f"need {HPW} query heads per KV head, got Hq={Hq} Hkv={Hkv}"
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert block_table.shape == (B, Hkv, S, topk) and block_table.dtype == torch.int32
    if softmax_scale is None:
        softmax_scale = D**-0.5
    remap = config.get("remap", "lpt")
    if remap == "block" and S % (NXCD * block_size) != 0:
        remap = "none"
    q, k, v, block_table = (t.contiguous() for t in (q, k, v, block_table))

    o = torch.empty_like(q)
    lse = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)
    slse = torch.empty((S, B, Hq, topk) if return_slot_lse else (1,), dtype=torch.float32, device=q.device)
    check_buffer_bytes(q=q, k=k, v=v, block_table=block_table, slot_lse=slse)
    key = (Hkv, topk, block_size, float(softmax_scale), remap, bool(return_slot_lse))
    entry = _CACHE.get(key)
    args = (q, k, v, block_table, o, lse, slse, int(S), int(B), torch.cuda.current_stream())
    if entry is None:
        fn = build_fwd_wave32(
            Hkv, topk, block_size, float(softmax_scale), remap=remap, emit_slot_lse=bool(return_slot_lse)
        )
        entry = fn.compile(*args)
        _CACHE[key] = entry
    entry(*args)
    return (o, lse, slse) if return_slot_lse else (o, lse)
