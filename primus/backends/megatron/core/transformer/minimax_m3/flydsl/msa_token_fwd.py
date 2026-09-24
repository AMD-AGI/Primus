###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention forward, one work-group per (query token, GQA group).

The reference selects KV blocks per query *token* (``block_indices`` is
``[B, n_kv_heads, S, topk]``) and shares the selection across the GQA group's
query heads, so the natural kernel tile is one token x the group's 16 heads:
the heads fill the 16-row side of a 16x16 MFMA, and every tile walked belongs
to that token's own selection -- no union, no per-row mask. This is the
structure of Primus-Turbo's DeepSeek-V4 sparse-MLA forward (heads as M, one
token per work-group), with MSA's differences: separate K and V at
head_dim 128, and contiguous 128-key blocks instead of gathered rows.

Walk: the indexer left-packs valid block ids by score and pads with -1, and a
causal query in block ``t // 128`` can see exactly ``min(t // 128 + 1, topk)``
blocks, so that is the trip count. Each block is read in steps of
``step_keys`` keys (16 or 32), masked by key position (``key <= t``): that is
the token-level causal mask on the diagonal block and also covers a partial
last block. A -1 slot is masked whole.

Per step, the wave stages the step's K and V rows in LDS (V has to go through
LDS anyway: PV needs it transposed, via ``ds_read_tr16_b64``), then runs
QK -> online softmax -> PV, while the next step's rows are already in flight.
The work-group is a single wave, whose LDS accesses execute in order, so no
barriers are needed; latency is hidden by occupancy, which LDS (160 KB per CU)
caps, so the per-wave footprint is kept small.

``remap`` places work on XCDs: ``block`` sends whole 128-token query blocks to
one XCD, round-robin, so the tokens that share a diagonal block (and usually
more of their selection) hit the same L2 while every XCD still gets a mix of
early (cheap) and late (expensive) tokens.

Layouts are Megatron's: q/k/v and o are ``[S, B, H, D]``, so o feeds
``linear_proj`` without a permute; lse is ``[S, B, Hq]`` fp32 (natural log).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_LOG2E = 1.4426950408889634
HPW = 16  # query heads per GQA group == MFMA rows
D = 128
TK = 16  # keys per MFMA tile
KS = D // 32  # QK MFMA K-steps
DT = D // 16  # PV output d-tiles
THREADS = 64
NXCD = 8
# LDS row stride in elements: 72 dwords, == 8 mod 32, so the transposed PV reads
# hit 16 distinct banks (same rule as Turbo's sparse-MLA D_LDS).
STRIDE = D + 16


def build_fwd(
    num_kv_heads: int,
    topk: int,
    block_size: int,
    scale: float,
    step_keys: int = 16,
    k_via_lds: bool = True,
    remap: str = "none",
    pipeline: bool = True,
    barriers: bool = False,
    emit_slot_lse: bool = False,
):
    """Build the launcher.

    ``emit_slot_lse`` also writes, per (token, head, slot), the logsumexp of the
    scaled scores over that slot's keys, so ``exp(slot_lse - lse)`` is the share
    of the head's attention the slot's block received (-inf past the visible
    slots). The sparse indexer loss is built from it.

    Tuning knobs, defaults measured fastest on MI355X (B=1, Hq=64, 4k-32k):
    ``step_keys`` keys per softmax step (16 or 32); ``k_via_lds`` stages K in LDS
    (else K is read from global straight into MFMA operands); ``pipeline``
    prefetches the next step's rows into registers and double-buffers LDS;
    ``barriers`` fences LDS reuse, which a one-wave work-group does not need;
    ``remap`` as in the module docstring.
    """
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    assert step_keys in (16, 32) and block_size % step_keys == 0
    assert remap in ("none", "block")
    assert k_via_lds or not pipeline, "pipeline prefetches the LDS-staged rows"
    NT = step_keys // TK  # MFMA tiles per step
    SPB = block_size // step_keys  # steps per block
    LPR = THREADS // step_keys  # loader lanes per key row
    CPL = D // LPR  # elements each loader lane moves per row
    NV8 = CPL // 8
    NBUF = 2 if pipeline else 1
    BUF = step_keys * STRIDE

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_token_fwd_smem")
    v_off = allocator._align(allocator.ptr, 16)
    k_off = allocator._align(v_off + NBUF * BUF * 2, 16)
    allocator.ptr = allocator._align(k_off + (NBUF * BUF * 2 if k_via_lds else 0), 16)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
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
        v8 = Vec.make_type(8, elem)
        v4 = Vec.make_type(4, elem)
        v4f = Vec.make_type(4, fx.Float32)
        lds_v = SmemPtr(allocator.get_base(), v_off, elem.ir_type, shape=(NBUF * BUF,)).get()
        if const_expr(k_via_lds):
            lds_k = SmemPtr(allocator.get_base(), k_off, elem.ir_type, shape=(NBUF * BUF,)).get()

        lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)

        Sn = fx.Index(S)
        Bn = fx.Index(B)
        raw = fx.Index(gpu.block_idx.x)
        if const_expr(remap == "block"):
            # XCD x runs query blocks x, x+8, ...: w-th work-group on it -> block
            # x + 8*(w // block_size), token w % block_size within it.
            w = raw // fx.Index(NXCD)
            qblk = raw % fx.Index(NXCD) + fx.Index(NXCD) * (w // fx.Index(block_size))
            tok = qblk * fx.Index(block_size) + w % fx.Index(block_size)
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
        c_big_neg = fx.Float32(-3.0e38)  # finite running-max init: avoids -inf - -inf
        c_zero = fx.Float32(0.0)

        # ---- Q, register-resident as the B operand: head = g*16 + lo ----
        head = g * fx.Index(HPW) + lo
        q_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)
        q_packs = [
            buffer_ops.buffer_load(
                q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]

        # ---- trip count: the causally visible blocks, left-packed by the indexer ----
        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = fx.Int32(ArithValue(n_vis < fx.Int32(topk)).select(_raw(n_vis), _raw(fx.Int32(topk))))
        n_steps = fx.Index(n_valid) * fx.Index(SPB)
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)

        def kv_offset(key):
            return ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)

        # loader: lane -> (row = lane // LPR, CPL-element slice = lane % LPR)
        ld_row = lane // fx.Index(LPR)
        ld_col = (lane % fx.Index(LPR)) * fx.Index(CPL)
        ld_lds = ld_row * fx.Index(STRIDE) + ld_col

        _pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")

        def _rswap(x, op):
            v_i32 = _raw(ArithValue(_raw(x)).bitcast(fx.Int32.ir_type))
            sw = op(_pair_ty, v_i32, v_i32, False, True)
            a = llvm.extractvalue(fx.Int32.ir_type, sw, [0])
            c = llvm.extractvalue(fx.Int32.ir_type, sw, [1])
            af = fx.Float32(_raw(ArithValue(a).bitcast(fx.Float32.ir_type)))
            cf = fx.Float32(_raw(ArithValue(c).bitcast(fx.Float32.ir_type)))
            return af, cf

        def crossgrp_max(x):
            a, c = _rswap(x, rocdl.permlane16_swap)
            m = fx.Float32(arith.MaxNumFOp(_raw(a), _raw(c)).result)
            a2, c2 = _rswap(m, rocdl.permlane32_swap)
            return fx.Float32(arith.MaxNumFOp(_raw(a2), _raw(c2)).result)

        def crossgrp_sum(x):
            a, c = _rswap(x, rocdl.permlane16_swap)
            m = fx.Float32(arith.AddFOp(_raw(a), _raw(c)).result)
            a2, c2 = _rswap(m, rocdl.permlane32_swap)
            return fx.Float32(arith.AddFOp(_raw(a2), _raw(c2)).result)

        def tr16(byte_addr):
            ptr = buffer_ops.create_llvm_ptr(_raw(byte_addr), address_space=3)
            return Vec(rocdl.ds_read_tr16_b64(v4, ptr).result).bitcast(fx.Int16)

        # PV transposed-read base: lane reads row grp*4 + lo//4, cols (lo%4)*4 of each tile
        pv_base = fx.Int64(
            ((grp * fx.Index(4) + lo // fx.Index(4)) * fx.Index(STRIDE) + (lo % fx.Index(4)) * fx.Index(4))
            * fx.Index(2)
            + fx.Index(v_off)
        )

        def load_blk(step):
            return fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + step // fx.Index(SPB), vec_width=1, dtype=fx.Int32)
            )

        def step_kv0(blk, step):
            safe = fx.Int32(ArithValue(blk >= fx.Int32(0)).select(_raw(blk), _raw(fx.Int32(0))))
            return safe * fx.Int32(block_size) + fx.Int32(step % fx.Index(SPB)) * fx.Int32(step_keys)

        def load_rows(kv0):
            # the loader's slice of the step's V (and K) rows, for staging in LDS
            src = kv_offset(fx.Index(kv0) + ld_row) + ld_col
            regs = [
                buffer_ops.buffer_load(v_rsrc, src + fx.Index(c * 8), vec_width=8, dtype=elem)
                for c in range_constexpr(NV8)
            ]
            if const_expr(k_via_lds):
                regs = regs + [
                    buffer_ops.buffer_load(k_rsrc, src + fx.Index(c * 8), vec_width=8, dtype=elem)
                    for c in range_constexpr(NV8)
                ]
            return regs

        NR = NV8 * (2 if k_via_lds else 1)
        NC = NR + 1 if pipeline else 0  # carried prefetch: rows + block id
        head_row = (tok * Bn + b) * fx.Index(Hq) + head
        c_zero_v4 = Vec.filled(4, 0.0, fx.Float32)
        init = [c_big_neg, c_zero] + [c_zero_v4 for _ in range_constexpr(DT)]
        if const_expr(pipeline):
            blk0 = load_blk(fx.Index(0))
            init = init + load_rows(step_kv0(blk0, fx.Index(0))) + [blk0]
        if const_expr(emit_slot_lse):
            init = init + [c_zero]  # the current slot's partial sum, like l_run
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            m_run = it[0]
            l_run = it[1]
            o_acc = [it[2 + dt] for dt in range_constexpr(DT)]

            if const_expr(pipeline):
                rows = [it[2 + DT + c] for c in range_constexpr(NR)]
                blk = it[2 + DT + NR]
                # issue the next step's loads now so they fly under this step's MFMAs
                jn = j + fx.Index(1)
                jn = fx.Index(ArithValue(jn < n_steps).select(_raw(jn), _raw(j)))
                blk_n = load_blk(jn)
                carry_next = load_rows(step_kv0(blk_n, jn)) + [blk_n]
                buf = (j % fx.Index(NBUF)) * fx.Index(BUF)
            else:
                blk = load_blk(j)
                rows = load_rows(step_kv0(blk, j))
                carry_next = []
                buf = fx.Index(0)
            blk_ok = ArithValue(blk >= fx.Int32(0))
            kv0 = step_kv0(blk, j)
            v_regs = rows[:NV8]
            k_regs = rows[NV8:]
            ld_lds_b = buf + ld_lds
            pv_base_b = pv_base + fx.Int64(buf * fx.Index(2))

            # ---- stage the step's rows in LDS (V always, K when k_via_lds) ----
            if const_expr(k_via_lds):
                k_ops = None
            else:
                k_ops = [
                    buffer_ops.buffer_load(
                        k_rsrc,
                        kv_offset(fx.Index(kv0) + fx.Index(t * TK) + lo)
                        + grp * fx.Index(8)
                        + fx.Index(ks * 32),
                        vec_width=8,
                        dtype=elem,
                    )
                    for t in range_constexpr(NT)
                    for ks in range_constexpr(KS)
                ]
            if const_expr(barriers):
                gpu.barrier()  # WAR: the previous step's LDS reads are done
            for c in range_constexpr(NV8):
                Vec(v_regs[c]).store(lds_v, [ld_lds_b + fx.Index(c * 8)])
                if const_expr(k_via_lds):
                    Vec(k_regs[c]).store(lds_k, [ld_lds_b + fx.Index(c * 8)])
            if const_expr(barriers):
                gpu.barrier()

            # ---- S^T = K Q^T per tile: lane holds S[key = t*16 + grp*4 + i, head = lo] ----
            s_all = []
            for t in range_constexpr(NT):
                acc = Vec.filled(4, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    if const_expr(k_via_lds):
                        a_k = _raw(
                            Vec.load(
                                v8,
                                lds_k,
                                [
                                    buf
                                    + (fx.Index(t * TK) + lo) * fx.Index(STRIDE)
                                    + fx.Index(ks * 32)
                                    + grp * fx.Index(8)
                                ],
                            )
                        )
                    else:
                        a_k = k_ops[t * KS + ks]
                    acc = rocdl.mfma_f32_16x16x32_bf16(v4f, [a_k, q_packs[ks], acc])
                key0 = kv0 + fx.Int32(t * TK) + fx.Int32(grp) * fx.Int32(4)
                for i in range_constexpr(4):
                    ok = ArithValue(
                        arith.AndIOp(_raw(blk_ok), _raw(ArithValue(key0 + fx.Int32(i) <= tok_i32))).result
                    )
                    bias = fx.Float32(ok.select(_raw(c_zero), _raw(c_neg_inf)))
                    s_all.append(fx.Float32(_raw(Vec(acc)[i])) + bias)

            # ---- online softmax over the step's keys (unscaled score space) ----
            lmax = s_all[0]
            for x in s_all[1:]:
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(x)).result)
            m_new = fx.Float32(arith.MaxNumFOp(_raw(m_run), _raw(crossgrp_max(lmax))).result)
            alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_run - m_new) * c_sl)))
            p = [fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((x - m_new) * c_sl))) for x in s_all]
            lsum = p[0]
            for x in p[1:]:
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(x)).result)
            l_new = fx.Float32(alpha * l_run + lsum)  # per-grp partial, summed once at the end

            # ---- O^T += V^T P: A = V^T via transposed LDS reads, B = P ----
            alpha_v = Vec.from_elements([alpha, alpha, alpha, alpha], fx.Float32)
            new_o = []
            if const_expr(NT == 1):
                pB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in p], elem).bitcast(fx.Int16))
                for dt in range_constexpr(DT):
                    a_v = _raw(tr16(pv_base_b + fx.Int64(dt * 32)))
                    new_o.append(
                        rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_v, pB, Vec(o_acc[dt]) * Vec(alpha_v)])
                    )
            else:
                pB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in p], elem))
                for dt in range_constexpr(DT):
                    va = tr16(pv_base_b + fx.Int64(dt * 32))
                    vb = tr16(pv_base_b + fx.Int64(TK * STRIDE * 2 + dt * 32))
                    a_v = _raw(
                        Vec.from_elements(
                            [va[0], va[1], va[2], va[3], vb[0], vb[1], vb[2], vb[3]], fx.Int16
                        ).bitcast(elem)
                    )
                    new_o.append(rocdl.mfma_f32_16x16x32_bf16(v4f, [a_v, pB, Vec(o_acc[dt]) * Vec(alpha_v)]))

            # ---- per-slot logsumexp: the slot's partial sum, rescaled like l ----
            extra = []
            if const_expr(emit_slot_lse):
                sub = j % fx.Index(SPB)
                ls_run = it[2 + DT + NC]
                ls_prev = fx.Float32(ArithValue(sub == fx.Index(0)).select(_raw(c_zero), _raw(ls_run)))
                ls_new = fx.Float32(alpha * ls_prev + lsum)
                slot_lse = fmath.log(crossgrp_sum(ls_new)) + fx.Float32(m_new * c_scale)
                last = ArithValue(sub == fx.Index(SPB - 1))
                buffer_ops.buffer_store(
                    slot_lse,
                    slse_rsrc,
                    (head_row * fx.Index(topk) + j // fx.Index(SPB)) * fx.Index(4),
                    mask=_raw(
                        ArithValue(arith.AndIOp(_raw(last), _raw(ArithValue(grp == fx.Index(0)))).result)
                    ),
                    offset_is_bytes=True,
                )
                extra = [ls_new]

            results = yield [m_new, l_new] + new_o + carry_next + extra

        # ---- epilogue: lane holds O[head = lo, d = dt*16 + grp*4 + i] ----
        m_run = results[0]
        l_sum = crossgrp_sum(results[1])
        o_acc = [results[2 + dt] for dt in range_constexpr(DT)]
        inv = fx.Float32(rocdl.rcp(fx.Float32.ir_type, _raw(l_sum)))
        inv_v = Vec.from_elements([inv, inv, inv, inv], fx.Float32)
        o_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)
        for dt in range_constexpr(DT):
            ov = Vec(o_acc[dt]) * inv_v
            pk0 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1]))
            pk1 = rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3]))
            buffer_ops.buffer_store(
                _raw(Vec.from_elements([fx.Int32(_raw(pk0)), fx.Int32(_raw(pk1))], fx.Int32)),
                o_rsrc,
                (o_row + fx.Index(dt * 16) + grp * fx.Index(4)) * fx.Index(2),
                offset_is_bytes=True,
            )
        lse_val = fx.Float32(m_run * c_scale) + fmath.log(l_sum)
        buffer_ops.buffer_store(
            lse_val,
            lse_rsrc,
            ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(4),
            mask=_raw(ArithValue(grp == fx.Index(0))),
            offset_is_bytes=True,
        )
        if const_expr(emit_slot_lse):
            # the slots past the visible blocks were never walked
            for s in range_constexpr(topk):
                unvisited = ArithValue(fx.Int32(s) >= n_valid)
                buffer_ops.buffer_store(
                    c_neg_inf,
                    slse_rsrc,
                    (head_row * fx.Index(topk) + fx.Index(s)) * fx.Index(4),
                    mask=_raw(
                        ArithValue(arith.AndIOp(_raw(unvisited), _raw(ArithValue(grp == fx.Index(0)))).result)
                    ),
                    offset_is_bytes=True,
                )

    @flyc.jit
    def launch(Q, K, V, TBL, O, LSE, SLSE, S, B, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, TBL, O, LSE, SLSE, S, B).launch(
            grid=(fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1), block=(THREADS, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CACHE: dict = {}


def msa_token_fwd(q, k, v, block_table, softmax_scale=None, block_size=128, return_slot_lse=False, **config):
    """MSA forward.

    Args:
        q: ``[S, B, Hq, 128]`` bf16, post-rope.
        k, v: ``[S, B, Hkv, 128]`` bf16, ``Hq == 16 * Hkv``.
        block_table: ``[B, Hkv, S, topk]`` int32 block ids, left-packed, -1 padded
            (the indexer's ``block_indices``).
        softmax_scale: defaults to ``1 / sqrt(128)``.
        return_slot_lse: also return ``slot_lse`` ``[S, B, Hq, topk]`` fp32, the
            logsumexp over each slot's keys; ``exp(slot_lse - lse)`` is the slot's
            share of the head's attention.
        config: ``build_fwd`` tuning knobs (``step_keys``, ``k_via_lds``, ``remap``).

    Returns:
        ``o`` ``[S, B, Hq, 128]`` bf16 and ``lse`` ``[S, B, Hq]`` fp32 (and
        ``slot_lse``).
    """
    S, B, Hq, Dq = q.shape
    Hkv = k.shape[2]
    topk = block_table.shape[-1]
    assert Dq == D and k.shape[-1] == D and v.shape[-1] == D, f"head_dim must be {D}"
    assert Hq == HPW * Hkv, f"need {HPW} query heads per KV head, got Hq={Hq} Hkv={Hkv}"
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert block_table.shape == (B, Hkv, S, topk) and block_table.dtype == torch.int32
    if softmax_scale is None:
        softmax_scale = D**-0.5
    if config.get("remap", "none") == "block" and S % (NXCD * block_size) != 0:
        config = {**config, "remap": "none"}
    q, k, v, block_table = (t.contiguous() for t in (q, k, v, block_table))

    o = torch.empty_like(q)
    lse = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)
    slse = torch.empty((S, B, Hq, topk) if return_slot_lse else (1,), dtype=torch.float32, device=q.device)
    config = {**config, "emit_slot_lse": bool(return_slot_lse)}
    key = (Hkv, topk, block_size, float(softmax_scale), tuple(sorted(config.items())))
    entry = _CACHE.get(key)
    args = (q, k, v, block_table, o, lse, slse, int(S), int(B), torch.cuda.current_stream())
    if entry is None:
        fn = build_fwd(Hkv, topk, block_size, float(softmax_scale), **config)
        entry = fn.compile(*args)
        _CACHE[key] = entry
    entry(*args)
    return (o, lse, slse) if return_slot_lse else (o, lse)
