###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""gfx1250 MSA forward: one wave32 per query token, WMMA 16x16x32 bf16.

Same contract as ``msa_token_fwd``, on the wave32 layout documented in ``msa_w32``:

* A step is 32 keys of a block, two tiles of 16.
* QK is ``S[key, head] = K[key] · Q[head]``, one tile at a time. A is the key
  row (two ``ds_load_b128`` from LDS), B is the query row read as a column.
  Each lane then holds 8 keys of one head per tile.
* The keys of a step reduce with one ``permlane16`` (low 8 + high 8).
* PV is ``O[head, d] += P[head, key] * V[key, d]``, one K=32 WMMA per d-tile:
  each lane group's B is one tile's ``P``, gathered with a ``permlane16_swap``,
  and the A operand ``V^T`` is read from row-major LDS with ``ds_load_tr16_b128``.
  Lane row ``half*16 + lane%8`` (then +8), column ``((lane//8)%2)*8`` gives each
  lane keys ``half*16 + 0..15`` of column ``d = lane%16``: the K order of a
  transposed copy, so the WMMA sums in the same order.
* The epilogue stores the 8 contiguous ``d`` values each lane holds.
* Staging: each step's K and V tiles ([32 keys, 128], 1 KB apart in global) are
  copied to a per-wave 2-stage LDS ring by the Tensor Data Mover
  (``tensor_load_to_lds``): no VGPRs, no per-lane address math, no register
  transposes. Step ``j+1`` is in flight while step ``j`` computes. Keys past ``S``
  are zero-filled, as an out-of-range buffer load reads 0.
* ``slot_lse`` is written once per block, on its last step.
* A workgroup holds ``FWD_TOKENS_PER_WG`` waves, adjacent tokens of one KV group (one query
  block, so the same step count). They step in lockstep (one barrier per step), and a step's
  K/V tile that several of them select at the same step is staged once, by the first of them,
  and read from its LDS slot by all: the kernel is L2->LDS bandwidth-bound, and the forced
  local block and similar selections of neighbouring tokens make such repeats common.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import ArithValue
from flydsl.expr.utils.arith import _to_raw as _raw
from primus_turbo.flydsl.utils import buffer_ops

from primus.backends.megatron.core.fusions import bitwise_check
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
    pack_pair_bf16,
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
    tokens_per_wg: int = 1,
):
    """Build the wave32 forward kernel; ``tokens_per_wg`` adjacent tokens share a workgroup.

    Why the lockstep layout (``tokens_per_wg > 1``) cannot deadlock or hang the GPU:

    * The only synchronization it adds is one ``gpu.barrier()`` per loop iteration. There is no
      spin-wait, no flag, counter or atomic, and no wait on state another workgroup writes.
    * Every wave of a workgroup runs the same number of barriers. The loop runs ``n_steps``
      iterations and has no early exit, ``break`` or wave-dependent branch around the barrier.
      ``n_steps = min(tok_wg // block_size + 1, topk) * steps_per_block`` is computed from the
      workgroup's grid position alone (``tok_wg`` is the workgroup's highest token,
      ``S - 1 - (block_idx.x // (B * Hkv)) * tokens_per_wg``), ``S`` and build constants: never
      from the block table or any tensor contents, and never from the wave's index. A ``-1``
      padded table entry only masks that step's keys (``blk_ok``); it does not shorten the loop.
      Query blocks with fewer visible blocks than ``topk`` give a smaller ``n_steps``, the same for
      all waves of the workgroup; ``n_steps`` is at least one block's steps.
    * ``tensor_wait`` only waits for Tensor Data Mover copies the wave itself issued (its own
      counter), so no wave waits on a copy whose issue is another wave's decision: a wave that
      does not own a tile issues nothing, waits for nothing, and reads the owner's slot only
      after the barrier that follows the owner's own ``tensor_wait``.
      (``tensor_wait`` with nothing issued is relied on to return at once, as the tables whose
      tiles are all shared exercise: most waves then never issue.)
    * Which wave owns a tile is a pure function of the (read-only) table rows of the workgroup's
      tokens, evaluated identically by every wave; it decides who copies, not how many barriers
      run. A wrong or adversarial table (out-of-range ids) reads zero-filled keys, as in the
      one-token kernel; it cannot change the trip count.
    * Eligibility: the dispatcher builds this layout only when ``S`` and ``block_size`` are
      multiples of ``tokens_per_wg`` (then the workgroup's tokens share one query block, and
      the group's ``n_steps`` is each token's own, which the bitwise self-check also proves on
      the first call); otherwise it runs the one-token kernel, and the launcher refuses
      (``ValueError``) an ``S`` the layout does not divide before anything is launched.
      ``tokens_per_wg=1`` emits no barrier at all.
    """
    NBUF = 2
    # waves per workgroup = adjacent tokens of one (b, kv group); they run their steps in
    # lockstep and a step's K/V tile is staged once for every wave that needs it
    WPG = tokens_per_wg
    assert WPG == 1 or (remap == "lpt" and block_size % WPG == 0)
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    KPS = 2 * TK  # keys per step
    assert block_size % KPS == 0
    assert remap in ("none", "block", "lpt")
    SPB = block_size // KPS
    # per-wave LDS ring of NBUF stages: K [KPS, KROW] then V [KPS, VROW] (bf16, padded rows)
    KROW = D + 8
    VROW = D + 16
    K_BYTES = KPS * KROW * 2
    STAGE = K_BYTES + KPS * VROW * 2
    LDS_BYTES = NBUF * STAGE

    @flyc.kernel(known_block_size=[WAVE * WPG, 1, 1])
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
        v8_ty = ir.VectorType.get([8], elem.ir_type)
        # Everything the shared layout adds is under a build-time `WPG > 1`: tokens_per_wg=1 emits
        # the one-token-per-workgroup kernel unchanged (no barrier, no owner lookup).
        smem = fx.SharedAllocator().allocate(LDS_BYTES * WPG)
        lds_wg = fx.Int32(fx.ptrtoint(smem.peek().ptr))

        def sel_i32(cond, a, b):
            return fx.Int32(ArithValue(cond).select(_raw(a), _raw(b)))

        def andb(a, b):
            return ArithValue(arith.AndIOp(_raw(a), _raw(b)).result)

        if const_expr(WPG > 1):
            tid = fx.Index(gpu.thread_idx.x)
            wave_u = fx.Index(
                fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, _raw(fx.Int32(tid // fx.Index(WAVE)))))
            )
            lane = tid % fx.Index(WAVE)
        else:
            lane = fx.Index(gpu.thread_idx.x)
        lo = lane % fx.Index(16)
        half = lane // fx.Index(16)  # 0: K 0..15, 1: K 16..31 of the WMMA operand

        Sn = fx.Index(S)
        Bn = fx.Index(B)
        raw = fx.Index(gpu.block_idx.x)
        if const_expr(remap == "lpt"):
            bgn = Bn * fx.Index(Hkv)
            if const_expr(WPG > 1):
                ti = (raw // bgn) * fx.Index(WPG) + wave_u
            else:
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

        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = sel_i32(ArithValue(n_vis < fx.Int32(topk)), n_vis, fx.Int32(topk))
        if const_expr(WPG > 1):
            # The trip count of the lockstep loop (one barrier per step) comes from the workgroup's
            # grid position, not from the wave: the workgroup's highest token. Its tokens share a
            # query block (S and block_size are multiples of WPG), so this is each token's own count.
            tok_wg = fx.Int32(Sn - fx.Index(1) - (raw // bgn) * fx.Index(WPG))
            n_vis_wg = tok_wg // fx.Int32(block_size) + fx.Int32(1)
            n_steps = fx.Index(
                sel_i32(ArithValue(n_vis_wg < fx.Int32(topk)), n_vis_wg, fx.Int32(topk))
            ) * fx.Index(SPB)
        else:
            n_steps = fx.Index(n_valid) * fx.Index(SPB)
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)
        # wave w has token tok_wave0 - w (all in one query block: same n_steps); tbl_row0 is the
        # lowest token's (the last wave's) table row
        if const_expr(WPG > 1):
            tbl_row0 = ((b * fx.Index(Hkv) + g) * Sn + (tok + wave_u - fx.Index(WPG - 1))) * fx.Index(topk)

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

        ptr3 = ir.Type.parse("!llvm.ptr<3>")

        def lds_ptr(addr):
            return llvm.inttoptr(ptr3, _raw(fx.Int32(addr)))

        def lds_b128(addr):
            return Vec(llvm.load(v8_ty, lds_ptr(addr), alignment=16))

        def lds_tr16(addr):
            r = _rocdl.ds_load_tr16_b128(v8_ty, lds_ptr(addr))
            return Vec(r if isinstance(r, ir.Value) else r.result)

        lds_ty = fx.PointerType.get(elem_ty=elem.ir_type, address_space=fx.AddressSpace.Shared, alignment=16)
        row_stride = Bn * fx.Index(Hkv) * fx.Index(D)  # elements between consecutive keys
        head_off = (b * fx.Index(Hkv) + g) * fx.Index(D)

        def tdm_tile(T_, kv0, lds_addr, row_elems):
            off = fx.Int64(fx.Int32(kv0)) * fx.Int64(fx.Int32(row_stride)) + fx.Int64(fx.Int32(head_off))
            gbase = fx.add_offset(fx.get_iter(T_), off)
            g_view = fx.Tensor(fx.make_view(gbase, fx.make_layout((KPS, D), (D, 1))))
            # Keys at or past S (a partial last block) are zero-filled by the TDM, as an
            # out-of-range buffer load reads 0: causality gives them P = 0, which must not meet
            # whatever memory follows K / V.
            rows = fx.Int32(S) - fx.Int32(kv0)
            rows = sel_i32(ArithValue(rows < fx.Int32(KPS)), rows, fx.Int32(KPS))
            rows = sel_i32(ArithValue(rows > c_i0), rows, c_i0)
            atom = fx.rocdl.make_tdm_atom(
                g_view,
                [rows, None],
                strides=[fx.Int32(row_stride), None],
                num_warps=1,
                pad_interval=D,
                pad_amount=row_elems - D,
            )
            l_view = fx.Tensor(
                fx.make_view(fx.inttoptr(lds_ty, lds_addr), fx.make_layout((KPS, D), (row_elems, 1)))
            )
            fx.copy_atom_call(atom, g_view, l_view)

        # LDS: NBUF stages x WPG slots of STAGE bytes. A step's tile (its kv0) is staged once per
        # workgroup, by the first wave that needs it, into that wave's slot; every wave reads the
        # owner's slot. Same tile bytes, same per-token step order: the math is unchanged.
        def stage_base(jj):
            return lds_wg + fx.Int32(jj % fx.Index(NBUF)) * fx.Int32(STAGE * WPG)

        def step_blk(jj):
            return fx.Int32(
                buffer_ops.buffer_load(t_rsrc, tbl_row + jj // fx.Index(SPB), vec_width=1, is_scalar=True)
            )

        def step_kv0(blk, jj):
            safe = sel_i32(ArithValue(blk >= c_i0), blk, c_i0)
            return safe * fx.Int32(block_size) + fx.Int32(jj % fx.Index(SPB)) * fx.Int32(KPS)

        def owner_of(jj):
            # first wave of the workgroup whose step-jj tile (kv0) equals this wave's
            # (wave_u itself when no lower wave shares it)
            mine = step_kv0(step_blk(jj), jj)
            own = fx.Int32(WPG - 1)
            for w2i in range_constexpr(WPG):
                w2 = WPG - 1 - w2i
                row2 = tbl_row0 + fx.Index((WPG - 1 - w2) * topk)  # wave w2's token is tok_wave0 - w2
                b2 = fx.Int32(
                    buffer_ops.buffer_load(t_rsrc, row2 + jj // fx.Index(SPB), vec_width=1, is_scalar=True)
                )
                own = sel_i32(ArithValue(step_kv0(b2, jj) == mine), fx.Int32(w2), own)
            return own, mine

        def issue(js, jj):
            # stage of step js, tile of step jj (jj = js, or the clamped last step)
            if const_expr(WPG > 1):
                own, kv0 = owner_of(jj)
                if own == fx.Int32(wave_u):
                    sb = stage_base(js) + fx.Int32(wave_u) * fx.Int32(STAGE)
                    tdm_tile(K, kv0, sb, KROW)
                    tdm_tile(V, kv0, sb + fx.Int32(K_BYTES), VROW)
            else:
                kv0 = step_kv0(step_blk(jj), jj)
                sb = stage_base(js)
                tdm_tile(K, kv0, sb, KROW)
                tdm_tile(V, kv0, sb + fx.Int32(K_BYTES), VROW)

        # per-lane LDS byte offsets: K A-operand row lo, d chunk half*16; V^T via ds_load_tr16_b128
        # with lane row half*16 + lane%8, column ((lane//8)%2)*8: the lane gets keys half*16 + 0..7
        # (then +8..15) of column d = lo.
        k_lane = fx.Int32(lo) * fx.Int32(KROW * 2) + fx.Int32(half) * fx.Int32(32)
        v_lane = (fx.Int32(half) * fx.Int32(16) + fx.Int32(lane % fx.Index(8))) * fx.Int32(
            VROW * 2
        ) + fx.Int32((lane // fx.Index(8)) % fx.Index(2)) * fx.Int32(16)

        grp0 = ArithValue(half == fx.Index(0))
        head_row = (tok * Bn + b) * fx.Index(Hq) + head
        c_zero_v8 = Vec.filled(8, 0.0, fx.Float32)
        init = [c_big_neg, c_zero] + [c_zero_v8 for _ in range_constexpr(DT)]
        if const_expr(emit_slot_lse):
            init = init + [c_zero]
        # prologue: stages 0 .. NBUF-2 in flight
        for pj in range_constexpr(NBUF - 1):
            jj = fx.Index(sel_i32(ArithValue(fx.Index(pj) < n_steps), fx.Int32(pj), fx.Int32(0)))
            issue(fx.Index(pj), jj)
        results = init
        for j, it in range(fx.Index(0), n_steps, fx.Index(1), init=init):
            m_cur = it[0]
            l_cur = it[1]
            o_cur = [it[2 + dt] for dt in range_constexpr(DT)]
            sub = j % fx.Index(SPB)
            blk = step_blk(j)
            blk_ok = ArithValue(blk >= c_i0)
            kv0 = step_kv0(blk, j)
            # step j's tiles landed; then put step j+NBUF-1 in flight into the stage step j-1 used
            # (its LDS reads were consumed by step j-1's WMMAs, so the wave is done with it)
            tdm_ops.tensor_wait(2 * (NBUF - 2))
            if const_expr(WPG > 1):
                gpu.barrier()  # every owner's step-j tile landed; every wave is done with step j-1's stage
            j_f = j + fx.Index(NBUF - 1)
            j_c = fx.Index(sel_i32(ArithValue(j_f < n_steps), fx.Int32(j_f), fx.Int32(j)))
            issue(j_f, j_c)
            if const_expr(WPG > 1):
                own_j, _ = owner_of(j)
                sb = stage_base(j) + own_j * fx.Int32(STAGE)
            else:
                sb = stage_base(j)
            kb = sb + k_lane
            vb = sb + fx.Int32(K_BYTES) + v_lane

            # S[key, head] = K[key] · Q[head]. A row = lane%16 = key in the tile.
            tiles = []
            for t in range_constexpr(KPS // TK):
                acc = Vec.filled(8, 0.0, fx.Float32)
                for ks in range_constexpr(KS):
                    ka = kb + fx.Int32((t * TK * KROW + ks * 32) * 2)
                    a_k = cat_bf16_16(lds_b128(ka), lds_b128(ka + fx.Int32(16)))
                    acc = wmma_f32_16x16x32_bf16(a_k, q_packs[ks], acc)
                # lane holds keys (t*16 + half*8 + i) of head lo
                scores = []
                key0 = kv0 + fx.Int32(t * TK) + fx.Int32(half) * fx.Int32(8)
                for i in range_constexpr(8):
                    ok = andb(blk_ok, ArithValue(key0 + fx.Int32(i) <= tok_i32))
                    raw_s = fx.Float32(_raw(Vec(acc)[i]))
                    scores.append(fx.Float32(ok.select(_raw(raw_s), _raw(c_neg_inf))))
                tiles.append(scores)

            lmax = tiles[0][0]
            for x in tiles[0][1:] + tiles[1]:
                lmax = fx.Float32(arith.MaxNumFOp(_raw(lmax), _raw(x)).result)
            m_new = fx.Float32(arith.MaxNumFOp(_raw(m_cur), _raw(crossgrp_max(lmax))).result)
            alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_cur - m_new) * c_sl)))
            p = [
                [fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((x - m_new) * c_sl))) for x in scores]
                for scores in tiles
            ]
            lsum = p[0][0]
            for x in p[0][1:] + p[1]:
                lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(x)).result)
            l_new = fx.Float32(alpha * l_cur + lsum)

            # B column = head lo; K 0..15 is the first tile's keys, K 16..31 the second's
            pB = pack_pair_bf16(p[0], p[1])

            alpha_v = Vec.from_elements([alpha] * 8, fx.Float32)
            new_o = []
            for dt in range_constexpr(DT):
                va = vb + fx.Int32(dt * 16 * 2)
                a_v = cat_bf16_16(lds_tr16(va), lds_tr16(va + fx.Int32(8 * VROW * 2)))
                new_o.append(wmma_f32_16x16x32_bf16(a_v, pB, Vec(o_cur[dt]) * Vec(alpha_v)))

            out = [m_new, l_new] + new_o
            if const_expr(emit_slot_lse):
                reset = ArithValue(sub == fx.Index(0))
                ls_prev = fx.Float32(reset.select(_raw(c_zero), _raw(it[2 + DT])))
                ls_new = fx.Float32(alpha * ls_prev + lsum)
                slot = j // fx.Index(SPB)
                # only the block's last step stores; skip the log / cross-lane sum on the others
                if sub == fx.Index(SPB - 1):
                    slot_lse = fmath.log(crossgrp_sum(ls_new)) + fx.Float32(m_new * c_scale)
                    buffer_ops.buffer_store(
                        slot_lse,
                        slse_rsrc,
                        (head_row * fx.Index(topk) + slot) * fx.Index(4),
                        mask=_raw(andb(grp0, ArithValue(fx.Int32(slot) < n_valid))),
                        offset_is_bytes=True,
                    )
                out = out + [ls_new]
            results = yield out

        tdm_ops.tensor_wait(0)  # no TDM write may outlive the wave
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
            if WPG == 1:
                return (fx.Index(S) * fx.Index(B) * fx.Index(Hkv), 1, 1)
            return (fx.Index(S) // fx.Index(WPG) * fx.Index(B) * fx.Index(Hkv), 1, 1)
        return (fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1)

    @flyc.jit
    def launch(Q, K, V, TBL, O, LSE, SLSE, S, B, stream):
        k_fn(Q, K, V, TBL, O, LSE, SLSE, S, B).launch(
            grid=_grid(S, B), block=(WAVE * WPG, 1, 1), stream=stream
        )

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CACHE: dict = {}
FWD_TOKENS_PER_WG = 4


def fwd_tokens_per_wg(S: int, block_size: int, remap: str) -> int:
    t = FWD_TOKENS_PER_WG
    return t if remap == "lpt" and S % t == 0 and block_size % t == 0 else 1


def msa_token_fwd_wave32(
    q,
    k,
    v,
    block_table,
    softmax_scale=None,
    block_size=128,
    return_slot_lse=False,
    shared_tiles=False,
    **config,
):
    """Wave32 MSA forward. Same tensors as ``msa_token_fwd``; ``waves`` is ignored.

    ``shared_tiles`` (config ``msa_shared_tile_staging``) runs ``FWD_TOKENS_PER_WG`` adjacent
    tokens per workgroup with one staging of each K/V tile they share. The first call for a
    shape runs both layouts on that call's tensors and compares every bit of ``o``, ``lse`` and
    ``slot_lse`` (``bitwise_check``); on a mismatch the run keeps one token per workgroup. So
    does a FlyDSL whose build of the shared layout fails (it needs ``gpu.barrier`` and
    ``rocdl.readfirstlane``, which the one-token kernel does not use): the check takes a build
    error for a mismatch. ``tokens_per_wg=1`` builds the kernel this file had before the shared
    layout, statement for statement."""
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
    slse_shape = (S, B, Hq, topk) if return_slot_lse else (1,)

    def run(tpw):
        if tpw > 1 and (S % tpw or block_size % tpw or remap != "lpt"):
            # the lockstep layout's barrier count is workgroup-uniform only for these shapes
            raise ValueError(f"msa_shared_tile_staging forward: S={S} block_size={block_size} remap={remap}")
        o = torch.empty_like(q)
        lse = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)
        slse = torch.empty(slse_shape, dtype=torch.float32, device=q.device)
        check_buffer_bytes(q=q, k=k, v=v, block_table=block_table, slot_lse=slse)
        key = (Hkv, topk, block_size, float(softmax_scale), remap, bool(return_slot_lse), tpw)
        entry = _CACHE.get(key)
        args = (q, k, v, block_table, o, lse, slse, int(S), int(B), torch.cuda.current_stream())
        if entry is None:
            fn = build_fwd_wave32(
                Hkv,
                topk,
                block_size,
                float(softmax_scale),
                remap=remap,
                emit_slot_lse=bool(return_slot_lse),
                tokens_per_wg=tpw,
            )
            entry = fn.compile(*args)
            _CACHE[key] = entry
        entry(*args)
        return (o, lse, slse) if return_slot_lse else (o, lse)

    # eligibility first: the shared layout needs lpt order and an S / block size it divides
    tpw = fwd_tokens_per_wg(int(S), block_size, remap) if shared_tiles else 1
    if tpw > 1 and not bitwise_check.passed(
        "msa_shared_tile_staging forward",
        lambda: run(tpw),
        lambda: run(1),
        # everything the kernel is built from or lays its grid out by
        key=(S, B, Hkv, topk, block_size, float(softmax_scale), remap, bool(return_slot_lse), tpw),
    ):
        tpw = 1
    return run(tpw)
