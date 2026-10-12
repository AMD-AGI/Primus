###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MiniMax Sparse Attention forward, one work-group per (``waves`` adjacent query tokens, GQA group).

The reference selects KV blocks per query *token* (``block_indices`` is
``[B, n_kv_heads, S, topk]``) and shares the selection across the GQA group's
query heads, so the natural kernel tile is one token x the group's 16 heads:
the heads fill the 16-row side of a 16x16 MFMA.

A work-group holds ``waves`` waves, each owning ONE query token and its own
``o`` accumulator. The waves walk the UNION of their tokens' selected blocks,
stage each union block in LDS exactly once, and a wave that does not own a
union block skips QK/PV for it -- a wave-uniform skip, so the branch is scalar.
Adjacent tokens of the same (batch, kv-head) share most of their top-k
selection (and all of it below ``topk * block_size``, where every visible block
is selected), so the union is far shorter than the sum: the shared K/V tile is
read from L2 once instead of once per token.

Walk: the indexer left-packs valid block ids by score and pads with -1, and a
causal query in block ``t // 128`` can see exactly ``min(t // 128 + 1, topk)``
blocks. The union list, its length, and each wave's slot index for each union
entry are built once in LDS in the prologue (see ``union prologue`` below).
Each union block is read in steps of ``step_keys`` keys (16 or 32), masked by
key position (``key <= t``) with each wave using its OWN token: that is the
token-level causal mask on the diagonal block and also covers a partial last
block. The softmax is online with a running maximum, so no score range can
overflow or underflow ``exp``.

Per step, the work-group cooperatively stages the step's K and V rows in LDS
(V has to go through LDS anyway: PV needs it transposed, via
``ds_read_tr16_b64``), then every owning wave runs QK -> online softmax -> PV
against the one staged copy, while the next step's rows are already in flight.
With more than one wave the LDS buffer is shared, so the reuse is fenced by two
``gpu.barrier()`` per step (WAR before the writes, RAW after them).

``remap`` places work on XCDs: ``block`` sends whole 128-token query blocks to
one XCD, round-robin. ``lpt`` instead flattens the grid to 1-D with the
(batch, kv-head) index innermost and walks tokens from last to first, so the
longest work-groups are dispatched first. The union walk requires ``lpt`` (it
is what makes a work-group's tokens adjacent) and an ``S`` the wave count
divides; ``msa_token_fwd`` lowers the wave count until it does.

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
WAVE = 64
NXCD = 8
# The coalesced slot_lse store covers at most 4 lane groups x 4 slots per head,
# and the union prologue gives one lane to each of the 4 waves' table slots.
MAX_TOPK = 16
# LDS row stride in elements: 72 dwords, == 8 mod 32, so the transposed PV reads
# hit 16 distinct banks (same rule as Turbo's sparse-MLA D_LDS).
STRIDE = D + 16
# Buffer offsets are 32-bit byte offsets, and a masked lane parks its offset at
# 0x7FFFFFFF to fall off the buffer; that only works while every buffer is
# smaller than 2 GiB.
MAX_BUFFER_BYTES = 2**31


def check_buffer_bytes(**tensors):
    """Refuse any tensor a kernel would address past 32-bit buffer offsets."""
    for name, t in tensors.items():
        nbytes = t.numel() * t.element_size()
        assert nbytes < MAX_BUFFER_BYTES, (
            f"{name} is {nbytes} bytes; the flydsl MSA kernels address buffers with 32-bit "
            f"offsets and need every tensor under {MAX_BUFFER_BYTES} bytes (shape {tuple(t.shape)})"
        )


def build_fwd(
    num_kv_heads: int,
    topk: int,
    block_size: int,
    scale: float,
    step_keys: int = 16,
    k_via_lds: bool = True,
    remap: str = "lpt",
    pipeline: bool = True,
    barriers: bool = False,
    emit_slot_lse: bool = False,
    waves: int = 4,
):
    """Build the launcher.

    ``emit_slot_lse`` also writes, per (token, head, slot), the logsumexp of the
    scaled scores over that slot's keys, so ``exp(slot_lse - lse)`` is the share
    of the head's attention the slot's block received (-inf past the visible
    slots). The sparse indexer loss is built from it.

    Tuning knobs, defaults measured fastest on MI355X (B=1, Hq=64, 4k-32k):
    ``step_keys`` keys per softmax step (16 or 32); ``k_via_lds`` stages K in LDS
    (else K is read from global straight into MFMA operands); ``pipeline``
    prefetches the next step's rows into registers; ``waves`` query tokens per
    work-group, walking their union (``>1`` needs ``remap == "lpt"`` and an ``S``
    it divides); ``barriers`` fences LDS reuse, which a one-wave work-group does
    not need and a multi-wave one always does; ``remap`` as in the module
    docstring.
    """
    elem = fx.BFloat16
    Hkv = num_kv_heads
    Hq = Hkv * HPW
    assert step_keys in (16, 32) and block_size % step_keys == 0
    assert remap in ("none", "block", "lpt")
    assert k_via_lds or not pipeline, "pipeline prefetches the LDS-staged rows"
    NW = int(waves)
    assert NW >= 1
    assert NW == 1 or remap == "lpt", "the union walk pairs adjacent lpt tokens"
    assert NW * topk <= WAVE, "the union prologue gives one lane to each table slot"
    NTHREADS = WAVE * NW
    if NW > 1:
        barriers = True  # a shared LDS buffer is not fenced by program order
    NT = step_keys // TK  # MFMA tiles per step
    SPB = block_size // step_keys  # steps per block
    # Loader geometry: each lane moves exactly one 16-byte chunk, and LPR
    # consecutive lanes cover one key row contiguously, so a load instruction
    # addresses RPP whole rows instead of a 16-byte slice of every row. With NW
    # waves the whole work-group loads cooperatively, so RPP grows and the
    # per-lane instruction count falls.
    LPR = D // 8  # lanes per key row: 16 x 16 B == the 256-byte row
    RPP = NTHREADS // LPR  # key rows covered per load instruction
    assert step_keys % RPP == 0, "the step's rows must divide over the work-group"
    NV8 = step_keys // RPP  # load instructions per step, per tensor
    # LDS accesses retire in program order within a wave, and across waves the
    # two per-step barriers fence reuse, so a single buffer is safe and halves
    # the LDS footprint that caps occupancy.
    NBUF = 1
    BUF = step_keys * STRIDE
    USZ = NW * topk  # union slots: the sum of the waves' lists at worst
    NWT = NW * topk  # flattened table entries, one per lane

    # slot_lse written one fp32 per lane into rows 4*topk bytes apart would put 4 B
    # into each 64 B write granule, so it is staged per wave in LDS and emitted as
    # one contiguous run per head. SLS = topk+1 keeps the LDS rows on distinct banks
    # (lane lo reads at (lo*SLS + grp*4) % 32).
    SLS = topk + 1
    slot_coalesce = emit_slot_lse and topk % 4 == 0
    NPL = topk // 4 if slot_coalesce else 0  # lanes per head that store

    allocator = SmemAllocator(None, arch=get_hip_arch(), global_sym_name="msa_token_fwd_smem")
    v_off = allocator._align(allocator.ptr, 16)
    k_off = allocator._align(v_off + NBUF * BUF * 2, 16)
    allocator.ptr = allocator._align(k_off + (NBUF * BUF * 2 if k_via_lds else 0), 16)
    if NW > 1:
        # union tables. Each array carries one extra slot: a masked store parks
        # its address there instead of branching (flydsl has no masked ds_write).
        ub_off = allocator._align(allocator.ptr, 16)  # union block ids
        us_off = allocator._align(ub_off + (USZ + 1) * 4, 16)  # per-wave slot of each union entry
        tb_off = allocator._align(us_off + (NW * USZ + 1) * 4, 16)  # each wave's own left-packed list
        ui_off = allocator._align(tb_off + (NWT + 1) * 4, 16)  # union index of each table entry
        allocator.ptr = allocator._align(ui_off + (NWT + 1) * 4, 16)
    if slot_coalesce:
        sl_off = allocator._align(allocator.ptr, 16)  # per-wave slot_lse staging
        allocator.ptr = allocator._align(sl_off + (NW * HPW * SLS + 1) * 4, 16)

    @flyc.kernel(known_block_size=[NTHREADS, 1, 1])
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
        v1i = Vec.make_type(1, fx.Int32)
        lds_v = SmemPtr(allocator.get_base(), v_off, elem.ir_type, shape=(NBUF * BUF,)).get()
        if const_expr(k_via_lds):
            lds_k = SmemPtr(allocator.get_base(), k_off, elem.ir_type, shape=(NBUF * BUF,)).get()
        if const_expr(slot_coalesce):
            lds_sl = SmemPtr(
                allocator.get_base(), sl_off, fx.Float32.ir_type, shape=(NW * HPW * SLS + 1,)
            ).get()
        if const_expr(NW > 1):
            lds_ub = SmemPtr(allocator.get_base(), ub_off, fx.Int32.ir_type, shape=(USZ + 1,)).get()
            lds_us = SmemPtr(allocator.get_base(), us_off, fx.Int32.ir_type, shape=(NW * USZ + 1,)).get()
            lds_tb = SmemPtr(allocator.get_base(), tb_off, fx.Int32.ir_type, shape=(NWT + 1,)).get()
            lds_ui = SmemPtr(allocator.get_base(), ui_off, fx.Int32.ir_type, shape=(NWT + 1,)).get()

        def sel_i(cond, a, b):
            return fx.Index(ArithValue(cond).select(_raw(a), _raw(b)))

        def sel_i32(cond, a, b):
            return fx.Int32(ArithValue(cond).select(_raw(a), _raw(b)))

        def andb(a, b):
            return ArithValue(arith.AndIOp(_raw(a), _raw(b)).result)

        def wave_prefix(pred):
            """Exclusive count of lanes below this one for which ``pred`` holds."""
            i32t = fx.Int32.ir_type
            i64t = ir.IntegerType.get_signless(64)
            m = rocdl.ballot(i64t, _raw(pred))
            c32 = arith.ConstantOp(i64t, ir.IntegerAttr.get(i64t, 32)).result
            z = arith.ConstantOp(i32t, ir.IntegerAttr.get(i32t, 0)).result
            lo = arith.TruncIOp(i32t, m).result
            hi = arith.TruncIOp(i32t, arith.ShRUIOp(m, c32).result).result
            return fx.Int32(rocdl.mbcnt_hi(i32t, hi, rocdl.mbcnt_lo(i32t, lo, z)))

        def st_f32(ptr, idx, val):
            Vec.from_elements([val], fx.Float32).store(ptr, [idx])

        def ld_i32(ptr, idx):
            return fx.Int32(_raw(Vec.load(v1i, ptr, [idx])[0]))

        def st_i32(ptr, idx, val):
            Vec.from_elements([val], fx.Int32).store(ptr, [idx])

        # tid indexes the whole work-group (the cooperative loader); lane and
        # wave split it, and every compute address below is wave-relative -- grp
        # is the 0..3 MFMA row group and must not run past 3.
        tid = fx.Index(gpu.thread_idx.x)
        lane = tid % fx.Index(WAVE)
        wave = tid // fx.Index(WAVE)
        lo = lane % fx.Index(16)
        grp = lane // fx.Index(16)

        Sn = fx.Index(S)
        Bn = fx.Index(B)
        raw = fx.Index(gpu.block_idx.x)
        if const_expr(remap == "lpt"):
            # 1-D grid, (b, g) innermost: longest-processing-time-first dispatch.
            # One work-group carries NW adjacent tokens of the same (b, g).
            bgn = Bn * fx.Index(Hkv)
            ti = raw // bgn
            tok0 = Sn - fx.Index(1) - ti * fx.Index(NW)  # wave 0's token
            tok = tok0 - wave
        elif const_expr(remap == "block"):
            # XCD x runs query blocks x, x+8, ...: w-th work-group on it -> block
            # x + 8*(w // block_size), token w % block_size within it.
            w = raw // fx.Index(NXCD)
            qblk = raw % fx.Index(NXCD) + fx.Index(NXCD) * (w // fx.Index(block_size))
            tok = qblk * fx.Index(block_size) + w % fx.Index(block_size)
        else:
            tok = raw
        if const_expr(remap == "lpt"):
            bg = raw % (Bn * fx.Index(Hkv))
        else:
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
        c_i0 = fx.Int32(0)
        c_im1 = fx.Int32(-1)

        if const_expr(slot_coalesce):
            # -inf everywhere: the slots past the visible blocks are never written.
            # This wave only ever touches its own region, and LDS accesses retire
            # in program order within a wave, so no barrier is needed.
            SLW = HPW * SLS
            for cc in range_constexpr((SLW + WAVE - 1) // WAVE):
                qq = fx.Index(cc * WAVE) + lane
                st_f32(
                    lds_sl,
                    sel_i(ArithValue(qq < fx.Index(SLW)), wave * fx.Index(SLW) + qq, fx.Index(NW * SLW)),
                    c_neg_inf,
                )

        # ---- trip count: the causally visible blocks, left-packed by the indexer ----
        tok_i32 = fx.Int32(tok)
        n_vis = tok_i32 // fx.Int32(block_size) + fx.Int32(1)
        n_valid = sel_i32(ArithValue(n_vis < fx.Int32(topk)), n_vis, fx.Int32(topk))
        tbl_row = ((b * fx.Index(Hkv) + g) * Sn + tok) * fx.Index(topk)

        # ---- union prologue: merge the NW tokens' block lists, once, in LDS ----
        # Every wave runs the SAME merge over the same NWT = NW*topk entries, one
        # entry per lane, so there is no wave-asymmetric code: entry q is kept iff
        # it is live and no EARLIER entry holds the same block id, and its union
        # index is the exclusive prefix count of `keep` -- one `ballot` plus two
        # `mbcnt`, not an LDS scan.
        if const_expr(NW > 1):
            lane_ok = ArithValue(lane < fx.Index(topk))
            li = sel_i(lane_ok, lane, fx.Index(0))
            mine = fx.Int32(buffer_ops.buffer_load(t_rsrc, tbl_row + li, vec_width=1, dtype=fx.Int32))
            mine = sel_i32(andb(lane_ok, ArithValue(fx.Int32(lane) < n_valid)), mine, c_im1)
            st_i32(lds_tb, sel_i(lane_ok, wave * fx.Index(topk) + lane, fx.Index(NWT)), mine)
            # -1-fill this wave's slot map while the table write is in flight; the
            # barrier below fences it against the slot stores after the merge.
            c_sd = fx.Index(NW * USZ)  # the parking slot of lds_us
            for cc in range_constexpr((USZ + WAVE - 1) // WAVE):
                qq = fx.Index(cc * WAVE) + lane
                st_i32(lds_us, sel_i(ArithValue(qq < fx.Index(USZ)), wave * fx.Index(USZ) + qq, c_sd), c_im1)
            gpu.barrier()

            # lane q owns entry q of the flattened NW x topk table
            q_ok = ArithValue(lane < fx.Index(NWT))
            qi = sel_i(q_ok, lane, fx.Index(NWT))
            e_q = sel_i32(q_ok, ld_i32(lds_tb, qi), c_im1)
            live = ArithValue(e_q >= c_i0)
            # first earlier entry with the same id; only entries of earlier waves
            # can match, so the scan stops at (NW-1)*topk.
            dup = c_im1
            for jj in range_constexpr((NW - 1) * topk):
                aj = ld_i32(lds_tb, fx.Index(jj))
                hit = andb(andb(live, ArithValue(fx.Index(jj) < lane)), ArithValue(e_q == aj))
                dup = sel_i32(andb(hit, ArithValue(dup < c_i0)), fx.Int32(jj), dup)
            keep = andb(live, ArithValue(dup < c_i0))

            pos = wave_prefix(keep)  # exclusive count of kept entries below this lane
            n_u = fx.Int32(
                rocdl.readlane(
                    fx.Int32.ir_type,
                    _raw(fx.Int32(pos + sel_i32(keep, fx.Int32(1), c_i0))),
                    _raw(fx.Int32(WAVE - 1)),
                )
            )
            c_ud = fx.Index(USZ)  # the parking slot of lds_ub
            c_id = fx.Index(NWT)  # the parking slot of lds_ui
            st_i32(lds_ub, sel_i(keep, fx.Index(pos), c_ud), e_q)
            st_i32(lds_ui, sel_i(keep, qi, c_id), pos)
            gpu.barrier()

            # a duplicate entry inherits the union index of the entry it duplicates
            u_idx = sel_i32(keep, pos, ld_i32(lds_ui, sel_i(ArithValue(dup >= c_i0), fx.Index(dup), c_id)))
            # each wave writes only its own slot map: entry q belongs to wave q//topk
            own = andb(live, ArithValue(lane // fx.Index(topk) == wave))
            st_i32(
                lds_us,
                sel_i(own, wave * fx.Index(USZ) + fx.Index(u_idx), c_sd),
                fx.Int32(lane % fx.Index(topk)),
            )
            gpu.barrier()
            n_steps = fx.Index(n_u) * fx.Index(SPB)
        else:
            n_steps = fx.Index(n_valid) * fx.Index(SPB)

        def blk_at(q):
            if const_expr(NW > 1):
                return ld_i32(lds_ub, q)
            return fx.Int32(buffer_ops.buffer_load(t_rsrc, tbl_row + q, vec_width=1, dtype=fx.Int32))

        def slot_at(q):
            if const_expr(NW > 1):
                return ld_i32(lds_us, wave * fx.Index(USZ) + q)
            return fx.Int32(q)

        # ---- Q, register-resident as the B operand: head = g*16 + lo ----
        head = g * fx.Index(HPW) + lo
        q_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)
        q_packs = [
            buffer_ops.buffer_load(
                q_rsrc, q_row + fx.Index(ks * 32) + grp * fx.Index(8), vec_width=8, dtype=elem
            )
            for ks in range_constexpr(KS)
        ]

        def kv_offset(key):
            return ((key * Bn + b) * fx.Index(Hkv) + g) * fx.Index(D)

        # loader: tid -> (row = tid // LPR, 8-element slice = tid % LPR), over the
        # whole work-group
        ld_row = tid // fx.Index(LPR)
        ld_col = (tid % fx.Index(LPR)) * fx.Index(8)
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
            return blk_at(step // fx.Index(SPB))

        def step_kv0(blk, step):
            safe = sel_i32(ArithValue(blk >= c_i0), blk, c_i0)
            return safe * fx.Int32(block_size) + fx.Int32(step % fx.Index(SPB)) * fx.Int32(step_keys)

        def load_rows(kv0):
            # the loader's slice of the step's V (and K) rows, for staging in LDS
            def src(c):
                return kv_offset(fx.Index(kv0) + ld_row + fx.Index(c * RPP)) + ld_col

            regs = [
                buffer_ops.buffer_load(v_rsrc, src(c), vec_width=8, dtype=elem) for c in range_constexpr(NV8)
            ]
            if const_expr(k_via_lds):
                regs = regs + [
                    buffer_ops.buffer_load(k_rsrc, src(c), vec_width=8, dtype=elem)
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
            blk_ok = ArithValue(blk >= c_i0)
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
                Vec(v_regs[c]).store(lds_v, [ld_lds_b + fx.Index(c * RPP * STRIDE)])
                if const_expr(k_via_lds):
                    Vec(k_regs[c]).store(lds_k, [ld_lds_b + fx.Index(c * RPP * STRIDE)])
            if const_expr(barriers):
                gpu.barrier()

            # ---- wave-uniform owner skip: a 0- or 1-trip loop, so a wave that
            # does not own this union block pays neither the QK nor the PV MFMAs
            # and carries its accumulators through untouched ----
            myslot = slot_at(j // fx.Index(SPB))
            own_n = sel_i(ArithValue(myslot >= c_i0), fx.Index(1), fx.Index(0))
            if const_expr(emit_slot_lse):
                sub = j % fx.Index(SPB)
                ls_run = it[2 + DT + NC]
                ls_prev = fx.Float32(ArithValue(sub == fx.Index(0)).select(_raw(c_zero), _raw(ls_run)))
                inner_init = [m_run, l_run] + o_acc + [ls_prev]
            else:
                inner_init = [m_run, l_run] + o_acc
            inner = inner_init
            for _u, it2 in range(fx.Index(0), own_n, fx.Index(1), init=inner_init):
                m_cur = it2[0]
                l_cur = it2[1]
                o_cur = [it2[2 + dt] for dt in range_constexpr(DT)]

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
                m_new = fx.Float32(arith.MaxNumFOp(_raw(m_cur), _raw(crossgrp_max(lmax))).result)
                alpha = fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((m_cur - m_new) * c_sl)))
                p = [fx.Float32(rocdl.exp2(fx.Float32.ir_type, _raw((x - m_new) * c_sl))) for x in s_all]
                lsum = p[0]
                for x in p[1:]:
                    lsum = fx.Float32(arith.AddFOp(_raw(lsum), _raw(x)).result)
                l_new = fx.Float32(alpha * l_cur + lsum)  # per-grp partial, summed once at the end

                # ---- O^T += V^T P: A = V^T via transposed LDS reads, B = P ----
                alpha_v = Vec.from_elements([alpha, alpha, alpha, alpha], fx.Float32)
                new_o = []
                if const_expr(NT == 1):
                    pB = _raw(Vec.from_elements([fx.BFloat16(_raw(x)) for x in p], elem).bitcast(fx.Int16))
                    for dt in range_constexpr(DT):
                        a_v = _raw(tr16(pv_base_b + fx.Int64(dt * 32)))
                        new_o.append(
                            rocdl.mfma_f32_16x16x16bf16_1k(v4f, [a_v, pB, Vec(o_cur[dt]) * Vec(alpha_v)])
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
                        new_o.append(
                            rocdl.mfma_f32_16x16x32_bf16(v4f, [a_v, pB, Vec(o_cur[dt]) * Vec(alpha_v)])
                        )

                inner_out = [m_new, l_new] + new_o
                if const_expr(emit_slot_lse):
                    inner_out = inner_out + [fx.Float32(alpha * it2[2 + DT] + lsum)]
                inner = yield inner_out

            m_out = inner[0]
            l_out = inner[1]
            o_out = [inner[2 + dt] for dt in range_constexpr(DT)]

            # ---- per-slot logsumexp: the slot's partial sum, rescaled like l ----
            extra = []
            if const_expr(emit_slot_lse):
                ls_new = inner[2 + DT]
                # Only the slot's last step writes, so only it pays for the log and the
                # cross-group sum. `sub` is work-group-uniform, so this is a scalar branch.
                last = ArithValue(sub == fx.Index(SPB - 1))
                if last:
                    slot_lse = fmath.log(crossgrp_sum(ls_new)) + fx.Float32(m_out * c_scale)
                    if const_expr(slot_coalesce):
                        keep_sl = andb(ArithValue(myslot >= c_i0), ArithValue(grp == fx.Index(0)))
                        st_f32(
                            lds_sl,
                            sel_i(
                                keep_sl,
                                wave * fx.Index(HPW * SLS) + lo * fx.Index(SLS) + fx.Index(myslot),
                                fx.Index(NW * HPW * SLS),
                            ),
                            slot_lse,
                        )
                    else:
                        buffer_ops.buffer_store(
                            slot_lse,
                            slse_rsrc,
                            (head_row * fx.Index(topk) + fx.Index(myslot)) * fx.Index(4),
                            mask=_raw(andb(ArithValue(myslot >= c_i0), ArithValue(grp == fx.Index(0)))),
                            offset_is_bytes=True,
                        )
                extra = [ls_new]

            results = yield [m_out, l_out] + o_out + carry_next + extra

        # ---- epilogue: lane holds O[head = lo, d = dt*16 + grp*4 + i] ----
        m_run = results[0]
        l_sum = crossgrp_sum(results[1])
        o_acc = [results[2 + dt] for dt in range_constexpr(DT)]
        inv = fx.Float32(rocdl.rcp(fx.Float32.ir_type, _raw(l_sum)))
        inv_v = Vec.from_elements([inv, inv, inv, inv], fx.Float32)
        o_row = ((tok * Bn + b) * fx.Index(Hq) + head) * fx.Index(D)

        # A lane holds o[head = lo, d = dt*16 + grp*4 + 0..3], so one 8-byte store
        # per d-tile lays only a 32-byte contiguous run on each 256-byte head row:
        # four stores per 128-byte L2 line, measured as 4.75x L2 write amplification.
        # `permlane16_swap` hands every lane its row-partner's quad, so two adjacent
        # d-tiles become ONE 16-byte store per lane and the contiguous run per head
        # row doubles to 64 bytes while the store count halves, 8 -> 4.
        def _iswap(x):
            v = _raw(ArithValue(_raw(x)).bitcast(fx.Int32.ir_type))
            sw = rocdl.permlane16_swap(_pair_ty, v, v, False, True)
            return (
                llvm.extractvalue(fx.Int32.ir_type, sw, [0]),
                llvm.extractvalue(fx.Int32.ir_type, sw, [1]),
            )

        pk = []
        for dt in range_constexpr(DT):
            ov = Vec(o_acc[dt]) * inv_v
            pk.append(
                (
                    rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[0]), _raw(Vec(ov)[1])),
                    rocdl.cvt_pk_bf16_f32(_raw(Vec(ov)[2]), _raw(Vec(ov)[3])),
                )
            )
        odd = ArithValue(grp % fx.Index(2) == fx.Index(1))
        half = (grp // fx.Index(2)) * fx.Index(8)
        for dp in range_constexpr(DT // 2):
            a0, c0 = _iswap(pk[2 * dp][0])
            a1, c1 = _iswap(pk[2 * dp][1])
            b0, d0 = _iswap(pk[2 * dp + 1][0])
            b1, d1 = _iswap(pk[2 * dp + 1][1])
            w = [
                fx.Int32(odd.select(b0, a0)),
                fx.Int32(odd.select(b1, a1)),
                fx.Int32(odd.select(d0, c0)),
                fx.Int32(odd.select(d1, c1)),
            ]
            dbase = sel_i(odd, fx.Index((2 * dp + 1) * 16), fx.Index(2 * dp * 16)) + half
            buffer_ops.buffer_store(
                _raw(Vec.from_elements(w, fx.Int32)),
                o_rsrc,
                (o_row + dbase) * fx.Index(2),
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
        if const_expr(slot_coalesce):
            # NPL adjacent lanes cover one head's whole topk row, so the 16-lane run
            # is 4*topk contiguous bytes against gfx950's 64 B write granule instead
            # of the 4 B one dword-per-lane store laid down.
            sl4 = Vec.load(
                Vec.make_type(4, fx.Float32),
                lds_sl,
                [wave * fx.Index(HPW * SLS) + lo * fx.Index(SLS) + grp * fx.Index(4)],
            )
            buffer_ops.buffer_store(
                _raw(sl4),
                slse_rsrc,
                (head_row * fx.Index(topk) + grp * fx.Index(4)) * fx.Index(4),
                mask=_raw(ArithValue(grp < fx.Index(NPL))),
                offset_is_bytes=True,
            )
        elif const_expr(emit_slot_lse):
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

    def _grid(S, B):
        # remap is a build-time constant, so the shape is picked here, outside the trace.
        if remap == "lpt":
            return (fx.Index(S // NW) * fx.Index(B) * fx.Index(Hkv), 1, 1)
        return (fx.Index(S), fx.Index(B) * fx.Index(Hkv), 1)

    @flyc.jit
    def launch(Q, K, V, TBL, O, LSE, SLSE, S, B, stream):
        allocator.finalized = False
        with ir.InsertionPoint(CompilationContext.get_current().gpu_module_body):
            allocator.finalize()
        k_fn(Q, K, V, TBL, O, LSE, SLSE, S, B).launch(grid=_grid(S, B), block=(NTHREADS, 1, 1), stream=stream)

    launch.compile = lambda *a: flyc.compile(launch, *a)
    return launch


_CACHE: dict = {}


def msa_token_fwd(q, k, v, block_table, softmax_scale=None, block_size=128, return_slot_lse=False, **config):
    """MSA forward.

    Args:
        q: ``[S, B, Hq, 128]`` bf16, post-rope.
        k, v: ``[S, B, Hkv, 128]`` bf16, ``Hq == 16 * Hkv``.
        block_table: ``[B, Hkv, S, topk]`` int32 block ids, left-packed, -1 padded
            (the indexer's ``block_indices``), ``topk <= 16``.
        softmax_scale: defaults to ``1 / sqrt(128)``.
        return_slot_lse: also return ``slot_lse`` ``[S, B, Hq, topk]`` fp32, the
            logsumexp over each slot's keys; ``exp(slot_lse - lse)`` is the slot's
            share of the head's attention.
        config: ``build_fwd`` tuning knobs (``step_keys``, ``k_via_lds``, ``remap``,
            ``waves``).

    Returns:
        ``o`` ``[S, B, Hq, 128]`` bf16 and ``lse`` ``[S, B, Hq]`` fp32 (and
        ``slot_lse``).
    """
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
    if config.get("remap", "none") == "block" and S % (NXCD * block_size) != 0:
        config = {**config, "remap": "none"}
    # The union walk groups adjacent lpt tokens, so it needs lpt and an S the
    # wave count divides.
    nw = int(config.get("waves", 4))
    if config.get("remap", "lpt") != "lpt":
        nw = 1
    while nw > 1 and (S % nw != 0 or nw * topk > WAVE):
        nw //= 2
    config = {**config, "waves": nw}
    q, k, v, block_table = (t.contiguous() for t in (q, k, v, block_table))

    o = torch.empty_like(q)
    lse = torch.empty((S, B, Hq), dtype=torch.float32, device=q.device)
    slse = torch.empty((S, B, Hq, topk) if return_slot_lse else (1,), dtype=torch.float32, device=q.device)
    check_buffer_bytes(q=q, k=k, v=v, block_table=block_table, slot_lse=slse)
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
