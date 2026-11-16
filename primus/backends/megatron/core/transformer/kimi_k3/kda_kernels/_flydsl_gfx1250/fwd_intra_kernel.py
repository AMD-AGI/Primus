###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The forward intra step of the ``flydsl`` KDA chunk on gfx1250, as one FlyDSL kernel: what fla's
``chunk_kda_fwd_intra`` computes with ``safe_gate`` and ``disable_recompute`` (its
``chunk_kda_fwd_kernel_intra_sub_chunk``, ``chunk_kda_fwd_kernel_inter_solve_fused`` and ``recompute_w_u_fwd``),
K = V = 128, 64-row chunks of four 16-row sub-chunks, H == HV.

Per chunk and head, g the chunk-local log2 cumsum, with fla's reference rows (row 8 of sub-chunk i for the
diagonal block (i, i), row 0 of sub-chunk i for the blocks (i, j < i)):

    Aqk[r, c] = scale sum_k q_r k_c 2^(g_r - g_c)   (r >= c)
    L[r, c] = beta_r sum_k k_r k_c 2^(g_r - g_c)    (r > c),   Akk = (I + L)^-1
    u = bf16(Akk) bf16(v beta),  w = bf16(Akk) bf16(k beta 2^g),  qg = q 2^g,  kg = k 2^(g_last - g)

Aqk / L and the inverse are FP32 WMMA (fla keeps them in FP32), u / w bf16 WMMA. One 4-wave workgroup per
(chunk, head):
  0. wave w loads rows [16w, 16w + 16): q / k / g to LDS, qg / kg out, v beta and k beta 2^g kept for step 4.
  1. the 10 blocks (i, j <= i) of Aqk and L from LDS, three per wave (two waves repeat a block); Aqk goes out, L
     to LDS.
  2. wave w inverts diagonal block w by forward substitution, one lane per column.
  3. levels l = 1, 2, 3, a barrier each: wave i >= l, j = i - l: Ai_ij = -Ai_ii sum_{j <= m < i} L_im Ai_mj.
  4. wave w: rows [16w, 16w + 16) of Akk out (bf16), then those rows of u and w.

The 16x16x4 FP32 WMMA's reduction slots are any bijection that the A and B operands share: in step 1 lane half h
holds k in [64h, 64h + 64) of its row (contiguous loads), in step 3 the second product uses the first's C fragment
as its A operand directly (reduction 8h + [0, 8)).
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_fwd_intra"]

_BT = 64
_BC = 16
_D = 128
_NW = 4
_WAVE = 32
_MAX_VIEW_BYTES = (1 << 31) - 4096

# LDS (bytes): the chunk's q, k (bf16), g (FP32) and L; after step 1 the inverse takes q's place, v beta k's and
# k beta 2^g g's. Row pitches are 4 mod 64 dwords, so 16 lanes reading one column / one 16 B slot of 16 rows hit
# distinct banks. Step 1 reads its operands here: from global (each wave re-reading its blocks' rows, one memory
# round trip per k-chunk) it took 0.71 of 1.07 ms.
_ROWB = _D * 2 + 16
_GP = _D * 4 + 16
_AP = (_BT + 4) * 4
_OFF_Q = 0
_OFF_K = _OFF_Q + _BT * _ROWB
_OFF_G = _OFF_K + _BT * _ROWB
_OFF_L = _OFF_G + _BT * _GP
_LDS_BYTES = _OFF_L + _BT * _AP

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm, scf
    from flydsl.expr import arith, rocdl

    BF, F32, I32 = fx.BFloat16, fx.Float32, fx.Int32

    def v4f():
        return fx.Vector.make_type(4, F32)

    def v8b():
        return fx.Vector.make_type(8, BF)

    def v8f():
        return fx.Vector.make_type(8, F32)

    def _ir(v):
        return fx.as_ir_value(v)

    def _bv(t, nb, dt, vec):
        nt = (1 << 31) // (vec * (dt.width // 8))
        b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=nb)
        return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))

    def _atom(dt, vec):
        return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)

    def _ldv(buf, idx, dt, vec):
        f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
        fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (idx, None)), f)
        return fx.Vector(f.load())

    def _ld1(buf, idx):
        f = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
        fx.copy_atom_call(_atom(F32, 1), fx.slice(buf, (idx, None)), f)
        return fx.Vector(f.load())[0]

    def _ld8f(buf, idx4):
        lo, hi = _ldv(buf, idx4, F32, 4), _ldv(buf, idx4 + I32(1), F32, 4)
        return [lo[e] for e in range(4)] + [hi[e] for e in range(4)]

    def _stv(vals, buf, idx, dt):
        n = len(vals)
        f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
        fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
        fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (idx, None)))

    def _lds(addr):
        pt = fx.PointerType.get(I32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
        return fx.to_llvm_ptr(fx.inttoptr(pt, addr))

    def _ld32(addr):
        return F32(llvm.load(F32.ir_type, _lds(addr)))

    def _ld8(addr):
        lo = fx.Vector(llvm.load(v4f(), _lds(addr)))
        hi = fx.Vector(llvm.load(v4f(), _lds(addr + I32(16))))
        return [lo[e] for e in range(4)] + [hi[e] for e in range(4)]

    def _st8(vals, addr):
        llvm.store(_ir(fx.Vector.from_elements(vals[:4], dtype=F32)), _lds(addr))
        llvm.store(_ir(fx.Vector.from_elements(vals[4:], dtype=F32)), _lds(addr + I32(16)))

    def _exp2(x):
        return fx.math.exp2(x, fastmath=arith.FastMathFlags.afn)

    def _mmf(a2, b2, c):
        return rocdl.wmma_f32_16x16x4_f32(
            v8f(), _ir(fx.Vector.from_elements(a2, dtype=F32)), _ir(fx.Vector.from_elements(b2, dtype=F32)), c
        ).result

    def _mmb(a, b, c):
        return rocdl.wmma_f32_16x16x32_bf16(v8f(), _ir(a), _ir(b), c, reuseA=False, reuseB=False).result

    def _zero():
        return _ir(fx.Vector.filled(8, 0.0, F32))

    def _bar():
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)

    def _when(cond, fn):  # fn() only on the waves where the wave-uniform cond holds
        if_op = scf.IfOp(_ir(cond), [], has_else=False)
        if len(if_op.regions[0].blocks) == 0:
            if_op.regions[0].blocks.append(*[])
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            fn()
            scf.YieldOp([])

    def _tr2(addr, rowb):
        lo = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr)))
        hi = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr + I32(16 * rowb))))
        return lo.shuffle(hi, list(range(16)))

    def body(Q_, K_, V_, G_, BETA, AQK, AKK, W_, U_, QG, KG, scale, T, HV, nb_x, nb_x4, nb_a, nb_r):
        tid = I32(fx.thread_idx.x)
        lane = tid % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        h = lane // I32(16)
        i_t = I32(fx.block_idx.x)
        i_bh = I32(fx.block_idx.y)
        b = i_bh // HV
        hv = i_bh - b * HV
        rowhv = (b * T + i_t * I32(_BT)) * HV + hv
        g_q, g_k, g_v = _bv(Q_, nb_x, BF, 8), _bv(K_, nb_x, BF, 8), _bv(V_, nb_x, BF, 8)
        g_w, g_u, g_qg, g_kg = (
            _bv(W_, nb_x, BF, 8),
            _bv(U_, nb_x, BF, 8),
            _bv(QG, nb_x, BF, 8),
            _bv(KG, nb_x, BF, 8),
        )
        g_g = _bv(G_, nb_x4, F32, 4)
        g_beta = _bv(BETA, nb_r, F32, 1)
        g_aqk, g_akk = _bv(AQK, nb_a, BF, 8), _bv(AKK, nb_a, BF, 8)
        rs, bx = HV * I32(16), rowhv * I32(16)
        rs4, bx4 = HV * I32(32), rowhv * I32(32)
        rsa, ba = HV * I32(8), rowhv * I32(8)
        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))
        lq, lk, lg, ll = l0 + I32(_OFF_Q), l0 + I32(_OFF_K), l0 + I32(_OFF_G), l0 + I32(_OFF_L)
        lai, lvb, lkb = lq, lk, lg
        z = F32(0.0)

        # ---- step 1 operands: lane (row r of a sub-chunk, half h) holds k in [64h + 16kc, 64h + 16kc + 16)
        def ld16b(src, row, kc):
            a = src + row * I32(_ROWB) + (h * I32(64) + I32(16 * kc)) * I32(2)
            return fx.Vector(llvm.load(v8b(), _lds(a))), fx.Vector(llvm.load(v8b(), _lds(a + I32(16))))

        def ld16f(src, row, kc):
            a = src + row * I32(_GP) + (h * I32(64) + I32(16 * kc)) * I32(4)
            return [fx.Vector(llvm.load(v4f(), _lds(a + I32(16 * u)))) for u in range(4)]

        def tab(v):  # v[w] for the runtime wave index
            return (w < I32(1)).select(
                I32(v[0]), (w < I32(2)).select(I32(v[1]), (w < I32(3)).select(I32(v[2]), I32(v[3])))
            )

        # (i, j) per slot: wave 0 (0,0) (1,1) (3,3); 1 (1,0) (2,2) (3,2); 2 (2,0) (2,1) (3,3); 3 (3,0) (3,1) (3,2)
        slots = [(w, I32(0)), (tab([1, 2, 2, 3]), tab([1, 2, 1, 1])), (I32(3), I32(3) - w % I32(2))]
        slots = [(i, j, (i - j) < I32(1)) for i, j in slots]

        def p1_loads(slot, kc):
            i, j, d = slot
            rb, ra = i * I32(_BC) + r, j * I32(_BC) + r
            gnrow = i * I32(_BC) + d.select(I32(_BC // 2), I32(0))
            beta = _ld1(g_beta, rowhv + rb * HV) if kc == 0 else None
            return (
                ld16b(lq, rb, kc),
                ld16b(lk, rb, kc),
                ld16f(lg, rb, kc),
                ld16b(lk, ra, kc),
                ld16f(lg, ra, kc),
                ld16f(lg, gnrow, kc),
                beta,
            )

        def p1_step(ld, cq, ck):
            qb, kb, gb, ka, ga, gn, _ = ld
            fl = lambda p: [p[0][e] for e in range(8)] + [p[1][e] for e in range(8)]
            f4 = lambda p: [p[u][e] for u in range(4) for e in range(4)]
            qbv, kbv, kav = fl(qb), fl(kb), fl(ka)
            gbv, gav, gnv = f4(gb), f4(ga), f4(gn)
            for s in range(8):
                a2, bq2, bk2 = [], [], []
                for p in range(2):
                    t = 2 * s + p
                    eb = _exp2(gbv[t] - gnv[t])
                    a2.append(F32(kav[t]) * _exp2(gnv[t] - gav[t]))
                    bq2.append(F32(qbv[t]) * eb)
                    bk2.append(F32(kbv[t]) * eb)
                cq = _mmf(a2, bq2, cq)
                ck = _mmf(a2, bk2, ck)
            return cq, ck

        def p1_out(slot, beta, cq, ck):  # C: lane (row r of sub-chunk i, h) holds columns 16j + 8h + [0, 8)
            i, j, d = slot
            row = i * I32(_BC) + r
            cqv, ckv = fx.Vector(cq), fx.Vector(ck)
            oq, ok = [], []
            for e in range(8):
                c = h * I32(8) + I32(e)
                vq, vk = cqv[e] * scale, ckv[e] * beta
                oq.append(d.select((c < r + I32(1)).select(vq, z), vq).to(BF))
                ok.append(d.select((c < r).select(vk, z), vk))
            _stv(oq, g_aqk, ba + row * rsa + j * I32(2) + h, BF)
            _st8(ok, ll + row * I32(_AP) + (j * I32(_BC) + h * I32(8)) * I32(4))

        # ---- this wave's rows 16w + lane // 16 + 2u: q / k / g to LDS, qg / kg out, v beta / k beta 2^g kept
        c8 = lane % I32(16)
        gn_last = _ld8f(g_g, bx4 + I32(_BT - 1) * rs4 + c8 * I32(2))

        def stage_loads(u_):
            row = w * I32(_BC) + lane // I32(16) + I32(2 * u_)
            xi = bx + row * rs + c8
            return (
                row,
                xi,
                _ld1(g_beta, rowhv + row * HV),
                _ldv(g_v, xi, BF, 8),
                _ldv(g_k, xi, BF, 8),
                _ldv(g_q, xi, BF, 8),
                _ld8f(g_g, bx4 + row * rs4 + c8 * I32(2)),
            )

        def stage_out(group):
            kept = []
            for row, xi, bt, vv, kv, qv, gk in group:
                eg = [_exp2(gk[e]) for e in range(8)]
                llvm.store(_ir(qv), _lds(lq + row * I32(_ROWB) + c8 * I32(16)))
                llvm.store(_ir(kv), _lds(lk + row * I32(_ROWB) + c8 * I32(16)))
                _st8(gk, lg + row * I32(_GP) + c8 * I32(32))
                _stv([(F32(qv[e]) * eg[e]).to(BF) for e in range(8)], g_qg, xi, BF)
                _stv([(F32(kv[e]) * _exp2(gn_last[e] - gk[e])).to(BF) for e in range(8)], g_kg, xi, BF)
                kept.append(
                    (
                        row,
                        fx.Vector.from_elements([(F32(vv[e]) * bt).to(BF) for e in range(8)], dtype=BF),
                        fx.Vector.from_elements(
                            [(F32(kv[e]) * bt * eg[e]).to(BF) for e in range(8)], dtype=BF
                        ),
                    )
                )
            return kept

        kept = stage_out([stage_loads(u_) for u_ in range(8)])
        _bar()
        ld = p1_loads(slots[0], 0)

        # ---- step 1: Aqk / L blocks, 3 slots x 4 k-chunks, the next chunk's loads issued before this one's WMMAs
        steps = [(s, kc) for s in range(3) for kc in range(4)]
        beta = ld[6]
        cq = ck = None
        for n, (s, kc) in enumerate(steps):
            nxt = p1_loads(slots[steps[n + 1][0]], steps[n + 1][1]) if n + 1 < len(steps) else ld
            if kc == 0:
                cq, ck = _zero(), _zero()
            cq, ck = p1_step(ld, cq, ck)
            if kc == 3:
                p1_out(slots[s], beta, cq, ck)
            ld = nxt
            if ld[6] is not None:
                beta = ld[6]
        _bar()
        zeros4 = _ir(fx.Vector.filled(4, 0.0, F32))
        for (
            row,
            vb,
            kb,
        ) in kept:  # q / k / g are dead: v beta, k beta 2^g and this wave's inverse rows (zeroed)
            a0 = row * I32(_ROWB) + c8 * I32(16)
            llvm.store(_ir(vb), _lds(lvb + a0))
            llvm.store(_ir(kb), _lds(lkb + a0))
        for u_ in range(8):
            i = lane + I32(32 * u_)
            llvm.store(zeros4, _lds(lai + (w * I32(_BC) + i // I32(16)) * I32(_AP) + (i % I32(16)) * I32(16)))

        def inverse():
            # ---- step 2: wave w inverts L_ww: lane column n = r, x_m = -L[m, n] - sum_{j < m} L[m, j] x_j
            rb0 = w * I32(_BC)
            lrows = []
            for m in range(1, _BC):
                base = ll + (rb0 + I32(m)) * I32(_AP) + rb0 * I32(4)
                vs = [fx.Vector(llvm.load(v4f(), _lds(base + I32(16 * u)))) for u in range((m + 3) // 4)]
                lrows.append([vs[x // 4][x % 4] for x in range(m)])
            lmn = [_ld32(ll + (rb0 + I32(m)) * I32(_AP) + (rb0 + r) * I32(4)) for m in range(1, _BC)]
            xs = [z]
            for m in range(1, _BC):
                acc = z - lmn[m - 1]
                for jj in range(m):
                    acc = acc - lrows[m - 1][jj] * xs[jj]
                xs.append(acc)
            for e in range(8):  # half h stores rows 8h + e of column n; Ai = x + I
                mrow = h * I32(8) + I32(e)
                one = (mrow < r).select(z, (r < mrow).select(z, F32(1.0)))
                val = (h < I32(1)).select(xs[e], xs[8 + e]) + one
                llvm.store(_ir(val), _lds(lai + (rb0 + mrow) * I32(_AP) + (rb0 + r) * I32(4)))
            _bar()

            # ---- step 3: one level of the block recurrence per barrier
            def level(lv):
                i, j = w, w - I32(lv)
                a_ops, b_ops = [], []
                for t_ in range(lv):  # X = sum_m L_im Ai_mj, C[row, col]: A = L_im rows, B = Ai_mj columns
                    m = j + I32(t_)
                    for st in range(4):
                        kk = I32(4 * st) + h * I32(2)
                        la = ll + (i * I32(_BC) + r) * I32(_AP) + (m * I32(_BC) + kk) * I32(4)
                        a_ops.append([_ld32(la), _ld32(la + I32(4))])
                        lb = lai + (m * I32(_BC) + kk) * I32(_AP) + (j * I32(_BC) + r) * I32(4)
                        b_ops.append([_ld32(lb), _ld32(lb + I32(_AP))])
                aii = _ld8(lai + (i * I32(_BC) + r) * I32(_AP) + (i * I32(_BC) + h * I32(8)) * I32(4))
                acc = _zero()
                for a2, b2 in zip(a_ops, b_ops):
                    acc = _mmf(a2, b2, acc)
                xc = fx.Vector(acc)  # lane (column r, h): X[8h + e, r]
                p = _zero()  # Ai_ii X with reduction 8h + 2st + p: A = xc, B = Ai_ii[r, 8h + [0, 8)]
                for st in range(4):
                    p = _mmf([xc[2 * st], xc[2 * st + 1]], [aii[2 * st], aii[2 * st + 1]], p)
                pv = fx.Vector(p)  # lane (row r, h): (Ai_ii X)[r, 8h + e]
                _st8(
                    [z - pv[e] for e in range(8)],
                    lai + (i * I32(_BC) + r) * I32(_AP) + (j * I32(_BC) + h * I32(8)) * I32(4),
                )

            for lv in range(1, 4):
                _when(I32(lv - 1) < w, lambda lv=lv: level(lv))
                _bar()

        def out_wu():
            # ---- step 4: rows t = 16w + r of Akk (bf16) out, and u / w = bf16(Akk) (v beta / k beta 2^g)
            t = w * I32(_BC) + r
            a_f = []
            for js in range(2):
                lo = _ld8(lai + t * I32(_AP) + (I32(32 * js) + h * I32(8)) * I32(4))
                hi = _ld8(lai + t * I32(_AP) + (I32(32 * js + 16) + h * I32(8)) * I32(4))
                vals = [x.to(BF) for x in lo] + [x.to(BF) for x in hi]
                a_f.append(fx.Vector.from_elements(vals, dtype=BF))
                _stv(vals[:8], g_akk, ba + t * rsa + I32(4 * js) + h, BF)
                _stv(vals[8:], g_akk, ba + t * rsa + I32(4 * js + 2) + h, BF)
            lane_r = h * I32(8) + lane % I32(8)
            lane_c = ((lane // I32(8)) % I32(2)) * I32(8)

            def opld(src, ct):
                return [
                    _tr2(src + I32(32 * js * _ROWB + 32 * ct) + lane_r * I32(_ROWB) + lane_c * I32(2), _ROWB)
                    for js in range(2)
                ]

            tiles = [(lvb, g_u, ct) for ct in range(8)] + [(lkb, g_w, ct) for ct in range(8)]
            ops = opld(tiles[0][0], 0)
            for n, (src, dst, ct) in enumerate(tiles):
                nxt = opld(tiles[n + 1][0], tiles[n + 1][2]) if n + 1 < len(tiles) else ops
                acc = _zero()
                for js in range(2):
                    acc = _mmb(ops[js], a_f[js], acc)
                ops = nxt
                c = fx.Vector(acc)
                _stv([c[e].to(BF) for e in range(8)], dst, bx + t * rs + I32(2 * ct) + h, BF)

        inverse()
        out_wu()

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_fwd_intra(
        Q_: fx.Tensor,
        K_: fx.Tensor,
        V_: fx.Tensor,
        G_: fx.Tensor,
        BETA: fx.Tensor,
        AQK: fx.Tensor,
        AKK: fx.Tensor,
        W_: fx.Tensor,
        U_: fx.Tensor,
        QG: fx.Tensor,
        KG: fx.Tensor,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_r: fx.Int64,
    ):
        body(Q_, K_, V_, G_, BETA, AQK, AKK, W_, U_, QG, KG, scale, T, HV, nb_x, nb_x4, nb_a, nb_r)

    @flyc.jit
    def launch(
        Q_,
        K_,
        V_,
        G_,
        BETA,
        AQK,
        AKK,
        W_,
        U_,
        QG,
        KG,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_r: fx.Int64,
        nt: fx.Int32,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_fwd_intra(
            Q_, K_, V_, G_, BETA, AQK, AKK, W_, U_, QG, KG, scale, T, HV, nb_x, nb_x4, nb_a, nb_r
        ).launch(grid=(nt, nbh, 1), block=(_NW * _WAVE, 1, 1), stream=stream)

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def kda_fwd_intra(q: Tensor, k: Tensor, v: Tensor, gk: Tensor, beta: Tensor, scale: float):
    """``(w, u, qg, kg, Aqk, Akk)`` as fla's ``chunk_kda_fwd_intra`` returns them (Aqk above its block diagonal
    is left unwritten, as fla's)."""
    B, T, HV, K = k.shape
    Aqk = k.new_empty(B, T, HV, _BT)
    Akk = k.new_empty(B, T, HV, _BT)
    w, u, qg, kg = torch.empty_like(k), torch.empty_like(v), torch.empty_like(q), torch.empty_like(k)
    args = (
        q,
        k,
        v,
        gk,
        beta,
        Aqk,
        Akk,
        w,
        u,
        qg,
        kg,
        float(scale),
        T,
        HV,
        k.numel() * 2,
        gk.numel() * 4,
        Aqk.numel() * 2,
        beta.numel() * 4,
        T // _BT,
        B * HV,
        torch.cuda.current_stream(),
    )
    flyc, launch = _kernels()
    key = (k.device.index,)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf
    return w, u, qg, kg, Aqk, Akk
