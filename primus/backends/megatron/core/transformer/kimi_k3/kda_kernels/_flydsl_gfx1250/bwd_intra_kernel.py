###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fla's ``chunk_kda_bwd_kernel_intra`` (``SAFE_GATE`` path) on gfx1250 FP32 WMMA (FlyDSL), at K3's KDA shape
(K = 128, 64-row chunks of four 16-row sub-chunks, H == HV).

For sub-chunk i (rows c), with fla's reference rows (gn_a = first row of i, gn_z = last, gn_m = row 8):

    dq2 = exp2(g_c - gn_a) sum_{j < i} dAqk[c, j] k_j exp2(gn_a - g_j) + exp2(g_c - gn_m) (tril dAqk_ii) k_i exp2(gn_m - g_i)
    dk2 = the same with dAkk;  db = rowsum(dk2 k_c) + db;  dk2 *= beta_c
    dkt = exp2(gn_z - g_c) sum_{j > i} (dAqk[j, c]^T q_j + dAkk[j, c]^T k_j beta_j) exp2(g_j - gn_z) + diagonal (triu)
    dq2 += dq;  dg2 = q_c dq2 + (dk2 - dkt) k_c + dg;  dk2 += dk + dkt

All products are FP32 (fla keeps these dots in FP32 for precision). One 4-wave workgroup per (chunk, head): the
chunk's q / k / g / dAqk / dAkk / beta are staged in LDS once, wave w owns sub-chunk w. The off-diagonal blocks
run as three static slots per wave: slot s is block s of the j < w sum when s < w, else block s + 1 of the j > w
sum, so every wave does exactly three blocks. The reduction runs over the 16 rows of a block, so its operand
(the scaled k / q columns) is read out of LDS transposed, two rows per lane per 16x16x4 step.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_bwd_intra"]

_BT = 64
_BC = 16
_D = 128
_NW = 4
_WAVE = 32
_MAX_VIEW_BYTES = (1 << 31) - 4096
# Which two reduction rows a lane holds in a 16x16x4 FP32 WMMA operand: 2h + {0, 1} (pairs) or h + {0, 2}
# (strided), h = lane // 16. scripts/probe_wmma_gfx1250.py measures it.
_F32_STRIDED = False

# LDS (bytes); row pitches are 16 mod 64 dwords, so the two half-waves of a transposed read hit disjoint banks
_QK_PITCH = (_D + 16) * 4
_DA_PITCH = (_BT + 16) * 4
_OFF_K = 0
_OFF_Q = _OFF_K + _BT * _QK_PITCH
_OFF_G = _OFF_Q + _BT * _QK_PITCH
_OFF_DAQ = _OFF_G + _BT * _QK_PITCH
_OFF_DAK = _OFF_DAQ + _BT * _DA_PITCH
_OFF_B = _OFF_DAK + _BT * _DA_PITCH
_LDS_BYTES = _OFF_B + _BT * 4

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import arith, range_constexpr, rocdl

    BF, F32, I32 = fx.BFloat16, fx.Float32, fx.Int32

    def v4f():
        return fx.Vector.make_type(4, F32)

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

    def _mma(a2, b2, c):
        return rocdl.wmma_f32_16x16x4_f32(
            v8f(), _ir(fx.Vector.from_elements(a2, dtype=F32)), _ir(fx.Vector.from_elements(b2, dtype=F32)), c
        ).result

    def _zero():
        return _ir(fx.Vector.filled(8, 0.0, F32))

    def _jj(h, u):
        return h + I32(2 * u) if _F32_STRIDED else h * I32(2) + I32(u)

    def body(Q, Kt, Gt, BETA, DAQ, DAK, DQ, DK, DG, DB, DQ2, DK2, DG2, DBO, T, HV, nb_x, nb_x4, nb_a4, nb_r):
        tid = I32(fx.thread_idx.x)
        lane = tid % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        h = lane // I32(16)
        i_t = I32(fx.block_idx.x)
        i_bh = I32(fx.block_idx.y)
        b = i_bh // HV
        hv = i_bh - b * HV
        row0 = b * T + i_t * I32(_BT)
        rowhv = row0 * HV + hv

        g_q = _bv(Q, nb_x, BF, 8)
        g_k = _bv(Kt, nb_x, BF, 8)
        g_g = _bv(Gt, nb_x4, F32, 4)
        g_beta = _bv(BETA, nb_r, F32, 1)
        g_daq = _bv(DAQ, nb_a4, F32, 4)
        g_dak = _bv(DAK, nb_a4, F32, 4)
        g_dq = _bv(DQ, nb_x4, F32, 4)
        g_dk = _bv(DK, nb_x4, F32, 4)
        g_dg = _bv(DG, nb_x4, F32, 4)
        g_db = _bv(DB, nb_r, F32, 1)
        g_dq2 = _bv(DQ2, nb_x4, F32, 4)
        g_dk2 = _bv(DK2, nb_x4, F32, 4)
        g_dg2 = _bv(DG2, nb_x4, F32, 4)
        g_dbo = _bv(DBO, nb_r, F32, 1)
        rs = HV * I32(16)
        bx = rowhv * I32(16)
        rs4 = HV * I32(32)
        bx4 = rowhv * I32(32)
        rsa4 = HV * I32(16)
        ba4 = rowhv * I32(16)

        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))
        lq, lk, lg = l0 + I32(_OFF_Q), l0 + I32(_OFF_K), l0 + I32(_OFF_G)
        laq, lak, lb = l0 + I32(_OFF_DAQ), l0 + I32(_OFF_DAK), l0 + I32(_OFF_B)

        # ---- stage rows [16w, 16w + 16): q, k as FP32, g, dAqk, dAkk, beta
        srcs = [(g_q, lq), (g_k, lk)]
        for u in range_constexpr(8):
            i = lane + I32(32 * u)
            row = w * I32(_BC) + i // I32(16)
            cc = i % I32(16)
            for si in range_constexpr(2):
                vec = _ldv(srcs[si][0], bx + row * rs + cc, BF, 8)
                _st8([F32(vec[e]) for e in range(8)], srcs[si][1] + row * I32(_QK_PITCH) + cc * I32(32))
        for u in range_constexpr(16):
            i = lane + I32(32 * u)
            row = w * I32(_BC) + i // I32(32)
            cc = i % I32(32)
            vec = _ldv(g_g, bx4 + row * rs4 + cc, F32, 4)
            llvm.store(_ir(vec), _lds(lg + row * I32(_QK_PITCH) + cc * I32(16)))
        das = [(g_daq, laq), (g_dak, lak)]
        for u in range_constexpr(8):
            i = lane + I32(32 * u)
            row = w * I32(_BC) + i // I32(16)
            cc = i % I32(16)
            for si in range_constexpr(2):
                vec = _ldv(das[si][0], ba4 + row * rsa4 + cc, F32, 4)
                llvm.store(_ir(vec), _lds(das[si][1] + row * I32(_DA_PITCH) + cc * I32(16)))
        crow = w * I32(_BC) + r
        f = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
        fx.copy_atom_call(_atom(F32, 1), fx.slice(g_beta, (rowhv + crow * HV, None)), f)
        beta_c = fx.Vector(f.load())[0]
        llvm.store(_ir(beta_c), _lds(lb + crow * I32(4)))
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)

        # ---- B operands (rows c, reduction over a block's rows j), kept across the k tiles
        c = crow  # this lane's output row
        i0 = w * I32(_BC)
        slot_p1, slot_rows, slot_bq, slot_bk, slot_bt = [], [], [], [], []
        for s in range_constexpr(3):
            p1 = I32(s) < w  # block s of the j < w sum, else block s + 1 of the j > w sum
            jb = p1.select(I32(s), I32(s + 1))
            rows, bq, bk, bt = [], [], [], []
            for st in range_constexpr(4):
                jr = [jb * I32(_BC) + I32(4 * st) + _jj(h, u) for u in range(2)]
                rows.append(jr)
                aq = [
                    p1.select(c * I32(_DA_PITCH) + jr[u] * I32(4), jr[u] * I32(_DA_PITCH) + c * I32(4))
                    for u in range(2)
                ]
                bq.append([_ld32(laq + aq[u]) for u in range(2)])
                bk.append([_ld32(lak + aq[u]) for u in range(2)])
                bt.append([p1.select(F32(1.0), _ld32(lb + jr[u] * I32(4))) for u in range(2)])
            slot_p1.append(p1)
            slot_rows.append(rows)
            slot_bq.append(bq)
            slot_bk.append(bk)
            slot_bt.append(bt)
        dg_rows, bq2, bk2, bq5, bk5, bt5 = [], [], [], [], [], []
        for st in range_constexpr(4):
            jl = [I32(4 * st) + _jj(h, u) for u in range(2)]
            jr = [i0 + jl[u] for u in range(2)]
            dg_rows.append(jr)
            lo_ = [r >= jl[u] for u in range(2)]  # tril(dAqk_ii)[c, j]
            up_ = [r <= jl[u] for u in range(2)]  # triu: dAqk[j, c] with c <= j
            z = F32(0.0)
            bq2.append([lo_[u].select(_ld32(laq + c * I32(_DA_PITCH) + jr[u] * I32(4)), z) for u in range(2)])
            bk2.append([lo_[u].select(_ld32(lak + c * I32(_DA_PITCH) + jr[u] * I32(4)), z) for u in range(2)])
            bq5.append([up_[u].select(_ld32(laq + jr[u] * I32(_DA_PITCH) + c * I32(4)), z) for u in range(2)])
            bk5.append([up_[u].select(_ld32(lak + jr[u] * I32(_DA_PITCH) + c * I32(4)), z) for u in range(2)])
            bt5.append([_ld32(lb + jr[u] * I32(4)) for u in range(2)])

        ra, rz, rm = i0, i0 + I32(_BC - 1), i0 + I32(_BC // 2)  # gn_a, gn_z, gn_m rows

        def gload(kt):  # this lane's dq / dk / dg inputs of tile kt
            gi = bx4 + c * rs4 + kt * I32(4) + h * I32(2)
            vs = [_ldv(buf, gi + I32(o), F32, 4) for buf in (g_dq, g_dk, g_dg) for o in range(2)]
            return [vs[x // 4][x % 4] for x in range(24)]

        def kstep(kt, db_acc, pre):
            kcol4 = (kt * I32(16) + r) * I32(4)  # this lane's operand column, bytes
            ref_a = _ld32(lg + ra * I32(_QK_PITCH) + kcol4)
            ref_z = _ld32(lg + rz * I32(_QK_PITCH) + kcol4)
            ref_m = _ld32(lg + rm * I32(_QK_PITCH) + kcol4)
            acc_q, acc_k, acc_t = _zero(), _zero(), _zero()
            for s in range_constexpr(3):
                p1 = slot_p1[s]
                ref = p1.select(ref_a, ref_z)
                lsrc = p1.select(lk, lq)
                cur1, cur2 = _zero(), _zero()
                for st in range_constexpr(4):
                    x1, x2 = [], []
                    for u in range_constexpr(2):
                        ro = slot_rows[s][st][u] * I32(_QK_PITCH) + kcol4
                        gj = _ld32(lg + ro)
                        kj = _ld32(lk + ro)
                        xj = _ld32(lsrc + ro)
                        e = _exp2(p1.select(ref - gj, gj - ref))
                        x1.append(xj * e)
                        x2.append(kj * slot_bt[s][st][u] * e)
                    cur1 = _mma(x1, slot_bq[s][st], cur1)
                    cur2 = _mma(x2, slot_bk[s][st], cur2)
                m1 = p1.select(F32(1.0), F32(0.0))
                m4 = F32(1.0) - m1
                a1, a2 = fx.Vector(cur1), fx.Vector(cur2)
                vq, vk, vt = fx.Vector(acc_q), fx.Vector(acc_k), fx.Vector(acc_t)
                acc_q = _ir(fx.Vector.from_elements([vq[e] + m1 * a1[e] for e in range(8)], dtype=F32))
                acc_k = _ir(fx.Vector.from_elements([vk[e] + m1 * a2[e] for e in range(8)], dtype=F32))
                acc_t = _ir(
                    fx.Vector.from_elements([vt[e] + m4 * (a1[e] + a2[e]) for e in range(8)], dtype=F32)
                )
            d1, d2, d3 = _zero(), _zero(), _zero()
            for st in range_constexpr(4):
                xa, x1, x2 = [], [], []
                for u in range_constexpr(2):
                    ro = dg_rows[st][u] * I32(_QK_PITCH) + kcol4
                    gj = _ld32(lg + ro)
                    kj = _ld32(lk + ro)
                    qj = _ld32(lq + ro)
                    xa.append(kj * _exp2(ref_m - gj))
                    e = _exp2(gj - ref_m)
                    x1.append(qj * e)
                    x2.append(kj * bt5[st][u] * e)
                d1 = _mma(xa, bq2[st], d1)
                d2 = _mma(xa, bk2[st], d2)
                d3 = _mma(x1, bq5[st], d3)
                d3 = _mma(x2, bk5[st], d3)
            # ---- tile kt of row c: columns 16kt + 8h + [0, 8)
            col4 = (kt * I32(16) + h * I32(8)) * I32(4)
            gc = _ld8(lg + c * I32(_QK_PITCH) + col4)
            qc = _ld8(lq + c * I32(_QK_PITCH) + col4)
            kc = _ld8(lk + c * I32(_QK_PITCH) + col4)
            gna = _ld8(lg + ra * I32(_QK_PITCH) + col4)
            gnz = _ld8(lg + rz * I32(_QK_PITCH) + col4)
            gnm = _ld8(lg + rm * I32(_QK_PITCH) + col4)
            gi = bx4 + c * rs4 + kt * I32(4) + h * I32(2)
            pre = [F32(x) for x in pre]
            dqi, dki, dgi = pre[0:8], pre[8:16], pre[16:24]
            vq, vk, vt = fx.Vector(acc_q), fx.Vector(acc_k), fx.Vector(acc_t)
            w1, w2, w3 = fx.Vector(d1), fx.Vector(d2), fx.Vector(d3)
            o_q, o_k, o_g = [], [], []
            for e in range_constexpr(8):
                ea = _exp2(gc[e] - gna[e])
                em = _exp2(gc[e] - gnm[e])
                dq2 = vq[e] * ea + w1[e] * em
                dk2 = vk[e] * ea + w2[e] * em
                dkt = vt[e] * _exp2(gnz[e] - gc[e]) + w3[e] * _exp2(gnm[e] - gc[e])
                db_acc = db_acc + dk2 * kc[e]
                dk2b = dk2 * beta_c
                o_q.append(dq2 + dqi[e])
                o_g.append(qc[e] * dq2 + (dk2b - dkt) * kc[e] + dgi[e])
                o_k.append(dk2b + dki[e] + dkt)
            _stv(o_q[:4], g_dq2, gi, F32)
            _stv(o_q[4:], g_dq2, gi + I32(1), F32)
            _stv(o_k[:4], g_dk2, gi, F32)
            _stv(o_k[4:], g_dk2, gi + I32(1), F32)
            _stv(o_g[:4], g_dg2, gi, F32)
            _stv(o_g[4:], g_dg2, gi + I32(1), F32)
            return db_acc

        # The 8 k tiles as a runtime loop: 1.44 -> 1.31 ms against the unrolled loop (same occupancy at 466 vs 419
        # VGPR, an eighth of the code). The next tile's dq / dk / dg loads ride in the loop state.
        @flyc.jit
        def kloop(state):
            final = state
            for it, carried in range(fx.Index(I32(0)), fx.Index(I32(8)), 1, init=state):
                kt = I32(it)
                cur = list(carried)
                nxt = gload((kt + I32(1)) % I32(8))
                db = kstep(kt, F32(cur[0]), cur[1:])
                final = yield [_ir(db)] + [_ir(x) for x in nxt]
            return final

        res = kloop([_ir(F32(0.0))] + [_ir(x) for x in gload(I32(0))])
        db_acc = F32(list(res)[0])
        db_acc = db_acc + fx.gpu.shuffle_xor(db_acc, 16, _WAVE)
        f = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
        fx.copy_atom_call(_atom(F32, 1), fx.slice(g_db, (rowhv + c * HV, None)), f)
        dbv = db_acc + fx.Vector(f.load())[0]
        fo = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
        fx.memref_store_vec(fx.Vector.from_elements([dbv], dtype=F32), fo)
        fx.copy_atom_call(_atom(F32, 1), fo, fx.slice(g_dbo, (rowhv + c * HV, None)))

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_intra(
        Q: fx.Tensor,
        Kt: fx.Tensor,
        Gt: fx.Tensor,
        BETA: fx.Tensor,
        DAQ: fx.Tensor,
        DAK: fx.Tensor,
        DQ: fx.Tensor,
        DK: fx.Tensor,
        DG: fx.Tensor,
        DB: fx.Tensor,
        DQ2: fx.Tensor,
        DK2: fx.Tensor,
        DG2: fx.Tensor,
        DBO: fx.Tensor,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a4: fx.Int64,
        nb_r: fx.Int64,
    ):
        body(Q, Kt, Gt, BETA, DAQ, DAK, DQ, DK, DG, DB, DQ2, DK2, DG2, DBO, T, HV, nb_x, nb_x4, nb_a4, nb_r)

    @flyc.jit
    def launch(
        Q,
        Kt,
        Gt,
        BETA,
        DAQ,
        DAK,
        DQ,
        DK,
        DG,
        DB,
        DQ2,
        DK2,
        DG2,
        DBO,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a4: fx.Int64,
        nb_r: fx.Int64,
        nt: fx.Int32,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_intra(
            Q, Kt, Gt, BETA, DAQ, DAK, DQ, DK, DG, DB, DQ2, DK2, DG2, DBO, T, HV, nb_x, nb_x4, nb_a4, nb_r
        ).launch(grid=(nt, nbh, 1), block=(_NW * _WAVE, 1, 1), stream=stream)

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def kda_bwd_intra(
    q: Tensor,
    k: Tensor,
    g: Tensor,
    beta: Tensor,
    dAqk: Tensor,
    dAkk: Tensor,
    dq: Tensor,
    dk: Tensor,
    db: Tensor,
    dg: Tensor,
):
    """``(dq2, dk2, db, dg2)`` as fla's ``chunk_kda_bwd_intra`` returns them (db with the input db added)."""
    B, T, HV, K = g.shape
    dq2, dk2, dg2 = torch.empty_like(dq), torch.empty_like(dk), torch.empty_like(dg)
    dbo = torch.empty_like(db)
    args = (
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        dg,
        db,
        dq2,
        dk2,
        dg2,
        dbo,
        T,
        HV,
        q.numel() * 2,
        g.numel() * 4,
        dAqk.numel() * 4,
        beta.numel() * 4,
        T // _BT,
        B * HV,
        torch.cuda.current_stream(),
    )
    flyc, launch = _kernels()
    key = (q.device.index,)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf
    return dq2, dk2, dbo, dg2
