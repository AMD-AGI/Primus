###############################################################################
# Some parts of this code are adapted from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention, 0.5.2): chunk_kda_bwd_kernel_wy_dqkg_fused,
# rewritten as a FlyDSL kernel for gfx1250.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. Licensed under the MIT License.
#
# Modification Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""fla's ``chunk_kda_bwd_kernel_wy_dqkg_fused`` on gfx1250 WMMA (FlyDSL), at K3's KDA shape (K = V = 128,
64-row chunks, H == HV, h / dh in fla's ``[K, V]`` layout).

Per 64-row chunk and head (one 4-wave workgroup), with every product in FP32:

    dq = (do h^T) * exp2(g) * scale          dw = -(dv h^T)            dkgb = A^T dw
    dk = (v_new dh^T) * exp2(gn - g) + dkgb * exp2(g) * beta
    dg = q dq - k dk1 + kg dkgb beta + [last row] (sum_v h dh * exp2(gn) + sum_t k dk1),  kg = k exp2(g)
    dvb = A^T dv,  dv2 = dvb * beta,  db = rowsum(dvb v) + rowsum(dkgb kg)
    dA = mask(-(A^T (mask(dv v^T + dw kg^T) * beta^T) A^T)),  mask = strictly lower

Wave w owns k columns [32w, 32w + 32) of dq / dk / dg / dw over all 64 rows, so ``A^T dw`` needs no data from
another wave, and v columns [32w, 32w + 32) of ``A^T dv`` (dv's slice transposed through LDS; A^T is read out of
the chunk's A staged in LDS). The four partial ``dv v^T + dw kg^T`` (each over the wave's slice) and row sums meet
in LDS; wave w then runs the dA chain for dA columns [16w, 16w + 16). WMMA operands are 16 rows x 32 reduction elements
per lane pair; an FP32 C fragment puts the B operand's rows on lanes and 8 consecutive A-operand rows in each
lane, so two C fragments packed to bf16 are directly the operand that reduces over those rows. The bf16
roundings are fla's (dw, kg, the masked dA and the middle product of the chain); only the summation order of the
FP32 accumulations differs.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["wy_dqkg_fused"]

_BT = 64
_D = 128
_NW = 4
_WAVE = 32

# LDS (bytes). Rows are padded by 16 B so the 16 rows of a ds_load_tr16 start in distinct bank groups.
_DV_ROWB = 32 * 2 + 16  # one wave's dv column slice: 32 bf16 per row
_DV_SLICE = _BT * _DV_ROWB
_A_ROWB = _BT * 2 + 16  # the chunk's A, read transposed
_DA_ROWB = _BT * 4 + 16  # one wave's FP32 dA partial
_DA_PART = _BT * _DA_ROWB
_OFF_DV = 0
_OFF_A = _OFF_DV + _NW * _DV_SLICE
_OFF_DA = _OFF_A + _BT * _A_ROWB
_OFF_DGK = _OFF_DA + _NW * _DA_PART
_OFF_BETA = _OFF_DGK + _NW * 32 * 4
_OFF_DB = _OFF_BETA + _BT * 4
_LDS_BYTES = _OFF_DB + _NW * _BT * 4

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import arith, const_expr, range_constexpr, rocdl

    BF, F32, I32 = fx.BFloat16, fx.Float32, fx.Int32

    def v8b():
        return fx.Vector.make_type(8, BF)

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

    def _exp2(x):
        return fx.math.exp2(x, fastmath=arith.FastMathFlags.afn)

    def _mma(a, b, c):
        return rocdl.wmma_f32_16x16x32_bf16(v8f(), _ir(a), _ir(b), c, reuseA=False, reuseB=False).result

    def _zero():
        return _ir(fx.Vector.filled(8, 0.0, F32))

    def _pack(lo, hi, neg):
        """bf16 operand from two FP32 C fragments: row = the lane's C column, reduction = the tiles' rows."""
        a, b = fx.Vector(lo), fx.Vector(hi)
        z = F32(0.0)
        sa = [(z - a[e]) if const_expr(neg) else a[e] for e in range(8)]
        sb = [(z - b[e]) if const_expr(neg) else b[e] for e in range(8)]
        return fx.Vector.from_elements([x.to(BF) for x in sa + sb], dtype=BF)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_wy_dqkg(
        Q: fx.Tensor,
        Kt: fx.Tensor,
        Vt: fx.Tensor,
        VN: fx.Tensor,
        Gt: fx.Tensor,
        BETA: fx.Tensor,
        At: fx.Tensor,
        Ht: fx.Tensor,
        DOt: fx.Tensor,
        DHt: fx.Tensor,
        DVt: fx.Tensor,
        DQ: fx.Tensor,
        DKo: fx.Tensor,
        DV2: fx.Tensor,
        DG: fx.Tensor,
        DB: fx.Tensor,
        DA: fx.Tensor,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_a4: fx.Int64,
        nb_h: fx.Int64,
        nb_r: fx.Int64,
    ):
        tid = I32(fx.thread_idx.x)
        lane = tid % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        half = lane // I32(16)
        i_t = I32(fx.block_idx.x)
        i_bh = I32(fx.block_idx.y)
        b = i_bh // HV
        hv = i_bh - b * HV
        row0 = b * T + i_t * I32(_BT)
        rowhv = row0 * HV + hv

        g_q = _bv(Q, nb_x, BF, 8)
        g_k = _bv(Kt, nb_x, BF, 8)
        g_v = _bv(Vt, nb_x, BF, 8)
        g_vn = _bv(VN, nb_x, BF, 8)
        g_do = _bv(DOt, nb_x, BF, 8)
        g_dv = _bv(DVt, nb_x, BF, 8)
        g_dv2 = _bv(DV2, nb_x, BF, 8)
        g_g = _bv(Gt, nb_x4, F32, 4)
        g_dq = _bv(DQ, nb_x4, F32, 4)
        g_dk = _bv(DKo, nb_x4, F32, 4)
        g_dg = _bv(DG, nb_x4, F32, 4)
        g_a = _bv(At, nb_a, BF, 8)
        g_da = _bv(DA, nb_a4, F32, 4)
        g_h = _bv(Ht, nb_h, BF, 8)
        g_dh = _bv(DHt, nb_h, BF, 8)
        g_beta = _bv(BETA, nb_r, F32, 1)
        g_db = _bv(DB, nb_r, F32, 1)

        rs = HV * I32(16)  # 8-element vectors between rows of a 128-wide bf16 tensor
        bx = rowhv * I32(16)
        rs4 = HV * I32(32)  # 4-element vectors between rows of a 128-wide FP32 tensor
        bx4 = rowhv * I32(32)
        rsa = HV * I32(8)  # A rows (64 bf16)
        ba = rowhv * I32(8)
        rsa4 = HV * I32(16)  # dA rows (64 FP32)
        ba4 = rowhv * I32(16)
        bh = ((b * NT + i_t) * HV + hv) * I32(_D * 16)  # h / dh [K, V], 16 vectors per k row
        k0 = w * I32(32)  # the wave's k (and v) columns [32w, 32w + 32)
        kv8 = w * I32(4)  # ... as an 8-element vector offset
        kv4 = w * I32(8)  # ... as a 4-element vector offset

        def opf(buf, base, rs_, row, kvec):
            """Operand fragment: rows row + [0, 16) on lanes, reduction columns kvec*8 + {half*8, 16 + half*8} + [0, 8)."""
            idx = base + (row + r) * rs_ + kvec + half
            return _ldv(buf, idx, BF, 8).shuffle(_ldv(buf, idx + I32(2), BF, 8), list(range(16)))

        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))

        # ---- stage this wave's dv columns [32w, 32w + 32) of all 64 rows, and (all waves) the chunk's A
        l_dv = l0 + I32(_OFF_DV) + w * I32(_DV_SLICE)
        for u in range_constexpr(8):
            i = lane + I32(32 * u)
            rr = i // I32(4)
            cc = i % I32(4)
            vec = _ldv(g_dv, bx + rr * rs + kv8 + cc, BF, 8)
            llvm.store(_ir(vec), _lds(l_dv + rr * I32(_DV_ROWB) + cc * I32(16)))
        l_a = l0 + I32(_OFF_A)
        for u in range_constexpr(4):
            i = tid + I32(128 * u)
            rr = i // I32(8)
            cc = i % I32(8)
            vec = _ldv(g_a, ba + rr * rsa + cc, BF, 8)
            llvm.store(_ir(vec), _lds(l_a + rr * I32(_A_ROWB) + cc * I32(16)))
        lane_r = half * I32(8) + lane % I32(8)
        lane_c = ((lane // I32(8)) % I32(2)) * I32(8)

        def tr(base, rowb):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(base))).shuffle(
                fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(base + I32(16 * rowb)))), list(range(16))
            )

        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)
        # A^T as the operand that reduces over A's rows: rows t (A's columns), reduction j
        tr_a = l_a + lane_r * I32(_A_ROWB) + lane_c * I32(2)
        a_f = [[tr(tr_a + I32(32 * js * _A_ROWB + 32 * tt), _A_ROWB) for js in range(2)] for tt in range(4)]

        # ---- pass 1: dw in both orientations. dwt[kt][tt]: lanes t, elements k; dwk[tt][kt]: lanes k, elements t
        dwt = [[_zero() for _ in range(4)] for _ in range(2)]
        dwk = [[_zero() for _ in range(2)] for _ in range(4)]
        for s in range_constexpr(4):
            hf = [opf(g_h, bh, I32(16), k0 + I32(16 * kt), I32(4 * s)) for kt in range(2)]
            dvf = [opf(g_dv, bx, rs, I32(16 * tt), I32(4 * s)) for tt in range(4)]
            for kt in range_constexpr(2):
                for tt in range_constexpr(4):
                    dwt[kt][tt] = _mma(hf[kt], dvf[tt], dwt[kt][tt])
                    dwk[tt][kt] = _mma(dvf[tt], hf[kt], dwk[tt][kt])
        dwop_t = [_pack(dwt[0][tt], dwt[1][tt], True) for tt in range(4)]  # rows t, reduction k (slice)
        dwop_k = [[_pack(dwk[2 * js][kt], dwk[2 * js + 1][kt], True) for js in range(2)] for kt in range(2)]
        # dkgb = A dw: lanes t, elements k
        dkgb = [[_zero() for _ in range(4)] for _ in range(2)]
        for kt in range_constexpr(2):
            for tt in range_constexpr(4):
                for js in range_constexpr(2):
                    dkgb[kt][tt] = _mma(dwop_k[kt][js], a_f[tt][js], dkgb[kt][tt])

        # ---- pass 2: dq0 = do h^T, dk0 = v_new dh^T (lanes t, elements k), and sum_v h dh per k row
        dq0 = [[_zero() for _ in range(4)] for _ in range(2)]
        dk0 = [[_zero() for _ in range(4)] for _ in range(2)]
        pg = [F32(0.0), F32(0.0)]
        for s in range_constexpr(4):
            hf = [opf(g_h, bh, I32(16), k0 + I32(16 * kt), I32(4 * s)) for kt in range(2)]
            dhf = [opf(g_dh, bh, I32(16), k0 + I32(16 * kt), I32(4 * s)) for kt in range(2)]
            dof = [opf(g_do, bx, rs, I32(16 * tt), I32(4 * s)) for tt in range(4)]
            vnf = [opf(g_vn, bx, rs, I32(16 * tt), I32(4 * s)) for tt in range(4)]
            for kt in range_constexpr(2):
                hx = fx.Vector(hf[kt])
                dhx = fx.Vector(dhf[kt])
                acc = pg[kt]
                for e in range_constexpr(16):
                    acc = acc + F32(hx[e]) * F32(dhx[e])
                pg[kt] = acc
                for tt in range_constexpr(4):
                    dq0[kt][tt] = _mma(hf[kt], dof[tt], dq0[kt][tt])
                    dk0[kt][tt] = _mma(dhf[kt], vnf[tt], dk0[kt][tt])
        # sum_v h dh of k row 16kt + r: both halves add up; through LDS to the lanes that hold k on elements
        l_dgk = l0 + I32(_OFF_DGK) + w * I32(32 * 4)
        for kt in range_constexpr(2):
            tot = pg[kt] + fx.gpu.shuffle_xor(pg[kt], 16, _WAVE)
            llvm.store(_ir(tot), _lds(l_dgk + (I32(16 * kt) + r) * I32(4)))
        rocdl.sched_barrier(0)
        hdh = []
        for kt in range_constexpr(2):
            a0 = l_dgk + (I32(16 * kt) + half * I32(8)) * I32(4)
            lo = fx.Vector(llvm.load(v4f(), _lds(a0)))
            hi = fx.Vector(llvm.load(v4f(), _lds(a0 + I32(16))))
            hdh.append([lo[e] for e in range(4)] + [hi[e] for e in range(4)])

        # ---- epilogue of the k slice: dq, dk, dg; kg (bf16) for dw kg^T; the row sums of dkgb kg
        beta_t = []
        for tt in range_constexpr(4):
            f = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
            fx.copy_atom_call(_atom(F32, 1), fx.slice(g_beta, (rowhv + (I32(16 * tt) + r) * HV, None)), f)
            beta_t.append(fx.Vector(f.load())[0])
        db2 = [F32(0.0) for _ in range(4)]
        kgb = [[None] * 4 for _ in range(2)]
        for kt in range_constexpr(2):
            cv = kv4 + I32(4 * kt) + half * I32(2)  # 4-element vector of k = 32w + 16kt + 8half
            c8 = kv8 + I32(2 * kt) + half  # ... as an 8-element vector
            gl = _ldv(g_g, bx4 + I32(_BT - 1) * rs4 + cv, F32, 4)
            gh_ = _ldv(g_g, bx4 + I32(_BT - 1) * rs4 + cv + I32(1), F32, 4)
            gn = [gl[e] for e in range(4)] + [gh_[e] for e in range(4)]
            dgkc = [hdh[kt][e] * _exp2(gn[e]) for e in range(8)]
            colsum = [F32(0.0) for _ in range(8)]
            o_dg = []
            for tt in range_constexpr(4):
                t = I32(16 * tt) + r
                glo = _ldv(g_g, bx4 + t * rs4 + cv, F32, 4)
                ghi = _ldv(g_g, bx4 + t * rs4 + cv + I32(1), F32, 4)
                gv = [glo[e] for e in range(4)] + [ghi[e] for e in range(4)]
                qv = _ldv(g_q, bx + t * rs + c8, BF, 8)
                kv = _ldv(g_k, bx + t * rs + c8, BF, 8)
                cq = fx.Vector(dq0[kt][tt])
                ck = fx.Vector(dk0[kt][tt])
                cb = fx.Vector(dkgb[kt][tt])
                bt = beta_t[tt]
                o_dq, o_dk, o_dg, o_kg = [], [], [], []
                acc_db = db2[tt]
                for e in range_constexpr(8):
                    gk = _exp2(gv[e])
                    dqe = cq[e] * gk * scale
                    dk1 = ck[e] * _exp2(gn[e] - gv[e])
                    ke = F32(kv[e])
                    kge = ke * gk
                    kdk = ke * dk1
                    cbe = cb[e]
                    o_dq.append(dqe)
                    o_dk.append(dk1 + cbe * gk * bt)
                    o_dg.append(F32(qv[e]) * dqe - kdk + kge * cbe * bt)
                    o_kg.append(kge.to(BF))
                    acc_db = acc_db + cbe * kge
                    colsum[e] = colsum[e] + kdk
                db2[tt] = acc_db
                kgb[kt][tt] = o_kg
                _stv(o_dq[:4], g_dq, bx4 + t * rs4 + cv, F32)
                _stv(o_dq[4:], g_dq, bx4 + t * rs4 + cv + I32(1), F32)
                _stv(o_dk[:4], g_dk, bx4 + t * rs4 + cv, F32)
                _stv(o_dk[4:], g_dk, bx4 + t * rs4 + cv + I32(1), F32)
                if const_expr(tt < 3):
                    _stv(o_dg[:4], g_dg, bx4 + t * rs4 + cv, F32)
                    _stv(o_dg[4:], g_dg, bx4 + t * rs4 + cv + I32(1), F32)
            # o_dg is row tile 3's, stored below with the last row's term. Sum over the 64 rows of this lane's 8 k columns: the 16 lanes of the half, 4 row tiles done
            for e in range_constexpr(8):
                cs = colsum[e]
                for sft in range_constexpr(4):
                    cs = cs + fx.gpu.shuffle_xor(cs, 1 << sft, _WAVE)
                dgkc[e] = dgkc[e] + cs
            is_last = r == I32(15)  # row 63 is row tile 3, lane row 15
            t3 = I32(48) + r
            fin = [o_dg[e] + is_last.select(dgkc[e], F32(0.0)) for e in range(8)]
            _stv(fin[:4], g_dg, bx4 + t3 * rs4 + cv, F32)
            _stv(fin[4:], g_dg, bx4 + t3 * rs4 + cv + I32(1), F32)
        for tt in range_constexpr(4):
            db2[tt] = db2[tt] + fx.gpu.shuffle_xor(db2[tt], 16, _WAVE)

        # ---- dA partial over the k slice: dw kg^T, as lanes t, elements j
        kgop = [
            fx.Vector.from_elements(kgb[0][jt] + kgb[1][jt], dtype=BF) for jt in range(4)
        ]  # rows j, red. k
        da = [[_mma(kgop[jt], dwop_t[tt], _zero()) for tt in range(4)] for jt in range(4)]

        # ---- v slice: dvb = A^T dv (lanes t, elements v) from dv's slice transposed out of LDS; dv2; rowsum(dvb v)
        tr_dv = l_dv + lane_r * I32(_DV_ROWB) + lane_c * I32(2)
        rocdl.sched_barrier(0)
        dvt = [
            [tr(tr_dv + I32(32 * js * _DV_ROWB + 32 * vt), _DV_ROWB) for js in range(2)] for vt in range(2)
        ]
        db1 = [F32(0.0) for _ in range(4)]
        for vt in range_constexpr(2):
            for tt in range_constexpr(4):
                acc = _zero()
                for js in range_constexpr(2):
                    acc = _mma(dvt[vt][js], a_f[tt][js], acc)
                c = fx.Vector(acc)
                t = I32(16 * tt) + r
                vi = bx + t * rs + kv8 + I32(2 * vt) + half
                vv = _ldv(g_v, vi, BF, 8)
                bt = beta_t[tt]
                _stv([(c[e] * bt).to(BF) for e in range(8)], g_dv2, vi, BF)
                s_ = db1[tt]
                for e in range_constexpr(8):
                    s_ = s_ + c[e] * F32(vv[e])
                db1[tt] = s_
        for tt in range_constexpr(4):
            db1[tt] = db1[tt] + fx.gpu.shuffle_xor(db1[tt], 16, _WAVE)
        # dA partial over the v slice: dv v^T
        vfj = [opf(g_v, bx, rs, I32(16 * jt), kv8) for jt in range(4)]
        dvw = [opf(g_dv, bx, rs, I32(16 * tt), kv8) for tt in range(4)]
        for jt in range_constexpr(4):
            for tt in range_constexpr(4):
                da[jt][tt] = _mma(vfj[jt], dvw[tt], da[jt][tt])

        # ---- partials to LDS: dA [t][j], the row sums, beta
        l_da = l0 + I32(_OFF_DA) + w * I32(_DA_PART)
        for jt in range_constexpr(4):
            for tt in range_constexpr(4):
                c = fx.Vector(da[jt][tt])
                a0 = l_da + (I32(16 * tt) + r) * I32(_DA_ROWB) + (I32(16 * jt) + half * I32(8)) * I32(4)
                llvm.store(_ir(fx.Vector.from_elements([c[e] for e in range(4)], dtype=F32)), _lds(a0))
                llvm.store(
                    _ir(fx.Vector.from_elements([c[e] for e in range(4, 8)], dtype=F32)), _lds(a0 + I32(16))
                )
        for tt in range_constexpr(4):
            t = I32(16 * tt) + r
            llvm.store(_ir(db1[tt] + db2[tt]), _lds(l0 + I32(_OFF_DB) + w * I32(_BT * 4) + t * I32(4)))
            llvm.store(_ir(beta_t[tt]), _lds(l0 + I32(_OFF_BETA) + t * I32(4)))
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)

        # ---- dA columns [16w, 16w + 16): Y = X A^T, Z = A^T Y, X = mask(sum of partials) * beta^T (bf16)
        aj = [opf(g_a, ba, rsa, w * I32(16), I32(4 * ps)) for ps in range(2)]  # A rows j, reduction p

        def xop(tt, ps):
            t = I32(16 * tt) + r
            vals = []
            for hk in range_constexpr(2):
                p0 = I32(32 * ps + 16 * hk) + half * I32(8)
                acc = [F32(0.0) for _ in range(8)]
                for wv in range_constexpr(_NW):
                    a0 = l0 + I32(_OFF_DA + wv * _DA_PART) + t * I32(_DA_ROWB) + p0 * I32(4)
                    lo = fx.Vector(llvm.load(v4f(), _lds(a0)))
                    hi = fx.Vector(llvm.load(v4f(), _lds(a0 + I32(16))))
                    for e in range_constexpr(4):
                        acc[e] = acc[e] + lo[e]
                        acc[4 + e] = acc[4 + e] + hi[e]
                b0 = l0 + I32(_OFF_BETA) + p0 * I32(4)
                blo = fx.Vector(llvm.load(v4f(), _lds(b0)))
                bhi = fx.Vector(llvm.load(v4f(), _lds(b0 + I32(16))))
                bp = [blo[e] for e in range(4)] + [bhi[e] for e in range(4)]
                for e in range_constexpr(8):
                    vals.append((t > p0 + I32(e)).select(acc[e] * bp[e], F32(0.0)).to(BF))
            return fx.Vector.from_elements(vals, dtype=BF)

        y = []
        for tt in range_constexpr(4):
            acc = _zero()
            for ps in range_constexpr(2):
                acc = _mma(xop(tt, ps), aj[ps], acc)
            y.append(acc)  # lanes j, elements t
        yt = [_pack(y[2 * qs], y[2 * qs + 1], False) for qs in range(2)]  # rows j, reduction q
        j0 = w * I32(16) + half * I32(8)
        for tt in range_constexpr(4):
            acc = _zero()
            for qs in range_constexpr(2):
                acc = _mma(yt[qs], a_f[tt][qs], acc)
            c = fx.Vector(acc)  # lanes t, elements j
            t = I32(16 * tt) + r
            z = F32(0.0)
            o = [(t > j0 + I32(e)).select(z - c[e], z) for e in range(8)]
            di = ba4 + t * rsa4 + w * I32(4) + half * I32(2)
            _stv(o[:4], g_da, di, F32)
            _stv(o[4:], g_da, di + I32(1), F32)
        # db of rows [16w, 16w + 16)
        t = w * I32(16) + r
        s_ = F32(0.0)
        for wv in range_constexpr(_NW):
            s_ = s_ + F32(llvm.load(F32.ir_type, _lds(l0 + I32(_OFF_DB + wv * _BT * 4) + t * I32(4))))
        f = fx.make_rmem_tensor(fx.make_layout(1, 1), F32)
        fx.memref_store_vec(fx.Vector.from_elements([s_], dtype=F32), f)
        fx.copy_atom_call(_atom(F32, 1), f, fx.slice(g_db, (rowhv + t * HV, None)))

    @flyc.jit
    def launch(
        Q,
        Kt,
        Vt,
        VN,
        Gt,
        BETA,
        At,
        Ht,
        DOt,
        DHt,
        DVt,
        DQ,
        DKo,
        DV2,
        DG,
        DB,
        DA,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_a4: fx.Int64,
        nb_h: fx.Int64,
        nb_r: fx.Int64,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_wy_dqkg(
            Q,
            Kt,
            Vt,
            VN,
            Gt,
            BETA,
            At,
            Ht,
            DOt,
            DHt,
            DVt,
            DQ,
            DKo,
            DV2,
            DG,
            DB,
            DA,
            scale,
            T,
            HV,
            NT,
            nb_x,
            nb_x4,
            nb_a,
            nb_a4,
            nb_h,
            nb_r,
        ).launch(grid=(NT, nbh, 1), block=(_NW * _WAVE, 1, 1), stream=stream)

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def wy_dqkg_fused(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    v_new: Tensor,
    g: Tensor,
    beta: Tensor,
    A: Tensor,
    h: Tensor,
    do: Tensor,
    dh: Tensor,
    dv: Tensor,
    scale: float,
):
    """``(dq, dk, dv2, db, dg, dA)`` exactly as fla's ``chunk_kda_bwd_wy_dqkg_fused`` returns them."""
    B, T, HV, K = q.shape
    NT = T // _BT
    dq = torch.empty(B, T, HV, K, device=q.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    dv2 = torch.empty_like(v)
    db = torch.empty_like(beta, dtype=torch.float32)
    dA = torch.empty_like(A, dtype=torch.float32)
    nb_x = q.numel() * 2
    nb_a = A.numel() * 2
    args = (
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        do,
        dh,
        dv,
        dq,
        dk,
        dv2,
        dg,
        db,
        dA,
        float(scale),
        T,
        HV,
        NT,
        nb_x,
        nb_x * 2,
        nb_a,
        nb_a * 2,
        h.numel() * 2,
        beta.numel() * 4,
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
    return dq, dk, dv2, db, dg, dA
