###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fla's ``chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`` for KDA on gfx1250 (FlyDSL), K = V = 128, 64-row chunks.

The backward chunk-state recurrence, per (batch, head), from the last chunk to the first (qg / kg = fla's gated
q / k, gk = the chunk-local log2 decay):

    dh[t] = state;  dv2 = kg bf16(state) + dv;  state = state * exp2(gk_last) + scale qg^T do - w^T bf16(dv2)

Same structure as ``_flydsl_kda_fwd_h_gfx1250``: one 8-wave workgroup per (batch, head), wave w owns state columns
[16w, 16w + 16) as FP32 C fragments; chunk t - 1's kg / qg / w tiles and gk_last are loaded while chunk t
computes and written to the other half of a 2-stage LDS ring; dv and do are carried in registers, and dv / do /
dv2 / the stored state pass through per-wave LDS transposes.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_bwd_dhu"]

_BT = 64
_D = 128
_NW = 8
_WAVE = 32
_MAX_VIEW_BYTES = (1 << 31) - 4096

_ROWB = _D * 2 + 16
_TILE = _BT * _ROWB
_STAGE = 3 * _TILE + _D * 4  # kg, qg, w tiles, then gk's last row
_U_ROWB = 16 * 2 + 16  # [64 t][16 v]
_VN_ROWB = _BT * 2 + 16  # [16 v][64 t]
_HS_ROWB = _D * 2 + 16  # [16 v][128 k]
_WAVE_B = 2 * _BT * _U_ROWB + 16 * _VN_ROWB + 16 * _HS_ROWB
_OFF_WAVE = 2 * _STAGE
_LDS_BYTES = _OFF_WAVE + _NW * _WAVE_B

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

    def _bar():
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)

    def body(Q_, K_, W_, GK, DO_, DV_, DH_, DV2, scale, T, HV, NT, nb_x, nb_x4, nb_h):
        tid = I32(fx.thread_idx.x)
        lane = tid % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        h = lane // I32(16)
        bh = I32(fx.block_idx.x)
        b = bh // HV
        hv = bh - b * HV
        g_q = _bv(Q_, nb_x, BF, 8)
        g_k = _bv(K_, nb_x, BF, 8)
        g_w = _bv(W_, nb_x, BF, 8)
        g_do = _bv(DO_, nb_x, BF, 8)
        g_dv = _bv(DV_, nb_x, BF, 8)
        g_dv2 = _bv(DV2, nb_x, BF, 8)
        g_gk = _bv(GK, nb_x4, F32, 4)
        g_dh = _bv(DH_, nb_h, BF, 8)

        def x8(t, i, c8):
            return ((b * T + t * I32(_BT) + i) * HV + hv) * I32(16) + c8

        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))
        ldvi = l0 + I32(_OFF_WAVE) + w * I32(_WAVE_B)  # dv in, [64 t][16 v]
        ldo = ldvi + I32(_BT * _U_ROWB)  # do, [64 t][16 v]
        ldv2 = ldo + I32(_BT * _U_ROWB)  # dv2 out, [16 v][64 t]
        lhs = ldv2 + I32(16 * _VN_ROWB)  # dh out, [16 v][128 k]
        lane_r = h * I32(8) + lane % I32(8)
        lane_c = ((lane // I32(8)) % I32(2)) * I32(8)

        def tr1(addr):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr)))

        def tr2(addr, rowb):
            return tr1(addr).shuffle(tr1(addr + I32(16 * rowb)), list(range(16)))

        def prefetch(tn):
            kv, qv, wv, dvv, dov = [], [], [], [], []
            for u_ in range_constexpr(4):
                idx = tid + I32(256 * u_)
                i, c = idx // I32(16), idx % I32(16)
                kv.append(_ldv(g_k, x8(tn, i, c), BF, 8))
                qv.append(_ldv(g_q, x8(tn, i, c), BF, 8))
                wv.append(_ldv(g_w, x8(tn, i, c), BF, 8))
            gl = _ldv(
                g_gk, ((b * T + tn * I32(_BT) + I32(_BT - 1)) * HV + hv) * I32(32) + tid % I32(32), F32, 4
            )
            for u_ in range_constexpr(4):
                idx = lane + I32(32 * u_)
                ix = x8(tn, idx // I32(2), w * I32(2) + idx % I32(2))
                dvv.append(_ldv(g_dv, ix, BF, 8))
                dov.append(_ldv(g_do, ix, BF, 8))
            return kv, qv, wv, gl, dvv, dov

        def stage_store(off, kv, qv, wv, gl):
            for u_ in range_constexpr(4):
                idx = tid + I32(256 * u_)
                a0 = l0 + off + (idx // I32(16)) * I32(_ROWB) + (idx % I32(16)) * I32(16)
                llvm.store(_ir(kv[u_]), _lds(a0))
                llvm.store(_ir(qv[u_]), _lds(a0 + I32(_TILE)))
                llvm.store(_ir(wv[u_]), _lds(a0 + I32(2 * _TILE)))
            llvm.store(_ir(gl), _lds(l0 + off + I32(3 * _TILE) + (tid % I32(32)) * I32(16)))

        def step(st, it):
            hs = list(st[:8])
            dvv = [fx.Vector(x) for x in st[8:12]]
            dov = [fx.Vector(x) for x in st[12:16]]
            t = NT - I32(1) - it
            cur = (it % I32(2)) * I32(_STAGE)
            lk = l0 + cur
            lq = lk + I32(_TILE)
            lw = lk + I32(2 * _TILE)
            lgk = lk + I32(3 * _TILE)
            tn = (t > I32(0)).select(t - I32(1), t)
            nkv, nqv, nwv, ngl, ndvv, ndov = prefetch(tn)
            for u_ in range_constexpr(4):
                idx = lane + I32(32 * u_)
                a0 = (idx // I32(2)) * I32(_U_ROWB) + (idx % I32(2)) * I32(16)
                llvm.store(_ir(dvv[u_]), _lds(ldvi + a0))
                llvm.store(_ir(dov[u_]), _lds(ldo + a0))
            # dh[t] (bf16) through [16 v][128 k]
            for kt in range_constexpr(8):
                c = fx.Vector(hs[kt])
                vals = fx.Vector.from_elements([c[e].to(BF) for e in range(8)], dtype=BF)
                llvm.store(_ir(vals), _lds(lhs + r * I32(_HS_ROWB) + (I32(16 * kt) + h * I32(8)) * I32(2)))
            rocdl.sched_barrier(0)
            hb = ((b * NT + t) * HV + hv) * I32(_D * 16)
            # every phase issues its LDS reads before using any of them (one wait instead of one per read)
            hvals = [tr1(lhs + lane_r * I32(_HS_ROWB) + lane_c * I32(2) + I32(32 * cb)) for cb in range(8)]
            for cb in range_constexpr(8):
                _stv(
                    [hvals[cb][e] for e in range(8)],
                    g_dh,
                    hb + (I32(16 * cb) + r) * I32(16) + w * I32(2) + h,
                    BF,
                )
            # dv2 = kg bf16(state) + dv: lanes v, elements t
            hop = [
                fx.Vector.from_elements(
                    [fx.Vector(hs[2 * ks])[e].to(BF) for e in range(8)]
                    + [fx.Vector(hs[2 * ks + 1])[e].to(BF) for e in range(8)],
                    dtype=BF,
                )
                for ks in range(4)
            ]
            afr = []
            for tt in range_constexpr(4):
                rowa = lk + (I32(16 * tt) + r) * I32(_ROWB) + h * I32(16)
                for ks in range_constexpr(4):
                    a0 = rowa + I32(64 * ks)
                    afr.append(
                        fx.Vector(llvm.load(v8b(), _lds(a0))).shuffle(
                            fx.Vector(llvm.load(v8b(), _lds(a0 + I32(32)))), list(range(16))
                        )
                    )
            vp = [_zero() for _ in range(4)]
            for tt in range_constexpr(4):
                for ks in range_constexpr(4):
                    vp[tt] = _mma(afr[4 * tt + ks], hop[ks], vp[tt])
            rocdl.sched_barrier(0)
            dvi = [
                tr2(ldvi + I32(32 * ts * _U_ROWB) + lane_r * I32(_U_ROWB) + lane_c * I32(2), _U_ROWB)
                for ts in range(2)
            ]
            dot = [
                tr2(ldo + I32(32 * ts * _U_ROWB) + lane_r * I32(_U_ROWB) + lane_c * I32(2), _U_ROWB)
                for ts in range(2)
            ]
            dv2 = []
            for tt in range_constexpr(4):
                p = fx.Vector(vp[tt])
                di = dvi[tt // 2]
                dv2.append([(p[e] + F32(di[(tt % 2) * 8 + e])).to(BF) for e in range(8)])
            for tt in range_constexpr(4):
                llvm.store(
                    _ir(fx.Vector.from_elements(dv2[tt], dtype=BF)),
                    _lds(ldv2 + r * I32(_VN_ROWB) + (I32(16 * tt) + h * I32(8)) * I32(2)),
                )
            rocdl.sched_barrier(0)
            vvals = [tr1(ldv2 + lane_r * I32(_VN_ROWB) + lane_c * I32(2) + I32(32 * cb)) for cb in range(4)]
            for cb in range_constexpr(4):
                _stv([vvals[cb][e] for e in range(8)], g_dv2, x8(t, I32(16 * cb) + r, w * I32(2) + h), BF)
            # state = state * exp2(gk_last) + scale qg^T do - w^T bf16(dv2)
            dvop = [fx.Vector.from_elements(dv2[2 * ts] + dv2[2 * ts + 1], dtype=BF) for ts in range(2)]
            out = []
            for kt in range_constexpr(8):
                if const_expr(kt % 4 == 0):  # the reads of four k tiles at a time (all eight would spill)
                    gks, qt, wt_ = {}, {}, {}
                    for k2 in range_constexpr(kt, kt + 4):
                        g0 = lgk + (I32(16 * k2) + h * I32(8)) * I32(4)
                        gks[k2] = (
                            fx.Vector(llvm.load(v4f(), _lds(g0))),
                            fx.Vector(llvm.load(v4f(), _lds(g0 + I32(16)))),
                        )
                        offs = [
                            I32(32 * ts * _ROWB + 32 * k2) + lane_r * I32(_ROWB) + lane_c * I32(2)
                            for ts in range(2)
                        ]
                        qt[k2] = [tr2(lq + o, _ROWB) for o in offs]
                        wt_[k2] = [tr2(lw + o, _ROWB) for o in offs]
                glo, ghi = gks[kt]
                gv = [glo[e] for e in range(4)] + [ghi[e] for e in range(4)]
                aq, aw = _zero(), _zero()
                for ts in range_constexpr(2):
                    aq = _mma(qt[kt][ts], dot[ts], aq)
                    aw = _mma(wt_[kt][ts], dvop[ts], aw)
                c, cq, cw = fx.Vector(hs[kt]), fx.Vector(aq), fx.Vector(aw)
                out.append(
                    _ir(
                        fx.Vector.from_elements(
                            [c[e] * _exp2(gv[e]) + cq[e] * scale - cw[e] for e in range(8)], dtype=F32
                        )
                    )
                )
            stage_store((I32(1) - it % I32(2)) * I32(_STAGE), nkv, nqv, nwv, ngl)
            _bar()
            return out + [_ir(x) for x in ndvv] + [_ir(x) for x in ndov]

        @flyc.jit
        def steps(state, n):
            final = state
            for it, carried in range(fx.Index(I32(0)), fx.Index(n), 1, init=state):
                res = step(list(carried), I32(it))
                final = yield res
            return final

        kv, qv, wv, gl, dvv, dov = prefetch(NT - I32(1))
        stage_store(I32(0), kv, qv, wv, gl)
        _bar()
        steps([_zero() for _ in range(8)] + [_ir(x) for x in dvv] + [_ir(x) for x in dov], NT)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_bwd_dhu(
        Q_: fx.Tensor,
        K_: fx.Tensor,
        W_: fx.Tensor,
        GK: fx.Tensor,
        DO_: fx.Tensor,
        DV_: fx.Tensor,
        DH_: fx.Tensor,
        DV2: fx.Tensor,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_h: fx.Int64,
    ):
        body(Q_, K_, W_, GK, DO_, DV_, DH_, DV2, scale, T, HV, NT, nb_x, nb_x4, nb_h)

    @flyc.jit
    def launch(
        Q_,
        K_,
        W_,
        GK,
        DO_,
        DV_,
        DH_,
        DV2,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_h: fx.Int64,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_bwd_dhu(Q_, K_, W_, GK, DO_, DV_, DH_, DV2, scale, T, HV, NT, nb_x, nb_x4, nb_h).launch(
            grid=(nbh, 1, 1), block=(_NW * _WAVE, 1, 1), stream=stream
        )

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def kda_bwd_dhu(q: Tensor, k: Tensor, w: Tensor, do: Tensor, dv: Tensor, gk: Tensor, scale: float):
    """``(dh, None, dv2)`` as fla's ``chunk_gated_delta_rule_bwd_dhu`` returns them for KDA."""
    B, T, HV, K = q.shape
    NT = T // _BT
    dh = q.new_empty(B, NT, HV, K, do.shape[-1])
    dv2 = torch.empty_like(dv)
    args = (
        q,
        k,
        w,
        gk,
        do,
        dv,
        dh,
        dv2,
        float(scale),
        T,
        HV,
        NT,
        q.numel() * 2,
        gk.numel() * 4,
        dh.numel() * 2,
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
    return dh, None, dv2
