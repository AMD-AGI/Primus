###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fla's ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64`` for KDA on gfx1250 (FlyDSL), K = V = 128, 64-row chunks.

The chunk-state recurrence, per (batch, head), chunk by chunk (gk = the chunk-local log2 decay, kg = fla's
k * exp2(gk_last - gk)):

    h[t] = state;  v_new = u - w bf16(state);  state = state * exp2(gk_last) + kg^T bf16(v_new)

One 8-wave workgroup per (batch, head); wave w owns state columns [16w, 16w + 16) as eight FP32 C fragments
(lanes = columns, elements = 8 consecutive state rows), so packed to bf16 they are directly the operand of
w state, and the update's C fragments land back in the same layout: the recurrence never leaves registers.
fla's Triton kernel issues each chunk's loads after the previous chunk's math (384 programs, one chain each);
here chunk t + 1's w / kg / gk_last are loaded while chunk t computes and written into the other half of a
2-stage LDS ring at the end of the step (one workgroup barrier per chunk), and u is carried in registers. u,
v_new and the stored h pass through small per-wave LDS transposes so every global access is a 16-byte vector.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_fwd_h"]

_BT = 64
_D = 128
_NW = 8
_WAVE = 32
_MAX_VIEW_BYTES = (1 << 31) - 4096

# LDS (bytes). Ring stage: w and kg tiles [64][128] bf16 (16 B row pad), then gk's last row (FP32).
_ROWB = _D * 2 + 16
_TILE = _BT * _ROWB
_STAGE = 2 * _TILE + _D * 4
# Per wave: u [64 t][16 v], v_new [16 v][64 t], h [16 v][128 k] (bf16, padded rows).
_U_ROWB = 16 * 2 + 16
_VN_ROWB = _BT * 2 + 16
_HS_ROWB = _D * 2 + 16
_WAVE_B = _BT * _U_ROWB + 16 * _VN_ROWB + 16 * _HS_ROWB
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
    from flydsl.expr import arith, range_constexpr, rocdl

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

    def body(K_, W_, U_, GK, H_, VN, T, HV, NT, nb_x, nb_x4, nb_h):
        tid = I32(fx.thread_idx.x)
        lane = tid % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        h = lane // I32(16)
        bh = I32(fx.block_idx.x)
        b = bh // HV
        hv = bh - b * HV
        g_k = _bv(K_, nb_x, BF, 8)
        g_w = _bv(W_, nb_x, BF, 8)
        g_u = _bv(U_, nb_x, BF, 8)
        g_vn = _bv(VN, nb_x, BF, 8)
        g_gk = _bv(GK, nb_x4, F32, 4)
        g_h = _bv(H_, nb_h, BF, 8)

        def x8(t, i, c8):  # 8-element vector index of (chunk t row i, head hv, column 8 * c8)
            return ((b * T + t * I32(_BT) + i) * HV + hv) * I32(16) + c8

        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))
        lu = l0 + I32(_OFF_WAVE) + w * I32(_WAVE_B)
        lvn = lu + I32(_BT * _U_ROWB)
        lhs = lvn + I32(16 * _VN_ROWB)
        lane_r = h * I32(8) + lane % I32(8)
        lane_c = ((lane // I32(8)) % I32(2)) * I32(8)

        def tr1(addr):
            return fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr)))

        def tr2(addr, rowb):
            return tr1(addr).shuffle(tr1(addr + I32(16 * rowb)), list(range(16)))

        def prefetch(tn):
            """Chunk tn's w and kg tiles and gk's last row (shared by the workgroup), and this wave's u columns."""
            wv, kv, uv = [], [], []
            for u_ in range_constexpr(4):
                idx = tid + I32(256 * u_)
                i, c = idx // I32(16), idx % I32(16)
                wv.append(_ldv(g_w, x8(tn, i, c), BF, 8))
                kv.append(_ldv(g_k, x8(tn, i, c), BF, 8))
            gl = _ldv(
                g_gk, ((b * T + tn * I32(_BT) + I32(_BT - 1)) * HV + hv) * I32(32) + tid % I32(32), F32, 4
            )
            for u_ in range_constexpr(4):
                idx = lane + I32(32 * u_)
                uv.append(_ldv(g_u, x8(tn, idx // I32(2), w * I32(2) + idx % I32(2)), BF, 8))
            return wv, kv, gl, uv

        def stage_store(off, wv, kv, gl):
            for u_ in range_constexpr(4):
                idx = tid + I32(256 * u_)
                a0 = l0 + off + (idx // I32(16)) * I32(_ROWB) + (idx % I32(16)) * I32(16)
                llvm.store(_ir(wv[u_]), _lds(a0))
                llvm.store(_ir(kv[u_]), _lds(a0 + I32(_TILE)))
            llvm.store(_ir(gl), _lds(l0 + off + I32(2 * _TILE) + (tid % I32(32)) * I32(16)))

        def step(st, t):
            hs = list(st[:8])
            uv = [fx.Vector(x) for x in st[8:12]]
            cur = (t % I32(2)) * I32(_STAGE)
            lw = l0 + cur
            lkg = lw + I32(_TILE)
            lgk = lw + I32(2 * _TILE)
            tn = (t + I32(1) < NT).select(t + I32(1), t)
            nwv, nkv, ngl, nuv = prefetch(tn)
            # this chunk's u columns -> [64 t][16 v]
            for u_ in range_constexpr(4):
                idx = lane + I32(32 * u_)
                llvm.store(_ir(uv[u_]), _lds(lu + (idx // I32(2)) * I32(_U_ROWB) + (idx % I32(2)) * I32(16)))
            # h[t] (bf16): C fragments -> [16 v][128 k] -> rows of 8 consecutive v
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
                    g_h,
                    hb + (I32(16 * cb) + r) * I32(16) + w * I32(2) + h,
                    BF,
                )
            # v_pred = w bf16(h): lanes v, elements t
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
                rowa = lw + (I32(16 * tt) + r) * I32(_ROWB) + h * I32(16)
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
            # u in the same layout (tr16 over [64 t][16 v]); v_new = u - v_pred
            rocdl.sched_barrier(0)
            uop = [
                tr2(lu + I32(32 * ts * _U_ROWB) + lane_r * I32(_U_ROWB) + lane_c * I32(2), _U_ROWB)
                for ts in range(2)
            ]
            vn = []
            for tt in range_constexpr(4):
                p = fx.Vector(vp[tt])
                uo = uop[tt // 2]
                vn.append([(F32(uo[(tt % 2) * 8 + e]) - p[e]).to(BF) for e in range(8)])
            # v_new: [16 v][64 t] -> rows of 8 consecutive v
            for tt in range_constexpr(4):
                llvm.store(
                    _ir(fx.Vector.from_elements(vn[tt], dtype=BF)),
                    _lds(lvn + r * I32(_VN_ROWB) + (I32(16 * tt) + h * I32(8)) * I32(2)),
                )
            rocdl.sched_barrier(0)
            vvals = [tr1(lvn + lane_r * I32(_VN_ROWB) + lane_c * I32(2) + I32(32 * cb)) for cb in range(4)]
            for cb in range_constexpr(4):
                _stv([vvals[cb][e] for e in range(8)], g_vn, x8(t, I32(16 * cb) + r, w * I32(2) + h), BF)
            # state = state * exp2(gk_last) + kg^T bf16(v_new)
            vnop = [fx.Vector.from_elements(vn[2 * ts] + vn[2 * ts + 1], dtype=BF) for ts in range(2)]
            gks, kgt = [], []
            for kt in range_constexpr(8):
                g0 = lgk + (I32(16 * kt) + h * I32(8)) * I32(4)
                gks.append(
                    (fx.Vector(llvm.load(v4f(), _lds(g0))), fx.Vector(llvm.load(v4f(), _lds(g0 + I32(16)))))
                )
                kgt.append(
                    [
                        tr2(
                            lkg + I32(32 * ts * _ROWB + 32 * kt) + lane_r * I32(_ROWB) + lane_c * I32(2),
                            _ROWB,
                        )
                        for ts in range(2)
                    ]
                )
            out = []
            for kt in range_constexpr(8):
                glo, ghi = gks[kt]
                gv = [glo[e] for e in range(4)] + [ghi[e] for e in range(4)]
                c = fx.Vector(hs[kt])
                acc = _ir(fx.Vector.from_elements([c[e] * _exp2(gv[e]) for e in range(8)], dtype=F32))
                for ts in range_constexpr(2):
                    acc = _mma(kgt[kt][ts], vnop[ts], acc)
                out.append(acc)
            stage_store((I32(1) - t % I32(2)) * I32(_STAGE), nwv, nkv, ngl)
            _bar()
            return out + [_ir(x) for x in nuv]

        @flyc.jit
        def steps(state, n):
            final = state
            for it, carried in range(fx.Index(I32(0)), fx.Index(n), 1, init=state):
                res = step(list(carried), I32(it))
                final = yield res
            return final

        wv, kv, gl, uv = prefetch(I32(0))
        stage_store(I32(0), wv, kv, gl)
        _bar()
        steps([_zero() for _ in range(8)] + [_ir(x) for x in uv], NT)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_fwd_h(
        K_: fx.Tensor,
        W_: fx.Tensor,
        U_: fx.Tensor,
        GK: fx.Tensor,
        H_: fx.Tensor,
        VN: fx.Tensor,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_h: fx.Int64,
    ):
        body(K_, W_, U_, GK, H_, VN, T, HV, NT, nb_x, nb_x4, nb_h)

    @flyc.jit
    def launch(
        K_,
        W_,
        U_,
        GK,
        H_,
        VN,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_h: fx.Int64,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_fwd_h(K_, W_, U_, GK, H_, VN, T, HV, NT, nb_x, nb_x4, nb_h).launch(
            grid=(nbh, 1, 1), block=(_NW * _WAVE, 1, 1), stream=stream
        )

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def kda_fwd_h(k: Tensor, w: Tensor, u: Tensor, gk: Tensor):
    """``(h, v_new, None)`` as fla's ``chunk_gated_delta_rule_fwd_h`` returns them for KDA."""
    B, T, HV, K = k.shape
    NT = T // _BT
    h = k.new_empty(B, NT, HV, K, u.shape[-1])
    v_new = torch.empty_like(u)
    args = (
        k,
        w,
        u,
        gk,
        h,
        v_new,
        T,
        HV,
        NT,
        k.numel() * 2,
        gk.numel() * 4,
        h.numel() * 2,
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
    return h, v_new, None
