###############################################################################
# Some parts of this code are adapted from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention, 0.5.2): chunk_gla_fwd_kernel_o,
# rewritten as a FlyDSL kernel for gfx1250.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. Licensed under the MIT License.
#
# Modification Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""The output step of the ``flydsl`` KDA chunk forward on gfx1250 (FlyDSL), K = V = 128, 64-row chunks, H == HV.

``kda_fwd_o`` (what fla's ``chunk_gla_fwd_kernel_o`` computes): o = scale bf16(q exp2(g)) h + tril(A) v_new.

One 4-wave workgroup per (chunk, head), wave w owns rows [16w, 16w + 16) of the output. The operands that the
GEMMs reduce over their rows (h and v_new) are staged in LDS once and read transposed; the output comes out of the
C fragments as 8 consecutive columns of a row.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_fwd_o"]

_BT = 64
_D = 128
_NW = 4
_WAVE = 32
_ROWB = _D * 2 + 16

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

    def _common(HV, T):
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
        lane_r = h * I32(8) + lane % I32(8)
        lane_c = ((lane // I32(8)) % I32(2)) * I32(8)
        return tid, lane, w, r, h, b, hv, i_t, rowhv, lane_r, lane_c

    def _tr2(addr, rowb):
        lo = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr)))
        hi = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr + I32(16 * rowb))))
        return lo.shuffle(hi, list(range(16)))

    # ------------------------------------------------------------------ recompute_w_u
    def o_body(Q_, G_, V_, H_, A_, O_, scale, T, HV, NT, nb_x, nb_x4, nb_a, nb_h):
        tid, lane, w, r, h, b, hv, i_t, rowhv, lane_r, lane_c = _common(HV, T)
        g_q, g_vn, g_o = _bv(Q_, nb_x, BF, 8), _bv(V_, nb_x, BF, 8), _bv(O_, nb_x, BF, 8)
        g_g = _bv(G_, nb_x4, F32, 4)
        g_a = _bv(A_, nb_a, BF, 8)
        g_h = _bv(H_, nb_h, BF, 8)
        rs, bx = HV * I32(16), rowhv * I32(16)
        rs4, bx4 = HV * I32(32), rowhv * I32(32)
        rsa, ba = HV * I32(8), rowhv * I32(8)
        bh = ((b * NT + i_t) * HV + hv) * I32(_D * 16)
        smem = fx.SharedAllocator().allocate((_D + _BT) * _ROWB)
        lh = I32(fx.ptrtoint(smem.peek().ptr))
        lvn = lh + I32(_D * _ROWB)
        for u_ in range_constexpr(16):  # h [128][128]: 2048 vectors over 128 threads
            i = tid + I32(128 * u_)
            vec = _ldv(g_h, bh + i, BF, 8)
            llvm.store(_ir(vec), _lds(lh + (i // I32(16)) * I32(_ROWB) + (i % I32(16)) * I32(16)))
        for u_ in range_constexpr(8):  # v_new [64][128]
            i = tid + I32(128 * u_)
            vec = _ldv(g_vn, bx + (i // I32(16)) * rs + i % I32(16), BF, 8)
            llvm.store(_ir(vec), _lds(lvn + (i // I32(16)) * I32(_ROWB) + (i % I32(16)) * I32(16)))
        t0 = w * I32(16)
        t = t0 + r
        # qg operands: rows t, reduction k
        qg = []
        for ks in range_constexpr(4):
            vals = []
            for hk in range_constexpr(2):
                c8 = I32(4 * ks + 2 * hk) + h
                qv = _ldv(g_q, bx + t * rs + c8, BF, 8)
                gv = _ld8f(g_g, bx4 + t * rs4 + c8 * I32(2))
                vals += [(F32(qv[e]) * _exp2(gv[e])).to(BF) for e in range(8)]
            qg.append(fx.Vector.from_elements(vals, dtype=BF))
        a_f = []
        for js in range_constexpr(2):  # tril(A) rows t, reduction j
            raw = _ldv(g_a, ba + t * rsa + I32(4 * js) + h, BF, 8).shuffle(
                _ldv(g_a, ba + t * rsa + I32(4 * js) + h + I32(2), BF, 8), list(range(16))
            )
            jidx = [I32(32 * js) + h * I32(8) + I32(e) for e in range(8)] + [
                I32(32 * js + 16) + h * I32(8) + I32(e) for e in range(8)
            ]
            a_f.append(
                fx.Vector.from_elements([(t >= jidx[e]).select(raw[e], BF(0.0)) for e in range(16)], dtype=BF)
            )
        _bar()
        for ct in range_constexpr(8):
            acc = _zero()
            for ks in range_constexpr(4):
                op = _tr2(lh + I32(32 * ks * _ROWB + 32 * ct) + lane_r * I32(_ROWB) + lane_c * I32(2), _ROWB)
                acc = _mma(op, qg[ks], acc)
            c = fx.Vector(acc)
            acc = _ir(fx.Vector.from_elements([c[e] * scale for e in range(8)], dtype=F32))
            for js in range_constexpr(2):
                op = _tr2(lvn + I32(32 * js * _ROWB + 32 * ct) + lane_r * I32(_ROWB) + lane_c * I32(2), _ROWB)
                acc = _mma(op, a_f[js], acc)
            c = fx.Vector(acc)
            _stv([c[e].to(BF) for e in range(8)], g_o, bx + t * rs + I32(2 * ct) + h, BF)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_o(
        Q_: fx.Tensor,
        G_: fx.Tensor,
        V_: fx.Tensor,
        H_: fx.Tensor,
        A_: fx.Tensor,
        O_: fx.Tensor,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_h: fx.Int64,
    ):
        o_body(Q_, G_, V_, H_, A_, O_, scale, T, HV, NT, nb_x, nb_x4, nb_a, nb_h)

    @flyc.jit
    def launch_o(
        Q_,
        G_,
        V_,
        H_,
        A_,
        O_,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        NT: fx.Int32,
        nb_x: fx.Int64,
        nb_x4: fx.Int64,
        nb_a: fx.Int64,
        nb_h: fx.Int64,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_o(Q_, G_, V_, H_, A_, O_, scale, T, HV, NT, nb_x, nb_x4, nb_a, nb_h).launch(
            grid=(NT, nbh, 1), block=(_NW * _WAVE, 1, 1), stream=stream
        )

    return flyc, launch_o


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def _run(launcher_idx, key, args):
    k = _kernels()
    flyc, launch = k[0], k[launcher_idx]
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf


def kda_fwd_o(q: Tensor, v: Tensor, g: Tensor, A: Tensor, h: Tensor, scale: float):
    """``o`` as fla's ``chunk_gla_fwd_o_gk`` returns it (state [K, V] layout)."""
    B, T, HV, V = v.shape
    o = torch.empty_like(v)
    args = (
        q,
        g,
        v,
        h,
        A,
        o,
        float(scale),
        T,
        HV,
        T // _BT,
        q.numel() * 2,
        g.numel() * 4,
        A.numel() * 2,
        h.numel() * 2,
        B * HV,
        torch.cuda.current_stream(),
    )
    _run(1, ("o", q.device.index), args)
    return o
