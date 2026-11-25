###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""fla's ``chunk_kda_bwd_kernel_dAv`` on gfx1250 (FlyDSL), K = V = 128, 64-row chunks, H == HV.

Per chunk and head: dA = scale * tril(do v^T) (FP32) and dv = tril(A)^T do (bf16), v = the forward's v_new,
A = Aqk. One 4-wave workgroup per (chunk, head), wave w owns rows [16w, 16w + 16) of both outputs. dA's operands
are do's and v's rows as they are stored; dv reduces over the rows of A and do, so both are staged in LDS once and
read transposed.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["kda_bwd_dav"]

_BT = 64
_D = 128
_NW = 4
_WAVE = 32

_DO_ROWB = _D * 2 + 16
_A_ROWB = _BT * 2 + 16
_OFF_A = _BT * _DO_ROWB
_LDS_BYTES = _OFF_A + _BT * _A_ROWB

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import range_constexpr, rocdl

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

    def _stv(vals, buf, idx, dt):
        n = len(vals)
        f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
        fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
        fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (idx, None)))

    def _lds(addr):
        pt = fx.PointerType.get(I32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
        return fx.to_llvm_ptr(fx.inttoptr(pt, addr))

    def _mma(a, b, c):
        return rocdl.wmma_f32_16x16x32_bf16(v8f(), _ir(a), _ir(b), c, reuseA=False, reuseB=False).result

    def _zero():
        return _ir(fx.Vector.filled(8, 0.0, F32))

    def body(V_, A_, DO_, DV_, DA_, scale, T, HV, nb_x, nb_a, nb_a4):
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
        g_v = _bv(V_, nb_x, BF, 8)
        g_do = _bv(DO_, nb_x, BF, 8)
        g_dv = _bv(DV_, nb_x, BF, 8)
        g_a = _bv(A_, nb_a, BF, 8)
        g_da = _bv(DA_, nb_a4, F32, 4)
        rs, bx = HV * I32(16), rowhv * I32(16)
        rsa, ba = HV * I32(8), rowhv * I32(8)
        rsa4, ba4 = HV * I32(16), rowhv * I32(16)

        smem = fx.SharedAllocator().allocate(_LDS_BYTES)
        l0 = I32(fx.ptrtoint(smem.peek().ptr))
        lda_ = l0 + I32(_OFF_A)

        def opf(buf, base, rs_, row, kvec):
            idx = base + (row + r) * rs_ + kvec + h
            return _ldv(buf, idx, BF, 8).shuffle(_ldv(buf, idx + I32(2), BF, 8), list(range(16)))

        # Every load of a phase is issued before its first use: a load that follows a store in program order is
        # not hoisted above it, and each use then waits for the whole load counter.
        do_i = [tid + I32(128 * u) for u in range(8)]  # do [64][128]: 1024 vectors over 128 threads
        a_i = [tid + I32(128 * u) for u in range(4)]  # A [64][64]
        do_st = [_ldv(g_do, bx + (i // I32(16)) * rs + i % I32(16), BF, 8) for i in do_i]
        a_st = [_ldv(g_a, ba + (i // I32(8)) * rsa + i % I32(8), BF, 8) for i in a_i]
        t0 = w * I32(16)
        # dA rows [16w, 16w + 16): C[m = j][n = t] = sum_v v[j, v] do[t, v]
        dof = [opf(g_do, bx, rs, t0, I32(4 * s)) for s in range(4)]
        vop = [opf(g_v, bx, rs, I32(0), I32(4 * s)) for s in range(4)]
        for i, vec in zip(do_i, do_st):
            llvm.store(_ir(vec), _lds(l0 + (i // I32(16)) * I32(_DO_ROWB) + (i % I32(16)) * I32(16)))
        for i, vec in zip(a_i, a_st):
            llvm.store(_ir(vec), _lds(lda_ + (i // I32(8)) * I32(_A_ROWB) + (i % I32(8)) * I32(16)))
        t = t0 + r
        for jt in range_constexpr(4):
            nxt = [opf(g_v, bx, rs, I32(16 * jt + 16), I32(4 * s)) for s in range(4)] if jt < 3 else vop
            acc = _zero()
            for s in range_constexpr(4):
                acc = _mma(vop[s], dof[s], acc)
            vop = nxt
            c = fx.Vector(acc)
            j0 = I32(16 * jt) + h * I32(8)
            z = F32(0.0)
            o = [(t >= j0 + I32(e)).select(c[e] * scale, z) for e in range(8)]
            di = ba4 + t * rsa4 + I32(4 * jt) + h * I32(2)
            _stv(o[:4], g_da, di, F32)
            _stv(o[4:], g_da, di + I32(1), F32)
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)
        # dv rows [16w, 16w + 16): C[m = v][n = a] = sum_b do[b, v] A[b, a], b >= a
        lane_r = h * I32(8) + lane % I32(8)
        lane_c = ((lane // I32(8)) % I32(2)) * I32(8)

        def tr2(addr, rowb):
            lo = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr)))
            hi = fx.Vector(rocdl.ds_load_tr16_b128(v8b(), _lds(addr + I32(16 * rowb))))
            return lo.shuffle(hi, list(range(16)))

        def dld(vt):
            return [
                tr2(
                    l0 + I32(32 * bs * _DO_ROWB + 32 * vt) + lane_r * I32(_DO_ROWB) + lane_c * I32(2),
                    _DO_ROWB,
                )
                for bs in range(2)
            ]

        a_row = t0 + r  # this lane's output row a
        raws = [
            tr2(
                lda_ + I32(32 * bs * _A_ROWB) + t0 * I32(2) + lane_r * I32(_A_ROWB) + lane_c * I32(2), _A_ROWB
            )
            for bs in range(2)
        ]
        dop = dld(0)
        at = []
        for bs in range_constexpr(2):  # A^T rows a, reduction b in [32 bs, 32 bs + 32), masked to b >= a
            raw = raws[bs]
            bidx = [I32(32 * bs) + h * I32(8) + I32(e) for e in range(8)] + [
                I32(32 * bs + 16) + h * I32(8) + I32(e) for e in range(8)
            ]
            z = BF(0.0)
            at.append(
                fx.Vector.from_elements([(bidx[e] >= a_row).select(raw[e], z) for e in range(16)], dtype=BF)
            )
        for vt in range_constexpr(8):
            nxt = dld(vt + 1) if vt < 7 else dop
            acc = _zero()
            for bs in range_constexpr(2):
                acc = _mma(dop[bs], at[bs], acc)
            dop = nxt
            c = fx.Vector(acc)  # lanes a, elements v = 16 vt + 8h + [0, 8)
            _stv([c[e].to(BF) for e in range(8)], g_dv, bx + a_row * rs + I32(2 * vt) + h, BF)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_kda_dav(
        V_: fx.Tensor,
        A_: fx.Tensor,
        DO_: fx.Tensor,
        DV_: fx.Tensor,
        DA_: fx.Tensor,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_a: fx.Int64,
        nb_a4: fx.Int64,
    ):
        body(V_, A_, DO_, DV_, DA_, scale, T, HV, nb_x, nb_a, nb_a4)

    @flyc.jit
    def launch(
        V_,
        A_,
        DO_,
        DV_,
        DA_,
        scale: fx.Float32,
        T: fx.Int32,
        HV: fx.Int32,
        nb_x: fx.Int64,
        nb_a: fx.Int64,
        nb_a4: fx.Int64,
        nt: fx.Int32,
        nbh: fx.Int32,
        stream: fx.Stream,
    ):
        k_kda_dav(V_, A_, DO_, DV_, DA_, scale, T, HV, nb_x, nb_a, nb_a4).launch(
            grid=(nt, nbh, 1), block=(_NW * _WAVE, 1, 1), stream=stream
        )

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def kda_bwd_dav(v: Tensor, do: Tensor, A: Tensor, scale: float):
    """``(dA, dv)`` as fla's ``chunk_kda_bwd_dAv`` returns them."""
    B, T, HV, V = v.shape
    dA = v.new_empty(B, T, HV, _BT, dtype=torch.float32)
    dv = torch.empty_like(do)
    args = (
        v,
        A,
        do,
        dv,
        dA,
        float(scale),
        T,
        HV,
        v.numel() * 2,
        A.numel() * 2,
        A.numel() * 4,
        T // _BT,
        B * HV,
        torch.cuda.current_stream(),
    )
    flyc, launch = _kernels()
    key = (v.device.index,)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf
    return dA, dv
