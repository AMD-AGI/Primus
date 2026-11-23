###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The FP32 MoE router's gating GEMM on gfx1250 WMMA (FlyDSL): FP32 ``logits = x W^T`` from BF16 x [M, K] and
W [N, K].

hipBLASLt's BF16 x BF16 -> FP32 kernels are slow on gfx1250 at router shapes (many tokens, a few hundred experts),
and an FP32 router keeps FP32 logits for its top-k. One 4-wave workgroup computes 128 tokens x 128 experts, each
wave 64 x 64 (16 accumulators); the WMMA's A operand is W (experts), B is x (tokens), so a lane's C fragment is 8
consecutive experts of one token (two 16 B stores). The K loop runs over 32-wide k-steps. The workgroup's 128 x 32
token and expert tiles of a k-step are loaded once (each thread 4 + 4 16 B pieces) into a 3-stage LDS ring, and the
waves read their operands there, with one barrier per k-step. Reading the operands straight from global memory, with
every tile read by two waves, is about twice as slow.

:func:`router_logits_unsupported_reason` says which operands it takes. Kernels are built on first use and compiled
once per device; the shapes, buffer sizes and the stream are runtime arguments.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["flydsl_router_logits", "router_logits_unsupported_reason"]

_TM = 128  # tokens per workgroup
_TN = 128  # experts per workgroup
_KS = 32  # k per step (one bf16 WMMA)
_NW = 4
_WAVE = 32
_MAX_VIEW_BYTES = (1 << 31) - 4096
# k-steps of the LDS ring's global loads in flight, LDS stages, and the row pitch of a 32-wide bf16 k-step in LDS
# in bytes (64 + 16 pad). 4 / 3 is the fastest at 8192 x 896 x 7168 (141 us; 1 / 3: 150 us; 4 stages: the same).
# The ring needs 3 stages: with 2, a stage is refilled while other waves still read it.
_LH = 4
_NS = 3
_LP = 80
assert _NS >= 3

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import llvm
    from flydsl.expr import rocdl

    BF, F32, I32 = fx.BFloat16, fx.Float32, fx.Int32

    def v8f():
        return fx.Vector.make_type(8, F32)

    def v8b():
        return fx.Vector.make_type(8, BF)

    def _lds(addr):
        pt = fx.PointerType.get(I32.ir_type, address_space=fx.AddressSpace.Shared, alignment=4)
        return fx.to_llvm_ptr(fx.inttoptr(pt, addr))

    def _bar():
        rocdl.sched_barrier(0)
        fx.gpu.barrier(address_spaces=ir.Attribute.parse("[#gpu.address_space<workgroup>]"))
        rocdl.sched_barrier(0)

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

    def _mma(a, b, c):
        return rocdl.wmma_f32_16x16x32_bf16(v8f(), _ir(a), _ir(b), c, reuseA=False, reuseB=False).result

    def _zero():
        return _ir(fx.Vector.filled(8, 0.0, F32))

    def _rt_loop(step, state, n):
        @flyc.jit
        def loop(st, n_):
            final = st
            for it, carried in range(fx.Index(I32(0)), fx.Index(n_), 1, init=st):
                res = step(list(carried), I32(it))
                final = yield res
            return final

        res = loop(state, n)
        return list(res) if isinstance(res, (list, tuple)) else [res]

    def _slot_loop(sub, acc, load, n, u):
        # u k-steps per iteration, slot j only ever holds slot j's loads (no register rotation, whose copies wait
        # for every load in flight); load(j) -> slot j's first loads, sub(acc, slot, ks) -> (acc, next loads of the
        # slot); needs n % u == 0
        # (sched_barriers keep each slot's loads in order, or the waits at the loop head merge to 0)
        slots = []
        for j in range(u):
            if j:
                rocdl.sched_barrier(0)
            slots.append(load(j))
        s = len(slots[0])

        def it_body(cur, it):
            a, sl = cur[:16], [cur[16 + s * j : 16 + s * (j + 1)] for j in range(u)]
            for j in range(u):
                if j:
                    rocdl.sched_barrier(0)
                a, sl[j] = sub(a, sl[j], it * I32(u) + I32(j))
            return a + [x for q in sl for x in q]

        return _rt_loop(it_body, acc + [x for q in slots for x in q], n // I32(u))[:16]

    def body(X, W, Y, M, N, K, nb_x, nb_w, nb_y):
        lane = I32(fx.thread_idx.x) % I32(_WAVE)
        w = I32(rocdl.wave_id())
        r = lane % I32(16)
        h = lane // I32(16)
        t0 = I32(fx.block_idx.y) * I32(_TM) + (w // I32(2)) * I32(64)  # this wave's 64 tokens
        e0 = I32(fx.block_idx.x) * I32(_TN) + (w % I32(2)) * I32(64)  # and 64 experts
        g_x, g_w = _bv(X, nb_x, BF, 8), _bv(W, nb_w, BF, 8)
        g_y = _bv(Y, nb_y, F32, 4)
        rs = K // I32(8)  # 8-element vectors per row of x / W

        nks = K // I32(_KS)

        def lds_loop(g_x, g_w, rs, nks):
            # the workgroup's 128 x 32 token and expert tiles of a k-step go to an _NS-stage LDS ring once (each
            # thread 4 + 4 16 B pieces), the waves read their operands there; one barrier per k-step
            tid = I32(fx.thread_idx.x)
            bt0 = I32(fx.block_idx.y) * I32(_TM)
            be0 = I32(fx.block_idx.x) * I32(_TN)
            smem = fx.SharedAllocator().allocate(_NS * 2 * _TM * _LP)
            l0 = I32(fx.ptrtoint(smem.peek().ptr))
            vs = [tid + I32(128 * u) for u in range(4)]  # 512 16 B pieces per tile: row v // 4, piece v % 4

            def gl(ks):
                return [_ldv(g_x, (bt0 + v // I32(4)) * rs + ks * I32(4) + v % I32(4), BF, 8) for v in vs] + [
                    _ldv(g_w, (be0 + v // I32(4)) * rs + ks * I32(4) + v % I32(4), BF, 8) for v in vs
                ]

            def st(stage, regs):
                for i, v in enumerate(vs):
                    a = (v // I32(4)) * I32(_LP) + (v % I32(4)) * I32(16)
                    llvm.store(_ir(regs[i]), _lds(l0 + stage * I32(2 * _TM * _LP) + a))
                    llvm.store(_ir(regs[4 + i]), _lds(l0 + stage * I32(2 * _TM * _LP) + I32(_TM * _LP) + a))

            def lop(
                stage, part, row
            ):  # operand from LDS: rows row + [0, 16), k 8h + [0, 8) and 16 + 8h + [0, 8)
                a = (
                    l0
                    + stage * I32(2 * _TM * _LP)
                    + I32(part * _TM * _LP)
                    + (row + r) * I32(_LP)
                    + h * I32(16)
                )
                lo = fx.Vector(llvm.load(v8b(), _lds(a)))
                hi = fx.Vector(llvm.load(v8b(), _lds(a + I32(32))))
                return lo.shuffle(hi, list(range(16)))

            lt = (w // I32(2)) * I32(64)  # this wave's rows inside the tiles
            le = (w % I32(2)) * I32(64)
            for p in range(_NS - 1):
                st(I32(p), gl(I32(p)))
            _bar()

            # k-step ks: LDS holds steps ks .. ks + _NS - 2, the slot step ks + _NS - 1 (loaded _LH steps ago); stage
            # (ks - 1) % _NS was last read before the previous barrier
            def lsub(acc, slot, ks):
                stage = ks % I32(_NS)
                op = [lop(stage, 1, le + I32(16 * et)) for et in range(4)] + [
                    lop(stage, 0, lt + I32(16 * tt)) for tt in range(4)
                ]
                out = [_mma(op[et], op[4 + tt], acc[4 * tt + et]) for tt in range(4) for et in range(4)]
                _bar()
                st((ks + I32(_NS - 1)) % I32(_NS), slot)
                return out, [_ir(x) for x in gl((ks + I32(_NS - 1 + _LH)) % nks)]

            def first(p):
                return [_ir(x) for x in gl(I32(_NS - 1 + p) % nks)]

            return _slot_loop(lsub, [_zero() for _ in range(16)], first, nks, _LH)

        res = lds_loop(g_x, g_w, rs, nks)
        n4 = N // I32(4)
        for tt in range(4):
            t = t0 + I32(16 * tt) + r
            for et in range(4):
                c = fx.Vector(res[4 * tt + et])  # lane (token t, h): experts e0 + 16 et + 8h + [0, 8)
                yi = t * n4 + (e0 + I32(16 * et) + h * I32(8)) // I32(4)
                _stv([c[e] for e in range(4)], g_y, yi, F32)
                _stv([c[e] for e in range(4, 8)], g_y, yi + I32(1), F32)

    @flyc.kernel(known_block_size=[_NW * _WAVE, 1, 1])
    def k_router_logits(
        X: fx.Tensor,
        W: fx.Tensor,
        Y: fx.Tensor,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        nb_x: fx.Int64,
        nb_w: fx.Int64,
        nb_y: fx.Int64,
    ):
        body(X, W, Y, M, N, K, nb_x, nb_w, nb_y)

    @flyc.jit
    def launch(
        X,
        W,
        Y,
        M: fx.Int32,
        N: fx.Int32,
        K: fx.Int32,
        nb_x: fx.Int64,
        nb_w: fx.Int64,
        nb_y: fx.Int64,
        gn: fx.Int32,
        gm: fx.Int32,
        stream: fx.Stream,
    ):
        k_router_logits(X, W, Y, M, N, K, nb_x, nb_w, nb_y).launch(
            grid=(gn, gm, 1), block=(_NW * _WAVE, 1, 1), stream=stream
        )

    return flyc, launch


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def router_logits_unsupported_reason(
    x: Tensor, weight: Tensor, bias: Optional[Tensor], out_dtype
) -> Optional[str]:
    """``None`` when :func:`flydsl_router_logits` takes ``x [.., K] @ weight [N, K]^T`` in ``out_dtype``."""
    if bias is not None or out_dtype != torch.float32:
        return "bias or a non-FP32 output"
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return "dtypes (bf16 x / weight)"
    K = weight.shape[1]
    M = x.numel() // K if K else 0
    N = weight.shape[0]
    if x.shape[-1] != K or M % _TM or N % _TN or K % (_KS * _LH):
        return f"shape M {M} / N {N} / K {K} (needs M % {_TM}, N % {_TN}, K % {_KS * _LH} == 0)"
    if not (x.is_contiguous() and weight.is_contiguous()):
        return "non-contiguous"
    if x.numel() * 2 > _MAX_VIEW_BYTES or M * N * 4 > _MAX_VIEW_BYTES:
        return "a tensor exceeds one 2 GiB buffer view"
    if not x.is_cuda:
        return "CPU tensors"
    arch = str(getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        return f"device arch is {arch!r}, not gfx1250"
    return None


def flydsl_router_logits(x: Tensor, weight: Tensor) -> Tensor:
    """``(x.reshape(-1, K) @ weight^T)`` in FP32, ``[M, N]``."""
    K = weight.shape[1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], weight.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.float32)
    args = (
        x2,
        weight,
        y,
        M,
        N,
        K,
        x2.numel() * 2,
        weight.numel() * 2,
        y.numel() * 4,
        N // _TN,
        M // _TM,
        torch.cuda.current_stream(),
    )
    flyc, launch = _kernels()
    key = (x.device.index,)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf
    return y
