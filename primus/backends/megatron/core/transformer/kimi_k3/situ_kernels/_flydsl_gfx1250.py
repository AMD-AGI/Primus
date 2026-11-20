###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused FlyDSL ``situ`` for gfx1250: one launch forward, one launch backward.

Same function as :func:`...situ_activation.situ_pre_mul_fused` on a ``[..., 2I]``
``[gate | up]`` bf16 tensor::

    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    out    = situ_a * (linear_beta * tanh(up / linear_beta))

The eager version materialises every intermediate in fp32 (``.to(float32)``,
two ``tanh``, a ``sigmoid``, four multiplies, and autograd's saved tensors and
backward ops on top). On the routed experts that is ``[tokens * top_k, 2 *
moe_ffn]`` -- about 0.7 G elements per MoE layer and micro-batch at the 4-layer
proxy -- so every intermediate is a multi-GB fp32 round trip. Here the forward
reads the bf16 input once and writes the bf16 output once; the backward saves
only the input, recomputes the activation in registers and writes ``d[gate |
up]`` once.

Lanes move 16 B vectors of 8 elements, interleaved so a wave's accesses coalesce;
when a row's vector count is a multiple of 4 each lane takes 4 of them and
issues every load before the first store. The row count is a runtime argument, so one compiled
kernel serves every routed-token count.
A buffer view addresses at most 2 GiB, so the torch wrapper splits larger
tensors into row chunks.

``tanh(x)`` is ``1 - 2 / (exp(2x) + 1)`` on the hardware ``v_exp_f32``: exact at
both saturations (``exp`` -> 0 gives -1, ``exp`` -> inf gives 1) and within fp32
rounding of ``torch.tanh`` elsewhere; the absolute error near 0 is ~1e-7.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "flydsl_situ_pre_mul_fused",
    "flydsl_situ_with_probs",
    "supports_situ_inputs",
    "supports_situ_probs_inputs",
]

_BLOCK = 256
_VEC = 8
# situ * probs forward: 4 vectors of 8 columns per lane (1 selects the one-vector kernel; 4: 0.71 -> 0.58 ms on the
# routed experts' [229376, 2 x 3072], one vector per lane leaves too few loads in flight)
_FGRP = 4
# tanh / sigmoid divide with v_rcp_f32 (~1 ulp) instead of the IEEE division sequence (about 10 instructions each,
# most of the kernels' instructions)
_RCP = True
# plain situ forward / backward: 4 vectors of 8 columns per lane (1 selects the one-vector kernels; 4: fwd + bwd
# 1.59 -> 1.47 ms at [229376, 3072], 1.08 -> 1.01 at the dense layer's [14336, 33792])
_PLAIN_GRP = 4
_LOG2E = 1.4426950408889634
_MAX_VIEW_BYTES = (1 << 31) - 4096

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_COMPILE_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import range_constexpr

    def _bv(t, nbytes, dt, vec):
        nt = (1 << 31) // (vec * (dt.width // 8))
        b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=nbytes)
        return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))

    def _atom(dt, vec):
        return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)

    def _ldv(buf, tile, dt, vec):
        f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
        fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (tile, None)), f)
        return fx.Vector(f.load().ir_value())

    def _stv(vals, buf, tile, dt):
        n = len(vals)
        f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
        fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
        fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (tile, None)))

    def _exp2(x):
        return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))

    one = 1.0
    two = 2.0

    def _rcp(x):
        return fx.Float32(fx.rocdl.rcp(fx.Float32.ir_type, x.ir_value()))

    def _tanh(x, inv_bound2):
        # tanh(x / bound) with inv_bound2 = 2 * log2(e) / bound
        e = _exp2(x * inv_bound2)
        if _RCP:
            return fx.Float32(one) - fx.Float32(two) * _rcp(e + fx.Float32(one))
        return fx.Float32(one) - fx.Float32(two) / (e + fx.Float32(one))

    def _sigmoid(x):
        if _RCP:
            return _rcp(fx.Float32(one) + _exp2(x * fx.Float32(-_LOG2E)))
        return fx.Float32(one) / (fx.Float32(one) + _exp2(x * fx.Float32(-_LOG2E)))

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situ_fwd(
        X: fx.Tensor,
        Y: fx.Tensor,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gy = _bv(Y, y_bytes, fx.BFloat16, _VEC)
        v = bid * fx.Int32(_BLOCK) + tid
        # Tail lanes redo the last vector and store the same bytes: branch-free.
        vc = (v < n_vec).select(v, n_vec - fx.Int32(1))
        row = vc // vpr
        col = vc - row * vpr
        gi = row * (vpr + vpr) + col
        g8 = _ldv(gx, gi, fx.BFloat16, _VEC)
        u8 = _ldv(gx, gi + vpr, fx.BFloat16, _VEC)
        outs = []
        for e in range_constexpr(_VEC):
            g = fx.Float32(g8[e])
            u = fx.Float32(u8[e])
            a = beta * _tanh(g, g_scale) * _sigmoid(g)
            lu = lbeta * _tanh(u, u_scale)
            outs.append((a * lu).to(fx.BFloat16))
        _stv(outs, gy, vc, fx.BFloat16)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situ_bwd(
        X: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gdy = _bv(DY, y_bytes, fx.BFloat16, _VEC)
        gdx = _bv(DX, x_bytes, fx.BFloat16, _VEC)
        v = bid * fx.Int32(_BLOCK) + tid
        vc = (v < n_vec).select(v, n_vec - fx.Int32(1))
        row = vc // vpr
        col = vc - row * vpr
        gi = row * (vpr + vpr) + col
        g8 = _ldv(gx, gi, fx.BFloat16, _VEC)
        u8 = _ldv(gx, gi + vpr, fx.BFloat16, _VEC)
        d8 = _ldv(gdy, vc, fx.BFloat16, _VEC)
        dgs = []
        dus = []
        for e in range_constexpr(_VEC):
            g = fx.Float32(g8[e])
            u = fx.Float32(u8[e])
            dy = fx.Float32(d8[e])
            tg = _tanh(g, g_scale)
            sg = _sigmoid(g)
            a = beta * tg * sg
            # d(beta * tanh(g / beta)) / dg = 1 - tanh^2; d sigmoid / dg = sg (1 - sg)
            dadg = (fx.Float32(one) - tg * tg) * sg + a * (fx.Float32(one) - sg)
            tu = _tanh(u, u_scale)
            lu = lbeta * tu
            dlu = fx.Float32(one) - tu * tu
            dgs.append((dy * lu * dadg).to(fx.BFloat16))
            dus.append((dy * a * dlu).to(fx.BFloat16))
        _stv(dgs, gdx, gi, fx.BFloat16)
        _stv(dus, gdx, gi + vpr, fx.BFloat16)

    @flyc.jit
    def launch_fwd(
        X,
        Y,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situ_fwd(X, Y, n_vec, vpr, x_bytes, y_bytes, beta, g_scale, lbeta, u_scale).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_bwd(
        X,
        DY,
        DX,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situ_bwd(X, DY, DX, n_vec, vpr, x_bytes, y_bytes, beta, g_scale, lbeta, u_scale).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    # ---- situ(x) * p[row] (routed experts: the router probability folded in) ----------------------------
    # Bodies are plain helpers: FlyDSL rewrites a kernel's own AST, these run as ordinary Python at trace time.
    _PGRP = 8  # vectors (of 8 columns) per lane in the backward: each lane owns 64 columns of one row, interleaved

    def _act(g, u, beta, g_scale, lbeta, u_scale):
        tg = _tanh(g, g_scale)
        sg = _sigmoid(g)
        a = beta * tg * sg
        tu = _tanh(u, u_scale)
        return tg, sg, a, tu, lbeta * tu

    def _situp_fwd_body(gx, gp, gy, n_vec, vpr, beta, g_scale, lbeta, u_scale):
        v = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        vc = (v < n_vec).select(v, n_vec - fx.Int32(1))
        row = vc // vpr
        col = vc - row * vpr
        gi = row * (vpr + vpr) + col
        g8 = _ldv(gx, gi, fx.BFloat16, _VEC)
        u8 = _ldv(gx, gi + vpr, fx.BFloat16, _VEC)
        pr = fx.Float32(_ldv(gp, row, fx.Float32, 1)[0])
        outs = []
        for e in range(_VEC):
            _, _, a, _, lu = _act(fx.Float32(g8[e]), fx.Float32(u8[e]), beta, g_scale, lbeta, u_scale)
            outs.append((a * lu * pr).to(fx.BFloat16))
        _stv(outs, gy, vc, fx.BFloat16)

    def _situp_bwd_body(gx, gp, gdy, gdx, gpp, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale):
        lane = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        lc = (lane < n_lanes).select(lane, n_lanes - fx.Int32(1))
        row = lc // gpr
        grp = lc - row * gpr
        pr = fx.Float32(_ldv(gp, row, fx.Float32, 1)[0])
        acc = fx.Float32(0.0)
        # interleaved, so neighbouring lanes touch neighbouring vectors at every j (coalesced)
        cols = [grp + fx.Int32(j) * gpr for j in range(_PGRP)]
        gis = [row * (vpr + vpr) + col for col in cols]
        # Every load before the first store: a load of X / DY cannot be moved above a store to DX (the compiler
        # cannot prove they do not alias), so interleaving them would serialise the lane.
        loads = [
            (
                _ldv(gx, gi, fx.BFloat16, _VEC),
                _ldv(gx, gi + vpr, fx.BFloat16, _VEC),
                _ldv(gdy, row * vpr + col, fx.BFloat16, _VEC),
            )
            for gi, col in zip(gis, cols)
        ]
        for j in range(_PGRP):
            gi = gis[j]
            g8, u8, d8 = loads[j]
            dgs, dus = [], []
            for e in range(_VEC):
                tg, sg, a, tu, lu = _act(fx.Float32(g8[e]), fx.Float32(u8[e]), beta, g_scale, lbeta, u_scale)
                dout = fx.Float32(d8[e])
                acc = acc + dout * (a * lu)
                dyf = dout * pr
                dadg = (fx.Float32(one) - tg * tg) * sg + a * (fx.Float32(one) - sg)
                dgs.append((dyf * lu * dadg).to(fx.BFloat16))
                dus.append((dyf * a * (fx.Float32(one) - tu * tu)).to(fx.BFloat16))
            _stv(dgs, gdx, gi, fx.BFloat16)
            _stv(dus, gdx, gi + vpr, fx.BFloat16)
        _stv([acc], gpp, lc, fx.Float32)

    def _situp_fwd_g_body(gx, gp, gy, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale):
        """_FGRP vectors per lane (columns grp + j * gpr, so neighbouring lanes stay on neighbouring vectors), all
        loads first: one 16 B vector per lane left too few bytes in flight (about 5 TB/s)."""
        lane = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        lc = (lane < n_lanes).select(lane, n_lanes - fx.Int32(1))
        row = lc // gpr
        grp = lc - row * gpr
        cols = [grp + fx.Int32(j) * gpr for j in range(_FGRP)]
        base = row * (vpr + vpr)
        loads = [
            (_ldv(gx, base + col, fx.BFloat16, _VEC), _ldv(gx, base + col + vpr, fx.BFloat16, _VEC))
            for col in cols
        ]
        pr = fx.Float32(_ldv(gp, row, fx.Float32, 1)[0])
        for col, (g8, u8) in zip(cols, loads):
            outs = []
            for e in range(_VEC):
                _, _, a, _, lu = _act(fx.Float32(g8[e]), fx.Float32(u8[e]), beta, g_scale, lbeta, u_scale)
                outs.append((a * lu * pr).to(fx.BFloat16))
            _stv(outs, gy, row * vpr + col, fx.BFloat16)

    def _situ_g_rows(n_lanes, gpr):
        lane = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        lc = (lane < n_lanes).select(lane, n_lanes - fx.Int32(1))
        row = lc // gpr
        return row, [lc - row * gpr + fx.Int32(j) * gpr for j in range(_PLAIN_GRP)]

    def _situ_fwd_g_body(gx, gy, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale):
        """Plain situ forward with _PLAIN_GRP interleaved vectors per lane, all loads first."""
        row, cols = _situ_g_rows(n_lanes, gpr)
        base = row * (vpr + vpr)
        loads = [
            (_ldv(gx, base + col, fx.BFloat16, _VEC), _ldv(gx, base + col + vpr, fx.BFloat16, _VEC))
            for col in cols
        ]
        for col, (g8, u8) in zip(cols, loads):
            outs = []
            for e in range(_VEC):
                _, _, a, _, lu = _act(fx.Float32(g8[e]), fx.Float32(u8[e]), beta, g_scale, lbeta, u_scale)
                outs.append((a * lu).to(fx.BFloat16))
            _stv(outs, gy, row * vpr + col, fx.BFloat16)

    def _situ_bwd_g_body(gx, gdy, gdx, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale):
        """Plain situ backward with _PLAIN_GRP interleaved vectors per lane, all loads first."""
        row, cols = _situ_g_rows(n_lanes, gpr)
        base = row * (vpr + vpr)
        loads = [
            (
                _ldv(gx, base + col, fx.BFloat16, _VEC),
                _ldv(gx, base + col + vpr, fx.BFloat16, _VEC),
                _ldv(gdy, row * vpr + col, fx.BFloat16, _VEC),
            )
            for col in cols
        ]
        for col, (g8, u8, d8) in zip(cols, loads):
            dgs, dus = [], []
            for e in range(_VEC):
                tg, sg, a, tu, lu = _act(fx.Float32(g8[e]), fx.Float32(u8[e]), beta, g_scale, lbeta, u_scale)
                dy = fx.Float32(d8[e])
                dadg = (fx.Float32(one) - tg * tg) * sg + a * (fx.Float32(one) - sg)
                dgs.append((dy * lu * dadg).to(fx.BFloat16))
                dus.append((dy * a * (fx.Float32(one) - tu * tu)).to(fx.BFloat16))
            _stv(dgs, gdx, base + col, fx.BFloat16)
            _stv(dus, gdx, base + col + vpr, fx.BFloat16)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situ_fwd_g(
        X: fx.Tensor,
        Y: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gy = _bv(Y, y_bytes, fx.BFloat16, _VEC)
        _situ_fwd_g_body(gx, gy, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situ_bwd_g(
        X: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gdy = _bv(DY, y_bytes, fx.BFloat16, _VEC)
        gdx = _bv(DX, x_bytes, fx.BFloat16, _VEC)
        _situ_bwd_g_body(gx, gdy, gdx, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale)

    @flyc.jit
    def launch_situ_fwd_g(
        X,
        Y,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situ_fwd_g(X, Y, n_lanes, vpr, gpr, x_bytes, y_bytes, beta, g_scale, lbeta, u_scale).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_situ_bwd_g(
        X,
        DY,
        DX,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situ_bwd_g(X, DY, DX, n_lanes, vpr, gpr, x_bytes, y_bytes, beta, g_scale, lbeta, u_scale).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situp_fwd_g(
        X: fx.Tensor,
        P: fx.Tensor,
        Y: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gy = _bv(Y, y_bytes, fx.BFloat16, _VEC)
        gp = _bv(P, p_bytes, fx.Float32, 1)
        _situp_fwd_g_body(gx, gp, gy, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale)

    @flyc.jit
    def launch_situp_fwd_g(
        X,
        P,
        Y,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situp_fwd_g(
            X, P, Y, n_lanes, vpr, gpr, x_bytes, y_bytes, p_bytes, beta, g_scale, lbeta, u_scale
        ).launch(grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situp_fwd(
        X: fx.Tensor,
        P: fx.Tensor,
        Y: fx.Tensor,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gy = _bv(Y, y_bytes, fx.BFloat16, _VEC)
        gp = _bv(P, p_bytes, fx.Float32, 1)
        _situp_fwd_body(gx, gp, gy, n_vec, vpr, beta, g_scale, lbeta, u_scale)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_situp_bwd(
        X: fx.Tensor,
        P: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        PP: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        pp_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gdy = _bv(DY, y_bytes, fx.BFloat16, _VEC)
        gdx = _bv(DX, x_bytes, fx.BFloat16, _VEC)
        gp = _bv(P, p_bytes, fx.Float32, 1)
        gpp = _bv(PP, pp_bytes, fx.Float32, 1)
        _situp_bwd_body(gx, gp, gdy, gdx, gpp, n_lanes, vpr, gpr, beta, g_scale, lbeta, u_scale)

    @flyc.jit
    def launch_situp_fwd(
        X,
        P,
        Y,
        n_vec: fx.Int32,
        vpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situp_fwd(X, P, Y, n_vec, vpr, x_bytes, y_bytes, p_bytes, beta, g_scale, lbeta, u_scale).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_situp_bwd(
        X,
        P,
        DY,
        DX,
        PP,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        gpr: fx.Int32,
        x_bytes: fx.Int64,
        y_bytes: fx.Int64,
        p_bytes: fx.Int64,
        pp_bytes: fx.Int64,
        beta: fx.Float32,
        g_scale: fx.Float32,
        lbeta: fx.Float32,
        u_scale: fx.Float32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_situp_bwd(
            X,
            P,
            DY,
            DX,
            PP,
            n_lanes,
            vpr,
            gpr,
            x_bytes,
            y_bytes,
            p_bytes,
            pp_bytes,
            beta,
            g_scale,
            lbeta,
            u_scale,
        ).launch(grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream)

    return (
        flyc,
        launch_fwd,
        launch_bwd,
        launch_situp_fwd,
        launch_situp_bwd,
        launch_situp_fwd_g,
        launch_situ_fwd_g,
        launch_situ_bwd_g,
    )


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def _run(name, exe, *args):
    """Compile once per launcher and device (every shape argument is runtime), then reuse.

    The first call compiles under a lock; ``flyc.compile`` launches it on its arguments
    (``flydsl.compiler.jit_function._compile_impl``), so it runs once."""
    flyc = _kernels()[0]
    key = (name, args[0].device)
    cf = _COMPILED.get(key)
    if cf is None:
        with _COMPILE_LOCK:
            cf = _COMPILED.get(key)
            if cf is None:
                cf = flyc.compile(exe, *args)
                if cf is not None:
                    _COMPILED[key] = cf
                return
    cf(*args)


def _aligned(t: Tensor) -> Tensor:
    """``t``, or a contiguous copy when its first element is not 16-byte aligned.

    The kernels move 16 bytes per lane per access, and ``contiguous()`` keeps a view's
    storage offset; same rule as the vector loads in ``weight_deosc.py``.
    """
    return t.clone(memory_format=torch.contiguous_format) if t.data_ptr() % 16 else t


def supports_situ_inputs(x: Tensor) -> Optional[str]:
    """``None`` when the kernel can run ``x``, else why it cannot."""
    if not x.is_cuda:
        return "x is on CPU"
    if x.dtype != torch.bfloat16:
        return f"dtype {x.dtype} is not bf16"
    if x.shape[-1] % (2 * _VEC) != 0:
        return f"last dim {x.shape[-1]} is not a multiple of {2 * _VEC}"
    arch = str(getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        return f"device arch is {arch!r}, not gfx1250"
    return None


def _row_chunks(rows: int, row_bytes: int):
    step = max(1, _MAX_VIEW_BYTES // row_bytes)
    for r0 in range(0, rows, step):
        yield r0, min(rows, r0 + step)


def _scalars(beta: float, linear_beta: float):
    return (
        float(beta),
        float(2.0 * _LOG2E / beta),
        float(linear_beta),
        float(2.0 * _LOG2E / linear_beta),
    )


class _SituFlydsl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, beta: float, linear_beta: float):
        x = _aligned(x.contiguous())
        two_i = x.shape[-1]
        inter = two_i // 2
        rows = x.numel() // two_i
        y = torch.empty(*x.shape[:-1], inter, dtype=x.dtype, device=x.device)
        if rows:
            launch_fwd = _kernels()[1]
            x2, y2 = x.view(rows, two_i), y.view(rows, inter)
            vpr = inter // _VEC
            stream = torch.cuda.current_stream()
            for r0, r1 in _row_chunks(rows, two_i * x.element_size()):
                xs, ys = x2[r0:r1], y2[r0:r1]
                if _PLAIN_GRP > 1 and vpr % _PLAIN_GRP == 0:
                    gpr = vpr // _PLAIN_GRP
                    n_lanes = (r1 - r0) * gpr
                    _run(
                        "fwd_g",
                        _kernels()[6],
                        xs,
                        ys,
                        n_lanes,
                        vpr,
                        gpr,
                        xs.numel() * 2,
                        ys.numel() * 2,
                        *_scalars(beta, linear_beta),
                        (n_lanes + _BLOCK - 1) // _BLOCK,
                        stream,
                    )
                    continue
                n_vec = (r1 - r0) * vpr
                _run(
                    "fwd",
                    launch_fwd,
                    xs,
                    ys,
                    n_vec,
                    vpr,
                    xs.numel() * 2,
                    ys.numel() * 2,
                    *_scalars(beta, linear_beta),
                    (n_vec + _BLOCK - 1) // _BLOCK,
                    stream,
                )
        ctx.save_for_backward(x)
        ctx.betas = (beta, linear_beta)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        (x,) = ctx.saved_tensors
        beta, linear_beta = ctx.betas
        dy = _aligned(dy.contiguous())
        two_i = x.shape[-1]
        inter = two_i // 2
        rows = x.numel() // two_i
        dx = torch.empty_like(x)
        if rows:
            launch_bwd = _kernels()[2]
            x2, dy2, dx2 = x.view(rows, two_i), dy.view(rows, inter), dx.view(rows, two_i)
            vpr = inter // _VEC
            stream = torch.cuda.current_stream()
            for r0, r1 in _row_chunks(rows, two_i * x.element_size()):
                xs, dys, dxs = x2[r0:r1], dy2[r0:r1], dx2[r0:r1]
                if _PLAIN_GRP > 1 and vpr % _PLAIN_GRP == 0:
                    gpr = vpr // _PLAIN_GRP
                    n_lanes = (r1 - r0) * gpr
                    _run(
                        "bwd_g",
                        _kernels()[7],
                        xs,
                        dys,
                        dxs,
                        n_lanes,
                        vpr,
                        gpr,
                        xs.numel() * 2,
                        dys.numel() * 2,
                        *_scalars(beta, linear_beta),
                        (n_lanes + _BLOCK - 1) // _BLOCK,
                        stream,
                    )
                    continue
                n_vec = (r1 - r0) * vpr
                _run(
                    "bwd",
                    launch_bwd,
                    xs,
                    dys,
                    dxs,
                    n_vec,
                    vpr,
                    xs.numel() * 2,
                    dys.numel() * 2,
                    *_scalars(beta, linear_beta),
                    (n_vec + _BLOCK - 1) // _BLOCK,
                    stream,
                )
        return dx, None, None


def flydsl_situ_pre_mul_fused(x: Tensor, *, beta: float, linear_beta: Optional[float]) -> Tensor:
    """``situ`` on a fused ``[..., 2I]`` bf16 tensor, interchangeable with the eager entry.

    Raises:
        ValueError: when the inputs or the device are outside what the kernel supports, or
            ``linear_beta`` is None (the kernel always soft-clamps the up branch).
    """
    if linear_beta is None:
        raise ValueError("the FlyDSL situ kernel needs linear_beta; select situ_backend: eager")
    reason = supports_situ_inputs(x)
    if reason is not None:
        raise ValueError(
            f"the FlyDSL situ kernel cannot run these inputs: {reason}. Select situ_backend: eager."
        )
    return _SituFlydsl.apply(x, float(beta), float(linear_beta))


_PCOLS = 64  # columns per lane in the probs backward (_PGRP vectors of _VEC)


class _SituProbsFlydsl(torch.autograd.Function):
    """``situ(x) * probs[row]``: the routed experts' activation with the router probability folded in, so the
    fp32 ``[rows, I]`` product and its cast (forward), and the two products and the cast (backward), are gone.
    ``d probs[row] = sum_c dout[row, c] * situ(x)[row, c]`` is reduced over 64 columns per lane in the kernel
    and over the row's ``I / 64`` lane partials here."""

    @staticmethod
    def forward(ctx, x: Tensor, probs: Tensor, beta: float, linear_beta: float):
        x = _aligned(x.contiguous())
        p = probs.reshape(-1).float().contiguous()
        two_i = x.shape[-1]
        inter = two_i // 2
        rows = x.numel() // two_i
        y = torch.empty(*x.shape[:-1], inter, dtype=x.dtype, device=x.device)
        if rows:
            x2, y2 = x.view(rows, two_i), y.view(rows, inter)
            vpr = inter // _VEC
            stream = torch.cuda.current_stream()
            grouped = _FGRP > 1 and vpr % _FGRP == 0
            launch = _kernels()[5 if grouped else 3]
            for r0, r1 in _row_chunks(rows, two_i * x.element_size()):
                xs, ys, ps = x2[r0:r1], y2[r0:r1], p[r0:r1]
                if grouped:
                    gpr = vpr // _FGRP
                    n_lanes = (r1 - r0) * gpr
                    _run(
                        "situp_fwd_g",
                        launch,
                        xs,
                        ps,
                        ys,
                        n_lanes,
                        vpr,
                        gpr,
                        xs.numel() * 2,
                        ys.numel() * 2,
                        ps.numel() * 4,
                        *_scalars(beta, linear_beta),
                        (n_lanes + _BLOCK - 1) // _BLOCK,
                        stream,
                    )
                    continue
                n_vec = (r1 - r0) * vpr
                _run(
                    "situp_fwd",
                    launch,
                    xs,
                    ps,
                    ys,
                    n_vec,
                    vpr,
                    xs.numel() * 2,
                    ys.numel() * 2,
                    ps.numel() * 4,
                    *_scalars(beta, linear_beta),
                    (n_vec + _BLOCK - 1) // _BLOCK,
                    stream,
                )
        ctx.save_for_backward(x, p)
        ctx.betas = (beta, linear_beta)
        ctx.probs_shape, ctx.probs_dtype = probs.shape, probs.dtype
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, p = ctx.saved_tensors
        beta, linear_beta = ctx.betas
        dy = _aligned(dy.contiguous())
        two_i = x.shape[-1]
        inter = two_i // 2
        rows = x.numel() // two_i
        gpr = inter // _PCOLS
        dx = torch.empty_like(x)
        part = torch.empty(rows, gpr, dtype=torch.float32, device=x.device)
        if rows:
            launch = _kernels()[4]
            x2, dy2, dx2 = x.view(rows, two_i), dy.view(rows, inter), dx.view(rows, two_i)
            vpr = inter // _VEC
            stream = torch.cuda.current_stream()
            for r0, r1 in _row_chunks(rows, two_i * x.element_size()):
                xs, dys, dxs, ps, pps = x2[r0:r1], dy2[r0:r1], dx2[r0:r1], p[r0:r1], part[r0:r1]
                n_lanes = (r1 - r0) * gpr
                _run(
                    "situp_bwd",
                    launch,
                    xs,
                    ps,
                    dys,
                    dxs,
                    pps,
                    n_lanes,
                    vpr,
                    gpr,
                    xs.numel() * 2,
                    dys.numel() * 2,
                    ps.numel() * 4,
                    pps.numel() * 4,
                    *_scalars(beta, linear_beta),
                    (n_lanes + _BLOCK - 1) // _BLOCK,
                    stream,
                )
        dp = part.sum(-1).reshape(ctx.probs_shape).to(ctx.probs_dtype)
        return dx, dp, None, None


def supports_situ_probs_inputs(x: Tensor, probs: Tensor) -> Optional[str]:
    """``None`` when :func:`flydsl_situ_with_probs` can run, else why not."""
    reason = supports_situ_inputs(x)
    if reason is not None:
        return reason
    if (x.shape[-1] // 2) % _PCOLS:
        return f"intermediate width {x.shape[-1] // 2} is not a multiple of {_PCOLS}"
    if probs.numel() != x.numel() // x.shape[-1]:
        return f"probs has {probs.numel()} elements for {x.numel() // x.shape[-1]} rows"
    return None


def flydsl_situ_with_probs(x: Tensor, probs: Tensor, *, beta: float, linear_beta: Optional[float]) -> Tensor:
    """``situ(x) * probs`` (probs one per row, e.g. ``[rows, 1]``), the result in ``x``'s dtype."""
    if linear_beta is None:
        raise ValueError("the FlyDSL situ kernel needs linear_beta; select situ_backend: eager")
    reason = supports_situ_probs_inputs(x, probs)
    if reason is not None:
        raise ValueError(
            f"the FlyDSL situ-with-probs kernel cannot run these inputs: {reason}. Select situ_backend: eager."
        )
    return _SituProbsFlydsl.apply(x, probs, float(beta), float(linear_beta))
