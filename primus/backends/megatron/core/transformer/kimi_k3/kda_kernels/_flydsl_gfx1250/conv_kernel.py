###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Causal depthwise conv1d + SiLU for the KDA short convolutions, in FlyDSL for gfx1250.

``y[b, t, d] = silu(sum_w x[b, t - (W - 1) + w, d] * weight[d, w])`` with zero padding before ``t = 0``, on a
contiguous ``[B, T, D]`` bf16 tensor -- what ``fla.modules.convolution.causal_conv1d(..., activation="silu")``
computes for ``KimiDeltaAttention._short_conv`` (no bias, no packed sequences). The short convolutions of the
``flydsl`` KDA backend on gfx1250.

fla's Triton backward on gfx1250 runs ~10x off memory bandwidth at the K3 shape (``[2, 7168, 12288]``) and
re-runs the forward first. Here each lane owns 8 contiguous channels (16 B loads / stores) of one ``_TS``-step
time segment and keeps a ``W``-deep window of the input in registers: the forward reads ``x`` once and writes
``y`` once; the backward reads ``x`` and ``dy``, recomputes the pre-activation in registers, writes ``dx`` and
one fp32 ``[W, 8]`` weight-gradient partial per lane, and the partials are summed on the host side. All loads of
a lane are issued before its first store, and the sigmoid divides with ``v_rcp_f32`` (about 1 ulp) instead of the
IEEE division sequence (42% of the forward's instructions; forward + backward 0.407 -> 0.358 ms at the K3 shape).
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["flydsl_causal_conv1d_silu", "supports_conv_inputs"]

# Threads per block and time steps per lane (measured best at the K3 shape among blocks 32-256, TS 4-64).
_BLOCK = 64
_VEC = 8
_TS = 16
_LOG2E = 1.4426950408889634
_MAX_VIEW_BYTES = (1 << 31) - 4096

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_BUILT: Dict[int, tuple] = {}


def _build(W: int):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

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

    def _ld8f(buf4, idx8):
        """8 fp32 at vector index ``idx8`` of an fp32 view laid out 4 wide (16 B copies)."""
        lo = _ldv(buf4, idx8 * fx.Int32(2), fx.Float32, 4)
        hi = _ldv(buf4, idx8 * fx.Int32(2) + fx.Int32(1), fx.Float32, 4)
        return [fx.Float32(lo[e]) for e in range(4)] + [fx.Float32(hi[e]) for e in range(4)]

    def _st8f(vals, buf4, idx8):
        _stv(vals[:4], buf4, idx8 * fx.Int32(2), fx.Float32)
        _stv(vals[4:], buf4, idx8 * fx.Int32(2) + fx.Int32(1), fx.Float32)

    def _sigmoid(x):
        e = fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, (x * fx.Float32(-_LOG2E)).ir_value()))
        return fx.Float32(fx.rocdl.rcp(fx.Float32.ir_type, (fx.Float32(1.0) + e).ir_value()))

    def _lane(n_lanes, vpr, nseg):
        lane = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        # Tail lanes redo the last lane and store the same bytes: branch-free.
        lc = (lane < n_lanes).select(lane, n_lanes - fx.Int32(1))
        rest = lc // vpr
        dv = lc - rest * vpr
        b = rest // nseg
        seg = rest - b * nseg
        return dv, seg, b

    def _raw_x(gx, b, t, T, vpr, dv):
        """x[b, t, dv*8 : dv*8+8] as loaded (bf16) and its causal-padding mask (0 for t < 0)."""
        valid = t >= fx.Int32(0)
        tc = valid.select(t, fx.Int32(0))
        return _ldv(gx, (b * T + tc) * vpr + dv, fx.BFloat16, _VEC), valid.select(
            fx.Float32(1.0), fx.Float32(0.0)
        )

    def _raw_dy(gdy, b, t, T, vpr, dv):
        """dy[b, t, ...] as loaded and its mask (0 for t >= T: the last segment's look-ahead)."""
        in_t = t < T
        tc = in_t.select(t, T - fx.Int32(1))
        return _ldv(gdy, (b * T + tc) * vpr + dv, fx.BFloat16, _VEC), in_t.select(
            fx.Float32(1.0), fx.Float32(0.0)
        )

    def _cvt(raw):
        v8, m = raw
        return [fx.Float32(v8[e]) * m for e in range(_VEC)]

    # The kernel bodies below only set up views and call these: FlyDSL rewrites a kernel's own AST (its
    # ``for`` / ``if`` / ``continue``), while plain helpers run as ordinary Python at trace time, so the
    # loops here unroll and the conditions are decided when the kernel is built.
    def _weights(gw, vpr, dv):
        return [_ld8f(gw, fx.Int32(w) * vpr + dv) for w in range(W)]

    def _preact(win, wv, e):
        s = win[0][e] * wv[0][e]
        for w in range(1, W):
            s = s + win[w][e] * wv[w][e]
        return s

    def _fwd_body(gx, gy, gw, n_lanes, vpr, T, nseg):
        dv, seg, b = _lane(n_lanes, vpr, nseg)
        t0 = seg * fx.Int32(_TS)
        wv = _weights(gw, vpr, dv)
        n = _TS + W - 1
        ts = [t0 + fx.Int32(i - (W - 1)) for i in range(n)]
        # Every load is issued before the first store: the compiler cannot move a load of X above a store to Y
        # (it cannot prove they do not alias), so interleaving them serialises each lane.
        raws = [_raw_x(gx, b, t, T, vpr, dv) for t in ts]
        win = []
        for i in range(n):
            t = ts[i]
            win.append(_cvt(raws[i]))
            if len(win) > W:
                win.pop(0)
            if i >= W - 1:
                outs = []
                for e in range(_VEC):
                    s = _preact(win, wv, e)
                    outs.append((s * _sigmoid(s)).to(fx.BFloat16))
                _stv(outs, gy, (b * T + t) * vpr + dv, fx.BFloat16)

    def _bwd_body(gx, gdy, gdx, gw, gdw, n_lanes, vpr, T, nseg):
        dv, seg, b = _lane(n_lanes, vpr, nseg)
        t0 = seg * fx.Int32(_TS)
        wv = _weights(gw, vpr, dv)
        dw = [[fx.Float32(0.0) for _ in range(_VEC)] for _ in range(W)]
        win = []  # x(t - W + 1) .. x(t)
        dwin = []  # ds(t - W + 1) .. ds(t), only for t >= t0
        # t runs over [t0 - (W - 1), t0 + _TS + W - 2]: the pre-activation is needed for [t0, t0 + _TS + W - 1),
        # ds for the same range (dx of the last W - 1 rows looks ahead into the next segment).
        n = _TS + 2 * (W - 1)
        ts = [t0 + fx.Int32(i - (W - 1)) for i in range(n)]
        raws = [_raw_x(gx, b, t, T, vpr, dv) for t in ts]
        rawd = [None] * (W - 1) + [_raw_dy(gdy, b, ts[i], T, vpr, dv) for i in range(W - 1, n)]
        for i in range(n):
            t = ts[i]
            win.append(_cvt(raws[i]))
            if len(win) > W:
                win.pop(0)
            if i < W - 1:
                continue
            dyv = _cvt(rawd[i])
            ds = []
            for e in range(_VEC):
                s = _preact(win, wv, e)
                sg = _sigmoid(s)
                ds.append(dyv[e] * sg * (fx.Float32(1.0) + s * (fx.Float32(1.0) - sg)))
            dwin.append(ds)
            if len(dwin) > W:
                dwin.pop(0)
            if i < _TS + W - 1:  # t in [t0, t0 + _TS): this lane's own rows of the weight gradient
                for w in range(W):
                    for e in range(_VEC):
                        dw[w][e] = dw[w][e] + ds[e] * win[w][e]
            if i >= 2 * (W - 1):  # dx(t - (W - 1)) = sum_w ds(t - w) * weight[W - 1 - w]
                outs = []
                for e in range(_VEC):
                    acc = dwin[W - 1][e] * wv[0][e]
                    for w in range(1, W):
                        acc = acc + dwin[W - 1 - w][e] * wv[w][e]
                    outs.append(acc.to(fx.BFloat16))
                _stv(outs, gdx, (b * T + t - fx.Int32(W - 1)) * vpr + dv, fx.BFloat16)
        for w in range(W):
            _st8f(dw[w], gdw, ((b * nseg + seg) * fx.Int32(W) + fx.Int32(w)) * vpr + dv)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_conv_fwd(
        X: fx.Tensor,
        WT: fx.Tensor,
        Y: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        T: fx.Int32,
        nseg: fx.Int32,
        x_bytes: fx.Int64,
        w_bytes: fx.Int64,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gy = _bv(Y, x_bytes, fx.BFloat16, _VEC)
        gw = _bv(WT, w_bytes, fx.Float32, 4)
        _fwd_body(gx, gy, gw, n_lanes, vpr, T, nseg)

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_conv_bwd(
        X: fx.Tensor,
        WT: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        DW: fx.Tensor,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        T: fx.Int32,
        nseg: fx.Int32,
        x_bytes: fx.Int64,
        w_bytes: fx.Int64,
        dw_bytes: fx.Int64,
    ):
        gx = _bv(X, x_bytes, fx.BFloat16, _VEC)
        gdy = _bv(DY, x_bytes, fx.BFloat16, _VEC)
        gdx = _bv(DX, x_bytes, fx.BFloat16, _VEC)
        gw = _bv(WT, w_bytes, fx.Float32, 4)
        gdw = _bv(DW, dw_bytes, fx.Float32, 4)
        _bwd_body(gx, gdy, gdx, gw, gdw, n_lanes, vpr, T, nseg)

    @flyc.jit
    def launch_fwd(
        X,
        WT,
        Y,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        T: fx.Int32,
        nseg: fx.Int32,
        x_bytes: fx.Int64,
        w_bytes: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_conv_fwd(X, WT, Y, n_lanes, vpr, T, nseg, x_bytes, w_bytes).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    @flyc.jit
    def launch_bwd(
        X,
        WT,
        DY,
        DX,
        DW,
        n_lanes: fx.Int32,
        vpr: fx.Int32,
        T: fx.Int32,
        nseg: fx.Int32,
        x_bytes: fx.Int64,
        w_bytes: fx.Int64,
        dw_bytes: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_conv_bwd(X, WT, DY, DX, DW, n_lanes, vpr, T, nseg, x_bytes, w_bytes, dw_bytes).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    return flyc, launch_fwd, launch_bwd


def _kernels(W: int):
    with _LOCK:
        if W not in _BUILT:
            _BUILT[W] = _build(W)
        return _BUILT[W]


def _run(W, name, exe, *args):
    """Compile once per (width, launcher); every shape argument is runtime."""
    flyc = _kernels(W)[0]
    key = (W, name, args[0].device)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
        return
    cf = flyc.compile(exe, *args)
    if cf is not None:
        _COMPILED[key] = cf


def supports_conv_inputs(x: Tensor, weight: Tensor) -> Optional[str]:
    """``None`` when the kernel can run ``x`` ``[B, T, D]`` with ``weight`` ``[D, W]``, else why not."""
    if not x.is_cuda:
        return "x is on CPU"
    if x.dim() != 3 or weight.dim() != 2 or weight.shape[0] != x.shape[-1]:
        return f"needs x [B, T, D] and weight [D, W], got {tuple(x.shape)} / {tuple(weight.shape)}"
    if x.dtype != torch.bfloat16:
        return f"dtype {x.dtype} is not bf16"
    B, T, D = x.shape
    if D % _VEC or T % _TS:
        return f"needs D % {_VEC} == 0 and T % {_TS} == 0, got D={D} T={T}"
    if not 2 <= weight.shape[1] <= 4:
        return f"kernel width {weight.shape[1]} not in [2, 4]"
    if x.numel() * 2 > _MAX_VIEW_BYTES:
        return "x is larger than one 2 GiB buffer view"
    arch = str(getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        return f"device arch is {arch!r}, not gfx1250"
    return None


def _geometry(x: Tensor):
    B, T, D = x.shape
    vpr = D // _VEC
    nseg = T // _TS
    n_lanes = B * nseg * vpr
    return B, T, D, vpr, nseg, n_lanes, (n_lanes + _BLOCK - 1) // _BLOCK


class _CausalConvSiluFlydsl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor):
        x = x.contiguous()
        W = weight.shape[1]
        wt = weight.detach().float().t().contiguous()  # [W, D]
        y = torch.empty_like(x)
        B, T, D, vpr, nseg, n_lanes, nblk = _geometry(x)
        _, launch_fwd, _ = _kernels(W)
        _run(
            W,
            "fwd",
            launch_fwd,
            x,
            wt,
            y,
            n_lanes,
            vpr,
            T,
            nseg,
            x.numel() * 2,
            wt.numel() * 4,
            nblk,
            torch.cuda.current_stream(),
        )
        ctx.save_for_backward(x, weight)
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, weight = ctx.saved_tensors
        dy = dy.contiguous()
        W = weight.shape[1]
        wt = weight.detach().float().t().contiguous()
        B, T, D, vpr, nseg, n_lanes, nblk = _geometry(x)
        dx = torch.empty_like(x)
        dwp = torch.empty(B * nseg * W, D, dtype=torch.float32, device=x.device)
        _, _, launch_bwd = _kernels(W)
        _run(
            W,
            "bwd",
            launch_bwd,
            x,
            wt,
            dy,
            dx,
            dwp,
            n_lanes,
            vpr,
            T,
            nseg,
            x.numel() * 2,
            wt.numel() * 4,
            dwp.numel() * 4,
            nblk,
            torch.cuda.current_stream(),
        )
        dw = dwp.view(B * nseg, W, D).sum(0).t().to(weight.dtype)
        return dx, dw


def flydsl_causal_conv1d_silu(x: Tensor, weight: Tensor) -> Tensor:
    """``silu(causal_depthwise_conv1d(x, weight))`` on ``x`` ``[B, T, D]`` bf16, ``weight`` ``[D, W]``."""
    reason = supports_conv_inputs(x, weight)
    if reason is not None:
        raise ValueError(f"the FlyDSL causal conv1d kernel cannot run these inputs: {reason}")
    return _CausalConvSiluFlydsl.apply(x, weight)
