###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""The ops around the ``flydsl`` KDA chunk on gfx1250 (FlyDSL), forward and backward, on ``[B, T, H, 128]`` rows:

``l2norm``          ``y = x * rsqrt(sum x^2 + eps)`` (``kda_l2norm``), bf16 in and out
``kda_gate``        ``g = lower_bound * sigmoid(exp(A_log[h]) * (z + dt_bias))`` (``kda_gate``), bf16 ``z``, fp32 ``g``
``rms_norm_gated``  ``y = x * rsqrt(mean x^2 + eps) * weight * sigmoid(gate)`` (``KimiGatedRMSNorm``), bf16

Sixteen lanes own a row (8 contiguous channels each: 16-byte bf16 accesses) and reduce it with ``shuffle_xor``;
each lane walks 16 rows of one head (consecutive ``t``), so the parameter gradients (``dt_bias``, ``A_log``,
``weight``) are summed per lane in registers and the per-lane partials once on the host. The backward kernels
recompute the forward from its inputs (nothing but the inputs is kept). All math is fp32.
"""

from __future__ import annotations

import math
import threading
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = ["l2norm", "kda_gate", "rms_norm_gated", "ops_unsupported_reason"]

_D = 128
_VEC = 8
_LPR = _D // _VEC  # lanes per row
_R = 16  # rows (time steps) per lane
_PF = 4  # rows whose loads are issued together
_BLOCK = 256
_LOG2E = 1.0 / math.log(2.0)
_MAX_VIEW_BYTES = (1 << 31) - 4096

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_KERNELS = None


def _build():
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import gpu

    BF, F32, I32 = fx.BFloat16, fx.Float32, fx.Int32

    def _bv(t, nbytes, dt, vec):
        nt = (1 << 31) // (vec * (dt.width // 8))
        b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=nbytes)
        return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, vec), (vec, 1))))

    def _atom(dt, vec):
        return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * dt.width), dt)

    def _ldv(buf, idx, dt, vec):
        f = fx.make_rmem_tensor(fx.make_layout(vec, 1), dt)
        fx.copy_atom_call(_atom(dt, vec), fx.slice(buf, (idx, None)), f)
        return fx.Vector(f.load().ir_value())

    def _stv(vals, buf, idx, dt):
        n = len(vals)
        f = fx.make_rmem_tensor(fx.make_layout(n, 1), dt)
        fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=dt), f)
        fx.copy_atom_call(_atom(dt, n), f, fx.slice(buf, (idx, None)))

    def _f(v):
        return F32(v.ir_value() if hasattr(v, "ir_value") else v)

    def _ld8b(buf, idx):
        """8 bf16 (vector index ``idx`` of an 8-wide bf16 view) as fp32."""
        v = _ldv(buf, idx, BF, _VEC)
        return [F32(v[e]) for e in range(_VEC)]

    def _ld8f(buf4, idx8):
        lo = _ldv(buf4, idx8 * I32(2), F32, 4)
        hi = _ldv(buf4, idx8 * I32(2) + I32(1), F32, 4)
        return [F32(lo[e]) for e in range(4)] + [F32(hi[e]) for e in range(4)]

    def _st8b(vals, buf, idx):
        _stv([v.to(BF) for v in vals], buf, idx, BF)

    def _st8f(vals, buf4, idx8):
        _stv(vals[:4], buf4, idx8 * I32(2), F32)
        _stv(vals[4:], buf4, idx8 * I32(2) + I32(1), F32)

    def _rowsum(x):
        for dist in (1, 2, 4, 8):
            x = x + _f(gpu.shuffle_xor(x.ir_value(), dist, 32))
        return x

    def _dot(a, b):
        s = a[0] * b[0]
        for e in range(1, _VEC):
            s = s + a[e] * b[e]
        return s

    def _sigmoid(x):
        e = F32(fx.rocdl.exp2(F32.ir_type, (x * F32(-_LOG2E)).ir_value()))
        return F32(fx.rocdl.rcp(F32.ir_type, (F32(1.0) + e).ir_value()))

    def _rsq(x):
        return F32(fx.rocdl.rsq(F32.ir_type, x.ir_value()))

    def _lane(n_groups, H):
        """``(group, dv, rows)``: the lane's row group, its 8-channel slot, and the 8-vector index of its R rows."""
        lane = I32(fx.block_idx.x) * I32(_BLOCK) + I32(fx.thread_idx.x)
        grp = lane // I32(_LPR)
        dv = lane - grp * I32(_LPR)
        # Tail groups redo the last group and store the same bytes; whole groups, so the row sums stay exact.
        gc = (grp < n_groups).select(grp, n_groups - I32(1))
        bseg = gc // H
        h = gc - bseg * H
        rows = [((bseg * I32(_R) + I32(i)) * H + h) * I32(_LPR) + dv for i in range(_R)]
        return gc, dv, h, rows

    def _groups(rows, n):
        return [rows[i : i + n] for i in range(0, len(rows), n)]

    # ---------------------------------------------------------------- l2norm
    def _l2_fwd(gx, gy, n_groups, H, eps):
        _, _, _, rows = _lane(n_groups, H)
        for grp in _groups(rows, _PF):
            xs = [_ld8b(gx, r) for r in grp]
            for r, x in zip(grp, xs):
                rs = _rsq(_rowsum(_dot(x, x)) + eps)
                _st8b([v * rs for v in x], gy, r)

    def _l2_bwd(gx, gdy, gdx, n_groups, H, eps):
        _, _, _, rows = _lane(n_groups, H)
        for grp in _groups(rows, _PF):
            xs = [_ld8b(gx, r) for r in grp]
            dys = [_ld8b(gdy, r) for r in grp]
            for r, x, dy in zip(grp, xs, dys):
                rs = _rsq(_rowsum(_dot(x, x)) + eps)
                c = _rowsum(_dot(dy, x)) * rs * rs * rs
                _st8b([dy[e] * rs - x[e] * c for e in range(_VEC)], gdx, r)

    # ---------------------------------------------------------------- gate
    def _gate_fwd(gz, gcoef, gdt, gg, n_groups, H, lb):
        _, dv, h, rows = _lane(n_groups, H)
        ch = h * I32(_LPR) + dv
        coef, dt = _ld8f(gcoef, ch), _ld8f(gdt, ch)
        for grp in _groups(rows, _PF):
            zs = [_ld8b(gz, r) for r in grp]
            for r, z in zip(grp, zs):
                _st8f([lb * _sigmoid(coef[e] * (z[e] + dt[e])) for e in range(_VEC)], gg, r)

    def _gate_bwd(gz, gcoef, gdt, gdg, gdz, gpdt, gpa, n_groups, H, lb):
        gc, dv, h, rows = _lane(n_groups, H)
        ch = h * I32(_LPR) + dv
        coef, dt = _ld8f(gcoef, ch), _ld8f(gdt, ch)
        pdt = [F32(0.0) for _ in range(_VEC)]
        pa = [F32(0.0) for _ in range(_VEC)]
        for grp in _groups(rows, _PF):
            zs = [_ld8b(gz, r) for r in grp]
            dgs = [_ld8f(gdg, r) for r in grp]
            for r, z, dg in zip(grp, zs, dgs):
                dz = []
                for e in range(_VEC):
                    u = z[e] + dt[e]
                    s = _sigmoid(coef[e] * u)
                    d = dg[e] * lb * s * (F32(1.0) - s) * coef[e]
                    dz.append(d)
                    pdt[e] = pdt[e] + d
                    pa[e] = pa[e] + d * u
                _st8b(dz, gdz, r)
        # partials [n_groups, 128]: row gc of each, this lane's 8 channels
        _st8f(pdt, gpdt, gc * I32(_LPR) + dv)
        _st8f(pa, gpa, gc * I32(_LPR) + dv)

    # ---------------------------------------------------------------- gated RMSNorm
    def _norm_fwd(gx, ggate, gw, gy, n_groups, H, eps):
        _, dv, _, rows = _lane(n_groups, H)
        w = _ld8f(gw, dv)
        for grp in _groups(rows, _PF):
            xs = [_ld8b(gx, r) for r in grp]
            gs = [_ld8b(ggate, r) for r in grp]
            for r, x, gt in zip(grp, xs, gs):
                rs = _rsq(_rowsum(_dot(x, x)) * F32(1.0 / _D) + eps)
                _st8b([x[e] * rs * w[e] * _sigmoid(gt[e]) for e in range(_VEC)], gy, r)

    def _norm_bwd(gx, ggate, gw, gdy, gdx, gdgate, gpw, n_groups, H, eps):
        gc, dv, _, rows = _lane(n_groups, H)
        w = _ld8f(gw, dv)
        pw = [F32(0.0) for _ in range(_VEC)]
        for grp in _groups(rows, _PF):
            xs = [_ld8b(gx, r) for r in grp]
            gs = [_ld8b(ggate, r) for r in grp]
            dys = [_ld8b(gdy, r) for r in grp]
            for r, x, gt, dy in zip(grp, xs, gs, dys):
                rs = _rsq(_rowsum(_dot(x, x)) * F32(1.0 / _D) + eps)
                xh = [x[e] * rs for e in range(_VEC)]
                sg = [_sigmoid(gt[e]) for e in range(_VEC)]
                dgate, dxh = [], []
                for e in range(_VEC):
                    t = dy[e] * xh[e] * sg[e]
                    pw[e] = pw[e] + t
                    dgate.append(t * w[e] * (F32(1.0) - sg[e]))
                    dxh.append(dy[e] * w[e] * sg[e])
                m = _rowsum(_dot(dxh, xh)) * F32(1.0 / _D)
                _st8b([rs * (dxh[e] - xh[e] * m) for e in range(_VEC)], gdx, r)
                _st8b(dgate, gdgate, r)
        _st8f(pw, gpw, gc * I32(_LPR) + dv)

    blk = dict(known_block_size=[_BLOCK, 1, 1])

    @flyc.kernel(**blk)
    def k_l2_fwd(X: fx.Tensor, Y: fx.Tensor, n_groups: fx.Int32, H: fx.Int32, eps: fx.Float32, nb: fx.Int64):
        _l2_fwd(_bv(X, nb, BF, _VEC), _bv(Y, nb, BF, _VEC), n_groups, H, eps)

    @flyc.kernel(**blk)
    def k_l2_bwd(
        X: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
    ):
        _l2_bwd(_bv(X, nb, BF, _VEC), _bv(DY, nb, BF, _VEC), _bv(DX, nb, BF, _VEC), n_groups, H, eps)

    @flyc.kernel(**blk)
    def k_gate_fwd(
        Z: fx.Tensor,
        COEF: fx.Tensor,
        DT: fx.Tensor,
        G: fx.Tensor,
        n_groups: fx.Int32,
        H: fx.Int32,
        lb: fx.Float32,
        nb: fx.Int64,
        nbp: fx.Int64,
    ):
        _gate_fwd(
            _bv(Z, nb, BF, _VEC),
            _bv(COEF, nbp, F32, 4),
            _bv(DT, nbp, F32, 4),
            _bv(G, nb * 2, F32, 4),
            n_groups,
            H,
            lb,
        )

    @flyc.kernel(**blk)
    def k_gate_bwd(
        Z: fx.Tensor,
        COEF: fx.Tensor,
        DT: fx.Tensor,
        DG: fx.Tensor,
        DZ: fx.Tensor,
        PDT: fx.Tensor,
        PA: fx.Tensor,
        n_groups: fx.Int32,
        H: fx.Int32,
        lb: fx.Float32,
        nb: fx.Int64,
        nbp: fx.Int64,
        nbq: fx.Int64,
    ):
        _gate_bwd(
            _bv(Z, nb, BF, _VEC),
            _bv(COEF, nbp, F32, 4),
            _bv(DT, nbp, F32, 4),
            _bv(DG, nb * 2, F32, 4),
            _bv(DZ, nb, BF, _VEC),
            _bv(PDT, nbq, F32, 4),
            _bv(PA, nbq, F32, 4),
            n_groups,
            H,
            lb,
        )

    @flyc.kernel(**blk)
    def k_norm_fwd(
        X: fx.Tensor,
        GT: fx.Tensor,
        W: fx.Tensor,
        Y: fx.Tensor,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
    ):
        _norm_fwd(
            _bv(X, nb, BF, _VEC),
            _bv(GT, nb, BF, _VEC),
            _bv(W, _D * 4, F32, 4),
            _bv(Y, nb, BF, _VEC),
            n_groups,
            H,
            eps,
        )

    @flyc.kernel(**blk)
    def k_norm_bwd(
        X: fx.Tensor,
        GT: fx.Tensor,
        W: fx.Tensor,
        DY: fx.Tensor,
        DX: fx.Tensor,
        DGT: fx.Tensor,
        PW: fx.Tensor,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
        nbq: fx.Int64,
    ):
        _norm_bwd(
            _bv(X, nb, BF, _VEC),
            _bv(GT, nb, BF, _VEC),
            _bv(W, _D * 4, F32, 4),
            _bv(DY, nb, BF, _VEC),
            _bv(DX, nb, BF, _VEC),
            _bv(DGT, nb, BF, _VEC),
            _bv(PW, nbq, F32, 4),
            n_groups,
            H,
            eps,
        )

    grid = dict(block=(_BLOCK, 1, 1))

    @flyc.jit
    def l2_fwd(
        X,
        Y,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_l2_fwd(X, Y, n_groups, H, eps, nb).launch(grid=(nblk, 1, 1), stream=stream, **grid)

    @flyc.jit
    def l2_bwd(
        X,
        DY,
        DX,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_l2_bwd(X, DY, DX, n_groups, H, eps, nb).launch(grid=(nblk, 1, 1), stream=stream, **grid)

    @flyc.jit
    def gate_fwd(
        Z,
        COEF,
        DT,
        G,
        n_groups: fx.Int32,
        H: fx.Int32,
        lb: fx.Float32,
        nb: fx.Int64,
        nbp: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_gate_fwd(Z, COEF, DT, G, n_groups, H, lb, nb, nbp).launch(grid=(nblk, 1, 1), stream=stream, **grid)

    @flyc.jit
    def gate_bwd(
        Z,
        COEF,
        DT,
        DG,
        DZ,
        PDT,
        PA,
        n_groups: fx.Int32,
        H: fx.Int32,
        lb: fx.Float32,
        nb: fx.Int64,
        nbp: fx.Int64,
        nbq: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_gate_bwd(Z, COEF, DT, DG, DZ, PDT, PA, n_groups, H, lb, nb, nbp, nbq).launch(
            grid=(nblk, 1, 1), stream=stream, **grid
        )

    @flyc.jit
    def norm_fwd(
        X,
        GT,
        W,
        Y,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_norm_fwd(X, GT, W, Y, n_groups, H, eps, nb).launch(grid=(nblk, 1, 1), stream=stream, **grid)

    @flyc.jit
    def norm_bwd(
        X,
        GT,
        W,
        DY,
        DX,
        DGT,
        PW,
        n_groups: fx.Int32,
        H: fx.Int32,
        eps: fx.Float32,
        nb: fx.Int64,
        nbq: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_norm_bwd(X, GT, W, DY, DX, DGT, PW, n_groups, H, eps, nb, nbq).launch(
            grid=(nblk, 1, 1), stream=stream, **grid
        )

    return flyc, {
        "l2_fwd": l2_fwd,
        "l2_bwd": l2_bwd,
        "gate_fwd": gate_fwd,
        "gate_bwd": gate_bwd,
        "norm_fwd": norm_fwd,
        "norm_bwd": norm_bwd,
    }


def _kernels():
    global _KERNELS
    with _LOCK:
        if _KERNELS is None:
            _KERNELS = _build()
        return _KERNELS


def _run(name: str, *args):
    flyc, launchers = _kernels()
    key = (name, args[0].device.index)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
        return
    cf = flyc.compile(launchers[name], *args)
    if cf is not None:
        _COMPILED[key] = cf


def ops_unsupported_reason(x: Tensor) -> Optional[str]:
    """``None`` when these kernels take ``[B, T, H, 128]`` rows shaped like ``x``, else why not."""
    if x.dim() != 4 or x.shape[-1] != _D or x.shape[1] % _R:
        return f"needs [B, T, H, {_D}] with T % {_R} == 0; got {tuple(x.shape)}"
    if x.numel() * 4 > _MAX_VIEW_BYTES:
        return "larger than one 2 GiB buffer view (fp32)"
    if not x.is_cuda:
        return "CPU tensors"
    arch = str(getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", ""))
    if not arch.startswith("gfx1250"):
        return f"device arch is {arch!r}, not gfx1250"
    return None


def _geometry(x: Tensor):
    B, T, H, _ = x.shape
    n_groups = B * (T // _R) * H
    return H, n_groups, (n_groups * _LPR + _BLOCK - 1) // _BLOCK


def _check(x: Tensor):
    why = ops_unsupported_reason(x)
    if why is not None:
        raise ValueError(
            f"the gfx1250 FlyDSL KDA ops do not take this input: {why}. Select kda_backend: fla."
        )


class _L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, eps: float):
        x = x.contiguous().to(torch.bfloat16)
        H, n_groups, nblk = _geometry(x)
        y = torch.empty_like(x)
        _run("l2_fwd", x, y, n_groups, H, eps, x.numel() * 2, nblk, torch.cuda.current_stream())
        ctx.save_for_backward(x)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        H, n_groups, nblk = _geometry(x)
        dx = torch.empty_like(x)
        _run("l2_bwd", x, dy, dx, n_groups, H, ctx.eps, x.numel() * 2, nblk, torch.cuda.current_stream())
        return dx, None


class _KdaGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, A_log: Tensor, dt_bias: Optional[Tensor], lower_bound: float):
        z = z.contiguous().to(torch.bfloat16)
        H, n_groups, nblk = _geometry(z)
        coef = A_log.detach().float().exp().repeat_interleave(_D)
        dt = torch.zeros_like(coef) if dt_bias is None else dt_bias.detach().float().contiguous()
        g = torch.empty(z.shape, dtype=torch.float32, device=z.device)
        _run(
            "gate_fwd",
            z,
            coef,
            dt,
            g,
            n_groups,
            H,
            lower_bound,
            z.numel() * 2,
            coef.numel() * 4,
            nblk,
            torch.cuda.current_stream(),
        )
        ctx.save_for_backward(z, coef, dt)
        ctx.lower_bound = lower_bound
        ctx.has_dt = dt_bias is not None
        ctx.param_dtypes = (A_log.dtype, None if dt_bias is None else dt_bias.dtype)
        return g

    @staticmethod
    def backward(ctx, dg: Tensor):
        z, coef, dt = ctx.saved_tensors
        dg = dg.contiguous().float()
        H, n_groups, nblk = _geometry(z)
        dz = torch.empty_like(z)
        pdt = torch.empty(n_groups, _D, dtype=torch.float32, device=z.device)
        pa = torch.empty_like(pdt)
        _run(
            "gate_bwd",
            z,
            coef,
            dt,
            dg,
            dz,
            pdt,
            pa,
            n_groups,
            H,
            ctx.lower_bound,
            z.numel() * 2,
            coef.numel() * 4,
            pdt.numel() * 4,
            nblk,
            torch.cuda.current_stream(),
        )
        # partial row gc belongs to head gc % H
        d_dt = pdt.view(-1, H, _D).sum(0).reshape(-1)
        d_alog = pa.view(-1, H, _D).sum((0, 2))
        a_dtype, dt_dtype = ctx.param_dtypes
        return dz, d_alog.to(a_dtype), (d_dt.to(dt_dtype) if ctx.has_dt else None), None


class _RmsNormGated(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, gate: Tensor, weight: Tensor, eps: float):
        x = x.contiguous().to(torch.bfloat16)
        gate = gate.contiguous().to(torch.bfloat16)
        w = weight.detach().float().contiguous()
        H, n_groups, nblk = _geometry(x)
        y = torch.empty_like(x)
        _run("norm_fwd", x, gate, w, y, n_groups, H, eps, x.numel() * 2, nblk, torch.cuda.current_stream())
        ctx.save_for_backward(x, gate, w)
        ctx.eps = eps
        ctx.w_dtype = weight.dtype
        return y

    @staticmethod
    def backward(ctx, dy: Tensor):
        x, gate, w = ctx.saved_tensors
        dy = dy.contiguous().to(torch.bfloat16)
        H, n_groups, nblk = _geometry(x)
        dx, dgate = torch.empty_like(x), torch.empty_like(gate)
        pw = torch.empty(n_groups, _D, dtype=torch.float32, device=x.device)
        _run(
            "norm_bwd",
            x,
            gate,
            w,
            dy,
            dx,
            dgate,
            pw,
            n_groups,
            H,
            ctx.eps,
            x.numel() * 2,
            pw.numel() * 4,
            nblk,
            torch.cuda.current_stream(),
        )
        return dx, dgate, pw.sum(0).to(ctx.w_dtype), None


def l2norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """``kda_l2norm`` for bf16 ``[B, T, H, 128]``."""
    _check(x)
    return _L2Norm.apply(x, float(eps))


def kda_gate(
    z: Tensor, A_log: Tensor, dt_bias: Optional[Tensor] = None, lower_bound: Optional[float] = None
) -> Tensor:
    """``kda_gate`` (bounded form) for ``z`` ``[B, T, H, 128]``; fp32 output."""
    _check(z)
    if lower_bound is None:
        raise ValueError(
            "the gfx1250 FlyDSL KDA gate implements the bounded gate only (kda_gate_lower_bound set)."
        )
    return _KdaGate.apply(z, A_log, dt_bias, float(lower_bound))


def rms_norm_gated(x: Tensor, gate: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``KimiGatedRMSNorm.forward`` for bf16 ``[B, T, H, 128]``."""
    _check(x)
    return _RmsNormGated.apply(x, gate, weight, float(eps))
