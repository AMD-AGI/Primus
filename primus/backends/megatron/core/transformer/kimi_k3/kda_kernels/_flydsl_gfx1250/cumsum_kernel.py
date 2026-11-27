###############################################################################
# Some parts of this code are adapted from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention, 0.5.2): chunk_local_cumsum,
# rewritten as a FlyDSL kernel for gfx1250.
# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li. Licensed under the MIT License.
#
# Modification Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""Chunk-local cumulative sum of the KDA gate for the ``flydsl`` KDA chunk on gfx1250 (FlyDSL).

``y[b, t, h, :] = scale * sum_{s in chunk(t), s <= t} x[b, s, h, :]`` (forward) or ``sum_{s >= t}`` (reverse) over
64-row chunks of a contiguous fp32 ``[B, T, H, 128]`` tensor: what fla's ``chunk_local_cumsum`` computes for the
gate (forward, ``scale = 1 / ln 2``) and its gradient (reverse). Each lane owns 8 contiguous channels of one
(chunk, head) and walks the 64 rows, 16 at a time with all 16 rows' loads issued first; torch's cumsum over the
row axis of this layout runs at 1.7-4.2 TB/s.
"""

from __future__ import annotations

import threading
from typing import Dict, Tuple

import torch
from torch import Tensor

__all__ = ["chunk_local_cumsum"]

_BT = 64
_D = 128
_VEC = 8
_GROUP = 16
_BLOCK = 256

_COMPILED: Dict[Tuple, object] = {}
_LOCK = threading.Lock()
_BUILT: Dict[bool, tuple] = {}


def _build(reverse: bool):
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    vpr = _D // _VEC  # 8-wide vectors per (row, head)

    def _bv(t, nbytes):
        nt = (1 << 31) // 16
        b = fx.rocdl.make_buffer_tensor(t, num_records_bytes=nbytes)
        return fx.Tensor(fx.make_view(fx.get_iter(b), fx.make_layout((nt, 4), (4, 1))))

    def _atom():
        return fx.make_copy_atom(fx.rocdl.BufferCopy(4 * fx.Float32.width), fx.Float32)

    def _ld4(buf, idx4):
        f = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fx.copy_atom_call(_atom(), fx.slice(buf, (idx4, None)), f)
        return fx.Vector(f.load().ir_value())

    def _st4(vals, buf, idx4):
        f = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
        fx.memref_store_vec(fx.Vector.from_elements(vals, dtype=fx.Float32), f)
        fx.copy_atom_call(_atom(), f, fx.slice(buf, (idx4, None)))

    def _body(gx, gy, n_lanes, H, scale):
        lane = fx.Int32(fx.block_idx.x) * fx.Int32(_BLOCK) + fx.Int32(fx.thread_idx.x)
        # Tail lanes redo the last lane and store the same bytes: branch-free.
        lc = (lane < n_lanes).select(lane, n_lanes - fx.Int32(1))
        rest = lc // fx.Int32(vpr)
        dv = lc - rest * fx.Int32(vpr)
        chunk = rest // H
        h = rest - chunk * H
        # 4-float index of (row, h, 8 * dv): row r of the chunk is global row chunk * 64 + r (T % 64 == 0).
        base = ((chunk * fx.Int32(_BT)) * H + h) * fx.Int32(_D // 4) + dv * fx.Int32(2)
        pitch = H * fx.Int32(_D // 4)
        acc = [fx.Float32(0.0) for _ in range(_VEC)]
        rows = list(range(_BT))
        if reverse:
            rows.reverse()
        for g0 in range(0, _BT, _GROUP):
            grp = rows[g0 : g0 + _GROUP]
            raw = [
                (_ld4(gx, base + fx.Int32(r) * pitch), _ld4(gx, base + fx.Int32(r) * pitch + fx.Int32(1)))
                for r in grp
            ]
            for r, (lo, hi) in zip(grp, raw):
                x = [fx.Float32(lo[e]) for e in range(4)] + [fx.Float32(hi[e]) for e in range(4)]
                acc = [acc[e] + x[e] for e in range(_VEC)]
                out = [acc[e] * scale for e in range(_VEC)]
                _st4(out[:4], gy, base + fx.Int32(r) * pitch)
                _st4(out[4:], gy, base + fx.Int32(r) * pitch + fx.Int32(1))

    @flyc.kernel(known_block_size=[_BLOCK, 1, 1])
    def k_chunk_cumsum(
        X: fx.Tensor, Y: fx.Tensor, n_lanes: fx.Int32, H: fx.Int32, scale: fx.Float32, nbytes: fx.Int64
    ):
        _body(_bv(X, nbytes), _bv(Y, nbytes), n_lanes, H, scale)

    @flyc.jit
    def launch(
        X,
        Y,
        n_lanes: fx.Int32,
        H: fx.Int32,
        scale: fx.Float32,
        nbytes: fx.Int64,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_chunk_cumsum(X, Y, n_lanes, H, scale, nbytes).launch(
            grid=(nblk, 1, 1), block=(_BLOCK, 1, 1), stream=stream
        )

    return flyc, launch


def _kernels(reverse: bool):
    with _LOCK:
        if reverse not in _BUILT:
            _BUILT[reverse] = _build(reverse)
        return _BUILT[reverse]


def chunk_local_cumsum(x: Tensor, scale: float = 1.0, reverse: bool = False) -> Tensor:
    """Chunk-local (64-row) cumsum of a contiguous fp32 ``[B, T, H, 128]`` ``x``, times ``scale``, in fp32."""
    B, T, H, D = x.shape
    assert x.dtype == torch.float32 and x.is_contiguous() and D == _D and T % _BT == 0
    y = torch.empty_like(x)
    n_lanes = B * (T // _BT) * H * (_D // _VEC)
    args = (
        x,
        y,
        n_lanes,
        H,
        float(scale),
        x.numel() * 4,
        (n_lanes + _BLOCK - 1) // _BLOCK,
        torch.cuda.current_stream(),
    )
    flyc, launch = _kernels(reverse)
    key = (reverse, x.device.index)
    cf = _COMPILED.get(key)
    if cf is not None:
        cf(*args)
    else:
        cf = flyc.compile(launch, *args)
        if cf is not None:
            _COMPILED[key] = cf
    return y
