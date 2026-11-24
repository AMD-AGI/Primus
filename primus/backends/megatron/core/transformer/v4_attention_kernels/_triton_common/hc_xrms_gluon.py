###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Gluon form of ``hc_boundary._hc_expand_rms_kernel`` for gfx1250, byte-identical to it.

Computes, for one HC boundary (x [M, 4, D] bf16), the expanded streams
``new_h = post_h * out + (c_h0 x_0 + c_h1 x_1) + (c_h2 x_2 + c_h3 x_3)`` (bf16) together with the next HC's
rms-linear split-K partials ``P_RAW[s] = new_h[:, chunk] @ W^T[chunk]`` and ``P_SS[s] = sum(new_h[:, chunk]^2)``.

Same bits as the Triton kernel:
  * new_h, its bf16 store and the per-row sum of squares are computed in the Triton kernel's blocked layout
    (8 contiguous columns per lane, 16 lanes per row, warps along rows): same expression, same reduce tree;
  * P_RAW chains the same v_wmma_f32_16x16x32_bf16 over K in the same order (transposed WMMA layout as Triton).

Tiling: a program owns 64 rows x 1024 columns (one split-K chunk) x 4 streams with 16 warps. The 4 x planes, out
and the 4 W tiles of each 128-column step arrive by TDM (each tile issued by a single wave) into a 2-stage LDS
ring; the dot is one batched [4 streams, 64, 32] WMMA whose warps own distinct (stream, 16-row) tiles, so no
warp re-reads another warp's operands.
"""

import functools

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.experimental.gluon.language.amd.cdna5 import tdm

_BT, _BK, _CHUNK, _BN, _NW, _NS = 64, 128, 1024, 32, 16, 2


@gluon.jit
def _nb(x0, x1, x2, x3, o, c0, c1, c2, c3, p):
    return (p * o + ((c0 * x0 + c1 * x1) + (c2 * x2 + c3 * x3))).to(gl.bfloat16)


@gluon.jit
def _ldrow(PTR, r64, j: gl.constexpr, stride: gl.constexpr):
    return gl.expand_dims(gl.load(PTR + r64 * stride + j).to(gl.float32), 1)


@gluon.jit
def _xrms_gluon_kernel(
    X,
    OUT,
    POST,
    COMBT,
    NEW,
    W,
    P_RAW,
    P_SS,
    M,
    D,
    N,
    BT: gl.constexpr,
    BK: gl.constexpr,
    CHUNK: gl.constexpr,
    BN: gl.constexpr,
    NS: gl.constexpr,
    L8: gl.constexpr,
    WM3: gl.constexpr,
    OPA3: gl.constexpr,
    OPB3: gl.constexpr,
    SX: gl.constexpr,
    SNL: gl.constexpr,
    SWL3: gl.constexpr,
    L83: gl.constexpr,
):
    pid_t = gl.program_id(0)
    pid_c = gl.program_id(1)
    KD = 4 * D
    r0 = pid_t * BT
    c0w = pid_c * CHUNK
    rows = r0 + gl.arange(0, BT, layout=gl.SliceLayout(1, L8))
    r64 = rows.to(gl.int64)
    kc = gl.arange(0, BK, layout=gl.SliceLayout(0, L8))
    # comb^T[h, k] of each row (c<h><k>) and post_h, as [BT, 1] fp32 in the row slice of L8
    cb = COMBT
    c00 = _ldrow(cb, r64, 0, 16)
    c01 = _ldrow(cb, r64, 1, 16)
    c02 = _ldrow(cb, r64, 2, 16)
    c03 = _ldrow(cb, r64, 3, 16)
    c10 = _ldrow(cb, r64, 4, 16)
    c11 = _ldrow(cb, r64, 5, 16)
    c12 = _ldrow(cb, r64, 6, 16)
    c13 = _ldrow(cb, r64, 7, 16)
    c20 = _ldrow(cb, r64, 8, 16)
    c21 = _ldrow(cb, r64, 9, 16)
    c22 = _ldrow(cb, r64, 10, 16)
    c23 = _ldrow(cb, r64, 11, 16)
    c30 = _ldrow(cb, r64, 12, 16)
    c31 = _ldrow(cb, r64, 13, 16)
    c32 = _ldrow(cb, r64, 14, 16)
    c33 = _ldrow(cb, r64, 15, 16)
    p0 = _ldrow(POST, r64, 0, 4)
    p1 = _ldrow(POST, r64, 1, 4)
    p2 = _ldrow(POST, r64, 2, 4)
    p3 = _ldrow(POST, r64, 3, 4)
    xd = tdm.make_tensor_descriptor(base=X, shape=(M, KD), strides=(KD, 1), block_shape=(BT, BK), layout=SX)
    od = tdm.make_tensor_descriptor(base=OUT, shape=(M, D), strides=(D, 1), block_shape=(BT, BK), layout=SX)
    # W [N, 4 D] viewed as [4 streams, N, D]; rows N..BN-1 are zero-filled by TDM (out of bounds)
    wd = tdm.make_tensor_descriptor(
        base=W, shape=(4, N, D), strides=(D, KD, 1), block_shape=(4, BN, BK), layout=SWL3
    )
    xs = gl.allocate_shared_memory(gl.bfloat16, [NS * 5, BT, BK], SX)  # per stage: x_0..x_3, out
    ws = gl.allocate_shared_memory(gl.bfloat16, [NS, 4, BN, BK], SWL3)
    sn = gl.allocate_shared_memory(gl.bfloat16, [4, BT, BK], SNL)  # new_h of the step: the WMMA A operand
    NSTEP: gl.constexpr = CHUNK // BK
    for s in gl.static_range(NS - 1):
        c = c0w + s * BK
        tdm.async_load(xd, [r0, c], xs.index(s * 5 + 0), warp_used_hint=1)
        tdm.async_load(xd, [r0, D + c], xs.index(s * 5 + 1), warp_used_hint=2)
        tdm.async_load(xd, [r0, 2 * D + c], xs.index(s * 5 + 2), warp_used_hint=4)
        tdm.async_load(xd, [r0, 3 * D + c], xs.index(s * 5 + 3), warp_used_hint=8)
        tdm.async_load(od, [r0, c], xs.index(s * 5 + 4), warp_used_hint=16)
        tdm.async_load(wd, [0, 0, c], ws.index(s), warp_used_hint=32)
    acc = gl.zeros([4, BT, BN], gl.float32, layout=WM3)
    s0 = gl.zeros([BT], gl.float32, layout=gl.SliceLayout(1, L8))
    s1 = gl.zeros([BT], gl.float32, layout=gl.SliceLayout(1, L8))
    s2 = gl.zeros([BT], gl.float32, layout=gl.SliceLayout(1, L8))
    s3 = gl.zeros([BT], gl.float32, layout=gl.SliceLayout(1, L8))
    gl.static_assert(NSTEP % NS == 0)
    for st0 in range(0, NSTEP, NS):
        for slot in gl.static_range(NS):
            st = st0 + slot
            tdm.async_wait(NS - 2)
            gl.barrier()  # stage st landed for every wave; every read of the slot refilled below is done
            nst = st + NS - 1
            if nst < NSTEP:
                ns_ = nst % NS
                c = c0w + nst * BK
                tdm.async_load(xd, [r0, c], xs.index(ns_ * 5 + 0), warp_used_hint=1)
                tdm.async_load(xd, [r0, D + c], xs.index(ns_ * 5 + 1), warp_used_hint=2)
                tdm.async_load(xd, [r0, 2 * D + c], xs.index(ns_ * 5 + 2), warp_used_hint=4)
                tdm.async_load(xd, [r0, 3 * D + c], xs.index(ns_ * 5 + 3), warp_used_hint=8)
                tdm.async_load(od, [r0, c], xs.index(ns_ * 5 + 4), warp_used_hint=16)
                tdm.async_load(wd, [0, 0, c], ws.index(ns_), warp_used_hint=32)
            off = gl.expand_dims(r64 * KD, 1) + gl.expand_dims(c0w + st * BK + kc, 0)
            x0 = xs.index(slot * 5 + 0).load(L8).to(gl.float32)
            x1 = xs.index(slot * 5 + 1).load(L8).to(gl.float32)
            x2 = xs.index(slot * 5 + 2).load(L8).to(gl.float32)
            x3 = xs.index(slot * 5 + 3).load(L8).to(gl.float32)
            o = xs.index(slot * 5 + 4).load(L8).to(gl.float32)
            n0 = _nb(x0, x1, x2, x3, o, c00, c01, c02, c03, p0)
            n1 = _nb(x0, x1, x2, x3, o, c10, c11, c12, c13, p1)
            n2 = _nb(x0, x1, x2, x3, o, c20, c21, c22, c23, p2)
            n3 = _nb(x0, x1, x2, x3, o, c30, c31, c32, c33, p3)
            gl.store(NEW + off, n0)
            gl.store(NEW + off + D, n1)
            gl.store(NEW + off + 2 * D, n2)
            gl.store(NEW + off + 3 * D, n3)
            f = n0.to(gl.float32)
            s0 = s0 + gl.sum(f * f, axis=1)
            f = n1.to(gl.float32)
            s1 = s1 + gl.sum(f * f, axis=1)
            f = n2.to(gl.float32)
            s2 = s2 + gl.sum(f * f, axis=1)
            f = n3.to(gl.float32)
            s3 = s3 + gl.sum(f * f, axis=1)
            sn.slice(0, 1).store(gl.expand_dims(gl.convert_layout(n0, gl.SliceLayout(0, L83)), 0))
            sn.slice(1, 1).store(gl.expand_dims(gl.convert_layout(n1, gl.SliceLayout(0, L83)), 0))
            sn.slice(2, 1).store(gl.expand_dims(gl.convert_layout(n2, gl.SliceLayout(0, L83)), 0))
            sn.slice(3, 1).store(gl.expand_dims(gl.convert_layout(n3, gl.SliceLayout(0, L83)), 0))
            gl.barrier()
            a3 = sn.load(OPA3)
            b3 = ws.index(slot).permute([0, 2, 1]).load(OPB3)
            acc = gl.amd.cdna5.wmma(a3, b3, acc)
    # split index s = h * (D / CHUNK) + pid_c, as in the Triton kernel
    NCH = D // CHUNK
    LH: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, WM3))
    LR: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, WM3))
    LN: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, WM3))
    hh = gl.arange(0, 4, layout=LH)
    rr = r0 + gl.arange(0, BT, layout=LR)
    nn = gl.arange(0, BN, layout=LN)
    orow = gl.expand_dims(gl.expand_dims(hh * NCH + pid_c, 1), 2).to(gl.int64) * M + gl.expand_dims(
        gl.expand_dims(rr, 0), 2
    )
    gl.store(P_RAW + orow * BN + gl.expand_dims(gl.expand_dims(nn, 0), 1), acc)
    gl.store(P_SS + (0 * NCH + pid_c) * M + r64, s0)
    gl.store(P_SS + (1 * NCH + pid_c) * M + r64, s1)
    gl.store(P_SS + (2 * NCH + pid_c) * M + r64, s2)
    gl.store(P_SS + (3 * NCH + pid_c) * M + r64, s3)


@functools.lru_cache(maxsize=None)
def _layouts():
    wm3 = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]],
        reg_bases=[[0, 0, 1]],
        instr_shape=[16, 16, 32],
        rank=3,
    )
    return dict(
        L8=gl.BlockedLayout([1, 8], [2, 16], [_NW, 1], [1, 0]),
        WM3=wm3,
        OPA3=gl.DotOperandLayout(operand_index=0, parent=wm3, k_width=8),
        OPB3=gl.DotOperandLayout(operand_index=1, parent=wm3, k_width=8),
        SX=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]),
        SNL=gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, _BT, _BK], [2, 1, 0]),
        SWL3=gl.PaddedSharedLayout.with_identity_for([[128, 8]], [4, _BN, _BK], [2, 1, 0]),
        L83=gl.BlockedLayout([1, 1, 8], [1, 2, 16], [1, _NW, 1], [2, 1, 0]),
    )


def xrms_gluon_supported(x3, out_c, post_c, combt_c, next_w) -> bool:
    """Shapes / dtypes / layouts the Gluon kernel covers (the caller checks the Gluon gate): K = 4, M % 64 == 0,
    D % 1024 == 0, 16 < N <= 32 (the Triton kernel's BN is 32 then), contiguous bf16 x / out / W, bf16 or fp32
    post / comb^T, 16-byte aligned TDM bases."""
    M, K, D = x3.shape
    return (
        K == 4
        and M % _BT == 0
        and D % _CHUNK == 0
        and x3.is_contiguous()
        and out_c.is_contiguous()
        and out_c.numel() == M * D
        and out_c.shape[-1] == D
        and post_c.is_contiguous()
        and post_c.numel() == M * 4
        and combt_c.is_contiguous()
        and combt_c.numel() == M * 16
        and next_w.is_contiguous()
        and next_w.dim() == 2
        and 16 < next_w.shape[0] <= _BN
        and next_w.shape[1] == 4 * D
        and all(t.dtype == torch.bfloat16 for t in (x3, out_c, next_w))
        and post_c.dtype in (torch.bfloat16, torch.float32)
        and combt_c.dtype in (torch.bfloat16, torch.float32)
        and all(t.data_ptr() % 16 == 0 and t.data_ptr() != 0 for t in (x3, out_c, next_w))
    )


def expand_rms_gluon(x3, out_c, post_c, combt_c, next_w):
    """-> new [M, 4, D], p_raw [4 D / 1024, M, 32] fp32, p_ss [4 D / 1024, M] fp32 (see ``hc_boundary.expand_rms``)."""
    M, K, D = x3.shape
    N = next_w.shape[0]
    n_split = K * D // _CHUNK
    new = torch.empty_like(x3)
    p_raw = torch.empty((n_split, M, _BN), dtype=torch.float32, device=x3.device)
    p_ss = torch.empty((n_split, M), dtype=torch.float32, device=x3.device)
    _xrms_gluon_kernel[(M // _BT, D // _CHUNK)](
        x3,
        out_c,
        post_c,
        combt_c,
        new,
        next_w,
        p_raw,
        p_ss,
        M,
        D,
        N,
        BT=_BT,
        BK=_BK,
        CHUNK=_CHUNK,
        BN=_BN,
        NS=_NS,
        num_warps=_NW,
        **_layouts(),
    )
    return new, p_raw, p_ss
