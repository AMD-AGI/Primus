###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused per-head RMSNorm + interleaved partial RoPE (+ zero pad) for the V4 query.

The V4 query path used to run three HBM passes over ``[B, S, H, head_dim]`` per layer
in the forward pass -- per-head RMSNorm (``fused_rms_norm``), RoPE
(``RoPEFromPositionsFn``) and the sparse-MLA adapter copy into a zero-padded
``[B*S, H, head_dim + 64]`` buffer (``_rope_pad_q``) -- and two in the backward
(RoPE bwd, RMSNorm bwd). This module does each direction in ONE pass:

* forward: read ``x`` once, write the normalized + rotated query straight into a
  ``[B, S, H, head_dim + pad]`` buffer (pad block zeroed) and return the
  ``[..., :head_dim]`` view of it. The adapter recognises that buffer (see
  ``v4_sparse_mla_adapter._rope_pad_q``) and uses it as-is, so the pad copy is gone;
* backward: read ``dq`` (any row strides, e.g. the adapter's padded dq view) and ``x``
  once, apply the transposed rotation and the RMSNorm adjoint in registers, write ``dx``.

Numerics mirror the unfused kernels op for op: rstd in fp32, normalized value rounded
to the input dtype, rotation on the rounded values with cos/sin cast to the input
dtype, RMSNorm backward on the (rounded) rotation-adjoint gradient in fp32. Only the
fp32 summation order of the two row reductions can differ.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _pair_cs(pos, INVFREQ_PTR, HALF: tl.constexpr, NOPE: tl.constexpr):
    p = tl.arange(0, HALF)
    fi = p - NOPE // 2
    is_rope = fi >= 0
    ang = pos * tl.load(INVFREQ_PTR + fi, mask=is_rope, other=0.0).to(tl.float32)
    return tl.where(is_rope, tl.cos(ang), 1.0), tl.where(is_rope, tl.sin(ang), 0.0)


@triton.jit
def _q_norm_rope_fwd2_kernel(
    X_PTR,
    POS_PTR,
    INVFREQ_PTR,
    OUT_PTR,
    RSTD_PTR,
    H,
    EPS: tl.constexpr,
    D: tl.constexpr,
    NOPE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Pair-split variant: rotation on (even, odd) pairs from ``tl.split`` in fp32, one rounding;
    [4096, 64, 512] + pad 64: 67.0 -> 44.9 us (75% of HBM floor)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    DP: tl.constexpr = D + PAD
    x = tl.load(
        X_PTR + pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + EPS)
    tl.store(RSTD_PTR + pid_n.to(tl.int64) * H + h, rstd, mask=hm)
    c, s = _pair_cs(tl.load(POS_PTR + pid_n).to(tl.float32), INVFREQ_PTR, HALF, NOPE)
    a, b = tl.split(tl.reshape(x * rstd[:, None], (BLOCK_H, HALF, 2)))
    o = tl.reshape(tl.join(a * c[None, :] - b * s[None, :], a * s[None, :] + b * c[None, :]), (BLOCK_H, D))
    orow = OUT_PTR + pid_n.to(tl.int64) * H * DP + h[:, None].to(tl.int64) * DP
    tl.store(orow + d[None, :], o.to(OUT_PTR.dtype.element_ty), mask=hm[:, None])
    if PAD > 0:
        p = tl.arange(0, BLOCK_PAD)
        tl.store(
            orow + D + p[None, :],
            tl.zeros((BLOCK_H, BLOCK_PAD), dtype=OUT_PTR.dtype.element_ty),
            mask=hm[:, None] & (p[None, :] < PAD),
        )


@triton.jit
def _q_norm_rope_bwd2_kernel(
    DQ_PTR,
    X_PTR,
    RSTD_PTR,
    POS_PTR,
    INVFREQ_PTR,
    DX_PTR,
    H,
    SN,
    SH,
    D: tl.constexpr,
    NOPE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Pair-split backward: 69.3 -> 54.9 us (84% of HBM floor)."""
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    hm = h < H
    d = tl.arange(0, D)
    g = tl.load(
        DQ_PTR + pid_n.to(tl.int64) * SN + h[:, None].to(tl.int64) * SH + d[None, :],
        mask=hm[:, None],
        other=0.0,
    ).to(tl.float32)
    c, s = _pair_cs(tl.load(POS_PTR + pid_n).to(tl.float32), INVFREQ_PTR, HALF, NOPE)
    ga, gb = tl.split(tl.reshape(g, (BLOCK_H, HALF, 2)))
    dy = tl.reshape(
        tl.join(ga * c[None, :] + gb * s[None, :], gb * c[None, :] - ga * s[None, :]), (BLOCK_H, D)
    )
    xrow = pid_n.to(tl.int64) * H * D + h[:, None].to(tl.int64) * D + d[None, :]
    x = tl.load(X_PTR + xrow, mask=hm[:, None], other=0.0).to(tl.float32)
    rstd = tl.load(RSTD_PTR + pid_n.to(tl.int64) * H + h, mask=hm, other=0.0)
    coef = rstd * rstd * rstd * tl.sum(dy * x, axis=1) / D
    tl.store(
        DX_PTR + xrow, (rstd[:, None] * dy - x * coef[:, None]).to(DX_PTR.dtype.element_ty), mask=hm[:, None]
    )


class QNormRopePadFn(torch.autograd.Function):
    """``x [B,S,H,D]`` -> ``rope(rmsnorm(x))`` as the ``[..., :D]`` view of a zero-padded buffer."""

    @staticmethod
    def forward(ctx, x, pos_flat, inv_freq, rotary_dim, eps, pad):  # type: ignore[override]
        B, S, H, D = x.shape
        N = B * S
        x3 = x.reshape(N, H, D).contiguous()
        from .rope_interleaved_partial import rope_kernel_vectors

        pos_c, inv_c = rope_kernel_vectors(
            pos_flat.reshape(-1), inv_freq
        )  # PRIMUS_GLUE_ROPE_NOCAST: no casts
        buf = torch.empty((B, S, H, D + pad), dtype=x.dtype, device=x.device)
        rstd = torch.empty((N, H), dtype=torch.float32, device=x.device)
        # Forward tile: any BLOCK_H with num_warps >= 2 keeps the per-head reduction tree (8 elements x 32 lanes x
        # 2 warps over D = 512), hence the bits.
        bh2 = min(8, triton.next_power_of_2(max(H, 1)))
        _q_norm_rope_fwd2_kernel[(N, triton.cdiv(H, bh2))](
            x3,
            pos_c,
            inv_c,
            buf,
            rstd,
            H,
            EPS=float(eps),
            D=D,
            NOPE=D - rotary_dim,
            PAD=pad,
            BLOCK_PAD=max(triton.next_power_of_2(max(pad, 1)), 1),
            BLOCK_H=bh2,
            num_warps=2,
        )
        ctx.save_for_backward(x3, rstd, pos_c, inv_c)
        ctx.meta = (B, S, H, D, rotary_dim, x.dtype)
        if pad:
            buf._v4_rope_pad = pad  # lets the sparse-MLA adapter use the buffer without copying
        return buf[..., :D]

    @staticmethod
    def backward(ctx, dq):  # type: ignore[override]
        x3, rstd, pos_c, inv_c = ctx.saved_tensors
        B, S, H, D, rotary_dim, dtype = ctx.meta
        N = B * S
        try:
            dq3 = dq.view(N, H, D)
        except RuntimeError:
            dq3 = dq.reshape(N, H, D)
        if dq3.stride(2) != 1:
            dq3 = dq3.contiguous()
        dx = torch.empty((B, S, H, D), dtype=dtype, device=dq.device)
        bh2 = min(8, triton.next_power_of_2(max(H, 1)))
        _q_norm_rope_bwd2_kernel[(N, triton.cdiv(H, bh2))](
            dq3,
            x3,
            rstd,
            pos_c,
            inv_c,
            dx,
            H,
            dq3.stride(0),
            dq3.stride(1),
            D=D,
            NOPE=D - rotary_dim,
            BLOCK_H=bh2,
            num_warps=4,
        )
        return dx, None, None, None, None, None


def q_norm_rope_supported(x: torch.Tensor, rotary_dim: int) -> bool:
    D = x.shape[-1]
    return (
        os.environ.get("PRIMUS_V4_Q_NORM_ROPE_FUSED", "1") != "0"
        and x.is_cuda
        and x.dim() == 4
        and x.dtype in (torch.bfloat16, torch.float16)
        and D >= 2
        and (D & (D - 1)) == 0
        and 0 < rotary_dim <= D
        and rotary_dim % 2 == 0
    )


def q_norm_rope_pad(x, pos_flat, inv_freq, *, rotary_dim: int, eps: float, pad: int = 0):
    """Fused per-head RMSNorm + interleaved partial RoPE; output optionally zero-padded by ``pad``."""
    return QNormRopePadFn.apply(x, pos_flat, inv_freq, rotary_dim, eps, pad)


__all__ = ["QNormRopePadFn", "q_norm_rope_pad", "q_norm_rope_supported"]
