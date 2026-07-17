###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused HyperConnection ``collapse`` (small-kernel-fusion 2026-07-03).

Single FWD + single BWD kernel pair for the eager body of
:meth:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.collapse`::

    out[..., d] = sum_k pre[..., k] * x[..., k, d]

The eager body ``(pre.unsqueeze(-1) * x).sum(dim=-2)`` materialises a full
``[..., K, D]`` temporary (the broadcast multiply) and then reduces it — two
kernels + ``K*D`` of extra HBM traffic per call. This kernel streams ``D`` and
contracts the small ``K`` (constexpr) in registers, writing only the ``[..., D]``
result. Symmetric to the already-shipped ``expand`` fusion (``hc_expand.py``).

Gradients::

    dx[..., k, d] = pre[..., k] * g[..., d]
    dpre[..., k]  = sum_d x[..., k, d] * g[..., d]

Gating: routed through :func:`hc_collapse_triton` when
``PRIMUS_HC_COLLAPSE_TRITON != "0"`` (default-on); falls back to eager for
unsupported configs (CPU input, K out of range, dtype mismatch).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_SUPPORTED_K = (1, 2, 4, 8, 16)

_BLOCK_M = 16
_BLOCK_D = 128
_NUM_WARPS = 4


@triton.jit
def _hc_collapse_fwd_kernel(
    X_PTR,  # [M, K, D]
    PRE_PTR,  # [M, K]
    OUT_PTR,  # [M, D] OUT
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    mask2 = mask_m[:, None] & mask_d[None, :]
    mask3 = mask_m[:, None, None] & mask_d[None, None, :]

    KD = K * D
    x3 = tl.load(
        X_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask3,
        other=0.0,
    ).to(tl.float32)
    pre = tl.load(PRE_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    out = tl.sum(pre[:, :, None] * x3, axis=1)  # [BLOCK_M, BLOCK_D]
    tl.store(OUT_PTR + rm[:, None] * D + rd[None, :], out.to(OUT_PTR.dtype.element_ty), mask=mask2)


@triton.jit
def _hc_collapse_bwd_kernel(
    G_PTR,  # [M, D] grad of out
    X_PTR,  # [M, K, D]
    PRE_PTR,  # [M, K]
    DX_PTR,  # [M, K, D] OUT
    DPRE_P_PTR,  # [NUMD, M, K] partial OUT (reduced host-side)
    M,
    D,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    k_ax = tl.arange(0, K)
    mask_m = rm < M
    mask_d = rd < D
    mask2 = mask_m[:, None] & mask_d[None, :]
    mask3 = mask_m[:, None, None] & mask_d[None, None, :]

    KD = K * D
    x3 = tl.load(
        X_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask3,
        other=0.0,
    ).to(tl.float32)
    g = tl.load(G_PTR + rm[:, None] * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
    pre = tl.load(PRE_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)

    # dx[k, d] = pre[k] * g[d]
    dx = pre[:, :, None] * g[:, None, :]  # [BLOCK_M, K, BLOCK_D]
    tl.store(
        DX_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        dx.to(DX_PTR.dtype.element_ty),
        mask=mask3,
    )

    # dpre[k] = sum_d x[k, d] * g[d]   (partial over this d-block)
    dpre = tl.sum(x3 * g[:, None, :], axis=2)  # [BLOCK_M, K]
    tl.store(
        DPRE_P_PTR + pid_d * (M * K) + rm[:, None] * K + k_ax[None, :],
        dpre,
        mask=mask_m[:, None],
    )


class HCCollapseFn(torch.autograd.Function):
    """Autograd wrapper for the fused ``collapse`` FWD/BWD Triton kernels.

    ``out[..., d] = sum_k pre[..., k] * x[..., k, d]``
    """

    @staticmethod
    def forward(ctx, x, pre):  # type: ignore[override]
        K = x.shape[-2]
        D = x.shape[-1]
        leading = x.shape[:-2]
        M = 1
        for s in leading:
            M *= s

        x_c = x.contiguous()
        pre_c = pre.contiguous()
        out = torch.empty((*leading, D), dtype=x.dtype, device=x.device)

        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(D, _BLOCK_D))
        _hc_collapse_fwd_kernel[grid](
            x_c,
            pre_c,
            out,
            M,
            D,
            K=K,
            BLOCK_M=_BLOCK_M,
            BLOCK_D=_BLOCK_D,
            num_warps=_NUM_WARPS,
        )

        ctx.save_for_backward(x_c, pre_c)
        ctx.leading = tuple(leading)
        ctx.K = K
        ctx.D = D
        ctx.M = M
        return out

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        x_c, pre_c = ctx.saved_tensors
        leading, K, D, M = ctx.leading, ctx.K, ctx.D, ctx.M
        g = g.contiguous()

        dx = torch.empty_like(x_c)
        num_d = triton.cdiv(D, _BLOCK_D)
        dpre_p = torch.empty((num_d, M, K), dtype=torch.float32, device=x_c.device)

        grid = (triton.cdiv(M, _BLOCK_M), num_d)
        _hc_collapse_bwd_kernel[grid](
            g,
            x_c,
            pre_c,
            dx,
            dpre_p,
            M,
            D,
            K=K,
            BLOCK_M=_BLOCK_M,
            BLOCK_D=_BLOCK_D,
            num_warps=_NUM_WARPS,
        )

        d_pre = dpre_p.sum(dim=0).to(pre_c.dtype).view(*leading, K)
        return dx, d_pre


def is_triton_path_enabled() -> bool:
    """``PRIMUS_HC_COLLAPSE_TRITON`` knob; default-on, set ``"0"`` to disable."""
    return os.environ.get("PRIMUS_HC_COLLAPSE_TRITON", "1") != "0"


def is_triton_kernel_supported(x: torch.Tensor, pre: torch.Tensor) -> bool:
    if not (x.is_cuda and pre.is_cuda):
        return False
    if x.dim() < 2:
        return False
    K = x.shape[-2]
    if K not in _SUPPORTED_K:
        return False
    if pre.shape[-1] != K:
        return False
    if x.dtype != pre.dtype or x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


def eager_hc_collapse(x: torch.Tensor, pre: torch.Tensor) -> torch.Tensor:
    """Reference eager ``collapse`` (matches HyperMixer.collapse bit-for-bit)."""
    return (pre.unsqueeze(-1) * x).sum(dim=-2)


def hc_collapse_triton(x, pre):
    """Fused ``out[...,d] = sum_k pre[...,k] * x[...,k,d]`` with eager fallback."""
    if is_triton_path_enabled() and is_triton_kernel_supported(x, pre):
        return HCCollapseFn.apply(x, pre)
    return eager_hc_collapse(x, pre)


__all__ = [
    "HCCollapseFn",
    "hc_collapse_triton",
    "eager_hc_collapse",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
