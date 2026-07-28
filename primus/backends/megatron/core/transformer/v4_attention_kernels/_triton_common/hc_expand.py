###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused HyperConnection ``expand``.

Single FWD + single BWD kernel pair for the eager body of
:meth:`primus.backends.megatron.core.transformer.hyper_connection.HyperMixer.expand`::

    new[..., h, d] = post[..., h] * out[..., d] + sum_k comb[..., h, k] * x[..., k, d]

The contraction over the small ``K`` is done in registers (unrolled, ``K`` is a
constexpr) while ``D`` is streamed; the ``post`` outer-product and the final add
are folded into the same kernel.

Gating: routed through :func:`hc_expand_triton` when
``PRIMUS_HC_EXPAND_TRITON != "0"`` (default-on); falls back to eager for
unsupported configs (CPU input, K out of range, dtype mismatch).

Supported shapes:

* ``K`` (= ``hc_mult``) in ``{1, 2, 4, 8, 16}``.
* ``x``/``out``/``post``/``comb`` share one floating dtype and are CUDA tensors.
* Any leading shape (the wrapper flattens to ``[M, K, D]``).
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
def _hc_expand_fwd_kernel(
    X_PTR,  # [M, K, D]
    OUT_PTR,  # [M, D]
    POST_PTR,  # [M, K]
    COMB_PTR,  # [M, K, K]
    NEW_PTR,  # [M, K, D] OUT
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
    KK = K * K

    # x as [BLOCK_M, K(stream), BLOCK_D], loaded once.
    x3 = tl.load(
        X_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask3,
        other=0.0,
    ).to(tl.float32)
    out_tile = tl.load(OUT_PTR + rm[:, None] * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)

    for h in range(K):
        comb_h = tl.load(
            COMB_PTR + rm[:, None] * KK + h * K + k_ax[None, :], mask=mask_m[:, None], other=0.0
        ).to(
            tl.float32
        )  # [BLOCK_M, K]
        post_h = tl.load(POST_PTR + rm * K + h, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]
        # mix_h[d] = sum_k comb_h[k] * x3[k, d]
        mix_h = tl.sum(comb_h[:, :, None] * x3, axis=1)
        new_h = post_h[:, None] * out_tile + mix_h
        tl.store(
            NEW_PTR + rm[:, None] * KD + h * D + rd[None, :],
            new_h.to(NEW_PTR.dtype.element_ty),
            mask=mask2,
        )


@triton.jit
def _hc_expand_bwd_kernel(
    G_PTR,  # [M, K, D] grad of new
    X_PTR,  # [M, K, D]
    OUT_PTR,  # [M, D]
    POST_PTR,  # [M, K]
    COMB_PTR,  # [M, K, K]
    DX_PTR,  # [M, K, D] OUT
    DOUT_PTR,  # [M, D] OUT
    DPOST_P_PTR,  # [NUMD, M, K] partial OUT (reduced host-side)
    DCOMB_P_PTR,  # [NUMD, M, K, K] partial OUT (reduced host-side)
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
    KK = K * K

    # x3, g3 as [BLOCK_M, K(stream), BLOCK_D]; the stream axis of g indexes h.
    x3 = tl.load(
        X_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask3,
        other=0.0,
    ).to(tl.float32)
    g3 = tl.load(
        G_PTR + rm[:, None, None] * KD + k_ax[None, :, None] * D + rd[None, None, :],
        mask=mask3,
        other=0.0,
    ).to(tl.float32)
    out_tile = tl.load(OUT_PTR + rm[:, None] * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
    post = tl.load(POST_PTR + rm[:, None] * K + k_ax[None, :], mask=mask_m[:, None], other=0.0).to(
        tl.float32
    )  # [BLOCK_M, K(h)]

    # d_out[d] = sum_h post[h] * g[h, d]
    dout = tl.sum(post[:, :, None] * g3, axis=1)
    tl.store(DOUT_PTR + rm[:, None] * D + rd[None, :], dout.to(DOUT_PTR.dtype.element_ty), mask=mask2)

    # d_post[h] = sum_d out[d] * g[h, d]   (partial over this d-block)
    dpost = tl.sum(out_tile[:, None, :] * g3, axis=2)  # [BLOCK_M, K(h)]
    tl.store(
        DPOST_P_PTR + pid_d * (M * K) + rm[:, None] * K + k_ax[None, :],
        dpost,
        mask=mask_m[:, None],
    )

    # d_x[k, d] = sum_h comb[h, k] * g[h, d]
    for k in range(K):
        comb_col = tl.load(
            COMB_PTR + rm[:, None] * KK + k_ax[None, :] * K + k, mask=mask_m[:, None], other=0.0
        ).to(
            tl.float32
        )  # [BLOCK_M, K(h)]
        dxk = tl.sum(comb_col[:, :, None] * g3, axis=1)
        tl.store(DX_PTR + rm[:, None] * KD + k * D + rd[None, :], dxk.to(DX_PTR.dtype.element_ty), mask=mask2)

    # d_comb[h, k] = sum_d x[k, d] * g[h, d]   (partial over this d-block)
    for h in range(K):
        g_h = tl.load(G_PTR + rm[:, None] * KD + h * D + rd[None, :], mask=mask2, other=0.0).to(tl.float32)
        dcomb_h = tl.sum(x3 * g_h[:, None, :], axis=2)  # [BLOCK_M, K(k)]
        tl.store(
            DCOMB_P_PTR + pid_d * (M * KK) + rm[:, None] * KK + h * K + k_ax[None, :],
            dcomb_h,
            mask=mask_m[:, None],
        )


class HCExpandFn(torch.autograd.Function):
    """Autograd wrapper around the fused ``expand`` FWD/BWD Triton kernels.

    ``new[..., h, d] = post[..., h] * out[..., d] + sum_k comb[..., h, k] * x[..., k, d]``
    """

    @staticmethod
    def forward(ctx, x, out, post, comb):  # type: ignore[override]
        K = comb.shape[-1]
        D = x.shape[-1]
        leading = x.shape[:-2]
        M = 1
        for s in leading:
            M *= s

        x_c = x.contiguous()
        out_c = out.contiguous()
        post_c = post.contiguous()
        comb_c = comb.contiguous()

        new = torch.empty((*leading, K, D), dtype=x.dtype, device=x.device)

        grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(D, _BLOCK_D))
        _hc_expand_fwd_kernel[grid](
            x_c,
            out_c,
            post_c,
            comb_c,
            new,
            M,
            D,
            K=K,
            BLOCK_M=_BLOCK_M,
            BLOCK_D=_BLOCK_D,
            num_warps=_NUM_WARPS,
        )

        ctx.save_for_backward(x_c, out_c, post_c, comb_c)
        ctx.leading = tuple(leading)
        ctx.K = K
        ctx.D = D
        ctx.M = M
        return new

    @staticmethod
    def backward(ctx, g):  # type: ignore[override]
        x_c, out_c, post_c, comb_c = ctx.saved_tensors
        leading, K, D, M = ctx.leading, ctx.K, ctx.D, ctx.M
        g = g.contiguous()

        device = x_c.device
        dx = torch.empty_like(x_c)
        dout = torch.empty_like(out_c)
        num_d = triton.cdiv(D, _BLOCK_D)
        dpost_p = torch.empty((num_d, M, K), dtype=torch.float32, device=device)
        dcomb_p = torch.empty((num_d, M, K, K), dtype=torch.float32, device=device)

        grid = (triton.cdiv(M, _BLOCK_M), num_d)
        _hc_expand_bwd_kernel[grid](
            g,
            x_c,
            out_c,
            post_c,
            comb_c,
            dx,
            dout,
            dpost_p,
            dcomb_p,
            M,
            D,
            K=K,
            BLOCK_M=_BLOCK_M,
            BLOCK_D=_BLOCK_D,
            num_warps=_NUM_WARPS,
        )

        d_post = dpost_p.sum(dim=0).to(post_c.dtype).view(*leading, K)
        d_comb = dcomb_p.sum(dim=0).to(comb_c.dtype).view(*leading, K, K)
        return dx, dout, d_post, d_comb


def is_triton_path_enabled() -> bool:
    """``PRIMUS_HC_EXPAND_TRITON`` knob; default-on, set ``"0"`` to disable."""
    return os.environ.get("PRIMUS_HC_EXPAND_TRITON", "1") != "0"


def is_triton_kernel_supported(x: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> bool:
    if not (x.is_cuda and post.is_cuda and comb.is_cuda):
        return False
    K = comb.shape[-1]
    if K not in _SUPPORTED_K:
        return False
    if comb.shape[-2] != K or x.shape[-2] != K or post.shape[-1] != K:
        return False
    if not (x.dtype == post.dtype == comb.dtype) or x.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return False
    return True


def hc_expand_triton(x, out, post, comb):
    """Fused ``new[...,h,d] = post[h]*out[d] + sum_k comb[h,k]*x[k,d]``."""
    return HCExpandFn.apply(x, out, post, comb)


__all__ = [
    "HCExpandFn",
    "hc_expand_triton",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
