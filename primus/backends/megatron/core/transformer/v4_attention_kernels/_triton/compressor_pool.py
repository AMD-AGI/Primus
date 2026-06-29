# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
"""Triton-fused softmax-weighted pool for the V4 Compressor (forward burst).

Fuses the per-window-softmax pooling tail of :class:`Compressor.forward`::

    score = score + ape                                  # [B, N, W, hd]
    weights = softmax(score.float(), dim=2)              # over the window W
    pooled  = (kv * weights).sum(dim=2)                  # [B, N, hd]

into a single forward kernel: one program per ``(b, n)`` row computes, for each
channel ``d``, the softmax over the ``W`` window slots of ``score[..,:,d]+ape[:,d]``
and the weighted sum of ``kv[..,:,d]`` -- collapsing the ~5-launch forward burst
(add + cast + softmax + cast + mul + reduce) into one launch per compressed layer.

The forward runs the reduction in fp32 (more accurate than the eager bf16-weights
path, numerically equivalent within bf16 noise). The backward is the explicit
analytic gradient in eager torch (softmax jacobian + weighted-sum), so there is no
second Triton kernel to validate; the forward burst -- which is what dominates the
compressed-layer forward -- is the part fused here.

Gating: routed through :func:`fused_softmax_weighted_pool` when
``PRIMUS_COMPRESS_POOL_TRITON != "0"`` (default-on) for CUDA float16/bfloat16/float32
inputs; falls back to eager only for non-CUDA / unsupported dtypes. Handles BOTH
the HCA (non-overlap, window ``W = ratio``) and CSA (overlap, window ``W = 2*ratio``)
pools — the kernel softmaxes over whatever window axis ``W`` it is given (verified
fp32-exact at both W=128 and W=8).
"""
from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _compressor_pool_fwd_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    out_ptr,
    R,
    HD,
    stride_kv_bn,
    stride_kv_r,
    stride_kv_d,
    stride_sc_bn,
    stride_sc_r,
    stride_sc_d,
    stride_ape_r,
    stride_ape_d,
    stride_out_bn,
    stride_out_d,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]

    sc_ptrs = (
        score_ptr + pid_bn * stride_sc_bn + offs_r[:, None] * stride_sc_r + offs_d[None, :] * stride_sc_d
    )
    ape_ptrs = ape_ptr + offs_r[:, None] * stride_ape_r + offs_d[None, :] * stride_ape_d
    s = tl.load(sc_ptrs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(ape_ptrs, mask=mask, other=0.0).to(tl.float32)
    s = s + a

    # softmax over the window axis R (axis 0), masking padded rows to -inf.
    s = tl.where(r_mask[:, None], s, float("-inf"))
    s_max = tl.max(s, axis=0)
    e = tl.exp(s - s_max[None, :])
    e = tl.where(r_mask[:, None], e, 0.0)
    denom = tl.sum(e, axis=0)
    w = e / denom[None, :]  # [BLOCK_R, BLOCK_D] fp32 softmax weights

    kv_ptrs = kv_ptr + pid_bn * stride_kv_bn + offs_r[:, None] * stride_kv_r + offs_d[None, :] * stride_kv_d
    kv = tl.load(kv_ptrs, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.sum(kv * w, axis=0)  # [BLOCK_D]

    out_ptrs = out_ptr + pid_bn * stride_out_bn + offs_d * stride_out_d
    tl.store(out_ptrs, pooled.to(out_ptr.dtype.element_ty), mask=d_mask)


def _pool_fwd_triton(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """kv/score: ``[B, N, W, hd]``; ape: ``[W, hd]`` -> pooled ``[B, N, hd]``."""
    B, N, W, HD = kv.shape
    bn = B * N
    kv3 = kv.reshape(bn, W, HD)
    sc3 = score.reshape(bn, W, HD)
    out = torch.empty((bn, HD), dtype=kv.dtype, device=kv.device)
    BLOCK_R = max(16, triton.next_power_of_2(W))
    BLOCK_D = min(64, max(16, triton.next_power_of_2(HD)))
    grid = (bn, triton.cdiv(HD, BLOCK_D))
    _compressor_pool_fwd_kernel[grid](
        kv3,
        sc3,
        ape,
        out,
        W,
        HD,
        kv3.stride(0),
        kv3.stride(1),
        kv3.stride(2),
        sc3.stride(0),
        sc3.stride(1),
        sc3.stride(2),
        ape.stride(0),
        ape.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )
    return out.reshape(B, N, HD)


@triton.jit
def _compressor_pool_bwd_kernel(
    kv_ptr,
    score_ptr,
    ape_ptr,
    dout_ptr,
    dkv_ptr,
    dscore_ptr,
    R,
    HD,
    stride_kv_bn,
    stride_kv_r,
    stride_kv_d,
    stride_sc_bn,
    stride_sc_r,
    stride_sc_d,
    stride_ape_r,
    stride_ape_d,
    stride_do_bn,
    stride_do_d,
    stride_dkv_bn,
    stride_dkv_r,
    stride_dkv_d,
    stride_dsc_bn,
    stride_dsc_r,
    stride_dsc_d,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]

    # Recompute the fwd softmax weights w over the window axis R (axis 0), fp32.
    sc_ptrs = (
        score_ptr + pid_bn * stride_sc_bn + offs_r[:, None] * stride_sc_r + offs_d[None, :] * stride_sc_d
    )
    ape_ptrs = ape_ptr + offs_r[:, None] * stride_ape_r + offs_d[None, :] * stride_ape_d
    s = tl.load(sc_ptrs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(ape_ptrs, mask=mask, other=0.0).to(tl.float32)
    s = s + a
    s = tl.where(r_mask[:, None], s, float("-inf"))
    s_max = tl.max(s, axis=0)
    e = tl.exp(s - s_max[None, :])
    e = tl.where(r_mask[:, None], e, 0.0)
    denom = tl.sum(e, axis=0)
    w = e / denom[None, :]  # [BLOCK_R, BLOCK_D]

    kv_ptrs = kv_ptr + pid_bn * stride_kv_bn + offs_r[:, None] * stride_kv_r + offs_d[None, :] * stride_kv_d
    kv = tl.load(kv_ptrs, mask=mask, other=0.0).to(tl.float32)
    do_ptrs = dout_ptr + pid_bn * stride_do_bn + offs_d * stride_do_d
    dout = tl.load(do_ptrs, mask=d_mask, other=0.0).to(tl.float32)  # [BLOCK_D]

    # pooled = sum_r w*kv  ->  dkv = dout*w ; dw = dout*kv ;
    # softmax jacobian: dscore = w * (dw - sum_r(dw*w))
    dkv = dout[None, :] * w
    dw = dout[None, :] * kv
    sum_dw_w = tl.sum(dw * w, axis=0)  # [BLOCK_D]
    dscore = w * (dw - sum_dw_w[None, :])

    dkv_ptrs = (
        dkv_ptr + pid_bn * stride_dkv_bn + offs_r[:, None] * stride_dkv_r + offs_d[None, :] * stride_dkv_d
    )
    tl.store(dkv_ptrs, dkv.to(dkv_ptr.dtype.element_ty), mask=mask)
    dsc_ptrs = (
        dscore_ptr + pid_bn * stride_dsc_bn + offs_r[:, None] * stride_dsc_r + offs_d[None, :] * stride_dsc_d
    )
    tl.store(dsc_ptrs, dscore.to(dscore_ptr.dtype.element_ty), mask=mask)


def _pool_bwd_triton(dout, kv, score, ape):
    """Fused backward of the softmax-weighted pool. Returns (dkv, dscore, dape).

    Recomputes w in-register (like the fwd) and emits dkv + dscore in one kernel;
    dape = sum over the (B,N) rows of dscore is a single trailing reduce.
    """
    B, N, W, HD = kv.shape
    bn = B * N
    kv3 = kv.reshape(bn, W, HD)
    sc3 = score.reshape(bn, W, HD)
    do2 = dout.reshape(bn, HD)
    dkv = torch.empty_like(kv3)
    dscore = torch.empty_like(sc3)
    BLOCK_R = max(16, triton.next_power_of_2(W))
    BLOCK_D = min(64, max(16, triton.next_power_of_2(HD)))
    grid = (bn, triton.cdiv(HD, BLOCK_D))
    _compressor_pool_bwd_kernel[grid](
        kv3,
        sc3,
        ape,
        do2,
        dkv,
        dscore,
        W,
        HD,
        kv3.stride(0),
        kv3.stride(1),
        kv3.stride(2),
        sc3.stride(0),
        sc3.stride(1),
        sc3.stride(2),
        ape.stride(0),
        ape.stride(1),
        do2.stride(0),
        do2.stride(1),
        dkv.stride(0),
        dkv.stride(1),
        dkv.stride(2),
        dscore.stride(0),
        dscore.stride(1),
        dscore.stride(2),
        BLOCK_R=BLOCK_R,
        BLOCK_D=BLOCK_D,
    )
    dkv = dkv.reshape(B, N, W, HD)
    dscore = dscore.reshape(B, N, W, HD)
    dape = dscore.sum(dim=(0, 1)).to(ape.dtype)
    return dkv, dscore, dape


class _CompressorSoftmaxPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, score, ape):
        pooled = _pool_fwd_triton(kv, score, ape)
        ctx.save_for_backward(kv, score, ape)
        return pooled

    @staticmethod
    def backward(ctx, dout):
        kv, score, ape = ctx.saved_tensors
        # Fused Triton backward (default-on): recompute softmax + emit dkv/dscore in
        # one launch instead of the ~14-kernel eager analytic burst. Handles both the
        # HCA (W=ratio) and CSA (W=2*ratio overlap) pools -- the window fits one tile
        # in either case. PRIMUS_COMPRESS_POOL_TRITON_BWD=0 restores the eager path.
        if os.environ.get("PRIMUS_COMPRESS_POOL_TRITON_BWD", "1") != "0" and pool_supported(kv):
            return _pool_bwd_triton(dout.contiguous(), kv, score, ape)
        # Analytic gradient of pooled = sum_w softmax_w(score+ape) * kv (fp32).
        w = torch.softmax((score + ape).float(), dim=2)  # [B, N, W, hd]
        dout3 = dout.unsqueeze(2).float()  # [B, N, 1, hd]
        dkv = (dout3 * w).to(kv.dtype)
        dw = dout3 * kv.float()
        # softmax jacobian: dscore = w * (dw - sum_w(dw * w))
        dscore = (w * (dw - (dw * w).sum(dim=2, keepdim=True))).to(score.dtype)
        dape = dscore.sum(dim=(0, 1)).to(ape.dtype)
        return dkv, dscore, dape


def fused_softmax_weighted_pool(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """Fused ``(softmax(score+ape, dim=2) * kv).sum(dim=2)``.

    kv/score: ``[B, N, W, hd]``; ape: ``[W, hd]`` -> ``[B, N, hd]``.
    """
    return _CompressorSoftmaxPool.apply(kv, score, ape)


def pool_supported(kv: torch.Tensor) -> bool:
    return kv.is_cuda and kv.dtype in (torch.float16, torch.bfloat16, torch.float32)
