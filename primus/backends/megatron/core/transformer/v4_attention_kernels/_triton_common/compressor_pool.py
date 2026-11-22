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


# ----------------------------------------------------------------------------------------------------------------
# PRIMUS_GLUE_COMPRESS_POOL=1: the pool reads the fused wkv_gate projection directly. The window split, the overlap
# stitch (zeros + roll + cat of the two halves) and their backward (slice-grad zero fill + copy, chunk/split cats)
# become address arithmetic inside the two kernels; values, operation order and rounding are those of the kernels
# above, so the result is bit-identical.
# ----------------------------------------------------------------------------------------------------------------


@triton.jit
def _window_src(pid_bn, offs_r, NW, RATIO: tl.constexpr, HD: tl.constexpr, OVERLAP: tl.constexpr):
    """(source row in [B*S], kv channel offset, score channel offset, valid) of window slot r of window pid_bn."""
    n = pid_bn % NW
    if OVERLAP:
        is_a = offs_r < RATIO
        row = tl.where(is_a, pid_bn * RATIO + offs_r, (pid_bn - 1) * RATIO + offs_r - RATIO)
        valid = is_a | (n > 0)
        kv_c = tl.where(is_a, 0, HD)
        sc_c = tl.where(is_a, 2 * HD, 3 * HD)
    else:
        row = pid_bn * RATIO + offs_r
        valid = offs_r < RATIO
        kv_c = offs_r * 0
        sc_c = offs_r * 0 + HD
    return row, kv_c, sc_c, valid


@triton.jit
def _window_pool_fwd_kernel(P_PTR, APE_PTR, OUT_PTR, NW, R, stride_p, RATIO: tl.constexpr, HD: tl.constexpr,
                            OVERLAP: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bn = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]
    row, kv_c, sc_c, valid = _window_src(pid_bn, offs_r, NW, RATIO, HD, OVERLAP)
    lmask = mask & valid[:, None]
    rbase = row.to(tl.int64)[:, None] * stride_p + offs_d[None, :]
    s = tl.load(P_PTR + rbase + sc_c[:, None], mask=lmask, other=0.0).to(tl.float32)
    a = tl.load(APE_PTR + offs_r[:, None] * HD + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    s = s + a
    s = tl.where(r_mask[:, None], s, float("-inf"))
    s_max = tl.max(s, axis=0)
    e = tl.exp(s - s_max[None, :])
    e = tl.where(r_mask[:, None], e, 0.0)
    denom = tl.sum(e, axis=0)
    w = e / denom[None, :]
    kv = tl.load(P_PTR + rbase + kv_c[:, None], mask=lmask, other=0.0).to(tl.float32)
    pooled = tl.sum(kv * w, axis=0)
    tl.store(OUT_PTR + pid_bn * HD + offs_d, pooled.to(OUT_PTR.dtype.element_ty), mask=d_mask)


@triton.jit
def _window_pool_bwd_kernel(P_PTR, APE_PTR, DO_PTR, DP_PTR, DSC_PTR, NW, R, stride_p, RATIO: tl.constexpr,
                            HD: tl.constexpr, OVERLAP: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_bn = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]
    row, kv_c, sc_c, valid = _window_src(pid_bn, offs_r, NW, RATIO, HD, OVERLAP)
    lmask = mask & valid[:, None]
    rbase = row.to(tl.int64)[:, None] * stride_p + offs_d[None, :]
    s = tl.load(P_PTR + rbase + sc_c[:, None], mask=lmask, other=0.0).to(tl.float32)
    a = tl.load(APE_PTR + offs_r[:, None] * HD + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
    s = s + a
    s = tl.where(r_mask[:, None], s, float("-inf"))
    s_max = tl.max(s, axis=0)
    e = tl.exp(s - s_max[None, :])
    e = tl.where(r_mask[:, None], e, 0.0)
    denom = tl.sum(e, axis=0)
    w = e / denom[None, :]
    kv = tl.load(P_PTR + rbase + kv_c[:, None], mask=lmask, other=0.0).to(tl.float32)
    dout = tl.load(DO_PTR + pid_bn * HD + offs_d, mask=d_mask, other=0.0).to(tl.float32)
    dkv = dout[None, :] * w
    dw = dout[None, :] * kv
    sum_dw_w = tl.sum(dw * w, axis=0)
    dscore = w * (dw - sum_dw_w[None, :])
    dt = DP_PTR.dtype.element_ty
    # every (row, channel) of d(proj) belongs to exactly one window slot: plain stores, no accumulation
    tl.store(DP_PTR + rbase + kv_c[:, None], dkv.to(dt), mask=lmask)
    tl.store(DP_PTR + rbase + sc_c[:, None], dscore.to(dt), mask=lmask)
    # dscore in the [B*N, W, HD] layout of the unfused path, for the (unchanged) torch dape reduction
    tl.store(DSC_PTR + pid_bn * R * HD + offs_r[:, None] * HD + offs_d[None, :], dscore.to(DSC_PTR.dtype.element_ty),
             mask=mask)
    if OVERLAP:
        # the last window's "b" halves feed no window (the roll drops them): their gradient is zero
        if pid_bn % NW == NW - 1:
            zr = (pid_bn * RATIO + offs_r).to(tl.int64)
            zm = (offs_r < RATIO)[:, None] & d_mask[None, :]
            z = tl.zeros((BLOCK_R, BLOCK_D), dtype=dt)
            tl.store(DP_PTR + zr[:, None] * stride_p + HD + offs_d[None, :], z, mask=zm)
            tl.store(DP_PTR + zr[:, None] * stride_p + 3 * HD + offs_d[None, :], z, mask=zm)


@triton.jit
def _window_pool_bwd_dape_kernel(P_PTR, APE_PTR, DO_PTR, DP_PTR, PART_PTR, NBN, NW, R, stride_p, GW,
                                 RATIO: tl.constexpr, HD: tl.constexpr, OVERLAP: tl.constexpr,
                                 BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    """PRIMUS_FUSE2_WPOOL_DAPE: ``_window_pool_bwd_kernel`` for GW consecutive windows per program (each window's
    math and d(proj) stores unchanged), plus the window-sum of the bf16-rounded dscore in fp32 registers
    (fixed order: window by window), written as one [R, HD] fp32 partial per program."""
    pid_g = tl.program_id(0)
    pid_d = tl.program_id(1)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    r_mask = offs_r < R
    d_mask = offs_d < HD
    mask = r_mask[:, None] & d_mask[None, :]
    dt = DP_PTR.dtype.element_ty
    acc = tl.zeros((BLOCK_R, BLOCK_D), dtype=tl.float32)
    for g in range(GW):
        pid_bn = pid_g * GW + g
        inb = pid_bn < NBN
        row, kv_c, sc_c, valid = _window_src(pid_bn, offs_r, NW, RATIO, HD, OVERLAP)
        lmask = mask & valid[:, None] & inb
        rbase = row.to(tl.int64)[:, None] * stride_p + offs_d[None, :]
        s = tl.load(P_PTR + rbase + sc_c[:, None], mask=lmask, other=0.0).to(tl.float32)
        a = tl.load(APE_PTR + offs_r[:, None] * HD + offs_d[None, :], mask=mask, other=0.0).to(tl.float32)
        s = s + a
        s = tl.where(r_mask[:, None], s, float("-inf"))
        s_max = tl.max(s, axis=0)
        e = tl.exp(s - s_max[None, :])
        e = tl.where(r_mask[:, None], e, 0.0)
        denom = tl.sum(e, axis=0)
        w = e / denom[None, :]
        kv = tl.load(P_PTR + rbase + kv_c[:, None], mask=lmask, other=0.0).to(tl.float32)
        dout = tl.load(DO_PTR + pid_bn * HD + offs_d, mask=d_mask & inb, other=0.0).to(tl.float32)
        dkv = dout[None, :] * w
        dw = dout[None, :] * kv
        sum_dw_w = tl.sum(dw * w, axis=0)
        dscore = w * (dw - sum_dw_w[None, :])
        tl.store(DP_PTR + rbase + kv_c[:, None], dkv.to(dt), mask=lmask)
        tl.store(DP_PTR + rbase + sc_c[:, None], dscore.to(dt), mask=lmask)
        acc += tl.where(mask & inb, dscore.to(dt).to(tl.float32), 0.0)
        if OVERLAP:
            zr = (pid_bn * RATIO + offs_r).to(tl.int64)
            zm = (offs_r < RATIO)[:, None] & d_mask[None, :] & inb & (pid_bn % NW == NW - 1)
            z = tl.zeros((BLOCK_R, BLOCK_D), dtype=dt)
            tl.store(DP_PTR + zr[:, None] * stride_p + HD + offs_d[None, :], z, mask=zm)
            tl.store(DP_PTR + zr[:, None] * stride_p + 3 * HD + offs_d[None, :], z, mask=zm)
    tl.store(PART_PTR + pid_g.to(tl.int64) * (R * HD) + offs_r[:, None] * HD + offs_d[None, :], acc, mask=mask)


@triton.jit
def _dape_reduce_kernel(PART_PTR, OUT_PTR, NG, NUMEL, BETA: tl.constexpr, NG_P: tl.constexpr, BLOCK: tl.constexpr):
    """dape = bf16(sum over the NG partials) (the unfused path's bf16 sum), widened to OUT's dtype; BETA: added to
    OUT (main_grad accumulate) instead of overwriting it."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    gs = tl.arange(0, NG_P)
    m = offs < NUMEL
    p = tl.load(PART_PTR + gs[:, None].to(tl.int64) * NUMEL + offs[None, :], mask=(gs[:, None] < NG) & m[None, :],
                other=0.0)
    v = tl.sum(p, axis=0).to(tl.bfloat16).to(tl.float32)
    if BETA:
        v = v + tl.load(OUT_PTR + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT_PTR + offs, v.to(OUT_PTR.dtype.element_ty), mask=m)


def fuse2_wpool_dape_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSE2_WPOOL_DAPE", "0") == "1"


def _ape_main_grad_ok(ape) -> bool:
    mg = getattr(ape, "main_grad", None)
    return (
        isinstance(ape, torch.nn.Parameter)
        and hasattr(ape, "grad_added_to_main_grad")
        and isinstance(mg, torch.Tensor)
        and mg.shape == ape.shape
        and mg.dtype in (torch.float32, torch.bfloat16)
        and mg.is_contiguous()
        and not getattr(ape, "shared", False)
    )


class _WindowPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, proj, ape, ratio, hd, overlap):
        B, S, C = proj.shape
        NW = S // ratio
        R = 2 * ratio if overlap else ratio
        p2 = proj.reshape(B * S, C)
        out = torch.empty((B * NW, hd), dtype=proj.dtype, device=proj.device)
        BLOCK_R = max(16, triton.next_power_of_2(R))
        BLOCK_D = min(64, max(16, triton.next_power_of_2(hd)))
        grid = (B * NW, triton.cdiv(hd, BLOCK_D))
        _window_pool_fwd_kernel[grid](p2, ape, out, NW, R, p2.stride(0), RATIO=ratio, HD=hd, OVERLAP=bool(overlap),
                                      BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(p2, ape)
        ctx.meta = (B, S, C, NW, R, ratio, hd, bool(overlap), BLOCK_R, BLOCK_D)
        # PRIMUS_FUSE2_WPOOL_DAPE: dape summed in the bwd kernel (+ one small reduce); with a DDP main_grad the
        # reduce writes it there (first write overwrites, later ones add), like the fused-wgrad linears
        ctx.dape = fuse2_wpool_dape_enabled() and ape.requires_grad
        ctx.ape_param = None
        if ctx.dape and _ape_main_grad_ok(ape):
            ctx.ape_param = ape
            ape.grad_added_to_main_grad = True
            ape._primus_fwd_claim = True
        return out.view(B, NW, hd)

    @staticmethod
    def backward(ctx, dout):
        p2, ape = ctx.saved_tensors
        B, S, C, NW, R, ratio, hd, overlap, BLOCK_R, BLOCK_D = ctx.meta
        do2 = dout.reshape(B * NW, hd).contiguous()
        dp = torch.empty_like(p2)
        if ctx.dape:
            NBN = B * NW
            nd = triton.cdiv(hd, BLOCK_D)
            GW = max(1, min(8, (NBN * nd) // 2048))
            NG = triton.cdiv(NBN, GW)
            part = torch.empty((NG, R, hd), dtype=torch.float32, device=p2.device)
            _window_pool_bwd_dape_kernel[(NG, nd)](p2, ape, do2, dp, part, NBN, NW, R, p2.stride(0), GW, RATIO=ratio,
                                                   HD=hd, OVERLAP=overlap, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D)
            wp, ctx.ape_param = ctx.ape_param, None
            beta = False
            if wp is not None:
                from primus_turbo.pytorch.ops.utils import _get_dummy_wgrad, main_grad_first_write

                out = wp.main_grad
                beta = not main_grad_first_write(wp)
            else:
                out = torch.empty((R, hd), dtype=ape.dtype, device=p2.device)
            numel = R * hd
            ng_p = triton.next_power_of_2(NG)
            blk = max(16, min(1024, 8192 // ng_p))
            _dape_reduce_kernel[(triton.cdiv(numel, blk),)](part, out, NG, numel, BETA=beta, NG_P=ng_p, BLOCK=blk)
            dape = _get_dummy_wgrad(list(ape.shape), ape.dtype) if wp is not None else out
            return dp.view(B, S, C), dape, None, None, None
        dsc = torch.empty((B * NW, R, hd), dtype=p2.dtype, device=p2.device)
        grid = (B * NW, triton.cdiv(hd, BLOCK_D))
        _window_pool_bwd_kernel[grid](p2, ape, do2, dp, dsc, NW, R, p2.stride(0), RATIO=ratio, HD=hd,
                                      OVERLAP=overlap, BLOCK_R=BLOCK_R, BLOCK_D=BLOCK_D)
        dape = dsc.reshape(B, NW, R, hd).sum(dim=(0, 1)).to(ape.dtype)
        return dp.view(B, S, C), dape, None, None, None


def glue_window_pool_ok(proj: torch.Tensor, ape: torch.Tensor, ratio: int, hd: int, overlap: bool) -> bool:
    """PRIMUS_GLUE_COMPRESS_POOL=1 (default off) and the contiguous fused-projection layout."""
    if os.environ.get("PRIMUS_GLUE_COMPRESS_POOL", "0") != "1":
        return False
    if os.environ.get("PRIMUS_COMPRESS_POOL_TRITON", "1") == "0" or os.environ.get(
            "PRIMUS_COMPRESS_POOL_TRITON_BWD", "1") == "0":
        return False
    coff = 2 if overlap else 1
    return (
        proj.is_cuda and proj.dim() == 3 and proj.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and proj.is_contiguous() and proj.shape[-1] == 2 * coff * hd and proj.shape[1] % ratio == 0
        and proj.shape[1] >= ratio and ape.is_contiguous() and tuple(ape.shape) == ((2 * ratio if overlap else ratio), hd)
    )


def fused_window_pool(proj: torch.Tensor, ape: torch.Tensor, ratio: int, hd: int, overlap: bool) -> torch.Tensor:
    """``_pool_windows`` from the fused ``[B, S, 2*coff*hd]`` projection -> ``[B, S // ratio, hd]``."""
    return _WindowPool.apply(proj, ape, ratio, hd, overlap)


def fused_softmax_weighted_pool(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """Fused ``(softmax(score+ape, dim=2) * kv).sum(dim=2)``.

    kv/score: ``[B, N, W, hd]``; ape: ``[W, hd]`` -> ``[B, N, hd]``.
    """
    return _CompressorSoftmaxPool.apply(kv, score, ape)


def pool_supported(kv: torch.Tensor) -> bool:
    return kv.is_cuda and kv.dtype in (torch.float16, torch.bfloat16, torch.float32)
