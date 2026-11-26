###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark per-expert post norm, fused with the routing prob (Triton).

The latent MoE applies, to every expert output row ``y`` (all local experts in one
grouped GEMM output, ``[T, H]``; one norm weight ``w`` shared by the experts), and then
the routing prob ``p`` of the row (HF ``MuseSparkTextExperts`` order and rounding)::

    s   = round(y.float() * (1 + w.float()))          # InputScaledRMSNorm: scale, round
    n   = round(s.float() * rsqrt(mean(s.float()**2) + eps))
    out = round(n.float() * p.float())                # then the routing weight

``round`` is a cast to ``y.dtype``. Eager PyTorch runs this as about ten elementwise
kernels with fp32 ``[T, H]`` intermediates, and autograd keeps several of them
(at 256 experts, top-8 and H = 4096 that is several GiB per MoE layer). Here one
row-wise forward kernel and one backward kernel do it; the roundings are kept in
registers, so the result matches the eager path up to the fp32 reduction order.
Only ``y``, ``w``, ``p`` and the per-row ``rstd`` ([T] fp32) are saved.

Backward (the eager autograd graph, with the same roundings of the gradients)::

    dp  = sum(g.float() * n.float())                  # per row, fp32
    gn  = round(g.float() * p)
    ds  = round(r * gn - s * r**3 * mean(gn * s))     # r = rstd
    dy  = round(ds * (1 + w))
    dw  = sum_rows(ds * y.float())                    # per-program partials, then a sum

:func:`post_expert_norm_probs` is the entry point. It runs the kernel on 2-D GPU tensors
(CUDA / ROCm) of width <= ``MAX_FUSED_HIDDEN`` in fp32 / bf16 / fp16 when Triton is
available, and the eager formula (:func:`post_expert_norm_probs_reference`) otherwise.
"""

from __future__ import annotations

import torch

try:  # Triton ships with the ROCm / CUDA torch builds; the CPU path does not need it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only installs
    triton = None
    tl = None

__all__ = [
    "MAX_FUSED_HIDDEN",
    "fused_rms_norm_available",
    "post_expert_norm_probs",
    "post_expert_norm_probs_reference",
]

# One row is one Triton block; above this width the eager path is used.
MAX_FUSED_HIDDEN = 16384


def post_expert_norm_probs_reference(y, weight, probs, eps):
    """The eager path (``MuseSparkInputScaledRMSNorm`` then the prob), HF rounding order."""
    s = (y.float() * (1.0 + weight.float())).type_as(y)
    sf = s.float()
    n = (sf * torch.pow(sf.pow(2).mean(-1, keepdim=True) + eps, -0.5)).type_as(y)
    if probs is None:  # unit probs: n * 1 rounds to n
        return n
    # one prob per row: [T] and [T, 1] give [T, H], as the kernel's probs.reshape(t)
    return (n.float() * probs.float().reshape(*y.shape[:-1], 1)).to(y.dtype)


def fused_rms_norm_available(y) -> bool:
    """Whether the Muse Spark Triton norm kernels (post-expert, input-scaled, centered) can
    run on ``y``: Triton present, a 2-D GPU tensor of a supported width and dtype."""
    return (
        triton is not None
        and isinstance(y, torch.Tensor)
        and y.is_cuda
        and y.dim() == 2
        and 0 < y.shape[-1] <= MAX_FUSED_HIDDEN
        and y.dtype in (torch.float32, torch.bfloat16, torch.float16)
    )


if triton is not None:

    @triton.jit
    def _fwd_kernel(
        Y, W, P, OUT, RSTD, H, stride_y, stride_o, eps, BLOCK_H: tl.constexpr, HAS_P: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < H
        dt = Y.dtype.element_ty
        y = tl.load(Y + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
        wp1 = 1.0 + tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        s = (y * wp1).to(dt).to(tl.float32)
        ms = tl.sum(s * s, axis=0) / H
        r = tl.math.div_rn(1.0, tl.sqrt_rn(ms + eps))
        n = (s * r).to(dt).to(tl.float32)
        if HAS_P:  # scale by the row prob (without probs the multiply is skipped)
            n = n * tl.load(P + row).to(tl.float32)
        tl.store(OUT + row * stride_o + cols, n.to(OUT.dtype.element_ty), mask=mask)
        tl.store(RSTD + row, r)

    @triton.jit
    def _bwd_kernel(
        Y,
        W,
        P,
        G,
        RSTD,
        DY,
        DP,
        DWP,
        T,
        H,
        tiles_per_prog,
        stride_y,
        stride_g,
        stride_dy,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_P: tl.constexpr,
        HAS_DW: tl.constexpr,
    ):
        # each program walks ``tiles_per_prog`` tiles of BLOCK_T rows and keeps its own
        # fp32 partial of dw (one row of DWP); the host sums the partials
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        cmask = cols < H
        dt = Y.dtype.element_ty
        wp1 = 1.0 + tl.load(W + cols, mask=cmask, other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK_H], dtype=tl.float32)
        first = pid.to(tl.int64) * tiles_per_prog * BLOCK_T
        offs_t = tl.arange(0, BLOCK_T).to(tl.int64)
        for k in range(0, tiles_per_prog):
            rows = first + k * BLOCK_T + offs_t
            live = rows < T
            m = live[:, None] & cmask[None, :]
            y = tl.load(Y + rows[:, None] * stride_y + cols[None, :], mask=m, other=0.0).to(tl.float32)
            g = tl.load(G + rows[:, None] * stride_g + cols[None, :], mask=m, other=0.0).to(tl.float32)
            r = tl.load(RSTD + rows, mask=live, other=0.0)
            s = (y * wp1[None, :]).to(dt).to(tl.float32)
            if HAS_P:
                p = tl.load(P + rows, mask=live, other=0.0).to(tl.float32)
                n = (s * r[:, None]).to(dt).to(tl.float32)
                tl.store(DP + rows, tl.sum(g * n, axis=1).to(DP.dtype.element_ty), mask=live)
                gn = (g * p[:, None]).to(dt).to(tl.float32)
            else:  # unit probs: g * 1 is g (g is in dt already); no dp
                gn = g
            c = r * r * r * (tl.sum(gn * s, axis=1) / H)
            ds = (r[:, None] * gn - s * c[:, None]).to(dt).to(tl.float32)
            tl.store(
                DY + rows[:, None] * stride_dy + cols[None, :],
                (ds * wp1[None, :]).to(DY.dtype.element_ty),
                mask=m,
            )
            if HAS_DW:  # accumulate dw (skipped for a constant, no-grad weight)
                acc += tl.sum(ds * y, axis=0)  # dead rows add 0: their y is 0
        if HAS_DW:
            tl.store(DWP + pid.to(tl.int64) * H + cols, acc, mask=cmask)


# Launch tuning (rows of width BLOCK_H):
_MAX_BWD_ROWS = 4  # most rows one backward program loads per tile
_BWD_TILE_ELEMS = 4096  # backward tile size cap: rows per tile = this // BLOCK_H (1 at H >= 4096)
_MAX_WARPS = 16  # num_warps cap (the hardware limit is 16 wavefronts per block)
_COLS_PER_WARP = 256  # num_warps = BLOCK_H // this (256 columns per warp / wavefront)
_BWD_PROGRAMS_PER_SM = 4  # backward grid: this many programs per SM / CU (dw partial sums)
_FALLBACK_SMS = 4  # SM count when the device has none (the Triton interpreter on CPU)


def _launch_cfg(h):
    """(BLOCK_H, rows per backward tile, num_warps) for a row width ``h``."""
    block = triton.next_power_of_2(h)
    rows = max(1, min(_MAX_BWD_ROWS, _BWD_TILE_ELEMS // block))
    return block, rows, max(1, min(_MAX_WARPS, block // _COLS_PER_WARP))


def _bwd_programs(t, device):
    try:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:  # noqa: BLE001 - e.g. the Triton interpreter on CPU
        sms = _FALLBACK_SMS
    return max(1, min(t, _BWD_PROGRAMS_PER_SM * sms))


class _PostExpertNormProbs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, weight, probs, eps):
        y = y.contiguous()
        t, h = y.shape
        has_p = probs is not None
        ctx.probs_shape = probs.shape if has_p else None
        # no probs: the kernel never reads P (HAS_P=False); y stands in for the pointer
        probs = probs.reshape(t).contiguous() if has_p else y
        weight = weight.contiguous()
        out = torch.empty_like(y)
        rstd = torch.empty(t, dtype=torch.float32, device=y.device)
        if t:
            block, _, warps = _launch_cfg(h)
            _fwd_kernel[(t,)](
                y,
                weight,
                probs,
                out,
                rstd,
                h,
                y.stride(0),
                out.stride(0),
                float(eps),
                BLOCK_H=block,
                HAS_P=has_p,
                num_warps=warps,
            )
        ctx.has_p = has_p
        ctx.save_for_backward(y, weight, probs if has_p else None, rstd)
        return out

    @staticmethod
    def backward(ctx, grad):
        y, weight, probs, rstd = ctx.saved_tensors
        grad = grad.contiguous()
        t, h = y.shape
        dy = torch.empty_like(y)
        has_p = ctx.has_p
        dp = torch.empty_like(probs) if has_p else None
        n_prog = _bwd_programs(t, y.device)
        has_dw = ctx.needs_input_grad[1]
        dwp = None
        if has_dw:
            # t > 0: every program stores its whole dw row (grid (n_prog,), columns masked
            # to H), so no fill is needed; t == 0 launches nothing and the partial must be 0
            alloc = torch.empty if t else torch.zeros
            dwp = alloc((n_prog, h), dtype=torch.float32, device=y.device)
        if t:
            block, block_t, warps = _launch_cfg(h)
            tiles = triton.cdiv(triton.cdiv(t, block_t), n_prog)
            _bwd_kernel[(n_prog,)](
                y,
                weight,
                probs if has_p else y,
                grad,
                rstd,
                dy,
                dp if has_p else rstd,
                dwp if has_dw else rstd,  # not written without HAS_DW
                t,
                h,
                tiles,
                y.stride(0),
                grad.stride(0),
                dy.stride(0),
                BLOCK_T=block_t,
                BLOCK_H=block,
                HAS_P=has_p,
                HAS_DW=has_dw,
                num_warps=warps,
            )
        dw = dwp.sum(0).to(weight.dtype) if has_dw else None
        return dy, dw, dp.view(ctx.probs_shape) if has_p else None, None


def post_expert_norm_probs(y, weight, probs, eps):
    """``round(round(InputScaledRMSNorm(y)) * probs)`` per row: the fused kernel on GPU,
    the eager formula elsewhere (CPU, no Triton, width above ``MAX_FUSED_HIDDEN``).

    ``y``: ``[T, H]`` expert outputs; ``weight``: ``[H]``; ``probs``: ``[T]`` (or ``[T, 1]``) routing probs,
    or None for unit probs (the norm alone, ``round(InputScaledRMSNorm(y))``: the kernel
    neither loads probs nor computes their gradient).
    """
    if not fused_rms_norm_available(y):
        return post_expert_norm_probs_reference(y, weight, probs, eps)
    return _PostExpertNormProbs.apply(y, weight, probs, eps)
