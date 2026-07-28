###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused RMSNorm FWD/BWD (small-kernel-fusion campaign 2026-07-03).

Collapses the eager RMSNorm chain — ``x.float()`` (bf16→fp32 cast),
``x.pow(2)`` / ``x.square()``, ``.mean(-1)``, ``+ eps``, ``rsqrt``,
``* rstd``, ``.to(in_dtype)`` (fp32→bf16 cast), optional ``* weight`` —
into ONE Triton kernel (FWD) + ONE Triton kernel (BWD).

Every non-TE eager RMS site in the DeepSeek-V4 model body routes through
this kernel:

* ``_per_head_rms_norm`` (``deepseek_v4_attention``)   — no weight, out=in_dtype.
* ``LocalRMSNorm``       (``compressor.kv_norm`` etc.) — weight (+grad),
  mid-cast to in_dtype before the weight multiply, out=promote(in, weight).
* ``HyperMixer._packed_logits`` RMS                    — no weight, out=fp32.
* ``HyperHead.forward``  RMS                           — no weight, out=fp32.

The eager reference (matched bit-for-bit modulo fp32 accumulation order):

.. code-block:: python

    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    y = x32 * rstd                       # fp32 normalized value
    if mid_cast: y = y.to(in_dtype)      # LocalRMSNorm rounds here first
    if weight is not None: y = y * weight
    out = y.to(out_dtype)

Gradients (weight applied per channel, mid-cast treated as identity for
autograd — matches ``Tensor.to`` grad semantics):

.. code-block:: python

    # forward:  y_k = w_k * rstd * x_k,   rstd = (mean(x^2)+eps)^-1/2
    c   = sum_k( g_k * w_k * x_k )                 # reduce over D
    dx_k = rstd * g_k * w_k - x_k * rstd**3 * c / D
    dw_k = sum_over_rows( g_k * (x_k * rstd) )     # normalized value pre-weight

Gating: routed through :func:`fused_rms_norm` when ``PRIMUS_RMSNORM_TRITON
!= "0"`` (default-on) and the input is a supported CUDA/HIP float tensor;
otherwise the eager reference runs.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

_TORCH_TO_TL_DTYPE = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
}


def _triton_dtype(t: torch.dtype):
    try:
        return _TORCH_TO_TL_DTYPE[t]
    except KeyError as exc:
        raise TypeError(
            f"rmsnorm: unsupported dtype {t}; expected one of {list(_TORCH_TO_TL_DTYPE)}"
        ) from exc


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _rmsnorm_fwd_kernel(
    X_PTR,  # [N, D] contiguous
    W_PTR,  # [D] contiguous (fp32) or dummy when HAS_WEIGHT is False
    OUT_PTR,  # [N, D] contiguous (OUT_DTYPE)
    RSTD_PTR,  # [N] fp32 (saved for backward)
    N,
    D,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    MID_CAST: tl.constexpr,
    IN_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program per row; two passes over D (sum-of-squares, then write)."""
    row = tl.program_id(0)
    if row >= N:
        return
    x_row = X_PTR + row * D

    # Pass 1: sum of squares in fp32.
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)

    rstd = 1.0 / tl.sqrt(acc / D + EPS)
    tl.store(RSTD_PTR + row, rstd)

    # Pass 2: normalize (+ optional mid-cast, weight) and write.
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd
        if MID_CAST:
            y = y.to(IN_DTYPE).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)
            y = y * w
        tl.store(OUT_PTR + row * D + cols, y.to(OUT_DTYPE), mask=mask)


@triton.jit
def _rmsnorm_bwd_kernel(
    X_PTR,  # [N, D] contiguous (original input)
    W_PTR,  # [D] contiguous (fp32) or dummy
    RSTD_PTR,  # [N] fp32
    DY_PTR,  # [N, D] contiguous (upstream grad)
    DX_PTR,  # [N, D] contiguous (output, IN_DTYPE)
    DW_PTR,  # [D] fp32 accumulator (atomic) or dummy
    N,
    D,
    BLOCK_D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    IN_DTYPE: tl.constexpr,
):
    """One program per row. Computes dx (and atomic-accumulates dw)."""
    row = tl.program_id(0)
    if row >= N:
        return
    x_row = X_PTR + row * D
    dy_row = DY_PTR + row * D
    rstd = tl.load(RSTD_PTR + row)

    # Pass 1: c = sum_k( dy_k * w_k * x_k ).
    c = tl.zeros((), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)
            c += tl.sum(dy * w * x, axis=0)
        else:
            c += tl.sum(dy * x, axis=0)

    coef = rstd * rstd * rstd * c / D

    # Pass 2: dx_k = rstd*dy_k*w_k - x_k*coef ; accumulate dw_k += dy_k*x_k*rstd.
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(dy_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = tl.load(W_PTR + cols, mask=mask, other=0.0).to(tl.float32)
            dx = rstd * dy * w - x * coef
            tl.atomic_add(DW_PTR + cols, dy * (x * rstd), mask=mask)
        else:
            dx = rstd * dy - x * coef
        tl.store(DX_PTR + row * D + cols, dx.to(IN_DTYPE), mask=mask)


def _pick_block_d(d: int) -> int:
    """Tile the reduction axis; power-of-2, capped so registers stay bounded."""
    if d <= 2048:
        return triton.next_power_of_2(d)
    return 2048


# ---------------------------------------------------------------------------
# torch.autograd.Function
# ---------------------------------------------------------------------------


class FusedRMSNormFn(torch.autograd.Function):
    """Autograd wrapper for the fused RMSNorm FWD/BWD Triton kernels.

    ``apply(x, weight, eps, mid_cast, out_dtype)``:

    * ``x``: ``[..., D]`` float tensor (RMS over the last dim).
    * ``weight``: ``[D]`` tensor or ``None`` (parameter-less RMS).
    * ``eps``: numerical floor.
    * ``mid_cast``: when ``True``, round the normalized value to ``x.dtype``
      before the weight multiply (matches ``LocalRMSNorm``); no-op when
      ``weight is None``.
    * ``out_dtype``: dtype of the returned tensor.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        weight: Optional[torch.Tensor],
        eps: float,
        mid_cast: bool,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        D = x.shape[-1]
        x2 = x.reshape(-1, D)
        x2 = x2.contiguous()
        N = x2.shape[0]

        has_weight = weight is not None
        w32 = None
        if has_weight:
            if weight.shape[-1] != D:
                raise ValueError(f"rmsnorm: weight dim {tuple(weight.shape)} != x last dim {D}")
            w32 = weight.reshape(D).to(torch.float32).contiguous()

        out = torch.empty((N, D), dtype=out_dtype, device=x.device)
        rstd = torch.empty((N,), dtype=torch.float32, device=x.device)
        block_d = _pick_block_d(D)
        grid = (N,)
        _rmsnorm_fwd_kernel[grid](
            x2,
            w32 if has_weight else x2,  # dummy ptr when no weight
            out,
            rstd,
            N,
            D,
            EPS=float(eps),
            BLOCK_D=block_d,
            HAS_WEIGHT=has_weight,
            MID_CAST=bool(mid_cast) and has_weight,
            IN_DTYPE=_triton_dtype(x.dtype),
            OUT_DTYPE=_triton_dtype(out_dtype),
        )

        ctx.save_for_backward(x2, w32, rstd)
        ctx.has_weight = has_weight
        ctx.in_dtype = x.dtype
        ctx.weight_dtype = weight.dtype if has_weight else None
        ctx.weight_shape = tuple(weight.shape) if has_weight else None
        ctx.block_d = block_d
        ctx.D = D
        ctx.x_shape = tuple(x.shape)
        return out.reshape(ctx.x_shape)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        x2, w32, rstd = ctx.saved_tensors
        D = ctx.D
        N = x2.shape[0]
        has_weight = ctx.has_weight

        # dy comes in out_dtype; the kernel upcasts to fp32 internally, so any
        # float dtype is fine. Keep it as-is (contiguous) without a lossy cast.
        dy2 = dy.reshape(-1, D).contiguous()

        dx = torch.empty((N, D), dtype=ctx.in_dtype, device=dy.device)
        dw_acc = (
            torch.zeros((D,), dtype=torch.float32, device=dy.device)
            if has_weight
            else torch.empty((1,), dtype=torch.float32, device=dy.device)
        )
        grid = (N,)
        _rmsnorm_bwd_kernel[grid](
            x2,
            w32 if has_weight else x2,
            rstd,
            dy2,
            dx,
            dw_acc,
            N,
            D,
            BLOCK_D=ctx.block_d,
            HAS_WEIGHT=has_weight,
            IN_DTYPE=_triton_dtype(ctx.in_dtype),
        )

        dx = dx.reshape(ctx.x_shape)
        dweight = None
        if has_weight:
            dweight = dw_acc.to(ctx.weight_dtype).reshape(ctx.weight_shape)
        return dx, dweight, None, None, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Default-on; A/B toggle via ``PRIMUS_RMSNORM_TRITON=0``."""
    return os.environ.get("PRIMUS_RMSNORM_TRITON", "1") != "0"


def is_triton_kernel_supported(x: torch.Tensor, weight: Optional[torch.Tensor]) -> bool:
    """Supported iff CUDA/HIP float input (and matching-dtype-family weight)."""
    if not x.is_cuda:
        return False
    if x.dtype not in _TORCH_TO_TL_DTYPE:
        return False
    if x.shape[-1] == 0:
        return False
    if weight is not None and not weight.is_cuda:
        return False
    return True


def eager_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    eps: float,
    mid_cast: bool,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference eager RMSNorm matching the fused kernel's contract."""
    in_dtype = x.dtype
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x32 * rstd
    if weight is not None:
        if mid_cast:
            y = y.to(in_dtype)
        y = y.to(torch.float32) * weight.to(torch.float32)
    return y.to(out_dtype)


def fused_rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    *,
    eps: float,
    mid_cast: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dispatch: Triton path when enabled + supported, else eager reference.

    ``out_dtype`` defaults to the eager output dtype:
    ``promote_types(x.dtype, weight.dtype)`` when weighted, else ``x.dtype``.
    """
    if out_dtype is None:
        if weight is not None:
            out_dtype = torch.promote_types(x.dtype, weight.dtype)
        else:
            out_dtype = x.dtype

    if is_triton_path_enabled() and is_triton_kernel_supported(x, weight):
        return FusedRMSNormFn.apply(x, weight, float(eps), bool(mid_cast), out_dtype)
    return eager_rms_norm(x, weight, eps=eps, mid_cast=mid_cast, out_dtype=out_dtype)


__all__ = [
    "FusedRMSNormFn",
    "fused_rms_norm",
    "eager_rms_norm",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
