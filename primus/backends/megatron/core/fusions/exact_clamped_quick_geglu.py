###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Clamped quick-GEGLU (MiniMax-M3 ``swigluoai``) as one forward and one backward kernel,
bitwise equal to Megatron's eager GLU branch.

Megatron runs ``swigluoai`` (``quick_geglu`` + ``activation_func_clamp_value``) as::

    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
    x_glu = x_glu.clamp(max=L)
    x_linear = x_linear.clamp(-L, L)
    out = quick_gelu(x_glu) * (x_linear + offset)     # quick_gelu is torch.compile'd

That is ~5 forward kernels and ~10 backward kernels (clamp masks, where, muls, the compiled
quick_gelu backward and the ``cat`` that reassembles the two input-gradient halves), each
reading or writing an ffn-wide BF16 tensor.

The kernels here reproduce every rounding of that sequence, so outputs and gradients are
bit-identical:

* forward: ``a = bf16(qg(clamp(xg)))`` with Inductor's FP32 formula for ``quick_gelu``
  (``y * sigmoid(y * 1.702)`` in FP32, rounded once), ``b = bf16(clamp(xl) + offset)``,
  ``out = bf16(a * b)``.
* backward: ``MulBackward0`` gives ``ga = bf16(g_out * b)`` and ``gb = bf16(g_out * a)``;
  the compiled quick_gelu backward gives ``bf16(ga * s + (ga * y) * (s * (1 - s)) * 1.702)``
  in FP32 (``s`` the FP32 sigmoid); the clamp backwards pass the gradient only strictly inside the range
  (``xg < L``, ``-L < xl < L``: torch's clamp backward gives 0 at the bounds); both halves are written into one ``[.., 2F]`` gradient.

Only the input is saved for backward (the eager path keeps the clamped halves, ``a`` and ``b``).
"""

from __future__ import annotations

import torch

_kernels = None


def _get_kernels():
    global _kernels
    if _kernels is not None:
        return _kernels
    import triton
    import triton.language as tl

    @triton.jit
    def _qg_fwd(y):
        # Inductor: tmp1 = tmp0 * 1.702; tmp2 = sigmoid(tmp1); tmp3 = tmp0 * tmp2
        return y * tl.sigmoid(y * 1.702)

    @triton.jit
    def _fwd(X, Y, F, n_blocks_f, limit, offset, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = (pid // n_blocks_f).to(tl.int64)
        cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0)
        xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0)
        g = tl.minimum(xg.to(tl.float32), limit).to(tl.bfloat16)  # clamp(max=L): exact in bf16
        l = tl.minimum(tl.maximum(xl.to(tl.float32), -limit), limit)  # clamp(-L, L): exact
        a = _qg_fwd(g.to(tl.float32)).to(tl.bfloat16)
        b = (l + offset).to(tl.bfloat16)
        y = (a.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16)
        tl.store(Y + row * F + cols, y, mask=mask)

    @triton.jit
    def _bwd(X, DY, DX, F, n_blocks_f, limit, offset, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        row = (pid // n_blocks_f).to(tl.int64)
        cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0).to(tl.float32)
        xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + row * F + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.minimum(xg, limit)
        l = tl.minimum(tl.maximum(xl, -limit), limit)
        a = _qg_fwd(g).to(tl.bfloat16).to(tl.float32)
        b = (l + offset).to(tl.bfloat16).to(tl.float32)
        ga = (dy * b).to(tl.bfloat16).to(tl.float32)  # MulBackward0: grad * other
        gb = (dy * a).to(tl.bfloat16)  # MulBackward0: grad * self; AddBackward passes it
        s = tl.sigmoid(g * 1.702)
        gg = (ga * s + (ga * g) * (s * (1.0 - s)) * 1.702).to(tl.bfloat16)
        zero = tl.zeros_like(gg)
        dxg = tl.where(xg < limit, gg, zero)  # torch clamp backward: strictly inside
        dxl = tl.where((xl > -limit) & (xl < limit), gb, zero)
        tl.store(DX + row * 2 * F + cols, dxg, mask=mask)
        tl.store(DX + row * 2 * F + F + cols, dxl, mask=mask)

    @triton.jit
    def _fwd_p(X, P, Y, F, n_blocks_f, limit, offset, BLOCK: tl.constexpr):
        # _fwd followed by probs_mul's (f32(y) * p) -> bf16, y kept in registers
        pid = tl.program_id(0)
        row = (pid // n_blocks_f).to(tl.int64)
        cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0)
        xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0)
        p = tl.load(P + row)
        g = tl.minimum(xg.to(tl.float32), limit).to(tl.bfloat16)
        l = tl.minimum(tl.maximum(xl.to(tl.float32), -limit), limit)
        a = _qg_fwd(g.to(tl.float32)).to(tl.bfloat16)
        b = (l + offset).to(tl.bfloat16)
        y = (a.to(tl.float32) * b.to(tl.float32)).to(tl.bfloat16)
        tl.store(Y + row * F + cols, (y.to(tl.float32) * p).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _bwd_p(X, P, DOUT, DX, PROD, F, n_blocks_f, limit, offset, BLOCK: tl.constexpr):
        # probs_mul's backward (dy = bf16(f32(dout) * p), prod = f32(dout) * f32(y)) followed by _bwd
        pid = tl.program_id(0)
        row = (pid // n_blocks_f).to(tl.int64)
        cols = (pid % n_blocks_f) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        xg = tl.load(X + row * 2 * F + cols, mask=mask, other=0.0).to(tl.float32)
        xl = tl.load(X + row * 2 * F + F + cols, mask=mask, other=0.0).to(tl.float32)
        do = tl.load(DOUT + row * F + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.load(P + row)
        g = tl.minimum(xg, limit)
        l = tl.minimum(tl.maximum(xl, -limit), limit)
        a = _qg_fwd(g).to(tl.bfloat16).to(tl.float32)
        b = (l + offset).to(tl.bfloat16).to(tl.float32)
        y = (a * b).to(tl.bfloat16).to(tl.float32)
        tl.store(PROD + row * F + cols, do * y, mask=mask)
        dy = (do * p).to(tl.bfloat16).to(tl.float32)
        ga = (dy * b).to(tl.bfloat16).to(tl.float32)
        gb = (dy * a).to(tl.bfloat16)
        s = tl.sigmoid(g * 1.702)
        gg = (ga * s + (ga * g) * (s * (1.0 - s)) * 1.702).to(tl.bfloat16)
        zero = tl.zeros_like(gg)
        dxg = tl.where(xg < limit, gg, zero)
        dxl = tl.where((xl > -limit) & (xl < limit), gb, zero)
        tl.store(DX + row * 2 * F + cols, dxg, mask=mask)
        tl.store(DX + row * 2 * F + F + cols, dxl, mask=mask)

    _kernels = (_fwd, _bwd, _fwd_p, _bwd_p)
    return _kernels


_BLOCK = 1024


class ExactClampedQuickGeGLU(torch.autograd.Function):
    """``quick_gelu(clamp(x_glu, max=L)) * (clamp(x_linear, -L, L) + offset)``, bitwise as eager."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float, offset: float) -> torch.Tensor:
        import triton

        x = x.contiguous()
        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        y = torch.empty(*x.shape[:-1], f, dtype=x.dtype, device=x.device)
        nbf = triton.cdiv(f, _BLOCK)
        fwd = _get_kernels()[0]
        if rows:
            fwd[(rows * nbf,)](x, y, f, nbf, float(limit), float(offset), BLOCK=_BLOCK)
        ctx.save_for_backward(x)
        ctx.limit, ctx.offset = float(limit), float(offset)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        import triton

        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        dx = torch.empty_like(x)
        nbf = triton.cdiv(f, _BLOCK)
        bwd = _get_kernels()[1]
        if rows:
            bwd[(rows * nbf,)](x, dy, dx, f, nbf, ctx.limit, ctx.offset, BLOCK=_BLOCK)
        return dx, None, None


class ExactClampedQuickGeGLUProbs(torch.autograd.Function):
    """``(geglu(x) * probs).to(bf16)``: ExactClampedQuickGeGLU then probs_mul, one kernel each way, bitwise."""

    @staticmethod
    def forward(ctx, x, probs, limit, offset):
        import triton

        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        y = torch.empty(rows, f, dtype=x.dtype, device=x.device)
        nbf = triton.cdiv(f, _BLOCK)
        if rows:
            _get_kernels()[2][(rows * nbf,)](x, probs, y, f, nbf, float(limit), float(offset), BLOCK=_BLOCK)
        ctx.save_for_backward(x, probs)
        ctx.limit, ctx.offset = float(limit), float(offset)
        return y

    @staticmethod
    def backward(ctx, dout):
        import triton

        x, probs = ctx.saved_tensors
        dout = dout.contiguous()
        f = x.shape[-1] // 2
        rows = x.numel() // (2 * f)
        dx = torch.empty_like(x)
        prod = torch.empty(rows, f, dtype=torch.float32, device=x.device)
        nbf = triton.cdiv(f, _BLOCK)
        if rows:
            _get_kernels()[3][(rows * nbf,)](x, probs, dout, dx, prod, f, nbf, ctx.limit, ctx.offset, BLOCK=_BLOCK)
        # autograd's MulBackward0 + sum_to_size(probs.shape): the same ATen reduction as probs_mul
        dp = prod.sum(dim=1, keepdim=True) if ctx.needs_input_grad[1] else None
        return dx, dp, None, None


def exact_clamped_quick_geglu_probs(x, probs, limit, offset):
    """GEGLU + probs_mul in one pass, or None when the inputs are not the fused case."""
    import os

    if (
        os.environ.get("PRIMUS_FUSED_GEGLU_PROBS", "1") == "0"
        or os.environ.get("PRIMUS_FUSED_PROBS_MUL", "1") == "0"
        or x.dim() != 2
        or not x.is_contiguous()
        or probs.dtype != torch.float32
        or probs.shape != (x.shape[0], 1)
        or not probs.is_contiguous()
    ):
        return None
    return ExactClampedQuickGeGLUProbs.apply(x, probs, limit, offset)


def eligible_input(x: torch.Tensor) -> bool:
    return x.dtype == torch.bfloat16 and x.is_cuda and x.shape[-1] % 2 == 0


def exact_clamped_quick_geglu(x: torch.Tensor, limit: float, offset: float) -> torch.Tensor:
    return ExactClampedQuickGeGLU.apply(x, limit, offset)
