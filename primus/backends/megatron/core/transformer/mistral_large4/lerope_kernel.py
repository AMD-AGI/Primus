###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused LeRoPE rotary kernel for Mistral Large 4 (MLA queries / keys and the DSA indexer).

LeRoPE learns a per-frequency log multiplier ``alpha``: the inverse frequencies are
``inv_freq = yarn_inv_freq * exp(alpha)``. Megatron's fused MLA RoPE kernels read a cached
cos / sin table, which cannot change from step to step and has no gradient to ``alpha``. This op
computes the angles from the positions and the current ``inv_freq`` inside the kernel, applies
the DeepSeek-interleaved rotation (pairs ``(x[2i], x[2i+1])`` rotated, output ``[evens | odds]``,
as ``apply_rotary_pos_emb(..., mla_rotary_interleaved=True)``), and its backward returns the
gradient of ``x`` and, only when ``inv_freq`` requires a gradient, of ``inv_freq`` (autograd
chains it to ``alpha``). In the model ``alpha`` is a frozen buffer (as in the reference), so
training computes only the gradient of ``x``; the frequency gradient is there for a run that
makes ``alpha`` trainable, and is tested.

* ``lerope_rotary(x, positions, inv_freq)``: ``x`` ``[s, ..., 2F]`` (rope slice), ``positions``
  ``[s]`` (integer positions of the rows), ``inv_freq`` ``[F]`` fp32. Output: like ``x``.
* ``lerope_rotary_reference``: the eager (unfused) path the model runs without the kernel,
  ``positions * inv_freq`` -> ``cat(f, f)`` -> Megatron's ``_apply_rotary_pos_emb_bshd``.

New math: none beyond the reference. The kernel computes cos / sin in fp32 and rounds the output
once to the input dtype (the eager path rounds cos / sin to the input dtype first). The gradient
of ``inv_freq`` is reduced without atomics: each program writes its partial sum, and the partials
are summed in a fixed order, so the backward is deterministic.
"""

from __future__ import annotations

import torch
from megatron.core.models.common.embeddings.rope_utils import _apply_rotary_pos_emb_bshd

try:  # Triton is in the GPU images; CPU hosts fall back to the eager path
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    triton = None
    tl = None
    HAVE_TRITON = False

__all__ = ["lerope_rotary", "lerope_rotary_reference", "fused_lerope_available"]

BLOCK_ROWS = 16


def lerope_rotary_reference(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Eager LeRoPE rotation: the model's unfused path (``MistralLarge4RotaryEmbedding.freqs`` and
    Megatron's interleaved ``_apply_rotary_pos_emb_bshd``). Differentiable in ``x`` and
    ``inv_freq``."""
    f = positions.float()[:, None] * inv_freq.float()[None, :]  # [s, F]
    freqs = torch.cat((f, f), dim=-1)
    freqs = freqs.view(freqs.size(0), *([1] * (x.dim() - 2)), freqs.size(-1))
    return _apply_rotary_pos_emb_bshd(x, freqs, rotary_interleaved=False, mla_rotary_interleaved=True)


if HAVE_TRITON:

    @triton.jit
    def _lerope_fwd_kernel(
        X, Y, POS, INV, rows_per_pos, F: tl.constexpr, BLOCK_F: tl.constexpr, BLOCK_R: tl.constexpr
    ):
        s = tl.program_id(0).to(tl.int64)  # 64-bit offsets: long sequences x heads x 64 pass 2**31
        rb = tl.program_id(1)
        f = tl.arange(0, BLOCK_F)
        fm = f < F
        pos = tl.load(POS + s).to(tl.float32)
        inv = tl.load(INV + f, mask=fm, other=0.0)
        ang = pos * inv
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        rm = r < rows_per_pos
        m = rm[:, None] & fm[None, :]
        base = (s * rows_per_pos + r)[:, None] * (2 * F)
        x0 = tl.load(X + base + 2 * f[None, :], mask=m, other=0.0).to(tl.float32)
        x1 = tl.load(X + base + 2 * f[None, :] + 1, mask=m, other=0.0).to(tl.float32)
        ye = x0 * c - x1 * sn
        yo = x1 * c + x0 * sn
        tl.store(Y + base + f[None, :], ye.to(Y.dtype.element_ty), mask=m)
        tl.store(Y + base + F + f[None, :], yo.to(Y.dtype.element_ty), mask=m)

    @triton.jit
    def _lerope_bwd_kernel(
        DY,
        X,
        DX,
        DINV_PART,
        POS,
        INV,
        rows_per_pos,
        n_rb,
        F: tl.constexpr,
        BLOCK_F: tl.constexpr,
        BLOCK_R: tl.constexpr,
        NEED_DINV: tl.constexpr,
    ):
        s = tl.program_id(0).to(tl.int64)  # 64-bit offsets: long sequences x heads x 64 pass 2**31
        rb = tl.program_id(1)
        f = tl.arange(0, BLOCK_F)
        fm = f < F
        pos = tl.load(POS + s).to(tl.float32)
        inv = tl.load(INV + f, mask=fm, other=0.0)
        ang = pos * inv
        c = tl.cos(ang)[None, :]
        sn = tl.sin(ang)[None, :]
        r = rb * BLOCK_R + tl.arange(0, BLOCK_R)
        rm = r < rows_per_pos
        m = rm[:, None] & fm[None, :]
        base = (s * rows_per_pos + r)[:, None] * (2 * F)
        dye = tl.load(DY + base + f[None, :], mask=m, other=0.0).to(tl.float32)
        dyo = tl.load(DY + base + F + f[None, :], mask=m, other=0.0).to(tl.float32)
        # x = R^T y: dx0 = dye cos + dyo sin, dx1 = -dye sin + dyo cos
        dx0 = dye * c + dyo * sn
        dx1 = dyo * c - dye * sn
        tl.store(DX + base + 2 * f[None, :], dx0.to(DX.dtype.element_ty), mask=m)
        tl.store(DX + base + 2 * f[None, :] + 1, dx1.to(DX.dtype.element_ty), mask=m)
        if NEED_DINV:
            x0 = tl.load(X + base + 2 * f[None, :], mask=m, other=0.0).to(tl.float32)
            x1 = tl.load(X + base + 2 * f[None, :] + 1, mask=m, other=0.0).to(tl.float32)
            # d angle = dye * (-x0 sin - x1 cos) + dyo * (x0 cos - x1 sin); d inv = pos * d angle
            dang = dye * (-x0 * sn - x1 * c) + dyo * (x0 * c - x1 * sn)
            part = tl.sum(tl.where(m, dang, 0.0), axis=0) * pos
            tl.store(DINV_PART + (s * n_rb + rb) * BLOCK_F + f, part, mask=fm)


def fused_lerope_available(x: torch.Tensor) -> bool:
    """Whether ``lerope_rotary`` runs the Triton kernel for ``x`` (a GPU tensor, Triton present)."""
    return HAVE_TRITON and x.is_cuda


def _shape(x):
    s, width = x.size(0), x.size(-1)
    if width % 2:
        raise ValueError(f"LeRoPE: the rope slice width must be even, got {width}")
    return s, x.numel() // (s * width), width // 2


class _LeRoPEFn(torch.autograd.Function):
    # called only by lerope_rotary, which sends an empty input to the eager reference
    @staticmethod
    def forward(ctx, x, positions, inv_freq):
        s, rows, F = _shape(x)
        if inv_freq.numel() != F:
            raise ValueError(f"LeRoPE: {inv_freq.numel()} inverse frequencies for a rope width of {2 * F}")
        xc = x.contiguous()
        pos = positions.to(device=x.device).contiguous()
        inv = inv_freq.to(device=x.device, dtype=torch.float32).contiguous()
        y = torch.empty_like(xc)
        n_rb = triton.cdiv(rows, BLOCK_ROWS)
        _lerope_fwd_kernel[(s, n_rb)](
            xc, y, pos, inv, rows, F=F, BLOCK_F=triton.next_power_of_2(F), BLOCK_R=BLOCK_ROWS
        )
        ctx.save_for_backward(xc, pos, inv)
        ctx.inv_dtype = inv_freq.dtype
        ctx.inv_device = inv_freq.device
        return y

    @staticmethod
    def backward(ctx, dy):
        xc, pos, inv = ctx.saved_tensors
        s, rows, F = _shape(xc)
        need_dinv = ctx.needs_input_grad[2]
        dyc = dy.contiguous()
        dx = torch.empty_like(xc)
        block_f = triton.next_power_of_2(F)
        n_rb = triton.cdiv(rows, BLOCK_ROWS)
        # one partial row per program, summed in a fixed order below (no atomics: deterministic)
        part = torch.zeros((s * n_rb if need_dinv else 1, block_f), device=xc.device, dtype=torch.float32)
        _lerope_bwd_kernel[(s, n_rb)](
            dyc,
            xc,
            dx,
            part,
            pos,
            inv,
            rows,
            n_rb,
            F=F,
            BLOCK_F=block_f,
            BLOCK_R=BLOCK_ROWS,
            NEED_DINV=need_dinv,
        )
        dinv = None
        if need_dinv:
            dinv = part.sum(0)[:F].to(device=ctx.inv_device, dtype=ctx.inv_dtype)
        return dx, None, dinv


def lerope_rotary(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """LeRoPE rotation of ``x`` ``[s, ..., 2F]`` at ``positions`` ``[s]`` with ``inv_freq`` ``[F]``
    (fp32, ``yarn_inv_freq * exp(alpha)``): the fused kernel on a GPU, the eager reference
    elsewhere. Both are differentiable in ``x`` and ``inv_freq``."""
    if positions.dim() != 1 or positions.numel() != x.size(0):
        raise ValueError(
            f"LeRoPE: positions {tuple(positions.shape)} do not match the rows of x {tuple(x.shape)}"
        )
    if not fused_lerope_available(x) or x.numel() == 0:
        # an empty input (no rows) has nothing to rotate: the eager path returns it as it is
        return lerope_rotary_reference(x, positions, inv_freq)
    return _LeRoPEFn.apply(x, positions, inv_freq)
