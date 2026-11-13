###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Bit-exact ``h = residual + attn_out; y = pre_mlp_layernorm(h)`` in one pass each way.

Megatron runs the attention's bias-dropout-add (a compiled BF16 add when there is
no bias and no dropout) and then the Turbo RMSNorm of its result: two launches
and a re-read of ``h``. In the backward, ``h``'s two gradients (the MLP residual
path and the norm's dx) are summed by autograd in a third launch.

Here one kernel stores ``h`` (rounded to BF16, as the add does) and normalises
that rounded ``h``; one backward kernel computes the norm's dx, rounds it to
BF16 as Turbo does, and adds the residual-path gradient (one more BF16
rounding, as autograd's accumulate does). The kernels are Turbo's
``rmsnorm_fwd_kernel`` / ``rmsnorm_bwd_kernel_grid_stride`` verbatim around
those two extra elementwise steps, launched with Turbo's own configs (the
backward at the pinned ``PRIMUS_TURBO_RMSNORM_BWD_CONFIG``), and dgamma goes
through Turbo's ``_finalize_dgamma``. With the pin unset (autotuned Turbo) the
fusion stays off, so it never chooses a config Turbo would not.

``PRIMUS_FUSED_ADD_NORM=0`` turns it off.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ENABLED = os.environ.get("PRIMUS_FUSED_ADD_NORM", "1") != "0"

# id(norm module) -> (h, norm(h)) made by a bda for a norm that runs later in another module
# (the next layer's input norm inside linear_qkv, or the block's final norm)
PENDING: dict = {}
STATS = {"made": 0, "used": 0}


@triton.jit
def _add_norm_fwd(A_ptr, R_ptr, G_ptr, H_ptr, Y_ptr, RSTD_ptr, H: tl.constexpr, eps, BLOCK_H: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    a = tl.load(A_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R_ptr + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    hb = (a + r).to(H_ptr.dtype.element_ty)
    tl.store(H_ptr + row * H + offs, hb, mask=mask)
    # rmsnorm_fwd_kernel on the rounded h
    x = hb.to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * rstd * g).to(Y_ptr.dtype.element_ty)
    tl.store(Y_ptr + row * H + offs, y, mask=mask)
    tl.store(RSTD_ptr + row, rstd)


@triton.jit
def _norm_bwd_add_grid_stride(
    DY_ptr, X_ptr, G_ptr, RSTD_ptr, DRES_ptr, DX_ptr, DG_PART_ptr, stride_dgp,
    B, H: tl.constexpr, BLOCK_H: tl.constexpr, num_programs: tl.constexpr, HAS_DRES: tl.constexpr,
    ZERO_DRES: tl.constexpr = False,
):
    # ZERO_DRES: the residual gradient is an (unmaterialised) +0 tensor; dx + (+0) is applied as
    # -0 -> +0 (the only value an IEEE add of +0 changes), without loading anything.
    # rmsnorm_bwd_kernel_grid_stride, then dx(bf16) + dres. Same rows per program, same tl.sum tree and the
    # same dg_acc order (bitwise); the next row's loads are issued before the current row's reduction
    # (one wave per SIMD at grid = #CUs, so the plain loop was load-latency bound: 32 -> 21 us at 4096x6144).
    pid = tl.program_id(0)
    h_offs = tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    g = tl.load(G_ptr + h_offs, mask=h_mask, other=0.0).to(tl.float32)
    dg_acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    xn = tl.load(X_ptr + pid * H + h_offs, mask=h_mask & (pid < B), other=0.0)
    dyn = tl.load(DY_ptr + pid * H + h_offs, mask=h_mask & (pid < B), other=0.0)
    if HAS_DRES:
        drn = tl.load(DRES_ptr + pid * H + h_offs, mask=h_mask & (pid < B), other=0.0)
    for row in range(pid, B, num_programs):
        x = xn.to(tl.float32)
        dy = dyn.to(tl.float32)
        if HAS_DRES:
            dres = drn.to(tl.float32)
        nxt = row + num_programs
        xn = tl.load(X_ptr + nxt * H + h_offs, mask=h_mask & (nxt < B), other=0.0)
        dyn = tl.load(DY_ptr + nxt * H + h_offs, mask=h_mask & (nxt < B), other=0.0)
        if HAS_DRES:
            drn = tl.load(DRES_ptr + nxt * H + h_offs, mask=h_mask & (nxt < B), other=0.0)
        rstd = tl.load(RSTD_ptr + row).to(tl.float32)

        x_hat = x * rstd
        dxhat = dy * g
        m = tl.sum(dxhat * x_hat, axis=0) / H
        dx = (dxhat - x_hat * m) * rstd
        dxb = dx.to(DX_ptr.dtype.element_ty)
        if HAS_DRES:
            dxb = (dxb.to(tl.float32) + dres).to(DX_ptr.dtype.element_ty)
        if ZERO_DRES:
            dxf = dxb.to(tl.float32)
            dxb = tl.where(dxf == 0.0, 0.0, dxf).to(DX_ptr.dtype.element_ty)
        tl.store(DX_ptr + row * H + h_offs, dxb, mask=h_mask)

        dg_acc += dy * x_hat

    dgp_ptrs = DG_PART_ptr + pid * stride_dgp + h_offs
    tl.store(dgp_ptrs, dg_acc, mask=h_mask)


def _pinned():
    v = os.environ.get("PRIMUS_TURBO_RMSNORM_BWD_CONFIG")
    if not v:
        return None
    w, s = (int(t) for t in v.split(","))
    return w, s


class _AddNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, residual, gamma, eps):
        from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as R

        Hd = gamma.shape[0]
        B = a.numel() // Hd
        h = torch.empty_like(a)
        y = torch.empty_like(a)
        rstd = torch.empty(B, device=a.device, dtype=torch.float32)
        BLOCK_H, ROWS, num_warps, num_stages = R._pick_config(Hd, B)
        _add_norm_fwd[(B,)](a, residual, gamma, h, y, rstd, H=Hd, eps=eps, BLOCK_H=BLOCK_H,
                            num_warps=num_warps, num_stages=num_stages)
        ctx.save_for_backward(h, gamma, rstd)
        # an unused h (the final norm's) gets no materialised zero gradient; see ZERO_DRES
        ctx.set_materialize_grads(False)
        return h, y

    @staticmethod
    def backward(ctx, dh, dy):
        from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as R

        h, gamma, rstd = ctx.saved_tensors
        Hd = gamma.shape[0]
        B = h.numel() // Hd
        if dy is None:
            if dh is None:
                return None, None, None, None
            return dh, dh, None, None
        warps, stages = _pinned()
        mode, BLOCK_H, grid, _, _ = R._pick_bwd_config(Hd, B)
        dx = torch.empty_like(h)
        dg_partial = torch.empty(grid, Hd, device=h.device, dtype=torch.float32)
        dy = dy.contiguous()
        has = dh is not None
        dres = dh.contiguous() if has else dy
        _norm_bwd_add_grid_stride[(grid,)](
            dy, h, gamma, rstd, dres, dx, dg_partial, dg_partial.stride(0),
            B, H=Hd, BLOCK_H=BLOCK_H, num_programs=grid, HAS_DRES=has, ZERO_DRES=not has,
            num_warps=warps, num_stages=stages,
        )
        dg = R._finalize_dgamma(dg_partial, gamma.dtype)
        return dx, dx, dg, None


def eligible(norm, a, residual) -> bool:
    if not ENABLED or _pinned() is None:
        return False
    from primus.backends.megatron.core.extensions.primus_turbo import PrimusTurboRMSNorm

    if type(norm) is not PrimusTurboRMSNorm:
        return False
    try:
        from primus_turbo.pytorch.kernels.normalization import rmsnorm_impl as R
    except Exception:
        return False
    Hd = norm.weight.shape[0]
    if not (
        a.is_cuda and a.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16
        and a.shape == residual.shape and a.shape[-1] == Hd and a.is_contiguous() and residual.is_contiguous()
        and norm.weight.dtype == torch.bfloat16
    ):
        return False
    B = a.numel() // Hd
    return R._pick_config(Hd, B)[1] == 1 and R._pick_bwd_config(Hd, B)[0] == "grid"


def add_norm(norm, a, residual):
    """``(h, norm(h))`` with ``h = a + residual``, for :func:`eligible` inputs."""
    from primus.backends.megatron.core.extensions.step_weight_cache import plus_one

    gamma = plus_one(norm.weight) if norm.zero_centered_gamma else norm.weight
    return _AddNorm.apply(a, residual, gamma, norm.eps)


def take_pending(norm, x):
    """The pre-computed ``norm(x)`` if a bda made one for this very ``x``, else None."""
    p = PENDING.pop(id(norm), None)
    if p is not None and p[0] is x:
        STATS["used"] += 1
        return p[1]
    return None
