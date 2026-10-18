###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton kernel for one fused optimizer step per fp32 main param: grad clip + AdamW + model-param write.

One pass over g, p, m, v instead of three (TE ``multi_tensor_scale`` clip, TE ``multi_tensor_adam``, the
main -> model param copy). The clip coefficient is applied in registers; the grads are read, never written.

Math mirrors Transformer Engine's ROCm ``custom_adam_kernel`` (ADAM_MODE_1, HAS_MASTER=false):
  m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g*g
  denom = sqrt(v * beta2_corr_inv) + eps
  p = p - step_size*(m/denom) - lr*decay*p
with step_size = lr/bias_correction1 and beta2_corr_inv = 1/bias_correction2 in fp32, computed on the host
exactly as TE does (``te_scalars``). Contractions use explicit fma at the same places as TE's kernel; division
and sqrt are correctly rounded.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

# Launch config chosen on gfx1250 (MI455X). Any value is correct (masked); only speed depends on it. No
# in-process autotune.
BLOCK = 2048
NUM_WARPS = 8
# Elements per launch: the host cuts larger tensors so that in-kernel offsets fit in int32 (fp32 byte offsets
# < 2^30).
SLICE = 1 << 28


@triton.jit
def _adamw_clip_kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    v_ptr,
    out_ptr,
    n,
    clip,
    beta1,
    beta2,
    omb1,
    omb2,
    eps,
    step_size,
    bc2_inv,
    lr_decay,
    WRITE_OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    g = g * clip
    # Explicit fma placement reproduces TE's ROCm Adam kernel bit for bit.
    m = tl.fma(omb1, g, beta1 * m)
    v = tl.fma(omb2 * g, g, beta2 * v)
    denom = tl.sqrt_rn(v * bc2_inv) + eps
    q = tl.div_rn(m, denom)
    pn = tl.fma(-step_size, q, p)
    pn = tl.fma(-lr_decay, p, pn)
    tl.store(p_ptr + offs, pn, mask=mask)
    tl.store(m_ptr + offs, m, mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)
    if WRITE_OUT:
        tl.store(out_ptr + offs, pn.to(out_ptr.dtype.element_ty), mask=mask)


def te_scalars(lr, beta1, beta2, eps, weight_decay, step, bias_correction=True):
    """fp32 scalars exactly as TE's multi_tensor_adam_cuda_custom computes them on the host."""
    f = np.float32
    b1, b2, lr32, wd32 = f(beta1), f(beta2), f(lr), f(weight_decay)
    if bias_correction:
        bc1 = f(1.0 - float(b1) ** int(step))  # std::pow(float, int) -> double, then to float
        bc2 = f(1.0 - float(b2) ** int(step))
    else:
        bc1 = bc2 = f(1.0)
    return dict(
        beta1=float(b1),
        beta2=float(b2),
        omb1=float(f(1.0) - b1),
        omb2=float(f(1.0) - b2),
        eps=float(f(eps)),
        step_size=float(lr32 / bc1),
        bc2_inv=float(f(1.0) / bc2),
        lr_decay=float(lr32 * wd32),
    )


def _check(p, g, m, v, out):
    n = p.numel()
    for t, dt in ((p, torch.float32), (g, torch.float32), (m, torch.float32), (v, torch.float32)):
        if t.dtype != dt or not t.is_contiguous() or t.numel() != n or t.device != p.device:
            raise ValueError(f"fused AdamW wants contiguous fp32 tensors of {n} elements")
    if out is not None and (
        not out.is_contiguous()
        or out.numel() != n
        or out.device != p.device
        or out.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError("fused AdamW out must be a contiguous bf16/fp16/fp32 tensor of the same size")


def adamw_clip_(p, g, m, v, out, *, clip, scalars, check=True):
    """In place: p, m, v updated with g * clip; out (or None) gets p cast to its dtype. g is only read.

    ``scalars`` comes from ``te_scalars``; ``check=False`` skips the tensor validation (done by the caller).
    """
    if check:
        _check(p, g, m, v, out)
    n = p.numel()
    pf, gf, mf, vf = p.view(-1), g.view(-1), m.view(-1), v.view(-1)
    of = out.view(-1) if out is not None else pf
    for s in range(0, n, SLICE):
        e = min(n, s + SLICE)
        ns = e - s
        grid = (triton.cdiv(ns, BLOCK),)
        _adamw_clip_kernel[grid](
            pf[s:e],
            gf[s:e],
            mf[s:e],
            vf[s:e],
            of[s:e],
            ns,
            float(np.float32(clip)),
            scalars["beta1"],
            scalars["beta2"],
            scalars["omb1"],
            scalars["omb2"],
            scalars["eps"],
            scalars["step_size"],
            scalars["bc2_inv"],
            scalars["lr_decay"],
            WRITE_OUT=out is not None,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
