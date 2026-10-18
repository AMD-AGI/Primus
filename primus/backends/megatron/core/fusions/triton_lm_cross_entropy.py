###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Fused language-model cross entropy for tensor-parallel size 1 (Triton, one pinned launch config).

Logits [s, b, V] in bf16 / fp16 / fp32, labels [b, s], loss [b, s] in fp32 (Megatron's layout), fp32 math:

  forward   one pass per row: online max / sum-exp -> lse (saved), loss = lse - x[label].
  backward  one pass per row writing (exp(x - lse) - [i == label]) * g into the saved logits buffer, in place
            (the logits are dead after the loss, as in Transformer Engine's CE): no fp32 copy of the logits.

A label outside [0, V) contributes ``lse - max`` and no target term, like the target mask of Megatron's
vocab-parallel CE at TP = 1; it is never read out of bounds.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from primus.backends.megatron.core.fusions._notices import warn_fallback_once

# Launch config chosen on gfx1250 (MI455X) at the DeepSeek-V3 vocabulary (129280). Any value is correct (the
# column loop is masked); only speed depends on it. No in-process autotune.
BLOCK = 8192
NUM_WARPS = 8


@triton.jit
def ce_fwd_kernel(
    X, stride_x, Y, stride_yb, stride_ys, LOSS, stride_lb, stride_ls, LSE, B, n_cols, BLOCK: tl.constexpr
):
    """Row r of X = (s, b) = (r // B, r % B). One program per row."""
    row = tl.program_id(0).to(tl.int64)
    s_i = row // B
    b_i = row - s_i * B
    y = tl.load(Y + b_i * stride_yb + s_i * stride_ys).to(tl.int64)
    x_row = X + row * stride_x
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        x = tl.load(x_row + offs, mask=offs < n_cols, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(d)
    valid = (y >= 0) & (y < n_cols)
    y_safe = tl.where(valid, y, 0)
    x_y = tl.load(x_row + y_safe).to(tl.float32)
    x_y = tl.where(valid, x_y, m)
    tl.store(LOSS + b_i * stride_lb + s_i * stride_ls, lse - x_y)
    tl.store(LSE + row, lse)


@triton.jit
def ce_bwd_kernel(
    X, stride_x, Y, stride_yb, stride_ys, G, stride_gb, stride_gs, LSE, B, n_cols, BLOCK: tl.constexpr
):
    """In place: X[r, i] = (exp(X[r, i] - lse[r]) - [i == y_r]) * g[r]."""
    row = tl.program_id(0).to(tl.int64)
    s_i = row // B
    b_i = row - s_i * B
    y = tl.load(Y + b_i * stride_yb + s_i * stride_ys).to(tl.int64)
    g = tl.load(G + b_i * stride_gb + s_i * stride_gs).to(tl.float32)
    lse = tl.load(LSE + row)
    x_row = X + row * stride_x
    for i in range(0, n_cols, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(x - lse)
        p = tl.where(offs == y, p - 1.0, p)
        tl.store(x_row + offs, (p * g).to(X.dtype.element_ty), mask=mask)


def ineligible_reason(logits, labels) -> str | None:
    """Why the fused kernel must not run on these tensors (None = eligible); the caller checks the TP size."""
    if not (torch.is_tensor(logits) and logits.is_cuda):
        return "logits not a CUDA tensor"
    if logits.dim() != 3 or labels is None or labels.dim() != 2:
        return "logits / labels shape (want [s, b, V] / [b, s])"
    S, B, _ = logits.shape
    if tuple(labels.shape) != (B, S):
        return "labels shape != (b, s) of the logits"
    if logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return f"logits dtype {logits.dtype}"
    if not logits.is_contiguous():
        return "non-contiguous logits"
    if labels.dtype not in (torch.int64, torch.int32):
        return f"labels dtype {labels.dtype}"
    if labels.device != logits.device:
        return "labels on another device"
    return None


class DeferredGradCrossEntropy(torch.autograd.Function):
    """logits [s, b, V] (contiguous), labels [b, s] int -> loss [b, s] fp32 (Megatron's loss layout)."""

    @staticmethod
    def forward(ctx, logits, labels):
        S, B, V = logits.shape
        n_rows = S * B
        loss = torch.empty((B, S), dtype=torch.float32, device=logits.device)
        lse = torch.empty((n_rows,), dtype=torch.float32, device=logits.device)
        ce_fwd_kernel[(n_rows,)](
            logits,
            logits.stride(1),
            labels,
            labels.stride(0),
            labels.stride(1),
            loss,
            loss.stride(0),
            loss.stride(1),
            lse,
            B,
            V,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
        ctx.save_for_backward(logits, labels, lse)
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        logits, labels, lse = ctx.saved_tensors
        S, B, V = logits.shape
        g = grad_loss if grad_loss.dtype == torch.float32 else grad_loss.float()
        ce_bwd_kernel[(S * B,)](
            logits,
            logits.stride(1),
            labels,
            labels.stride(0),
            labels.stride(1),
            g,
            g.stride(0),
            g.stride(1),
            lse,
            B,
            V,
            BLOCK=BLOCK,
            num_warps=NUM_WARPS,
        )
        # The kernel overwrote logits through a raw pointer: bump its version so any other autograd node that
        # saved logits raises instead of reading the gradient.
        torch.autograd.graph.increment_version(logits)
        return logits, None


def fused_lm_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """[s, b, V] logits x [b, s] labels -> [b, s] fp32 loss (TP = 1); the backward overwrites logits."""
    return DeferredGradCrossEntropy.apply(logits, labels)


def _tp_size(module) -> int | None:
    tp = getattr(getattr(module, "pg_collection", None), "tp", None)
    if tp is not None:
        return tp.size()
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_world_size()
    except Exception:  # noqa: BLE001 - no parallel state: not eligible
        return None


def make_compute_language_model_loss(orig, key: str):
    """Drop-in for ``LanguageModule.compute_language_model_loss``: the fused CE for eligible calls, ``orig``
    otherwise (the reason logged once under the patch ``key``)."""

    @functools.wraps(orig)
    def compute_language_model_loss(self, labels, logits):
        why = ineligible_reason(logits, labels)
        if why is None:
            tp = _tp_size(self)
            if tp != 1:
                why = f"tensor-parallel size {tp}"
        if why is None:
            return fused_lm_cross_entropy(logits, labels)
        warn_fallback_once(key, why)
        return orig(self, labels, logits)

    return compute_language_model_loss
