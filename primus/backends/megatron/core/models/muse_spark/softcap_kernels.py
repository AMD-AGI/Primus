###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Muse Spark logit soft-cap ``cap * tanh(x * m / cap)`` as Triton kernels, bitwise the eager ops.

The eager soft-cap (HF order) rounds each step to the logits' dtype::

    t1 = round(x * m);  t2 = round(t1 / cap);  t3 = round(tanh(t2));  z = round(t3 * cap)

and its autograd::

    g1 = round(g * cap);  g2 = round(g1 * round(1 - t3 * t3));  g3 = round(g2 / cap);  dx = round(g3 * m)

(``aten.tanh_backward`` rounds ``1 - t3 * t3`` before the product).

* :func:`fused_softcap`: one kernel per pass (the forward writes z, the backward recomputes t3 from x).
* :func:`softcap_cross_entropy`: the soft-cap folded into the deferred TE cross entropy
  (``te_patches/cross_entropy_deferred_patches.py``). The forward reads x and never writes z; the backward
  chains the cross-entropy gradient of z (rounded like the installed TE: ``round_once``, as the deferred cross
  entropy decides) through the soft-cap backward and writes dx over x. Loss and dx are bitwise those of the
  soft-cap followed by the deferred cross entropy: the reductions keep TE's geometry, and the forward reads z
  from a 65536-entry table built once per ``(m, cap)`` with the eager ops (indexed by the logit's bit pattern).
"""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from triton.language.extra import libdevice

from primus.backends.megatron.core.fusions import bitwise_check
from primus.backends.megatron.patches.te_patches.cross_entropy_deferred_patches import (
    IGNORE_INDEX,
    MAX_BLOCK,
    NUM_WARPS,
)

__all__ = [
    "fused_softcap",
    "softcap_cross_entropy",
    "softcap_cross_entropy_matches",
    "softcap_matches_eager",
    "softcap_table",
]

# elementwise passes: the launch config changes no value
ELEMENTWISE_BLOCK = 2048
SOFTCAP_NUM_WARPS = 4
CE_GRAD_NUM_WARPS = 16
# bf16 aten.tanh_backward is wrong past the last multiple of 4096 elements on gfx1250 (ROCm torch 2.14
# nightly), so the eager reference of the one-time check only sees whole 4096-element blocks
CHECK_BLOCK = 4096
CHECK_ELEMENTS = 64 * CHECK_BLOCK
# leading sequence positions the cross-entropy check takes from the real logits
CHECK_ROWS = 16

_TABLES = {}


def softcap_table(m: float, cap: float, device) -> torch.Tensor:
    """z over all 65536 bf16 bit patterns (index = the pattern as an unsigned 16-bit integer), eager ops."""
    key = (float(m), float(cap), torch.device(device))
    table = _TABLES.get(key)
    if table is None:
        v = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(torch.bfloat16)
        table = _TABLES[key] = ((v * m).div_(cap).tanh_() * cap).contiguous()
    return table


@triton.jit
def _softcap_z(x, m, cap):
    t1 = (x * m).to(tl.bfloat16).to(tl.float32)
    t2 = tl.math.div_rn(t1, cap).to(tl.bfloat16).to(tl.float32)
    t3 = libdevice.tanh(t2).to(tl.bfloat16).to(tl.float32)
    return t3, (t3 * cap).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _softcap_dx(g, t3, m, cap):
    g1 = (g * cap).to(tl.bfloat16).to(tl.float32)
    g2 = (g1 * (1.0 - t3 * t3).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g3 = tl.math.div_rn(g2, cap).to(tl.bfloat16).to(tl.float32)
    return (g3 * m).to(tl.bfloat16)


@triton.jit
def _softcap_fwd_kernel(X, Z, n, m, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    _, z = _softcap_z(x, m, cap)
    tl.store(Z + offs, z.to(tl.bfloat16), mask=mask)


@triton.jit
def _softcap_bwd_kernel(X, G, DX, n, m, cap, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    t3, _ = _softcap_z(x, m, cap)
    tl.store(DX + offs, _softcap_dx(g, t3, m, cap), mask=mask)


class _FusedSoftcap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, m, cap):
        x = x.contiguous()
        z = torch.empty_like(x)
        n = x.numel()
        with torch.cuda.device(x.device):
            _softcap_fwd_kernel[(triton.cdiv(n, ELEMENTWISE_BLOCK),)](
                x, z, n, m, cap, BLOCK=ELEMENTWISE_BLOCK, num_warps=SOFTCAP_NUM_WARPS
            )
        ctx.save_for_backward(x)
        ctx.m, ctx.cap = m, cap
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        g = g.contiguous()
        dx = torch.empty_like(x)
        n = x.numel()
        with torch.cuda.device(x.device):
            _softcap_bwd_kernel[(triton.cdiv(n, ELEMENTWISE_BLOCK),)](
                x, g, dx, n, ctx.m, ctx.cap, BLOCK=ELEMENTWISE_BLOCK, num_warps=SOFTCAP_NUM_WARPS
            )
        return dx, None, None


def fused_softcap(x: torch.Tensor, m: float, cap: float) -> torch.Tensor:
    """``cap * tanh(x * m / cap)`` of bf16 CUDA logits, bitwise the eager ops (forward and gradient)."""
    return _FusedSoftcap.apply(x, float(m), float(cap))


@triton.jit
def _softcap_ce_stats_kernel(XI, ZT, Y, STATS, LOSS, n_cols, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    XI += row * n_cols
    y = tl.load(Y + row)
    # the deferred cross entropy's fp32 online recurrence, exp and reduction geometry, over z
    mx = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < n_cols
        xb = tl.load(XI + offsets, mask, other=0)
        z = tl.load(ZT + (xb.to(tl.int32) & 0xFFFF)).to(tl.float32)
        z = tl.where(mask, z, float("-inf"))
        m_new = tl.maximum(mx, tl.max(z))
        d = d * tl.exp(mx - m_new) + tl.sum(tl.exp(z - m_new))
        mx = m_new
    tl.store(STATS + row * 2, mx)
    tl.store(STATS + row * 2 + 1, d)
    valid = (y >= 0) & (y < n_cols)
    xyb = tl.load(XI + y, valid, other=0)
    zy = tl.where(valid, tl.load(ZT + (xyb.to(tl.int32) & 0xFFFF)).to(tl.float32), float("-inf"))
    loss = -(zy - mx - tl.log(d))
    tl.store(LOSS + row, tl.where(y == IGNORE, 0.0, loss))


@triton.jit
def _softcap_ce_grad_kernel(
    X, Y, STATS, UPSTREAM, n_cols, m, cap, IGNORE: tl.constexpr, ROUND_ONCE: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    X += row * n_cols
    y = tl.load(Y + row)
    mx = tl.load(STATS + row * 2)
    d = tl.load(STATS + row * 2 + 1)
    upstream = tl.load(UPSTREAM + row).to(tl.float32)
    for i in range(0, n_cols, BLOCK):
        offsets = i + tl.arange(0, BLOCK)
        mask = offsets < n_cols
        x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
        t3, z = _softcap_z(x, m, cap)
        p = tl.exp(z - mx) / d
        # the deferred cross entropy's gradient roundings
        if ROUND_ONCE:
            target_grad = p - 1.0
        else:
            p = p.to(tl.bfloat16).to(tl.float32)
            target_grad = (p - 1.0).to(tl.bfloat16).to(tl.float32)
        unscaled = tl.where(offsets == y, target_grad, p)
        unscaled = tl.where(y == IGNORE, 0.0, unscaled)
        g = (unscaled * upstream).to(tl.bfloat16).to(tl.float32)
        tl.store(X + offsets, _softcap_dx(g, t3, m, cap), mask)


class _SoftcapCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, target, m, cap, round_once):
        vocab = x.shape[-1]
        rows = target.numel()
        block = min(MAX_BLOCK, triton.next_power_of_2(vocab))
        table = softcap_table(m, cap, x.device)
        stats = torch.empty((rows, 2), dtype=torch.float32, device=x.device)
        loss = torch.empty(target.shape, dtype=torch.float32, device=x.device)
        with torch.cuda.device(x.device):
            _softcap_ce_stats_kernel[(rows,)](
                x.view(torch.int16),
                table,
                target,
                stats,
                loss,
                vocab,
                IGNORE_INDEX,
                block,
                num_warps=NUM_WARPS,
            )
        ctx.save_for_backward(x.detach(), target, stats)
        ctx.block, ctx.m, ctx.cap, ctx.round_once = block, m, cap, bool(round_once)
        ctx.consumed = False
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        if ctx.consumed:
            raise RuntimeError("the soft-capped deferred cross entropy supports one backward per forward")
        ctx.consumed = True
        x, target, stats = ctx.saved_tensors
        grad_output = grad_output.float().contiguous()
        with torch.cuda.device(x.device):
            _softcap_ce_grad_kernel[(target.numel(),)](
                x, target, stats, grad_output, x.shape[-1], ctx.m, ctx.cap, IGNORE_INDEX, ctx.round_once,
                min(ELEMENTWISE_BLOCK, ctx.block), num_warps=CE_GRAD_NUM_WARPS,
            )  # fmt: skip
        torch.autograd.graph.increment_version(x)
        return x, None, None, None, None


def softcap_cross_entropy(
    x: torch.Tensor, target: torch.Tensor, m: float, cap: float, round_once: bool
) -> torch.Tensor:
    """``loss[t] = CE(cap * tanh(x[t] * m / cap), target[t])`` (fp32, IGNORE_INDEX -> 0) for contiguous bf16 logits
    ``x`` ``[..., V]`` and int64 ``target`` ``[...]``; the backward writes dx over ``x``. ``round_once``: the
    deferred cross entropy's gradient rounding for the installed TE."""
    return _SoftcapCrossEntropy.apply(x, target, float(m), float(cap), bool(round_once))


def softcap_matches_eager(logits: torch.Tensor, m: float, cap: float, eager) -> bool:
    """Whether :func:`fused_softcap` is bit-identical to ``eager(x, m, cap)`` in this process (decided once per
    ``(m, cap, dtype)``): on the leading whole 4096-element blocks of ``logits`` and wide random values."""

    def probe():
        flat = logits.detach().reshape(-1)[:CHECK_ELEMENTS]
        flat = flat[: flat.numel() // CHECK_BLOCK * CHECK_BLOCK]
        wide = (300 * bitwise_check.randn((CHECK_ELEMENTS,), logits.device)).to(logits.dtype)
        return (torch.cat([flat, wide]),)

    return bitwise_check.passed(
        "muse_spark_triton_softcap",
        lambda: bitwise_check.outputs_and_grads(lambda x: fused_softcap(x, m, cap), *probe()),
        lambda: bitwise_check.outputs_and_grads(lambda x: eager(x, m, cap), *probe()),
        key=(float(m), float(cap), logits.dtype),
    )


def softcap_cross_entropy_matches(
    logits: torch.Tensor,
    target: torch.Tensor,
    m: float,
    cap: float,
    round_once: bool,
    cross_entropy,
    tp_group,
) -> bool:
    """Whether :func:`softcap_cross_entropy` is bit-identical to :func:`fused_softcap` followed by
    ``cross_entropy(z, target, tp_group, False)`` (Megatron's ``te_parallel_cross_entropy``: the deferred one, which
    checks itself against TE) in this process, decided once per ``(vocab, m, cap, round_once)``: on the leading
    positions of the real logits and as many wide random ones with one ignored label."""

    def probe():
        x, y = logits[:CHECK_ROWS].detach(), target[:CHECK_ROWS]
        wide = (300 * bitwise_check.randn(x.shape, x.device)).to(x.dtype)
        y_ignored = y.clone()
        y_ignored.view(-1)[0] = IGNORE_INDEX
        return torch.cat([x, wide]), torch.cat([y, y_ignored])

    return bitwise_check.passed(
        "muse_spark_softcap_cross_entropy",
        lambda: bitwise_check.outputs_and_grads(
            lambda x, y: softcap_cross_entropy(x, y, m, cap, round_once), *probe()
        ),
        lambda: bitwise_check.outputs_and_grads(
            lambda x, y: cross_entropy(fused_softcap(x, m, cap), y, tp_group, False), *probe()
        ),
        key=(logits.shape[-1], float(m), float(cap), bool(round_once)),
    )
