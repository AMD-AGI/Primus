###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Cross entropy for the 1-GPU proxy (g90mem, 2026-11-19): one logits read in forward, one read + one bf16 write
in backward.

TE's Triton path (``parallel_cross_entropy``) reads the logits twice in forward and writes an fp32 gradient
there ([T, V] fp32 = 2.1 GB at T=4096, V=129408), multiplies it by grad_output in backward (2.1 GB read +
write) and autograd then casts it to bf16 (another 3.2 GB): ~1.39 ms on MI455X for ~0.18 ms of compulsory traffic.

Here: forward = split-V partial (max, sum exp) + target logit per row, then a per-row combine (loss, lse);
backward = ``(exp(x - lse) - onehot(y)) * grad_out`` written straight in the logits dtype.
Same math as TE for label_smoothing == 0 (fp32 internally); world size 1 only, otherwise TE is called.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

CE_CFG = dict(fwd_block=16384, fwd_warps=8, bwd_block=8192, bwd_warps=8)


@triton.jit
def _ce_fwd_part_kernel(X, Y, PM, PD, PXY, V, stride_x, S2, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.program_id(1)
    cols = sp * BLOCK + tl.arange(0, BLOCK)
    msk = cols < V
    xb = X + row.to(tl.int64) * stride_x
    x = tl.load(xb + cols, mask=msk, other=-float("inf")).to(tl.float32)
    m = tl.max(x, axis=0)
    d = tl.sum(tl.exp(x - m), axis=0)
    tl.store(PM + row * S2 + sp, m)
    tl.store(PD + row * S2 + sp, d)
    y = tl.load(Y + row)
    if (y >= sp * BLOCK) & (y < sp * BLOCK + BLOCK) & (y != IGNORE):
        tl.store(PXY + row, tl.load(xb + y).to(tl.float32))


@triton.jit
def _ce_fwd_combine_kernel(PM, PD, PXY, Y, LOSS, LSE, T, S2: tl.constexpr, IGNORE: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < T
    s = tl.arange(0, S2)
    pm = tl.load(PM + rows[:, None] * S2 + s[None, :], mask=rm[:, None], other=-float("inf"))
    pd = tl.load(PD + rows[:, None] * S2 + s[None, :], mask=rm[:, None], other=0.0)
    m = tl.max(pm, axis=1)
    d = tl.sum(tl.where(pd > 0, pd * tl.exp(pm - m[:, None]), 0.0), axis=1)
    lse = m + tl.log(d)
    y = tl.load(Y + rows, mask=rm, other=IGNORE)
    valid = rm & (y != IGNORE)
    xy = tl.load(PXY + rows, mask=valid, other=0.0)
    tl.store(LOSS + rows, tl.where(valid, lse - xy, 0.0), mask=rm)
    tl.store(LSE + rows, lse, mask=rm)


@triton.jit
def _ce_bwd_kernel(X, Y, LSE, GO, DX, V, stride_x, stride_go, IGNORE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.program_id(1)
    cols = sp * BLOCK + tl.arange(0, BLOCK)
    msk = cols < V
    y = tl.load(Y + row)
    go = tl.load(GO + row * stride_go)
    go = tl.where(y != IGNORE, go, 0.0)
    lse = tl.load(LSE + row)
    r64 = row.to(tl.int64)
    x = tl.load(X + r64 * stride_x + cols, mask=msk, other=0.0).to(tl.float32)
    g = (tl.exp(x - lse) - tl.where(cols == y, 1.0, 0.0)) * go
    tl.store(DX + r64 * V + cols, g.to(DX.dtype.element_ty), mask=msk)


class PrimusCrossEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, target, ignore_idx):  # type: ignore[override]
        V = inp.shape[-1]
        x2d = inp.reshape(-1, V)
        if x2d.stride(-1) != 1:
            x2d = x2d.contiguous()
        y = target.reshape(-1).contiguous()
        T = x2d.shape[0]
        f = CE_CFG
        blk = f["fwd_block"]
        S = triton.cdiv(V, blk)
        S2 = triton.next_power_of_2(S)
        dev = inp.device
        pm = torch.full((T, S2), -float("inf"), dtype=torch.float32, device=dev)
        pd = torch.zeros((T, S2), dtype=torch.float32, device=dev)
        pxy = torch.empty((T,), dtype=torch.float32, device=dev)
        _ce_fwd_part_kernel[(T, S)](x2d, y, pm, pd, pxy, V, x2d.stride(0), S2, IGNORE=ignore_idx, BLOCK=blk,
                                    num_warps=f["fwd_warps"])
        loss = torch.empty((T,), dtype=torch.float32, device=dev)
        lse = torch.empty((T,), dtype=torch.float32, device=dev)
        _ce_fwd_combine_kernel[(triton.cdiv(T, 64),)](pm, pd, pxy, y, loss, lse, T, S2=S2, IGNORE=ignore_idx, BR=64,
                                                       num_warps=4)
        ctx.save_for_backward(x2d, y, lse)
        ctx.ignore_idx = ignore_idx
        ctx.inp_shape = inp.shape
        return loss.view(target.shape)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x2d, y, lse = ctx.saved_tensors
        T, V = x2d.shape
        go = grad_out.reshape(-1)
        if go.numel() == 1:
            go = go.expand(T)
        f = CE_CFG
        dx = torch.empty((T, V), dtype=x2d.dtype, device=x2d.device)
        _ce_bwd_kernel[(T, triton.cdiv(V, f["bwd_block"]))](
            x2d, y, lse, go, dx, V, x2d.stride(0), go.stride(0), IGNORE=ctx.ignore_idx, BLOCK=f["bwd_block"],
            num_warps=f["bwd_warps"],
        )
        return dx.view(ctx.inp_shape), None, None


def primus_ce_enabled() -> bool:
    return os.environ.get("PRIMUS_FUSED_CE", "0") == "1"


def primus_parallel_cross_entropy(inp, target, label_smoothing=0.0, reduce_loss=False, dist_process_group=None,
                                  ignore_idx=-100):
    """Drop-in for ``transformer_engine.pytorch.cross_entropy.parallel_cross_entropy`` (TP size 1, no smoothing)."""
    world = 1 if dist_process_group is None else torch.distributed.get_world_size(dist_process_group)
    if label_smoothing != 0.0 or world != 1 or not inp.is_cuda:
        from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

        return parallel_cross_entropy(inp, target, label_smoothing, reduce_loss, dist_process_group, ignore_idx)
    loss = PrimusCrossEntropyFn.apply(inp, target, ignore_idx)
    if reduce_loss:
        n = (target != ignore_idx).sum().clamp(min=1)
        return loss.sum() / n
    return loss


__all__ = ["primus_parallel_cross_entropy", "primus_ce_enabled", "PrimusCrossEntropyFn", "CE_CFG"]
