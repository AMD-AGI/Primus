###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Post-combine BF16 FC2 bias for four selected experts on an EP group."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _count_nonfinite(BIAS, COUNTS, H: tl.constexpr, E: tl.constexpr, BE: tl.constexpr, BH: tl.constexpr):
    col = tl.program_id(0) * BH + tl.arange(0, BH)
    expert = tl.arange(0, BE)
    bias = tl.load(
        BIAS + expert[:, None] * H + col[None, :], mask=(expert[:, None] < E) & (col[None, :] < H), other=0.0
    ).to(tl.float32)
    count = tl.sum((~(tl.abs(bias) < float("inf"))).to(tl.int32), axis=0)
    tl.store(COUNTS + col, count, mask=col < H)


@triton.jit
def _add_bias(Y, IDS, PROBS, BIAS, COUNTS, OUT, H: tl.constexpr, E: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    valid = col < H
    acc = tl.full((BH,), 0, tl.float32)
    seen_nonfinite = tl.full((BH,), 0, tl.int32)
    for slot in tl.static_range(4):
        expert = tl.load(IDS + row * 4 + slot).to(tl.int32)
        weight = tl.load(PROBS + row * E + expert)
        first = tl.full((), True, tl.int1)
        for earlier in tl.static_range(slot):
            prev = tl.load(IDS + row * 4 + earlier).to(tl.int32)
            first &= expert != prev
        bias = tl.load(BIAS + expert * H + col, mask=valid, other=0.0).to(tl.float32)
        acc = tl.where(first, tl.fma(weight, bias, acc), acc)
        seen_nonfinite += (first & ~(tl.abs(bias) < float("inf"))).to(tl.int32)
    # The dense reference also multiplies an unselected expert's bias by zero.
    # Keep its NaN propagation when that bias is infinite or NaN.
    total_nonfinite = tl.load(COUNTS + col, mask=valid, other=0)
    acc = tl.where(seen_nonfinite < total_nonfinite, float("nan"), acc)
    y = tl.load(Y + row * H + col, mask=valid, other=0.0).to(tl.float32)
    tl.store(OUT + row * H + col, y + acc, mask=valid)


@triton.jit
def _routing_gradient(DY, IDS, BIAS, DP, H: tl.constexpr, BH: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BH)
    dy = tl.load(DY + row * H + col, mask=col < H, other=0.0).to(tl.float32)
    for slot in tl.static_range(4):
        expert = tl.load(IDS + row * 4 + slot).to(tl.int32)
        bias = tl.load(BIAS + expert * H + col, mask=col < H, other=0.0).to(tl.float32)
        tl.store(DP + row * 4 + slot, tl.sum(dy * bias, axis=0))


def supports_sparse_expert_bias(x, indices, probabilities, bias, ep_group):
    """Only enable the qualified training geometry and FP32 accumulation policy."""
    return (
        x.is_cuda
        and torch.version.hip is not None
        and x.ndim == 2
        and x.shape[1] == 2880
        and 0 < x.shape[0] <= (2**31 - 1) // 2880
        and x.dtype == torch.bfloat16
        and bias is not None
        and bias.dtype == torch.bfloat16
        and bias.shape == (16, 2880)
        and ep_group is not None
        and ep_group.size() == 8
        and indices.shape == (x.shape[0], 4)
        and indices.dtype == torch.int64
        and probabilities.shape == indices.shape
        and probabilities.dtype == torch.float32
        and x.device == indices.device == probabilities.device == bias.device
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.is_autocast_enabled()
        and x.device.index == torch.cuda.current_device()
        and not torch.compiler.is_compiling()
        and not torch.cuda.is_current_stream_capturing()
        and getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(":")[0] == "gfx950"
    )


class SparseExpertBias(torch.autograd.Function):
    """Use four selected BF16 bias rows and leave the bias gradient reduction in FP32."""

    @staticmethod
    def forward(ctx, y, indices, probabilities, local_bias, group):
        tokens, hidden = y.shape
        experts = local_bias.shape[0] * group.size()
        bias = local_bias.new_empty((experts, hidden))
        dist.all_gather_into_tensor(bias, local_bias.contiguous(), group=group)
        dense = torch.zeros((tokens, experts), dtype=torch.float32, device=y.device)
        # Preserve the effective value of repeated IDs from the original scatter.
        dense.scatter_(1, indices, probabilities)
        indices = indices.contiguous()
        result = torch.empty_like(y, memory_format=torch.contiguous_format)
        counts = torch.empty(hidden, dtype=torch.int32, device=y.device)
        _count_nonfinite[(triton.cdiv(hidden, 32),)](
            bias, counts, hidden, experts, triton.next_power_of_2(experts), 32, num_warps=4
        )
        _add_bias[(tokens, triton.cdiv(hidden, 512))](
            y.contiguous(), indices, dense, bias, counts, result, hidden, experts, 512, num_warps=4
        )
        ctx.save_for_backward(indices, dense, bias)
        ctx.group = group
        return result

    @staticmethod
    @torch.no_grad()
    def backward(ctx, dy):
        indices, probabilities, bias = ctx.saved_tensors
        experts, hidden = bias.shape
        dprob = torch.empty_like(indices, dtype=torch.float32)
        _routing_gradient[(indices.shape[0],)](
            dy.contiguous(), indices, bias, dprob, hidden, triton.next_power_of_2(hidden), num_warps=8
        )
        dbias_full = probabilities.t() @ dy.float()
        dbias = dbias_full.new_empty((experts // ctx.group.size(), hidden))
        dist.reduce_scatter_tensor(dbias, dbias_full, group=ctx.group)
        return dy, None, dprob, dbias.to(bias.dtype), None
