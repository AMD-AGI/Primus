###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from Tencent HunyuanVideo-1.5 / HY-WorldPlay official implementation.

"""Primus process-group adapter for the transplanted WorldPlay model."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

_sp_group: Any = None


def set_sp_group(group: Any) -> None:
    global _sp_group
    _sp_group = group


def get_sp_world_size() -> int:
    return dist.get_world_size(_sp_group) if _sp_group is not None else 1


def get_sp_parallel_rank() -> int:
    return dist.get_rank(_sp_group) if _sp_group is not None else 0


def _all_to_all_4d(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    world_size = get_sp_world_size()
    if world_size == 1:
        return input_

    if scatter_dim == 2 and gather_dim == 1:
        sequence_lengths: list[int] = [0] * world_size
        dist.all_gather_object(sequence_lengths, input_.shape[1], group=_sp_group)
        gap = sequence_lengths[0] - sequence_lengths[-1]
        if gap:
            if get_sp_parallel_rank() == world_size - 1:
                input_ = F.pad(input_, (0, 0, 0, 0, 0, gap))
        batch, shard_sequence, heads, head_dim = input_.shape
        if heads % world_size:
            raise ValueError("attention head count must be divisible by sequence parallel size")
        shard_heads = heads // world_size
        input_t = (
            input_.reshape(batch, shard_sequence, world_size, shard_heads, head_dim)
            .transpose(0, 2)
            .contiguous()
        )
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=_sp_group)
        output = output.reshape(shard_sequence * world_size, batch, shard_heads, head_dim)
        output = output.transpose(0, 1).contiguous()
        return output[:, :-gap] if gap else output

    if scatter_dim == 1 and gather_dim == 2:
        batch, sequence, shard_heads, head_dim = input_.shape
        gap = (-sequence) % world_size
        if gap:
            input_ = F.pad(input_, (0, 0, 0, 0, 0, gap))
            sequence += gap
        shard_sequence = sequence // world_size
        input_t = (
            input_.reshape(batch, world_size, shard_sequence, shard_heads, head_dim)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
        )
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=_sp_group)
        output = output.reshape(shard_heads * world_size, shard_sequence, batch, head_dim)
        output = output.transpose(0, 2).contiguous()
        if gap and get_sp_parallel_rank() == world_size - 1:
            output = output[:, :-gap]
        return output

    raise ValueError("WorldPlay all-to-all only supports (2, 1) and (1, 2)")


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, scatter_dim: int, gather_dim: int):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        return _all_to_all_4d(input_, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _AllToAll4D.apply(
            grad_output, ctx.gather_dim, ctx.scatter_dim
        ), None, None


def sequence_model_parallel_all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> torch.Tensor:
    return _AllToAll4D.apply(input_, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int):
        dim %= input_.dim()
        ctx.dim = dim
        ctx.input_size = input_.shape[dim]
        world_size = get_sp_world_size()
        sizes: list[torch.Size] = [torch.Size()] * world_size
        dist.all_gather_object(sizes, input_.shape, group=_sp_group)
        max_size = max(size[dim] for size in sizes)
        pad = [0] * (2 * input_.dim())
        pad[2 * (input_.dim() - 1 - dim) + 1] = max_size - input_.shape[dim]
        padded = F.pad(input_, pad) if input_.shape[dim] != max_size else input_
        tensors = [torch.empty_like(padded) for _ in range(world_size)]
        dist.all_gather(tensors, padded.contiguous(), group=_sp_group)
        tensors = [
            tensor.narrow(dim, 0, size[dim])
            for tensor, size in zip(tensors, sizes)
        ]
        return torch.cat(tensors, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world_size = get_sp_world_size()
        sizes: list[int] = [0] * world_size
        dist.all_gather_object(sizes, ctx.input_size, group=_sp_group)
        return torch.split(grad_output, sizes, dim=ctx.dim)[get_sp_parallel_rank()], None


def sequence_model_parallel_all_gather(
    input_: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    if get_sp_world_size() == 1:
        return input_
    return _AllGather.apply(input_, dim)
