###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Ulysses Sequence Parallel primitives.

Follows the Wan2.2 official design: a self-contained ``distributed_attention``
function that wraps all-to-all + attention + all-to-all, plus ``sp_split`` /
``sp_gather`` / ``sp_unpad`` for slicing model inputs and gathering outputs.

The model code only needs to:
  1. ``sp_split`` inputs before the blocks
  2. call ``distributed_attention`` instead of ``attention`` in self-attn
  3. ``sp_gather`` + ``sp_unpad`` the output after the head

Gradient scaling (÷ sp_size in sp_slice backward, × sp_size in sp_gather
backward) ensures correctness with FSDP2 gradient averaging across the
combined DP+SP mesh.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Core autograd primitives
# ---------------------------------------------------------------------------


def _require_divisible_dim(x: torch.Tensor, dim: int, divisor: int, op_name: str) -> None:
    size = x.shape[dim]
    if size % divisor != 0:
        raise ValueError(
            f"{op_name} requires tensor dimension {dim} (size={size}) to be divisible by "
            f"sequence parallel size {divisor}. Got shape={tuple(x.shape)}."
        )


class _SeqAllToAll(torch.autograd.Function):
    """All-to-all with autograd.  Backward is the inverse (swap dims)."""

    @staticmethod
    def forward(ctx, group, x, scatter_dim, gather_dim):
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        sp_size = dist.get_world_size(group)
        _require_divisible_dim(x, scatter_dim, sp_size, "_SeqAllToAll")
        input_list = [t.contiguous() for t in x.tensor_split(sp_size, scatter_dim)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(sp_size)]
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            _SeqAllToAll.apply(ctx.group, grad_output, ctx.gather_dim, ctx.scatter_dim),
            None,
            None,
        )


class _SliceWithGather(torch.autograd.Function):
    """Forward: slice.  Backward: all-gather ÷ sp_size (FSDP2 compat)."""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim = dim
        ctx.group = group
        sp_size = dist.get_world_size(group)
        sp_rank = dist.get_rank(group)
        ctx.sp_size = sp_size
        _require_divisible_dim(x, dim, sp_size, "_SliceWithGather")
        chunk_size = x.shape[dim] // sp_size
        return x.narrow(dim, sp_rank * chunk_size, chunk_size).contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        gathered = [torch.empty_like(grad_output) for _ in range(ctx.sp_size)]
        dist.all_gather(gathered, grad_output.contiguous(), group=ctx.group)
        return torch.cat(gathered, dim=ctx.dim) / ctx.sp_size, None, None


class _GatherWithSlice(torch.autograd.Function):
    """Forward: all-gather.  Backward: slice × sp_size (FSDP2 compat)."""

    @staticmethod
    def forward(ctx, x, dim, group):
        ctx.dim = dim
        ctx.group = group
        sp_size = dist.get_world_size(group)
        sp_rank = dist.get_rank(group)
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.chunk_size = x.shape[dim]
        gathered = [torch.empty_like(x) for _ in range(sp_size)]
        dist.all_gather(gathered, x.contiguous(), group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        chunk = grad_output.narrow(ctx.dim, ctx.sp_rank * ctx.chunk_size, ctx.chunk_size).contiguous()
        return chunk * ctx.sp_size, None, None


# ---------------------------------------------------------------------------
# Public API — high-level
# ---------------------------------------------------------------------------


def distributed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    group: dist.ProcessGroup,
    attention_fn: Callable,
    **attention_kwargs,
) -> torch.Tensor:
    """
    Ulysses distributed attention (DeepSpeed Ulysses, arXiv:2309.14509).

    Wraps ``attention_fn`` with all-to-all communication so that each rank
    computes attention on the **full sequence** with a **subset of heads**.

    Input / output shapes: ``[B, S/P, H, D]`` (sharded-seq, full-heads).

    Args:
        q, k, v:  query / key / value  ``[B, S/P, H, D]``
        group:    Ulysses SP process group
        attention_fn:  e.g. ``primus.backends.diffusion.attention.attention``
        **attention_kwargs:  forwarded to ``attention_fn(q=, k=, v=, ...)``
    """
    # scatter heads, gather seq: [B, S/P, H, D] → [B, S, H/P, D]
    q = _SeqAllToAll.apply(group, q, 2, 1)
    k = _SeqAllToAll.apply(group, k, 2, 1)
    v = _SeqAllToAll.apply(group, v, 2, 1)
    # standard attention on full sequence with partial heads
    out = attention_fn(q=q, k=k, v=v, **attention_kwargs)
    # scatter seq, gather heads: [B, S, H/P, D] → [B, S/P, H, D]
    return _SeqAllToAll.apply(group, out, 1, 2)


def sp_split(
    tensors: List[torch.Tensor],
    dim: int,
    group: dist.ProcessGroup,
) -> Tuple[List[torch.Tensor], int]:
    """
    Pad + slice multiple tensors for sequence parallelism.

    Returns ``(sliced_tensors, original_size)`` where *original_size* is
    the size along *dim* before padding (needed by :func:`sp_unpad`).
    """
    sp_size = dist.get_world_size(group)
    original_size: Optional[int] = None
    results: List[torch.Tensor] = []
    for t in tensors:
        t, orig = _sp_pad(t, dim, sp_size)
        if original_size is None:
            original_size = orig
        results.append(_SliceWithGather.apply(t, dim, group))
    assert original_size is not None
    return results, original_size


# ---------------------------------------------------------------------------
# Public API — low-level (used directly for gathering the output)
# ---------------------------------------------------------------------------


def sp_slice(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """Slice ``x`` along *dim* for this SP rank (with autograd). Low-level."""
    return _SliceWithGather.apply(x, dim, group)


def sp_gather(x: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
    """All-gather ``x`` along *dim* from all SP ranks (with autograd)."""
    return _GatherWithSlice.apply(x, dim, group)


def sp_unpad(x: torch.Tensor, dim: int, original_size: int) -> torch.Tensor:
    """Remove padding added by :func:`sp_split`."""
    if x.shape[dim] == original_size:
        return x
    return x.narrow(dim, 0, original_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sp_pad(x: torch.Tensor, dim: int, sp_size: int) -> Tuple[torch.Tensor, int]:
    """Pad *x* along *dim* so its size is divisible by *sp_size*."""
    original_size = x.shape[dim]
    remainder = original_size % sp_size
    if remainder == 0:
        return x, original_size
    pad_amount = sp_size - remainder
    pad_config = [0] * (2 * x.dim())
    pos = 2 * (x.dim() - 1 - dim)
    pad_config[pos + 1] = pad_amount
    return F.pad(x, pad_config), original_size
