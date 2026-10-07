"""FlyDSL flash attention at the FLUX shape, as opaque custom ops for torch.compile."""

from __future__ import annotations

import functools
import math

import torch
from primus_turbo.flydsl.attention.flash_attn_bwd import _DO_STRIDED, flux_attn_bwd
from primus_turbo.flydsl.attention.flash_attn_fwd import (
    flux_attn_fwd,
    flux_attn_fwd_split,
)

# cudagraph_unsafe: the kernels keep module-level state a graph capture would strand.
_custom_op = functools.partial(torch.library.custom_op, tags=(torch._C.Tag.cudagraph_unsafe,))

# The backward's tile plan is pinned to this shape; anything else goes to AITER.
_SEQ_LEN = 512
_HEADS = 24
_HEAD_DIM = 128


@_custom_op("flux_flydsl::attn_fwd", mutates_args=(), device_types="cuda")
def _attn_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return flux_attn_fwd(q, k, v, 1.0 / math.sqrt(q.shape[-1]))


@_attn_fwd.register_fake
def _(q, k, v):
    B, S, H, _ = q.shape
    return torch.empty(q.shape, dtype=q.dtype, device=q.device), q.new_empty((B, H, S), dtype=torch.float32)


@_custom_op("flux_flydsl::attn_bwd", mutates_args=(), device_types="cuda")
def _attn_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = flux_attn_bwd(dout, q, k, v, out, lse, 1.0 / math.sqrt(q.shape[-1]))
    return dq.reshape(q.shape), dk.reshape(k.shape), dv.reshape(v.shape)


@_attn_bwd.register_fake
def _(dout, q, k, v, out, lse):
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(k.shape, dtype=k.dtype, device=k.device),
        torch.empty(v.shape, dtype=v.dtype, device=v.device),
    )


def _setup_context(ctx, inputs, output):
    q, k, v = inputs
    out, lse = output
    ctx.save_for_backward(q, k, v, out, lse)


def _backward(ctx, dout, _dlse):
    q, k, v, out, lse = ctx.saved_tensors
    return _attn_bwd(dout if _DO_STRIDED else dout.contiguous(), q, k, v, out, lse)


_attn_fwd.register_autograd(_backward, setup_context=_setup_context)


# The double block's joint attention with O returned as its txt rows [0, split_lt) and img rows, each
# contiguous: neither the consumers' reshape copies nor the backward's dO join is needed.
@_custom_op("flux_flydsl::attn_fwd_split", mutates_args=(), device_types="cuda")
def _attn_fwd_split(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, split_lt: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return flux_attn_fwd_split(q, k, v, 1.0 / math.sqrt(q.shape[-1]), split_lt)


@_attn_fwd_split.register_fake
def _(q, k, v, split_lt):
    B, S, H, D = q.shape
    return (
        q.new_empty((B, split_lt, H, D)),
        q.new_empty((B, S - split_lt, H, D)),
        q.new_empty((B, H, S), dtype=torch.float32),
    )


@_custom_op("flux_flydsl::attn_bwd_split", mutates_args=(), device_types="cuda")
def _attn_bwd_split(
    dout_a: torch.Tensor,
    dout_b: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_a: torch.Tensor,
    out_b: torch.Tensor,
    lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = flux_attn_bwd(
        dout_a, q, k, v, out_a, lse, 1.0 / math.sqrt(q.shape[-1]), out_b=out_b, dout_b=dout_b
    )
    return dq.reshape(q.shape), dk.reshape(k.shape), dv.reshape(v.shape)


@_attn_bwd_split.register_fake
def _(dout_a, dout_b, q, k, v, out_a, out_b, lse):
    return (
        torch.empty(q.shape, dtype=q.dtype, device=q.device),
        torch.empty(k.shape, dtype=k.dtype, device=k.device),
        torch.empty(v.shape, dtype=v.dtype, device=v.device),
    )


def _setup_context_split(ctx, inputs, output):
    q, k, v, _ = inputs
    out_a, out_b, lse = output
    ctx.save_for_backward(q, k, v, out_a, out_b, lse)


def _backward_split(ctx, dout_a, dout_b, _dlse):
    q, k, v, out_a, out_b, lse = ctx.saved_tensors
    return (*_attn_bwd_split(dout_a.contiguous(), dout_b.contiguous(), q, k, v, out_a, out_b, lse), None)


_attn_fwd_split.register_autograd(_backward_split, setup_context=_setup_context_split)


def eligible(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size):
    return (
        q.is_cuda
        and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and q.dim() == 4
        and q.shape == k.shape == v.shape
        and q.shape[1] == _SEQ_LEN
        and q.shape[2] == _HEADS
        and q.shape[3] == _HEAD_DIM
        and q_lens is None
        and k_lens is None
        and dropout_p == 0.0
        and q_scale is None
        and not causal
        and tuple(window_size)[:2] == (-1, -1)
        and (softmax_scale is None or abs(softmax_scale - 1.0 / math.sqrt(_HEAD_DIM)) < 1e-6)
    )


def flydsl_flux_attention(q, k, v):
    out, _ = _attn_fwd(q.contiguous(), k.contiguous(), v.contiguous())
    return out


def flydsl_flux_attention_split(q, k, v, split_lt):
    out_a, out_b, _ = _attn_fwd_split(q.contiguous(), k.contiguous(), v.contiguous(), split_lt)
    return out_a, out_b
