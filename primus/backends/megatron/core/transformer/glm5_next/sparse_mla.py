###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Absorbed sparse MLA for GLM-5.3 DSA layers.

Every query head attends over the same per-token selection of rows of one
shared latent (``kv_lora_rank`` wide) that serves as both K and V. That is the
exact contract of the DeepSeek-V4 ``triton_v2`` sparse-MLA kernel pair
(``v4_attention_kernels/_triton_v2``), so it is reused here with the sink off.
Those kernels take ``d_qk = kv_lora_rank + 64`` with a zero tail block, which
they skip (``HAS_ROPE=False``); the tail is appended here and dropped again.
"""

from __future__ import annotations

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
    sparse_mla_bwd_v4_triton,
    sparse_mla_fwd_v4_triton,
)

__all__ = ["sparse_mla", "sparse_mla_torch"]

_TAIL = 64


def _pad_tail(x: torch.Tensor) -> torch.Tensor:
    out = x.new_zeros(*x.shape[:-1], x.shape[-1] + _TAIL)
    out[..., : x.shape[-1]] = x
    return out


class _SparseMLAFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float):
        # q [T, H, D] bf16, kv [T, D] bf16, indices [T, K] int32 (-1 = skip)
        T, H, D = q.shape
        q_g = _pad_tail(q.contiguous())
        kv_g = _pad_tail(kv.contiguous()).unsqueeze(1)
        o, lse = sparse_mla_fwd_v4_triton(q_g, kv_g, indices, attn_sink=None, kv_lora_rank=D, scale=scale)
        ctx.save_for_backward(q_g, kv_g, o, lse, indices)
        ctx.scale = scale
        ctx.D = D
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q_g, kv_g, o, lse, indices = ctx.saved_tensors
        D = ctx.D
        dq, dkv, _ = sparse_mla_bwd_v4_triton(
            q_g, kv_g, o, do.contiguous(), indices, lse, attn_sink=None, kv_lora_rank=D, scale=ctx.scale
        )
        return dq[..., :D], dkv[:, 0, :D], None, None


def sparse_mla(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    """``softmax(q . kv[idx]^T * scale) @ kv[idx]`` per query token and head.

    Args:
        q: ``[T, H, D]`` bf16 absorbed queries.
        kv: ``[T, D]`` bf16 shared latent (K == V).
        indices: ``[T, K]`` int32 rows of ``kv`` each query attends to, ``-1`` = none.
        scale: softmax scale.

    Returns:
        ``[T, H, D]`` in ``q.dtype``.
    """
    return _SparseMLAFn.apply(q, kv, indices.to(torch.int32).contiguous(), float(scale))


def sparse_mla_torch(q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, scale: float) -> torch.Tensor:
    """Gather-based reference for :func:`sparse_mla` (fp32 softmax; small shapes only)."""
    valid = indices >= 0
    safe = torch.where(valid, indices, 0).long()
    kv_sel = kv[safe]  # [T, K, D]
    s = torch.einsum("thd,tkd->thk", q.float(), kv_sel.float()) * scale
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    p = torch.softmax(s, dim=-1)
    return torch.einsum("thk,tkd->thd", p, kv_sel.float()).to(q.dtype)
