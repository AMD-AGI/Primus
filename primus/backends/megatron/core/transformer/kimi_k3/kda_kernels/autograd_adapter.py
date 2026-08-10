###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-agnostic autograd wrapper for the KDA chunk kernels.

One :class:`torch.autograd.Function` serves every fused KDA backend. A
backend supplies a ``(fwd_fn, bwd_fn)`` pair and :func:`make_kda_chunk`
returns something with the shared backend signature documented in
:mod:`..kda_kernels`. This mirrors
``v4_attention_kernels/v4_sparse_mla_adapter.py:239-292``, where the
kernel pair is passed to the Function as non-tensor arguments so a new
kernel needs no new Function.

The contract between the adapter and a kernel pair::

    o, final_state, saved = fwd_fn(
        q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size
    )
    dq, dk, dv, dg, dbeta, dh0 = bwd_fn(saved, do, dht)

``saved`` is an opaque tuple of tensors the adapter stashes with
``ctx.save_for_backward`` — so it must contain **only** tensors (or
``None``) — plus an optional trailing non-tensor metadata dict, which the
adapter keeps on ``ctx`` instead. Any gradient the kernel does not
produce may be returned as ``None``.

``use_qk_l2norm_in_kernel`` is deliberately *not* part of the pair's
contract: the adapter applies :func:`..._eager.reference.kda_l2norm`
ahead of the Function using plain autograd-visible torch ops. The
normalisation is a cheap elementwise reduction, doing it outside keeps
one auditable implementation of it (the flag exists only so a backend is
signature-compatible with ``fla``'s ``chunk_kda``), and it means a kernel
never has to carry the ``rstd`` bookkeeping ``fla`` does.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch

from primus.backends.megatron.core.transformer.kimi_k3.kda_kernels._eager.reference import (
    kda_l2norm,
)

__all__ = ["make_kda_chunk"]


class _KDAChunkFn(torch.autograd.Function):
    """Bridge a ``(fwd_fn, bwd_fn)`` KDA kernel pair into autograd."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: Optional[torch.Tensor],
        output_final_state: bool,
        chunk_size: int,
        fwd_fn: Callable[..., Any],
        bwd_fn: Callable[..., Any],
    ):
        o, final_state, saved = fwd_fn(q, k, v, g, beta, scale, initial_state, output_final_state, chunk_size)
        meta: Dict[str, Any] = {}
        if saved and isinstance(saved[-1], dict):
            *saved_tensors, meta = saved
            saved = tuple(saved_tensors)
        ctx.save_for_backward(*saved)
        ctx.meta = meta
        ctx.bwd_fn = bwd_fn
        ctx.has_initial_state = initial_state is not None
        return o, final_state

    @staticmethod
    def backward(ctx, do: torch.Tensor, dht: Optional[torch.Tensor]):  # type: ignore[override]
        saved: Tuple[Any, ...] = tuple(ctx.saved_tensors)
        if ctx.meta:
            saved = saved + (ctx.meta,)
        dq, dk, dv, dg, dbeta, dh0 = ctx.bwd_fn(saved, do, dht)
        if not ctx.has_initial_state:
            dh0 = None
        # forward args: q, k, v, g, beta, scale, initial_state,
        #               output_final_state, chunk_size, fwd_fn, bwd_fn
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None


def make_kda_chunk(fwd_fn: Callable[..., Any], bwd_fn: Callable[..., Any]) -> Callable[..., Any]:
    """Build a KDA backend entry point from a kernel pair.

    The returned callable has the signature every KDA backend shares (see
    :mod:`..kda_kernels`), so it is interchangeable with
    :func:`..._eager.reference.eager_chunk_kda` and
    :func:`..._fla.fla_chunk_kda` at a call site.
    """

    def _chunk_kda(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: Optional[float] = None,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _validate_shapes(q, k, v, g, beta)
        if use_qk_l2norm_in_kernel:
            q, k = kda_l2norm(q), kda_l2norm(k)
        if scale is None:
            scale = q.shape[-1] ** -0.5
        return _KDAChunkFn.apply(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            initial_state,
            bool(output_final_state),
            int(chunk_size),
            fwd_fn,
            bwd_fn,
        )

    return _chunk_kda


def _validate_shapes(*tensors: torch.Tensor) -> None:
    q, k, v, g, beta = tensors
    if q.shape != k.shape or q.shape != g.shape:
        raise ValueError(
            f"q, k, g must share a shape; got {tuple(q.shape)}, {tuple(k.shape)}, {tuple(g.shape)}"
        )
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(f"v must agree with q on [B, T, H]; got {tuple(v.shape)} vs {tuple(q.shape)}")
    if beta.shape != q.shape[:3]:
        raise ValueError(f"beta must be [B, T, H]; got {tuple(beta.shape)}")
