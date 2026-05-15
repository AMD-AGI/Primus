###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Plan-8 P51 — autograd wrapper that ties the tilelang FWD + BWD.

Mirrors :class:`primus.backends.megatron.core.transformer.v4_attention_kernels.v4_attention.V4AttentionFn`
(the Triton autograd Function) but routes through the tilelang
FWD / BWD kernels.  The wrapper falls back to the Triton path
inside the FWD / BWD wrapper functions when the tilelang kernels
don't yet support the requested feature (e.g. additive_mask,
hca_local_seqlen > 0).

Implementation note: no ``from __future__ import annotations``
because tilelang's annotation eval is eager.
"""

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._tilelang.v4_attention_bwd_tilelang import (
    v4_attention_bwd_tilelang,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._tilelang.v4_attention_fwd_tilelang import (
    v4_attention_fwd_tilelang_with_lse,
)


class V4AttentionTilelangFn(torch.autograd.Function):
    """Autograd-aware V4 dense / SWA / sink attention via tilelang.

    FWD: :func:`v4_attention_fwd_tilelang_with_lse` returns
    ``(out, lse)``; we save the inputs + ``lse`` for backward.

    BWD: :func:`v4_attention_bwd_tilelang` consumes the saved
    tensors + the incoming ``dO``; returns ``(dq, dk, dv, dsink)``.
    For unsupported features (additive_mask, hca_local_seqlen),
    both wrappers fall back to the Triton kernels so this autograd
    path remains correct even when tilelang's scope is limited.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sink: Optional[torch.Tensor],
        additive_mask: Optional[torch.Tensor],
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
        hca_local_seqlen: int,
    ) -> torch.Tensor:
        out, lse = v4_attention_fwd_tilelang_with_lse(
            q,
            k,
            v,
            sink=sink,
            additive_mask=additive_mask,
            swa_window=int(swa_window),
            attn_dropout=float(attn_dropout),
            training=bool(training),
            scale=float(scale),
            hca_local_seqlen=int(hca_local_seqlen),
        )
        # Save tensors needed for BWD.  `additive_mask` and `sink`
        # may be None; save_for_backward only stores tensors, so we
        # stash None-ness on ctx and conditionally save.
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.sink = sink
        ctx.additive_mask = additive_mask
        ctx.swa_window = int(swa_window)
        ctx.scale = float(scale)
        ctx.hca_local_seqlen = int(hca_local_seqlen)
        return out

    @staticmethod
    def backward(ctx, d_out):  # type: ignore[override]
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv, dsink = v4_attention_bwd_tilelang(
            q,
            k,
            v,
            out,
            lse,
            d_out,
            sink=ctx.sink,
            additive_mask=ctx.additive_mask,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
            hca_local_seqlen=ctx.hca_local_seqlen,
        )
        # Match the input-arg count of `forward(...)`: q, k, v, sink,
        # additive_mask, swa_window, attn_dropout, training, scale,
        # hca_local_seqlen.  Non-tensor args get None.
        return (
            dq,
            dk,
            dv,
            dsink,
            None,  # additive_mask
            None,  # swa_window
            None,  # attn_dropout
            None,  # training
            None,  # scale
            None,  # hca_local_seqlen
        )


def v4_attention_tilelang(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: Optional[torch.Tensor] = None,
    additive_mask: Optional[torch.Tensor] = None,
    swa_window: int = 0,
    attn_dropout: float = 0.0,
    training: bool = False,
    scale: Optional[float] = None,
    hca_local_seqlen: int = 0,
) -> torch.Tensor:
    """Convenience wrapper matching the existing `v4_attention()`
    functional API but routing through the tilelang autograd
    Function."""
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    return V4AttentionTilelangFn.apply(
        q,
        k,
        v,
        sink,
        additive_mask,
        int(swa_window),
        float(attn_dropout),
        bool(training),
        float(scale),
        int(hca_local_seqlen),
    )


__all__ = ["V4AttentionTilelangFn", "v4_attention_tilelang"]
