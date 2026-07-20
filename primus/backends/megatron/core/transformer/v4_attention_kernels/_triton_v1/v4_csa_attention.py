###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 CSA (pool) attention — Triton **v1** autograd entry point (production, cr=4).

In-kernel top-k gather + pool scatter-add. Moved from the former top-level
``v4_csa_attention_v0.py`` during the triton v0/v1/v2 reorg; pairs with the
dense/HCA ``v4_attention_v1`` (also v1). See ``_triton_v0_deprecated`` for the deprecated
gathered path and ``_triton_v2`` for the fused sparse-MLA path.
"""
from __future__ import annotations

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention import (
    v4_attention_v1,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_csa_attention_bwd import (
    _launch_v4_csa_attention_pool_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_csa_attention_fwd import (
    _launch_v4_csa_attention_pool_fwd,
)


class V4CSAPoolAttentionFn(torch.autograd.Function):
    """Triton CSA attention with in-kernel topk gather and pool scatter-add."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        pool: torch.Tensor,
        topk_idxs: torch.Tensor,
        sink: Optional[torch.Tensor],
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "v4_csa_attention_v1 does not implement in-kernel attention "
                "dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = _launch_v4_csa_attention_pool_fwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )
        ctx.save_for_backward(q, k_local, v_local, pool, topk_idxs, out, lse, sink)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        q, k_local, v_local, pool, topk_idxs, out, lse, sink = ctx.saved_tensors
        sink_arg = None if ctx.sink_was_none else sink

        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk_local, dv_local, dpool, dsink = _launch_v4_csa_attention_pool_bwd(
            q,
            k_local,
            v_local,
            pool,
            topk_idxs,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
        )

        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk_local = None
        if not ctx.needs_input_grad[2]:
            dv_local = None
        if not ctx.needs_input_grad[3]:
            dpool = None
        if not ctx.needs_input_grad[5] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k_local, v_local, pool, topk_idxs, sink,
        # swa_window, attn_dropout, training, scale).
        return dq, dk_local, dv_local, dpool, None, dsink, None, None, None, None


def v4_csa_attention_v1(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    pool: torch.Tensor,  # [B, P, D]
    *,
    topk_idxs: torch.Tensor,  # [B, Sq, K_topk], -1 masks a slot
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    attn_dropout: float,
    training: bool,
    scale: float,
    use_tilelang: bool = False,
    use_flydsl: bool = False,  # accepted for call-site parity; no from-pool FlyDSL kernel, Triton path
) -> torch.Tensor:
    """Triton-backed CSA attention that gathers sparse keys in-kernel.

    ``pool`` is the compressed-pool tensor before per-query top-K gather.
    ``topk_idxs`` drives the sparse branch directly; negative entries are
    masked and contribute no probability mass. The backward kernel emits
    ``dpool`` with atomic scatter-add, avoiding the materialised
    ``[B, Sq, K_topk, D]`` gathered tensor and its autograd scatter.

    ``use_tilelang`` is reserved for the plan-8 P54 / P55 from-pool
    tilelang path (not landed); currently always falls through to
    :class:`V4CSAPoolAttentionFn` (Triton).
    """
    K_topk = topk_idxs.shape[2]
    if K_topk == 0:
        return v4_attention_v1(
            q,
            k_local,
            v_local,
            sink=sink,
            swa_window=swa_window,
            additive_mask=None,
            attn_dropout=attn_dropout,
            training=training,
            scale=scale,
        )

    # Plan-8 P57 close-out 2: from-pool tilelang path is not landed
    # (P54 / P55 descoped); the gated dispatcher always returns False
    # so this stays on the Triton autograd path.  We still consult
    # the dispatcher so a future P54 / P55 landing can flip behavior
    # without re-wiring this callsite.
    del use_tilelang
    return V4CSAPoolAttentionFn.apply(
        q,
        k_local,
        v_local,
        pool,
        topk_idxs,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
    )


__all__ = [
    "V4CSAPoolAttentionFn",
    "v4_csa_attention_v1",
]
