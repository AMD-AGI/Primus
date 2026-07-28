###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 CSA (gathered) attention — Triton **v0** autograd entry point. DEPRECATED.

The original ``gathered`` CSA path (takes a pre-gathered ``[B, Sq, K, D]``
tensor). It is ~30-260x slower than the v1 pool path (see attention_perf.md)
and is NOT used by the production dispatch — retained for reference/tests only.
Prefer ``_triton_v1`` (pool) or ``_triton_v2`` (fused sparse-MLA).
"""
from __future__ import annotations

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels import (
    _flydsl_v0_deprecated as _flydsl,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels import _tilelang
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated.v4_csa_attention_bwd import (
    _launch_v4_csa_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v0_deprecated.v4_csa_attention_fwd import (
    _launch_v4_csa_attention_fwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v1.v4_attention import (
    v4_attention_v1,
)


class V4CSAAttentionFn(torch.autograd.Function):
    """Triton FWD + Triton BWD for the CSA fused attention path."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,  # [B, H, Sq, D]
        k_local: torch.Tensor,  # [B, H, Sq, D]
        v_local: torch.Tensor,  # [B, H, Sq, D]
        gathered: torch.Tensor,  # [B, Sq, K_topk, D]
        sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
        sink: Optional[torch.Tensor],  # [H] or None
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            # Plan-4 P26 does not implement dropout in the kernel — V4
            # is trained with attn_dropout=0 so this branch is unreachable
            # in production. We refuse explicitly so a stray non-zero
            # dropout configuration raises rather than silently dropping
            # the kernel path.
            raise NotImplementedError(
                "v4_csa_attention_v0 does not implement in-kernel attention "
                "dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = _launch_v4_csa_attention_fwd(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
        )
        # Save tensors the BWD kernel needs. ``sink`` may be None; we
        # stash that fact on ``ctx`` because ``save_for_backward`` does
        # not accept ``None``.
        ctx.save_for_backward(q, k_local, v_local, gathered, sparse_mask, out, lse, sink)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        """Triton BWD: re-materialises joint ``P`` from saved LSE; emits all five gradients."""
        q, k_local, v_local, gathered, sparse_mask, out, lse, sink = ctx.saved_tensors

        sink_arg = None if ctx.sink_was_none else sink

        # Ensure dout is contiguous in the [B, H, Sq, D] layout the
        # kernel expects.
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk_local, dv_local, dgathered, dsink = _launch_v4_csa_attention_bwd(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            swa_window=ctx.swa_window,
            scale=ctx.scale,
        )

        # Honor needs_input_grad: zero-cost ``None`` for inputs that
        # don't want gradients. The kernel still computed them (the
        # main cost is the matmul, not the per-output cast), so this is
        # purely cleanliness.
        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk_local = None
        if not ctx.needs_input_grad[2]:
            dv_local = None
        if not ctx.needs_input_grad[3]:
            dgathered = None
        # sparse_mask (index 4) is built from the indexer's ``topk_idxs >=
        # 0`` test — it is NOT a learnable parameter and never needs a
        # gradient. The kernel does not produce one.
        if not ctx.needs_input_grad[5] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k_local, v_local, gathered, sparse_mask,
        # sink, swa_window, attn_dropout, training, scale).
        return dq, dk_local, dv_local, dgathered, None, dsink, None, None, None, None


def v4_csa_attention_v0(
    q: torch.Tensor,  # [B, H, Sq, D]
    k_local: torch.Tensor,  # [B, H, Sq, D]
    v_local: torch.Tensor,  # [B, H, Sq, D]
    gathered: torch.Tensor,  # [B, Sq, K_topk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    sparse_mask: torch.Tensor,  # [B, Sq, K_topk]
    attn_dropout: float,
    training: bool,
    scale: float,
    use_tilelang: bool = False,
    use_flydsl: bool = False,
) -> torch.Tensor:
    """Triton-backed V4 CSA fused attention.

    Drop-in replacement for :func:`eager_v4_csa_attention` with
    identical signature and dtype contract. Routes through
    :class:`V4CSAAttentionFn` so autograd works.

    When ``gathered.shape[2] == 0`` (degenerate Indexer state — no
    valid top-K positions) the wrapper short-circuits to the dense
    :func:`v4_attention_v1` kernel: the local SWA + sink path is exactly
    what CSA reduces to in that limit, and the dense kernel handles it
    natively. ``sparse_mask`` is unused on that path so it is allowed
    to be empty too.

    ``use_tilelang`` is plumbed by ``DeepseekV4Attention.forward``
    from the ``use_v4_tilelang_csa_attention`` config flag and only
    triggers a tilelang dispatch when the relevant plan-8 P54 / P55
    kernels are registered (otherwise the dispatcher warns once and
    falls back here).

    Returns ``[B, H, Sq, D]`` in ``v_local.dtype``.
    """
    K_topk = gathered.shape[2]
    if K_topk == 0:
        # Degenerate sparse branch — fall through to the dense kernel.
        # CSA's local SWA branch matches v4_attention_v1's dense+SWA+sink
        # path bit-identically when K_topk == 0 (the joint softmax
        # collapses to the local-only softmax).
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

    # Plan-8 P49 / P57 close-out 2: tilelang dispatcher hook for the
    # CSA family.  Defaults OFF; only fires when the caller passes
    # ``use_tilelang=True`` (i.e. the config flag is set).
    if _tilelang.should_dispatch("v4_csa_attention_fwd", enabled=use_tilelang):
        return _tilelang.v4_csa_attention_fwd_tilelang(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask=sparse_mask,
            sink=sink,
            swa_window=swa_window,
            attn_dropout=attn_dropout,
            training=training,
            scale=scale,
        )
    # FlyDSL CSA backend hook (forward-only; soft-dep, default off).
    # Short-circuits on enabled=False; falls back to Triton if the
    # runtime/kernel is unavailable. Inference/eval path -- training flows
    # through V4CSAAttentionFn below.
    if _flydsl.should_dispatch("v4_csa_attention_fwd", enabled=use_flydsl):
        return _flydsl.v4_csa_attention_fwd_flydsl(
            q,
            k_local,
            v_local,
            gathered,
            sparse_mask=sparse_mask,
            sink=sink,
            swa_window=swa_window,
            scale=scale,
            attn_dropout=attn_dropout,
            training=training,
        )
    return V4CSAAttentionFn.apply(
        q,
        k_local,
        v_local,
        gathered,
        sparse_mask,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
    )


__all__ = [
    "V4CSAAttentionFn",
    "v4_csa_attention_v0",
]
