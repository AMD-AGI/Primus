###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 attention Triton autograd entry point (plan-4 P25).

Public API:

* :func:`v4_attention` — functional API matching
  :func:`eager_v4_attention`'s signature; routes through
  :class:`V4AttentionFn` so autograd works.
* :class:`V4AttentionFn` — :class:`torch.autograd.Function` wrapping
  the Triton FWD + Triton BWD (re-materialises softmax from the saved
  LSE; sink gradient atomic-added per query head).

Dispatch contract (consumed by ``DeepseekV4Attention.forward``):

* ``compress_ratio == 0``: caller passes ``swa_window > 0`` and
  ``additive_mask=None`` so the kernel applies the SWA-causal mask
  in-place.
* ``compress_ratio == 128`` (HCA): caller pre-concatenates pool keys
  to local keys, passes the pool-only ``[Sq, P]`` additive mask, sets
  ``hca_local_seqlen=Sq``, and keeps ``swa_window > 0``. The kernel
  splits the loop into a pruned local SWA branch plus the compressed-pool
  suffix.

dtype contract (must match :func:`eager_v4_attention`):

* Q / K / V matmuls run on tensor cores in input dtype (bf16 in
  production); the matmul accumulator inside is fp32.
* The online-softmax accumulator (``m_running / l_running / acc``) is
  fp32 — the *only* fp32 step.
* Output is in ``v.dtype``.
* Saved LSE is fp32 (BWD re-materialises ``P`` from it).
"""

from __future__ import annotations

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels import _tilelang
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_bwd import (
    _launch_v4_attention_bwd,
)
from primus.backends.megatron.core.transformer.v4_attention_kernels._triton.v4_attention_fwd import (
    _launch_v4_attention_fwd,
)

# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------


class V4AttentionFn(torch.autograd.Function):
    """Triton FWD + (eager-recompute BWD until P25-stage-2 lands)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,  # [B, H, Sq, D]
        k: torch.Tensor,  # [B, K_H, Sk, D]
        v: torch.Tensor,  # [B, K_H, Sk, D]
        sink: Optional[torch.Tensor],  # [H] or None
        additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
        swa_window: int,
        attn_dropout: float,
        training: bool,
        scale: float,
        hca_local_seqlen: int,
    ) -> torch.Tensor:
        if attn_dropout > 0.0 and training:
            # Plan-4 P25 does not implement dropout in the kernel — V4
            # is trained with attn_dropout=0 so this branch is unreachable
            # in production. We refuse explicitly so a stray non-zero
            # dropout configuration raises rather than silently dropping
            # the kernel path.
            raise NotImplementedError(
                "v4_attention does not implement in-kernel attention "
                "dropout (V4 trains with attn_dropout=0). Got "
                f"attn_dropout={attn_dropout}, training={training}."
            )

        out, lse = _launch_v4_attention_fwd(
            q,
            k,
            v,
            sink=sink,
            swa_window=swa_window,
            additive_mask=additive_mask,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )
        # Save tensors the BWD kernel needs. None-typed args
        # (``sink`` / ``additive_mask``) are stashed on ``ctx`` because
        # ``save_for_backward`` does not accept ``None``.
        ctx.save_for_backward(q, k, v, out, lse, sink, additive_mask)
        ctx.swa_window = int(swa_window)
        ctx.attn_dropout = float(attn_dropout)
        ctx.training_mode = bool(training)
        ctx.scale = float(scale)
        ctx.hca_local_seqlen = int(hca_local_seqlen)
        ctx.sink_was_none = sink is None
        ctx.mask_was_none = additive_mask is None
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        """Triton BWD: re-materialises ``P`` from saved LSE; emits dq, dk, dv, dsink.

        Routes through :func:`_launch_v4_attention_bwd`, which:

        * runs a small fp32 pre-pass that computes the per-query
          ``D = (dout * out).sum(-1)`` scalar,
        * runs the main BWD kernel parallelized over query blocks,
          accumulating ``dQ`` in registers and atomic-adding into
          ``dK / dV / dsink``,
        * casts the fp32 accumulator buffers back to the input dtype
          for return.

        The Function returns gradients in the same positional order as
        the forward: ``(q, k, v, sink, additive_mask, swa_window,
        attn_dropout, training, scale)``. Non-tensor / non-grad inputs
        get ``None``.
        """
        q, k, v, out, lse, sink, additive_mask = ctx.saved_tensors

        sink_arg = None if ctx.sink_was_none else sink
        mask_arg = None if ctx.mask_was_none else additive_mask

        # Ensure dout is contiguous in the [B, H, Sq, D] layout the
        # kernel expects. The kernel reads strides from the tensor so
        # any contiguous-in-its-strides tensor would work, but a
        # ``.contiguous()`` here keeps the access pattern simple.
        if not grad_out.is_contiguous():
            grad_out = grad_out.contiguous()

        dq, dk, dv, dsink = _launch_v4_attention_bwd(
            q,
            k,
            v,
            out,
            grad_out,
            lse,
            sink=sink_arg,
            swa_window=ctx.swa_window,
            additive_mask=mask_arg,
            scale=ctx.scale,
            hca_local_seqlen=ctx.hca_local_seqlen,
        )

        # Honor needs_input_grad: zero-cost ``None`` for inputs that
        # don't want gradients. (We still computed them — the main
        # cost in the kernel is the matmul, not the per-output cast —
        # so this is purely a cleanliness touch.)
        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk = None
        if not ctx.needs_input_grad[2]:
            dv = None
        if not ctx.needs_input_grad[3] or ctx.sink_was_none:
            dsink = None

        # Forward signature: (q, k, v, sink, additive_mask, swa_window,
        # attn_dropout, training, scale, hca_local_seqlen)
        return dq, dk, dv, dsink, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def v4_attention(
    q: torch.Tensor,  # [B, H, Sq, D]
    k: torch.Tensor,  # [B, K_H, Sk, D]   K_H ∈ {1, H}
    v: torch.Tensor,  # [B, K_H, Sk, D]
    *,
    sink: Optional[torch.Tensor],  # [H] or None
    swa_window: int,
    additive_mask: Optional[torch.Tensor],  # [Sq, Sk] or None
    attn_dropout: float,
    training: bool,
    scale: float,
    hca_local_seqlen: int = 0,
) -> torch.Tensor:
    """Triton- or tilelang-backed V4 dense / HCA attention.

    Drop-in replacement for :func:`eager_v4_attention` with identical
    signature and dtype contract.

    Dispatch precedence (plan-8 P49):

    * If ``PRIMUS_V4_TILELANG_ATTN=1`` AND the plan-8 P50 / P51
      tilelang kernels are registered + tilelang importable at the
      pinned version, route through the tilelang FWD/BWD.
    * Otherwise, route through :class:`V4AttentionFn` (plan-4 P25
      Triton FWD + plan-5 P32 final split BWD).

    The MQA case (``K_H == 1``) is detected from ``k.shape[1]`` and
    each kernel internally broadcasts the single shared K / V head
    across the query heads.

    Returns ``[B, H, Sq, D]`` in ``v.dtype``.
    """
    # Plan-8 P49: tilelang dispatcher hook.  Defaults OFF until both
    # the env knob is set AND each plan-8 phase registers its kernel
    # via `register_available_kernel(...)`.  Falls back to the Triton
    # autograd path on any miss with a one-time rank-0 warning.
    if _tilelang.should_dispatch("v4_attention_fwd"):
        # P51: route through the tilelang autograd Function when both
        # FWD + BWD kernels are registered, so gradients flow through
        # the tilelang path correctly.  When only the FWD is registered
        # (P50-only state), fall through to the FWD-direct call — the
        # Triton BWD will still get called via the autograd graph that
        # the caller sets up (only the FWD wrapper is non-autograd in
        # that mode).
        # `_lazy_load` is idempotent + cheap on cache hit; calling it
        # here ensures `is_tilelang_kernel_available("v4_attention_bwd")`
        # reflects the post-import state.
        _tilelang._lazy_load("v4_attention_bwd")
        if _tilelang.is_tilelang_kernel_available("v4_attention_bwd"):
            from primus.backends.megatron.core.transformer.v4_attention_kernels._tilelang.v4_attention_autograd_tilelang import (
                v4_attention_tilelang,
            )

            return v4_attention_tilelang(
                q,
                k,
                v,
                sink=sink,
                additive_mask=additive_mask,
                swa_window=swa_window,
                attn_dropout=attn_dropout,
                training=training,
                scale=scale,
                hca_local_seqlen=hca_local_seqlen,
            )
        return _tilelang.v4_attention_fwd_tilelang(
            q,
            k,
            v,
            sink=sink,
            additive_mask=additive_mask,
            swa_window=swa_window,
            attn_dropout=attn_dropout,
            training=training,
            scale=scale,
            hca_local_seqlen=hca_local_seqlen,
        )
    return V4AttentionFn.apply(
        q,
        k,
        v,
        sink,
        additive_mask,
        swa_window,
        attn_dropout,
        training,
        scale,
        hca_local_seqlen,
    )


__all__ = [
    "V4AttentionFn",
    "v4_attention",
]
