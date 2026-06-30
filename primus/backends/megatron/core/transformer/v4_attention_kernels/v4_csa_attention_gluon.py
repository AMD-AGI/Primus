###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""V4 CSA attention via the Gluon sparse-MLA backend (gfx950).

Adapts Primus's CSA representation (per-head q + single MQA latent K=V +
compressed pool + per-query top-K pool indices, joint local-SWA + sparse
softmax with sink) to the aiter Gluon ``sparse_mla_{fwd,bwd}_v4`` kernels
(``_gluon_dsa``), then maps the gradients back. Numerically equivalent to
:func:`eager_v4_csa_attention` (== the official DeepSeek-V4 ``inference/model.py``
CSA path) — validated to ~1e-6 cos-err for O and all grads.

The mapping (the official V4 layout has head_dim=512 with RoPE applied
*in-place* to the last 64 channels, so K = V = 512, score over 512):

* The full 512 latent (RoPE already baked in) is fed as the gluon "lora"
  (``kv_lora_rank = head_dim``). The gluon kernel requires a separate rope
  block with ``D_ROPE > 0``, so a **zero rope pad** (``_ROPE_PAD``) is
  appended to q / kv — it contributes 0 to the score and is dropped from V.
* The kv buffer is ``[local latent (S) ++ compressed pool (P)]`` per batch
  (so ``num_kv = S + P > S``), and ``topk = [SWA window positions ++
  (S + pool top-k indices)]``; ``-1`` marks invalid slots.
* ``scale`` is passed through unchanged (Primus's ``_attention_scale``).

Constraints (satisfied by production V4-CSA): bf16; head_dim=512; the gluon
bwd dKV tiles by 64, so the per-query key count ``W + K`` must be >= 64
(CSA: 128 + 512/1024 = 640/1152 -> ok).
"""

from __future__ import annotations

from typing import Optional

import torch

from primus.backends.megatron.core.transformer.v4_attention_kernels._gluon_dsa import (
    sparse_mla_bwd_v4_gluon,
    sparse_mla_fwd_v4_gluon,
)

_ROPE_PAD = 64  # dummy separate-rope block (zeros); the gluon kernel needs D_ROPE > 0


def _build_topk(topk_idxs: torch.Tensor, S: int, P: int, W: int) -> torch.Tensor:
    """Build the gluon flat topk [B*S, W+K] over the per-batch [local ++ pool] buffer.

    ``topk_idxs`` [B, S, K] holds pool indices in [0, P) (or -1). Output indices
    address the flattened buffer where batch ``b`` occupies rows
    ``[b*(S+P) : (b+1)*(S+P))`` (local 0..S-1, pool S..S+P-1).
    """
    B, S_, K = topk_idxs.shape
    device = topk_idxs.device
    base = (torch.arange(B, device=device) * (S + P)).view(B, 1, 1)  # [B,1,1]

    # SWA-causal window positions (raw token indices), -1 where out of range.
    win_pos = torch.arange(S, device=device).view(S, 1) - W + 1 + torch.arange(W, device=device).view(1, W)
    win_valid = win_pos >= 0  # [S, W]
    win_idx = base + win_pos.view(1, S, W)  # [B, S, W]
    win_idx = torch.where(win_valid.view(1, S, W), win_idx, torch.full_like(win_idx, -1))

    # Sparse pool indices, offset into the pool region; -1 stays -1.
    pool_valid = topk_idxs >= 0
    pool_idx = torch.where(pool_valid, base + S + topk_idxs, torch.full_like(topk_idxs, -1))

    return torch.cat([win_idx, pool_idx], dim=2).reshape(B * S, W + K).to(torch.int32).contiguous()


class V4CSAGluonFn(torch.autograd.Function):
    """Autograd wrapper: Gluon sparse-MLA FWD/BWD for the V4 CSA layer."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_local_bh: torch.Tensor,  # [B, H, S, D]  (single MQA latent, head-broadcast)
        v_local_bh: torch.Tensor,  # [B, H, S, D]  (== k_local in V4)
        pool: torch.Tensor,  # [B, P, D]
        topk_idxs: torch.Tensor,  # [B, S, K]  pool indices, -1 = invalid
        sink: Optional[torch.Tensor],  # [H] fp32 or None
        swa_window: int,
        scale: float,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        P = pool.shape[1]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "gluon CSA requires bf16"
        assert W > 0, "gluon CSA requires swa_window > 0"

        latent = k_local_bh[:, 0, :, :]  # [B, S, D] (MQA: all heads identical)

        # Flatten batch into tokens and append a zero rope pad (D_ROPE > 0).
        z_q = torch.zeros(B * S, H, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        q_g = torch.cat([q_bh.permute(0, 2, 1, 3).reshape(B * S, H, D), z_q], dim=-1).contiguous()

        kv512 = torch.cat([latent, pool], dim=1).reshape(B * (S + P), 1, D)  # [local ++ pool]
        z_kv = torch.zeros(B * (S + P), 1, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()

        topk_g = _build_topk(topk_idxs, S, P, W)

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = sparse_mla_fwd_v4_gluon(
            q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale)
        )

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        ctx.shapes = (B, H, S, D, P, W)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None

        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, P, W = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        dq_g, dkv_g, dsink = sparse_mla_bwd_v4_gluon(
            q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
        )

        dq_bh = dq_g[:, :, :D].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

        dkv512 = dkv_g[:, 0, :D].reshape(B, S + P, D)
        dlatent = dkv512[:, :S, :]  # [B, S, D]
        dpool = dkv512[:, S:, :].contiguous()  # [B, P, D]

        # gluon dkv is the combined K+V grad w.r.t. the single latent. k_local /
        # v_local are head-broadcast views of that latent, so placing the whole
        # grad at head 0 (zeros elsewhere) makes the expand-backward sum to the
        # correct latent gradient; v_local then carries none.
        dk_local = torch.zeros(B, H, S, D, device=dq_bh.device, dtype=dq_bh.dtype)
        dk_local[:, 0, :, :] = dlatent.to(dq_bh.dtype)
        dv_local = torch.zeros(B, H, S, D, device=dq_bh.device, dtype=dq_bh.dtype)

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q_bh, k_local_bh, v_local_bh, pool, topk_idxs, sink, swa_window, scale)
        return dq_bh, dk_local, dv_local, dpool.to(dq_bh.dtype), None, dsink_out, None, None


def v4_csa_attention_gluon_from_pool(
    q_bh: torch.Tensor,
    k_local_bh: torch.Tensor,
    v_local_bh: torch.Tensor,
    pool: torch.Tensor,
    *,
    topk_idxs: torch.Tensor,
    sink: Optional[torch.Tensor],
    swa_window: int,
    attn_dropout: float,
    training: bool,
    scale: float,
) -> torch.Tensor:
    """Gluon-backed V4 CSA attention from the compressed pool + top-K indices.

    Drop-in for :func:`v4_csa_attention_from_pool` on the CSA (cr=4) layers when
    ``use_v4_gluon_csa_attention`` is set. Returns ``[B, H, S, D]``.
    """
    if attn_dropout > 0.0 and training:
        raise NotImplementedError(
            "v4_csa_attention_gluon does not implement in-kernel attention dropout "
            f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
        )
    return V4CSAGluonFn.apply(
        q_bh, k_local_bh, v_local_bh, pool, topk_idxs, sink, int(swa_window), float(scale)
    )


def _pad_topk_64(topk: torch.Tensor) -> torch.Tensor:
    """Pad the topk width to a multiple of 64 with -1 so the gluon bwd dKV
    tiling (TILE_K=64) is valid (e.g. HCA 128+32=160 -> 192)."""
    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat(
            [topk, torch.full((topk.shape[0], pad), -1, device=topk.device, dtype=topk.dtype)], dim=1
        )
    return topk.contiguous()


class V4GluonAttnFn(torch.autograd.Function):
    """Gluon sparse-MLA FWD/BWD for the V4 dense (cr=0) and HCA (cr=128) layers.

    ``k`` / ``v`` are the head-broadcast single latent (cr=0) or the
    ``[local ++ compressed pool]`` concat (cr=128, ``hca_local_seqlen = S``).
    topk = SWA window (++ causally-visible pool slots from ``additive_mask``).
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_bh: torch.Tensor,  # [B, H, Skv, D]  (Skv = S for cr=0; S+P for HCA)
        v_bh: torch.Tensor,  # [B, H, Skv, D]  (== k_bh in V4)
        sink: Optional[torch.Tensor],  # [H] fp32 or None
        swa_window: int,
        additive_mask: Optional[torch.Tensor],  # [S, P] pool-only mask (HCA) or None
        scale: float,
        hca_local_seqlen: int,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        Skv = k_bh.shape[2]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "gluon attn requires bf16"
        assert W > 0, "gluon attn requires swa_window > 0"

        device = q_bh.device
        base = (torch.arange(B, device=device) * Skv).view(B, 1, 1)
        win_pos = (
            torch.arange(S, device=device).view(S, 1) - W + 1 + torch.arange(W, device=device).view(1, W)
        )
        win_valid = win_pos >= 0
        win_idx = base + win_pos.view(1, S, W)
        win_idx = torch.where(win_valid.view(1, S, W), win_idx, torch.full_like(win_idx, -1))  # [B,S,W]

        if hca_local_seqlen > 0 and additive_mask is not None:
            P = Skv - int(hca_local_seqlen)
            vis = (additive_mask == 0).view(1, S, P)  # mask is 0 (visible) / -inf
            ps = torch.arange(P, device=device).view(1, 1, P)
            pool_idx = torch.where(
                vis, base + hca_local_seqlen + ps, torch.full((B, S, P), -1, device=device)
            )
            topk = torch.cat([win_idx, pool_idx], dim=2)
        else:
            topk = win_idx
        topk_g = _pad_topk_64(topk.reshape(B * S, -1).to(torch.int32))

        z_q = torch.zeros(B * S, H, _ROPE_PAD, device=device, dtype=q_bh.dtype)
        q_g = torch.cat([q_bh.permute(0, 2, 1, 3).reshape(B * S, H, D), z_q], dim=-1).contiguous()
        kv512 = k_bh[:, 0, :, :].reshape(B * Skv, 1, D)  # head-broadcast latent / [local++pool]
        z_kv = torch.zeros(B * Skv, 1, _ROPE_PAD, device=device, dtype=q_bh.dtype)
        kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = sparse_mla_fwd_v4_gluon(
            q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale)
        )

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        ctx.shapes = (B, H, S, D, Skv)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, Skv = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        dq_g, dkv_g, dsink = sparse_mla_bwd_v4_gluon(
            q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
        )

        dq_bh = dq_g[:, :, :D].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        dkv = dkv_g[:, 0, :D].reshape(B, Skv, D)
        dk_bh = torch.zeros(B, H, Skv, D, device=dq_bh.device, dtype=dq_bh.dtype)
        dk_bh[:, 0, :, :] = dkv.to(dq_bh.dtype)
        dv_bh = torch.zeros(B, H, Skv, D, device=dq_bh.device, dtype=dq_bh.dtype)

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q_bh, k_bh, v_bh, sink, swa_window, additive_mask, scale, hca_local_seqlen)
        return dq_bh, dk_bh, dv_bh, dsink_out, None, None, None, None


def v4_attention_gluon(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sink: Optional[torch.Tensor],
    swa_window: int,
    additive_mask: Optional[torch.Tensor],
    attn_dropout: float,
    training: bool,
    scale: float,
    hca_local_seqlen: int = 0,
) -> torch.Tensor:
    """Gluon-backed V4 dense (cr=0) / HCA (cr=128) attention. Returns [B, H, S, D]."""
    if attn_dropout > 0.0 and training:
        raise NotImplementedError(
            "v4_attention_gluon does not implement in-kernel attention dropout "
            f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
        )
    return V4GluonAttnFn.apply(
        q, k, v, sink, int(swa_window), additive_mask, float(scale), int(hca_local_seqlen)
    )


__all__ = [
    "V4CSAGluonFn",
    "v4_csa_attention_gluon_from_pool",
    "V4GluonAttnFn",
    "v4_attention_gluon",
]
