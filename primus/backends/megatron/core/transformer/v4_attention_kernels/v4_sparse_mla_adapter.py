###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Kernel-agnostic V4 attention adapters over a sparse-MLA fwd/bwd kernel pair.

Every "fused single-latent" V4 backend (gluon, triton-v2, flydsl-v2) speaks the
same sparse-MLA contract:

* ``fwd(q[T,H,Dqk], kv[T,1,Dqk], topk[T,TOPK], attn_sink, kv_lora_rank, scale)``
  -> ``(o[T,H,Dv], lse[T,H])``
* ``bwd(q, kv, o, do, topk, lse, attn_sink, kv_lora_rank, scale)``
  -> ``(dq, dkv, d_sink)``

This module maps Primus's V4 attention representations (per-head q, single MQA
latent K = V with RoPE baked in-place over ``head_dim = 512``, compressed pool,
per-query top-K, joint local-SWA + sparse softmax with sink) onto that contract
and maps gradients back — once — so each backend is just a kernel pair:

* :func:`make_csa_from_pool(fwd, bwd)`  -> CSA (cr=4) wrapper
* :func:`make_attention(fwd, bwd)`      -> dense (cr=0) / HCA (cr=128) wrapper

The fwd/bwd kernels are passed to the autograd Function as non-tensor args, so
the same Function serves all backends (backward returns ``None`` for them).
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

_ROPE_PAD = 64  # dummy separate-rope block (zeros); the kernels need D_ROPE > 0


def _pad_topk_64(topk: torch.Tensor) -> torch.Tensor:
    """Pad the topk width to a multiple of 64 with -1 so a backend whose dKV
    tiling is 64-wide (e.g. gluon) stays valid (HCA 128+32=160 -> 192)."""
    tk = topk.shape[1]
    pad = ((tk + 63) // 64) * 64 - tk
    if pad > 0:
        topk = torch.cat(
            [topk, torch.full((topk.shape[0], pad), -1, device=topk.device, dtype=topk.dtype)], dim=1
        )
    return topk.contiguous()


def _build_csa_topk(topk_idxs: torch.Tensor, S: int, P: int, W: int) -> torch.Tensor:
    """Flat topk [B*S, W+K] over the per-batch [local ++ pool] buffer.

    ``topk_idxs`` [B, S, K] holds pool indices in [0, P) (or -1). Batch ``b``
    occupies rows ``[b*(S+P) : (b+1)*(S+P))`` (local 0..S-1, pool S..S+P-1).
    """
    B, _, K = topk_idxs.shape
    device = topk_idxs.device
    base = (torch.arange(B, device=device) * (S + P)).view(B, 1, 1)

    win_pos = torch.arange(S, device=device).view(S, 1) - W + 1 + torch.arange(W, device=device).view(1, W)
    win_valid = win_pos >= 0
    win_idx = base + win_pos.view(1, S, W)
    win_idx = torch.where(win_valid.view(1, S, W), win_idx, torch.full_like(win_idx, -1))

    pool_valid = topk_idxs >= 0
    pool_idx = torch.where(pool_valid, base + S + topk_idxs, torch.full_like(topk_idxs, -1))

    return torch.cat([win_idx, pool_idx], dim=2).reshape(B * S, W + K).to(torch.int32).contiguous()


class _V4SparseMLACSAFn(torch.autograd.Function):
    """Autograd wrapper: sparse-MLA FWD/BWD for the V4 CSA (cr=4) layer."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_local_bh: torch.Tensor,  # [B, H, S, D] (single MQA latent, head-broadcast)
        v_local_bh: torch.Tensor,  # [B, H, S, D] (== k_local in V4)
        pool: torch.Tensor,  # [B, P, D]
        topk_idxs: torch.Tensor,  # [B, S, K] pool indices, -1 = invalid
        sink: Optional[torch.Tensor],  # [H] fp32 or None
        swa_window: int,
        scale: float,
        fwd_fn: Callable,
        bwd_fn: Callable,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        P = pool.shape[1]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "sparse-MLA adapter requires bf16"
        assert W > 0, "sparse-MLA adapter requires swa_window > 0"

        latent = k_local_bh[:, 0, :, :]  # [B, S, D]

        z_q = torch.zeros(B * S, H, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        q_g = torch.cat([q_bh.permute(0, 2, 1, 3).reshape(B * S, H, D), z_q], dim=-1).contiguous()

        kv512 = torch.cat([latent, pool], dim=1).reshape(B * (S + P), 1, D)
        z_kv = torch.zeros(B * (S + P), 1, _ROPE_PAD, device=q_bh.device, dtype=q_bh.dtype)
        kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()

        topk_g = _pad_topk_64(_build_csa_topk(topk_idxs, S, P, W))

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = fwd_fn(q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale))

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        ctx.shapes = (B, H, S, D, P, W)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        ctx.bwd_fn = bwd_fn
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, P, W = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        dq_g, dkv_g, dsink = ctx.bwd_fn(
            q_g, kv_g, o_g, grad_o_g, topk_g, lse, attn_sink=sink_arg, kv_lora_rank=D, scale=ctx.scale
        )

        dq_bh = dq_g[:, :, :D].reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()
        dkv512 = dkv_g[:, 0, :D].reshape(B, S + P, D)
        dlatent = dkv512[:, :S, :]
        dpool = dkv512[:, S:, :].contiguous()

        dk_local = torch.zeros(B, H, S, D, device=dq_bh.device, dtype=dq_bh.dtype)
        dk_local[:, 0, :, :] = dlatent.to(dq_bh.dtype)
        dv_local = torch.zeros(B, H, S, D, device=dq_bh.device, dtype=dq_bh.dtype)

        dsink_out = None
        if not ctx.sink_was_none and dsink is not None:
            dsink_out = dsink.to(sink_saved.dtype)

        # forward args: (q, k_local, v_local, pool, topk_idxs, sink, swa_window, scale, fwd_fn, bwd_fn)
        return dq_bh, dk_local, dv_local, dpool.to(dq_bh.dtype), None, dsink_out, None, None, None, None


class _V4SparseMLAAttnFn(torch.autograd.Function):
    """Sparse-MLA FWD/BWD for the V4 dense (cr=0) and HCA (cr=128) layers."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_bh: torch.Tensor,  # [B, H, S, D]
        k_bh: torch.Tensor,  # [B, H, Skv, D] (Skv = S for cr=0; S+P for HCA)
        v_bh: torch.Tensor,  # [B, H, Skv, D] (== k_bh in V4)
        sink: Optional[torch.Tensor],
        swa_window: int,
        additive_mask: Optional[torch.Tensor],  # [S, P] pool-only mask (HCA) or None
        scale: float,
        hca_local_seqlen: int,
        fwd_fn: Callable,
        bwd_fn: Callable,
    ) -> torch.Tensor:
        B, H, S, D = q_bh.shape
        Skv = k_bh.shape[2]
        W = int(swa_window)
        assert q_bh.dtype == torch.bfloat16, "sparse-MLA adapter requires bf16"
        assert W > 0, "sparse-MLA adapter requires swa_window > 0"

        device = q_bh.device
        base = (torch.arange(B, device=device) * Skv).view(B, 1, 1)
        win_pos = (
            torch.arange(S, device=device).view(S, 1) - W + 1 + torch.arange(W, device=device).view(1, W)
        )
        win_valid = win_pos >= 0
        win_idx = base + win_pos.view(1, S, W)
        win_idx = torch.where(win_valid.view(1, S, W), win_idx, torch.full_like(win_idx, -1))

        if hca_local_seqlen > 0 and additive_mask is not None:
            P = Skv - int(hca_local_seqlen)
            vis = (additive_mask == 0).view(1, S, P)
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
        kv512 = k_bh[:, 0, :, :].reshape(B * Skv, 1, D)
        z_kv = torch.zeros(B * Skv, 1, _ROPE_PAD, device=device, dtype=q_bh.dtype)
        kv_g = torch.cat([kv512, z_kv], dim=-1).contiguous()

        sink_arg = sink.float().contiguous() if sink is not None else None
        o_g, lse = fwd_fn(q_g, kv_g, topk_g, attn_sink=sink_arg, kv_lora_rank=D, scale=float(scale))

        ctx.save_for_backward(q_g, kv_g, o_g, lse, topk_g, sink_arg if sink is not None else q_g.new_empty(0))
        ctx.shapes = (B, H, S, D, Skv)
        ctx.scale = float(scale)
        ctx.sink_was_none = sink is None
        ctx.bwd_fn = bwd_fn
        return o_g.reshape(B, S, H, D).permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, grad_o_bh: torch.Tensor):  # type: ignore[override]
        q_g, kv_g, o_g, lse, topk_g, sink_saved = ctx.saved_tensors
        B, H, S, D, Skv = ctx.shapes
        sink_arg = None if ctx.sink_was_none else sink_saved

        grad_o_g = grad_o_bh.permute(0, 2, 1, 3).reshape(B * S, H, D).contiguous()
        dq_g, dkv_g, dsink = ctx.bwd_fn(
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

        # forward args: (q, k, v, sink, swa_window, additive_mask, scale, hca_local_seqlen, fwd_fn, bwd_fn)
        return dq_bh, dk_bh, dv_bh, dsink_out, None, None, None, None, None, None


def make_csa_from_pool(fwd_fn: Callable, bwd_fn: Callable) -> Callable:
    """Build a ``v4_csa_attention_v1``-style wrapper for a kernel pair."""

    def _csa_from_pool(
        q_bh,
        k_local_bh,
        v_local_bh,
        pool,
        *,
        topk_idxs,
        sink,
        swa_window,
        attn_dropout,
        training,
        scale,
    ):
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "sparse-MLA CSA adapter does not implement in-kernel attention dropout "
                f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
            )
        return _V4SparseMLACSAFn.apply(
            q_bh, k_local_bh, v_local_bh, pool, topk_idxs, sink, int(swa_window), float(scale), fwd_fn, bwd_fn
        )

    return _csa_from_pool


def make_attention(fwd_fn: Callable, bwd_fn: Callable) -> Callable:
    """Build a dense (cr=0) / HCA (cr=128) attention wrapper for a kernel pair."""

    def _attention(
        q,
        k,
        v,
        *,
        sink,
        swa_window,
        additive_mask,
        attn_dropout,
        training,
        scale,
        hca_local_seqlen=0,
    ):
        if attn_dropout > 0.0 and training:
            raise NotImplementedError(
                "sparse-MLA attention adapter does not implement in-kernel attention dropout "
                f"(V4 trains with attn_dropout=0). Got attn_dropout={attn_dropout}, training={training}."
            )
        return _V4SparseMLAAttnFn.apply(
            q, k, v, sink, int(swa_window), additive_mask, float(scale), int(hca_local_seqlen), fwd_fn, bwd_fn
        )

    return _attention


__all__ = ["make_csa_from_pool", "make_attention"]
