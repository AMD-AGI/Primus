###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton-fused V4 router post-logits chain (plan-6 P39).

Fuses the chain shared between
:class:`primus.backends.megatron.core.transformer.moe.v4_topk_router.DeepseekV4LearnedRouter`
and
:class:`primus.backends.megatron.core.transformer.moe.v4_hash_router.DeepseekV4HashRouter`:

.. code-block:: python

    scores = score_fn(logits)            # softmax / sigmoid / sqrtsoftplus
    weights = scores.gather(1, indices)  # [N, K]
    if score_fn != softmax:
        denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        weights = weights / denom
    weights *= topk_scaling_factor

    probs = zeros(N, E); probs.scatter_(1, indices, weights)
    routing_map = zeros(N, E); routing_map.scatter_(1, indices, True)

`indices` is provided by the host (hash router pre-computes via
``tid2eid[flat_ids]``; learned router pre-computes via
``torch.topk(sel_score, K)``).  Keeping `topk` on the host lets us
reuse the well-tuned cuBLAS / hip topk kernel and avoids stuffing
two unrelated kernels into one register file.

Gating: ``PRIMUS_V4_ROUTER_TRITON == "1"`` (**default-off**, P38 precedent).

Microbench at V4-Flash widths (N=4096, E=256, K=8) shows the kernel
wins on the V4 production score function:
- ``sqrtsoftplus`` (V4 default): 1.56x FWD / 1.22x BWD
- ``softmax`` (non-V4):           1.00x FWD / 0.73x BWD
- ``sigmoid``:                    near-parity

But the EP=8 proxy A/B (PRIMUS_V4_ROUTER_TRITON=1 vs =0, 10 iters
each) shows the microbench gain does **not** surface end-to-end:
~534 ms / iter both ways, lm_loss bit-identical iter-by-iter
(parity confirmed).  The per-call savings (~0.06 ms FWD+BWD ×
~16 router calls / iter ≈ 1 ms) are submerged in the ~500 ms /
step variance of the EP=8 dispatch + grouped-MLP pipeline.

P38-style descope: ship the kernel behind a knob, default OFF.
Available for future tuning (e.g. when the broader graph compresses
and exposes the per-call savings) and for small-shape paths where
the FWD win is closer to 1.5-2x.  Bit-identity makes flipping the
knob a safe operation.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Score function enum (matches FWD/BWD constexpr).
_SCORE_FN_SOFTMAX = 0
_SCORE_FN_SIGMOID = 1
_SCORE_FN_SQRTSOFTPLUS = 2
_SCORE_FN_MAP = {
    "softmax": _SCORE_FN_SOFTMAX,
    "sigmoid": _SCORE_FN_SIGMOID,
    "sqrtsoftplus": _SCORE_FN_SQRTSOFTPLUS,
}


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _v4_router_post_fwd_kernel(
    LOGITS_PTR,  # [N, E] fp32
    INDICES_PTR,  # [N, BLOCK_K] int64 (gather positions; padded to BLOCK_K)
    PROBS_PTR,  # [N, E] OUT_DTYPE (must be zero-init'd by caller)
    RMAP_PTR,  # [N, E] bool   (must be zero-init'd by caller)
    SCORES_OUT_PTR,  # [N, E] fp32 (saved-for-backward; full row of scores)
    WEIGHTS_OUT_PTR,  # [N, BLOCK_K] fp32 (saved-for-backward; gathered weights, post-denom, pre-scale)
    DENOM_OUT_PTR,  # [N] fp32  (saved-for-backward; the clamped denom)
    N,
    E,  # runtime int (real expert count; NOT required to be a power of 2)
    K,  # runtime int (real topk; NOT required to be a power of 2)
    SCORE_FN: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,  # next_pow2(E) — column block over the expert axis
    BLOCK_K: tl.constexpr,  # next_pow2(K) — column block over the topk axis
    OUT_DTYPE: tl.constexpr,
):
    """One program tile = ``BLOCK_N`` rows of the post-logits chain.

    Supports arbitrary (non-power-of-2) ``E`` / ``K``: the expert axis is
    tiled by ``BLOCK_E = next_pow2(E)`` and the topk axis by
    ``BLOCK_K = next_pow2(K)``, with masks zeroing the padded columns.
    ``INDICES_PTR`` / ``WEIGHTS_OUT_PTR`` are laid out with row stride
    ``BLOCK_K`` (host pads the indices tensor to ``[N, BLOCK_K]``).
    """

    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    e_idx = tl.arange(0, BLOCK_E)
    e_mask = e_idx < E
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < K

    # Load logits [BLOCK_N, BLOCK_E] in fp32 (padded cols masked out).
    logits = tl.load(
        LOGITS_PTR + n_offs[:, None] * E + e_idx[None, :],
        mask=n_mask[:, None] & e_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Apply score_fn (compile-time specialised). Padded expert columns must
    # not leak into the softmax reduction, so mask them to -inf first.
    if SCORE_FN == 0:  # softmax
        neg_inf = float("-inf")
        logits_m = tl.where(e_mask[None, :], logits, neg_inf)
        m = tl.max(logits_m, axis=1, keep_dims=True)
        e = tl.where(e_mask[None, :], tl.exp(logits_m - m), 0.0)
        s = tl.sum(e, axis=1, keep_dims=True)
        scores = e / s
    elif SCORE_FN == 1:  # sigmoid
        scores = tl.sigmoid(logits)
    else:  # sqrtsoftplus (SCORE_FN == 2)
        softplus = tl.log(1.0 + tl.exp(logits))
        # Stable: for large x, log(1+exp(x)) ≈ x.
        scores = tl.sqrt(softplus)

    # Zero padded expert columns so they never contribute to gather / denom.
    scores = tl.where(e_mask[None, :], scores, 0.0)

    # Save full scores row for backward (real columns only).
    tl.store(
        SCORES_OUT_PTR + n_offs[:, None] * E + e_idx[None, :],
        scores,
        mask=n_mask[:, None] & e_mask[None, :],
    )

    # Load indices [BLOCK_N, BLOCK_K] (int64; padded cols read as 0, masked below).
    indices = tl.load(
        INDICES_PTR + n_offs[:, None] * BLOCK_K + k_idx[None, :],
        mask=n_mask[:, None] & k_mask[None, :],
        other=0,
    )

    # Gather weights from scores at the K indices per row entirely in
    # registers (avoid store-then-reload-via-HBM hazard). Build
    # weights[BLOCK_N, BLOCK_K] via a static loop over BLOCK_K, extracting
    # each column from the already-loaded ``indices`` tile.
    weights = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k_off in tl.static_range(BLOCK_K):
        k_is_off = (k_idx == k_off).to(tl.float32)
        idx_col = tl.sum(tl.where(k_idx[None, :] == k_off, indices, 0), axis=1)
        # scores_at_idx[n] = scores[n, idx_col[n]]
        e_is_idx = e_idx[None, :] == idx_col[:, None]
        scores_at_idx = tl.sum(scores * e_is_idx.to(tl.float32), axis=1)
        weights += scores_at_idx[:, None] * k_is_off[None, :]

    # Zero padded topk columns (k >= K).
    weights = tl.where(k_mask[None, :], weights, 0.0)

    # If non-softmax: denom + normalize.
    if SCORE_FN != 0:
        s = tl.sum(weights, axis=1, keep_dims=True)
        denom = tl.maximum(s, EPS)
        weights = weights / denom
        denom_scalar = tl.reshape(denom, (BLOCK_N,))
    else:
        denom_scalar = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
    tl.store(DENOM_OUT_PTR + n_offs, denom_scalar, mask=n_mask)

    # Scale.
    weights = weights * SCALE

    # Save weights for backward.
    tl.store(
        WEIGHTS_OUT_PTR + n_offs[:, None] * BLOCK_K + k_idx[None, :],
        weights,
        mask=n_mask[:, None] & k_mask[None, :],
    )

    # Sparse scatter (probs[n, indices] = weights, rmap[n, indices] = True).
    # No atomics needed because each (n, e) is written at most once
    # per row (caller guarantees indices are unique within a row).  Padded
    # topk columns are masked out so they never write to expert 0.
    tl.store(
        PROBS_PTR + n_offs[:, None] * E + indices,
        weights.to(OUT_DTYPE),
        mask=n_mask[:, None] & k_mask[None, :],
    )
    tl.store(
        RMAP_PTR + n_offs[:, None] * E + indices,
        tl.full((BLOCK_N, BLOCK_K), 1, dtype=tl.int1),
        mask=n_mask[:, None] & k_mask[None, :],
    )


@triton.jit
def _v4_router_post_bwd_kernel(
    DPROBS_PTR,  # [N, E] OUT_DTYPE upstream grad
    INDICES_PTR,  # [N, BLOCK_K] int64 (padded)
    SCORES_PTR,  # [N, E] fp32 saved
    WEIGHTS_PTR,  # [N, BLOCK_K] fp32 saved (post-scaled, padded)
    DENOM_PTR,  # [N]   fp32 saved
    DLOGITS_PTR,  # [N, E] fp32 OUT
    N,
    E,  # runtime int
    K,  # runtime int
    SCORE_FN: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """VJP through the chain.

    FWD:
        scores[n, e]  = score_fn(logits[n, e])
        weights[n, k] = scores[n, indices[n, k]]
        if not softmax: weights /= sum_k weights[n, k]
        weights *= scale
        probs[n, indices[n, k]] = weights

    BWD (gather only the K positions touched by indices; others get 0):
        dprobs_at[n, k] = dprobs[n, indices[n, k]]
        dweights[n, k]  = dprobs_at[n, k] * SCALE   (chain through scale)
        If not softmax:
            saved weights = (gathered / denom) * SCALE
            d_pre_denom[n, k] = dweights[n, k] / denom
            d_denom[n] = -sum_k (dweights[n, k] * weights_scaled[n, k]) / (denom * scale)
                (because weights_scaled = gathered * scale / denom, so
                 ∂w/∂gathered = scale/denom, ∂w/∂denom = -gathered * scale / denom^2)
            d_gathered[n, k] = dweights[n, k] * scale / denom + d_denom[n] * 1
                Wait, denom = sum_k gathered, so ∂denom/∂gathered_k = 1.
                Therefore d_gathered[n, k] = dweights * scale/denom + d_denom (each gathered contributes to denom)
        else (softmax):
            d_gathered[n, k] = dweights[n, k]
        Scatter d_gathered back into d_scores at positions indices[n, k]
        (rest of d_scores is 0).
        Finally chain through score_fn:
            d_logits[n, e] = vjp_score_fn(d_scores[n, e], scores[n, e])
    """

    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    e_idx = tl.arange(0, BLOCK_E)
    e_mask = e_idx < E
    k_idx = tl.arange(0, BLOCK_K)
    k_mask = k_idx < K

    # Load saved state.
    scores = tl.load(
        SCORES_PTR + n_offs[:, None] * E + e_idx[None, :],
        mask=n_mask[:, None] & e_mask[None, :],
        other=0.0,
    )
    indices = tl.load(
        INDICES_PTR + n_offs[:, None] * BLOCK_K + k_idx[None, :],
        mask=n_mask[:, None] & k_mask[None, :],
        other=0,
    )

    # Gather upstream grad at the K indices per row.
    # dprobs_at = dprobs[n, indices[n, k]]   shape [BLOCK_N, BLOCK_K], fp32.
    # Padded topk columns are masked to 0 so they carry no gradient.
    dprobs_at = tl.load(
        DPROBS_PTR + n_offs[:, None] * E + indices,
        mask=n_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    # Chain through the SCALE multiply: forward was
    #     weights_post = weights_pre * SCALE
    # so d_weights_pre = d_weights_post * SCALE.
    dweights_pre_scale = dprobs_at * SCALE

    if SCORE_FN != 0:
        # Non-softmax: forward applied
        #     weights_pre = gathered / denom
        # so d_gathered_k = d_weights_pre_k / denom + d_denom
        # where d_denom = -sum_k d_weights_pre_k * gathered_k / denom^2.
        # Recover gathered_k from saved state:
        #     weights_saved_k = (gathered_k / denom) * SCALE
        # so gathered_k = weights_saved_k * denom / SCALE.
        denom = tl.load(DENOM_PTR + n_offs, mask=n_mask, other=1.0)
        weights_saved = tl.load(
            WEIGHTS_PTR + n_offs[:, None] * BLOCK_K + k_idx[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        gathered_k = weights_saved * (denom[:, None] / SCALE)
        d_denom_per_n = -tl.sum(dweights_pre_scale * gathered_k, axis=1) / (denom * denom)
        d_gathered = dweights_pre_scale / denom[:, None] + d_denom_per_n[:, None]
    else:
        # Softmax: weights_pre is just gathered directly.
        d_gathered = dweights_pre_scale

    # Zero padded topk columns so they contribute no gradient.
    d_gathered = tl.where(k_mask[None, :], d_gathered, 0.0)

    # Build d_scores_full entirely in registers using an explicit static
    # loop over BLOCK_K.  For each k, we add d_gathered[:, k] at the position
    # indices[:, k] in the E-axis using a broadcast compare.  This avoids
    # the round-trip-via-HBM hazard of a scatter-then-load pattern.  Padded
    # columns carry d_gathered == 0, so they add nothing even though their
    # index reads as 0.
    d_scores_full = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    for k_off in tl.static_range(BLOCK_K):
        idx_col = tl.sum(tl.where(k_idx[None, :] == k_off, indices, 0), axis=1)
        # Slice d_gathered[:, k_off] -> shape [BLOCK_N]
        grad_k = tl.sum(d_gathered * (k_idx[None, :] == k_off).to(tl.float32), axis=1)
        # Contribution: grad_k[:, None] where e_idx == idx_col[:, None]
        e_is_idx = e_idx[None, :] == idx_col[:, None]
        d_scores_full += tl.where(e_is_idx, grad_k[:, None], 0.0)

    if SCORE_FN == 0:  # softmax: d_logits = scores * (d_scores - sum(d_scores * scores))
        dot = tl.sum(d_scores_full * scores, axis=1, keep_dims=True)
        d_logits = scores * (d_scores_full - dot)
    elif SCORE_FN == 1:  # sigmoid
        d_logits = d_scores_full * scores * (1.0 - scores)
    else:  # sqrtsoftplus: y = sqrt(softplus(x)); dy/dx = (1 / (2 * sqrt(softplus(x)))) * sigmoid(x)
        # = sigmoid(x) / (2 * y); avoid div-by-zero by guarding small y.
        # sigmoid(x) = exp(x) / (1 + exp(x)); when y is close to 0, x is very negative,
        # and sigmoid(x) is also very small, so the limit is 0.
        # Use: dy/dx = sigmoid(x) / (2 * scores)
        # but we don't have x.  Recompute: scores = y = sqrt(softplus(x))
        # so softplus(x) = y^2 = scores^2, and exp(x) = exp(scores^2) - 1? No:
        # softplus(x) = log(1 + exp(x)) = y^2 -> exp(x) = exp(y^2) - 1 -> sigmoid(x) = (exp(y^2) - 1) / exp(y^2)
        # = 1 - exp(-y^2).
        sig_x = 1.0 - tl.exp(-scores * scores)
        y_safe = tl.maximum(scores, EPS)
        d_logits = d_scores_full * sig_x / (2.0 * y_safe)

    tl.store(
        DLOGITS_PTR + n_offs[:, None] * E + e_idx[None, :],
        d_logits,
        mask=n_mask[:, None] & e_mask[None, :],
    )


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------


class V4RouterPostFn(torch.autograd.Function):
    """Autograd-aware wrapper around the FWD/BWD Triton kernels.

    Inputs:
        logits [N, E] fp32  -- gate output (pre-score-fn).
        indices [N, K] long -- gather positions (from host-side topk
                               or hash table).  Must be unique per row.
        score_function: "softmax" | "sigmoid" | "sqrtsoftplus".
        topk_scaling_factor: float multiplier applied after denom.
        out_dtype: dtype of the returned `probs` tensor.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        logits: torch.Tensor,
        indices: torch.Tensor,
        score_function: str,
        topk_scaling_factor: float,
        out_dtype: torch.dtype,
    ):
        if logits.dim() != 2:
            raise ValueError(f"logits must be [N, E], got shape {tuple(logits.shape)}")
        if indices.dim() != 2:
            raise ValueError(f"indices must be [N, K], got shape {tuple(indices.shape)}")
        if logits.shape[0] != indices.shape[0]:
            raise ValueError(f"logits N={logits.shape[0]} != indices N={indices.shape[0]}")
        if score_function not in _SCORE_FN_MAP:
            raise ValueError(
                f"Unknown score_function: {score_function!r}; expected one of "
                f"{sorted(_SCORE_FN_MAP.keys())}"
            )
        N, E = logits.shape
        _, K = indices.shape
        # Arbitrary (non-power-of-2) E / K are supported: the kernel tiles
        # both axes by their next power of 2 and masks the padded columns.
        block_e = triton.next_power_of_2(E)
        block_k = triton.next_power_of_2(K)

        score_fn_enum = _SCORE_FN_MAP[score_function]
        logits_c = logits.contiguous().to(torch.float32)
        indices_c = indices.contiguous().to(torch.int64)
        # Pad the indices tensor to [N, BLOCK_K] so the kernel's BLOCK_K-wide
        # loads stay in-bounds; padded columns (value 0) are masked out
        # everywhere via k_mask so they never gather / scatter / grad.
        if block_k != K:
            indices_pad = torch.zeros((N, block_k), dtype=torch.int64, device=indices_c.device)
            indices_pad[:, :K] = indices_c
        else:
            indices_pad = indices_c

        device = logits_c.device
        probs = torch.zeros((N, E), dtype=out_dtype, device=device)
        routing_map = torch.zeros((N, E), dtype=torch.bool, device=device)
        scores_saved = torch.empty((N, E), dtype=torch.float32, device=device)
        weights_saved = torch.zeros((N, block_k), dtype=torch.float32, device=device)
        denom_saved = torch.empty((N,), dtype=torch.float32, device=device)

        # BLOCK_N heuristic: per-row state ~ BLOCK_E + BLOCK_K + few scalars.
        if block_e <= 64:
            block_n = 64
        elif block_e <= 256:
            block_n = 16
        else:
            block_n = 4
        grid = (triton.cdiv(N, block_n),)

        _v4_router_post_fwd_kernel[grid](
            logits_c,
            indices_pad,
            probs,
            routing_map,
            scores_saved,
            weights_saved,
            denom_saved,
            N,
            E,
            K,
            SCORE_FN=score_fn_enum,
            SCALE=float(topk_scaling_factor),
            EPS=1e-12,
            BLOCK_N=block_n,
            BLOCK_E=block_e,
            BLOCK_K=block_k,
            OUT_DTYPE={
                torch.float32: tl.float32,
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float64: tl.float64,
            }[out_dtype],
        )

        ctx.save_for_backward(indices_pad, scores_saved, weights_saved, denom_saved)
        ctx.score_fn_enum = score_fn_enum
        ctx.scale = float(topk_scaling_factor)
        ctx.E = E
        ctx.K = K
        ctx.block_e = block_e
        ctx.block_k = block_k
        ctx.block_n = block_n
        ctx.in_dtype = logits.dtype
        return probs, routing_map

    @staticmethod
    def backward(ctx, d_probs, d_routing_map):  # type: ignore[override]
        indices_pad, scores_saved, weights_saved, denom_saved = ctx.saved_tensors
        E = ctx.E
        K = ctx.K
        block_e = ctx.block_e
        block_k = ctx.block_k
        block_n = ctx.block_n
        score_fn_enum = ctx.score_fn_enum
        scale = ctx.scale
        in_dtype = ctx.in_dtype

        N = indices_pad.shape[0]
        device = indices_pad.device

        d_probs = d_probs.contiguous()
        # d_routing_map ignored (bool tensor, no grad flow).
        # Initialise d_logits to 0 so the scatter writes only at K positions per row.
        d_logits_fp32 = torch.zeros((N, E), dtype=torch.float32, device=device)

        grid = (triton.cdiv(N, block_n),)
        _v4_router_post_bwd_kernel[grid](
            d_probs,
            indices_pad,
            scores_saved,
            weights_saved,
            denom_saved,
            d_logits_fp32,
            N,
            E,
            K,
            SCORE_FN=score_fn_enum,
            SCALE=scale,
            EPS=1e-12,
            BLOCK_N=block_n,
            BLOCK_E=block_e,
            BLOCK_K=block_k,
        )

        return d_logits_fp32.to(in_dtype), None, None, None, None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def is_triton_path_enabled() -> bool:
    """Return True iff ``PRIMUS_V4_ROUTER_TRITON != "0"`` (default ``"1"``).

    Plan-8 P57 close-out 2 (2026-05-15): default flipped from ``"0"``
    to ``"1"``.  Microbench at V4-Flash widths is a clear positive on
    V4's production `sqrtsoftplus` score function (1.56x FWD / 1.22x
    BWD); EP=8 proxy A/B (P39 / P43) sat inside the proxy noise band,
    so we default the microbench-positive kernel ON to keep it on the
    production path.  Set ``PRIMUS_V4_ROUTER_TRITON=0`` to revert to
    the eager body.
    """
    return os.environ.get("PRIMUS_V4_ROUTER_TRITON", "1") != "0"


def v4_router_post_triton(
    logits: torch.Tensor,
    indices: torch.Tensor,
    *,
    score_function: str,
    topk_scaling_factor: float,
    out_dtype: torch.dtype,
):
    """Run the fused V4 router post-logits kernel.

    Returns ``(probs, routing_map)``.  Caller pre-computes ``indices``
    (hash router via ``tid2eid[flat_ids]``; learned router via
    ``torch.topk(sel_score, K).indices``).

    Arbitrary (non-power-of-2) ``E`` / ``K`` are supported — the kernel
    tiles both axes by the next power of 2 and masks the padded columns —
    so the only hard requirement is that the tensors live on the GPU.
    """
    assert logits.is_cuda and indices.is_cuda, "v4_router_post_triton requires CUDA / HIP tensors"
    return V4RouterPostFn.apply(logits, indices, score_function, topk_scaling_factor, out_dtype)


__all__ = [
    "V4RouterPostFn",
    "v4_router_post_triton",
    "is_triton_path_enabled",
]
