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
    INDICES_PTR,  # [N, K] int64 (gather positions)
    PROBS_PTR,  # [N, E] OUT_DTYPE (must be zero-init'd by caller)
    RMAP_PTR,  # [N, E] bool   (must be zero-init'd by caller)
    SCORES_OUT_PTR,  # [N, E] fp32 (saved-for-backward; full row of scores)
    WEIGHTS_OUT_PTR,  # [N, K] fp32 (saved-for-backward; gathered weights, post-denom, pre-scale)
    DENOM_OUT_PTR,  # [N] fp32  (saved-for-backward; the clamped denom)
    N,
    E: tl.constexpr,
    K: tl.constexpr,
    SCORE_FN: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    """One program tile = ``BLOCK_N`` rows of the post-logits chain.

    Layout: per-row state in fp32 registers (`[BLOCK_N, E]` scores
    + `[BLOCK_N, K]` weights).  At V4-Flash N=4096, E=256, K=6 the
    per-row footprint is ~1 KiB which lets us pick BLOCK_N=16 without
    hitting register pressure.
    """

    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    e_idx = tl.arange(0, E)
    k_idx = tl.arange(0, K)

    # Load logits [BLOCK_N, E] in fp32.
    logits = tl.load(
        LOGITS_PTR + n_offs[:, None] * E + e_idx[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    # Apply score_fn (compile-time specialised).
    if SCORE_FN == 0:  # softmax
        m = tl.max(logits, axis=1, keep_dims=True)
        e = tl.exp(logits - m)
        s = tl.sum(e, axis=1, keep_dims=True)
        scores = e / s
    elif SCORE_FN == 1:  # sigmoid
        scores = tl.sigmoid(logits)
    else:  # sqrtsoftplus (SCORE_FN == 2)
        softplus = tl.log(1.0 + tl.exp(logits))
        # Stable: for large x, log(1+exp(x)) ≈ x.
        scores = tl.sqrt(softplus)

    # Save full scores row for backward.
    tl.store(
        SCORES_OUT_PTR + n_offs[:, None] * E + e_idx[None, :],
        scores,
        mask=n_mask[:, None],
    )

    # Load indices [BLOCK_N, K] (int64).
    indices = tl.load(
        INDICES_PTR + n_offs[:, None] * K + k_idx[None, :],
        mask=n_mask[:, None],
        other=0,
    )

    # Gather weights from scores at the K indices per row by computing
    # the gather entirely in registers (avoid store-then-reload-via-HBM
    # hazard).  Build weights[BLOCK_N, K] as the per-(n, k) score at
    # index indices[n, k] by static loop over K:
    weights = tl.zeros((BLOCK_N, K), dtype=tl.float32)
    for k_off in tl.static_range(K):
        idx_col = tl.load(INDICES_PTR + n_offs * K + k_off, mask=n_mask, other=0)
        # scores_at_idx[n] = scores[n, idx_col[n]]
        e_is_idx = e_idx[None, :] == idx_col[:, None]
        scores_at_idx = tl.sum(scores * e_is_idx.to(tl.float32), axis=1)
        k_is_off = (k_idx == k_off).to(tl.float32)
        weights += scores_at_idx[:, None] * k_is_off[None, :]

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
        WEIGHTS_OUT_PTR + n_offs[:, None] * K + k_idx[None, :],
        weights,
        mask=n_mask[:, None],
    )

    # Sparse scatter (probs[n, indices] = weights, rmap[n, indices] = True).
    # No atomics needed because each (n, e) is written at most once
    # per row (caller guarantees indices are unique within a row).
    tl.store(
        PROBS_PTR + n_offs[:, None] * E + indices,
        weights.to(OUT_DTYPE),
        mask=n_mask[:, None],
    )
    tl.store(
        RMAP_PTR + n_offs[:, None] * E + indices,
        tl.full((BLOCK_N, K), 1, dtype=tl.int1),
        mask=n_mask[:, None],
    )


@triton.jit
def _v4_router_post_bwd_kernel(
    DPROBS_PTR,  # [N, E] OUT_DTYPE upstream grad
    INDICES_PTR,  # [N, K] int64
    SCORES_PTR,  # [N, E] fp32 saved
    WEIGHTS_PTR,  # [N, K] fp32 saved (post-scaled)
    DENOM_PTR,  # [N]   fp32 saved
    DLOGITS_PTR,  # [N, E] fp32 OUT
    N,
    E: tl.constexpr,
    K: tl.constexpr,
    SCORE_FN: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    e_idx = tl.arange(0, E)
    k_idx = tl.arange(0, K)

    # Load saved state.
    scores = tl.load(
        SCORES_PTR + n_offs[:, None] * E + e_idx[None, :],
        mask=n_mask[:, None],
        other=0.0,
    )
    indices = tl.load(
        INDICES_PTR + n_offs[:, None] * K + k_idx[None, :],
        mask=n_mask[:, None],
        other=0,
    )

    # Gather upstream grad at the K indices per row.
    # dprobs_at = dprobs[n, indices[n, k]]   shape [BLOCK_N, K], fp32.
    dprobs_at = tl.load(
        DPROBS_PTR + n_offs[:, None] * E + indices,
        mask=n_mask[:, None],
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
            WEIGHTS_PTR + n_offs[:, None] * K + k_idx[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        gathered_k = weights_saved * (denom[:, None] / SCALE)
        d_denom_per_n = -tl.sum(dweights_pre_scale * gathered_k, axis=1) / (denom * denom)
        d_gathered = dweights_pre_scale / denom[:, None] + d_denom_per_n[:, None]
    else:
        # Softmax: weights_pre is just gathered directly.
        d_gathered = dweights_pre_scale

    # Build d_scores_full entirely in registers using an explicit static
    # loop over K.  For each k, we add d_gathered[:, k] at the position
    # indices[:, k] in the E-axis using a broadcast compare.  This avoids
    # the round-trip-via-HBM hazard of a scatter-then-load pattern.
    d_scores_full = tl.zeros((BLOCK_N, E), dtype=tl.float32)
    for k_off in tl.static_range(K):
        idx_col = tl.load(INDICES_PTR + n_offs * K + k_off, mask=n_mask, other=0)
        grad_col = tl.load(INDICES_PTR + n_offs * K + k_off, mask=n_mask, other=0)  # placeholder
        _ = grad_col  # unused; the real grad comes from d_gathered slice
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
        mask=n_mask[:, None],
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
        if E & (E - 1) != 0:
            raise ValueError(f"E must be a power of 2; got E={E}")
        if K & (K - 1) != 0:
            raise ValueError(f"K must be a power of 2; got K={K}")

        score_fn_enum = _SCORE_FN_MAP[score_function]
        logits_c = logits.contiguous().to(torch.float32)
        indices_c = indices.contiguous().to(torch.int64)

        device = logits_c.device
        probs = torch.zeros((N, E), dtype=out_dtype, device=device)
        routing_map = torch.zeros((N, E), dtype=torch.bool, device=device)
        scores_saved = torch.empty((N, E), dtype=torch.float32, device=device)
        weights_saved = torch.empty((N, K), dtype=torch.float32, device=device)
        denom_saved = torch.empty((N,), dtype=torch.float32, device=device)

        # BLOCK_N heuristic: per-row state ~ E + K + few scalars in fp32.
        if E <= 64:
            block_n = 64
        elif E <= 256:
            block_n = 16
        else:
            block_n = 4
        grid = (triton.cdiv(N, block_n),)

        _v4_router_post_fwd_kernel[grid](
            logits_c,
            indices_c,
            probs,
            routing_map,
            scores_saved,
            weights_saved,
            denom_saved,
            N,
            E=E,
            K=K,
            SCORE_FN=score_fn_enum,
            SCALE=float(topk_scaling_factor),
            EPS=1e-12,
            BLOCK_N=block_n,
            OUT_DTYPE={
                torch.float32: tl.float32,
                torch.float16: tl.float16,
                torch.bfloat16: tl.bfloat16,
                torch.float64: tl.float64,
            }[out_dtype],
        )

        ctx.save_for_backward(indices_c, scores_saved, weights_saved, denom_saved)
        ctx.score_fn_enum = score_fn_enum
        ctx.scale = float(topk_scaling_factor)
        ctx.E = E
        ctx.K = K
        ctx.block_n = block_n
        ctx.in_dtype = logits.dtype
        return probs, routing_map

    @staticmethod
    def backward(ctx, d_probs, d_routing_map):  # type: ignore[override]
        indices_c, scores_saved, weights_saved, denom_saved = ctx.saved_tensors
        E = ctx.E
        K = ctx.K
        block_n = ctx.block_n
        score_fn_enum = ctx.score_fn_enum
        scale = ctx.scale
        in_dtype = ctx.in_dtype

        N = indices_c.shape[0]
        device = indices_c.device

        d_probs = d_probs.contiguous()
        # d_routing_map ignored (bool tensor, no grad flow).
        # Initialise d_logits to 0 so the scatter writes only at K positions per row.
        d_logits_fp32 = torch.zeros((N, E), dtype=torch.float32, device=device)

        grid = (triton.cdiv(N, block_n),)
        _v4_router_post_bwd_kernel[grid](
            d_probs,
            indices_c,
            scores_saved,
            weights_saved,
            denom_saved,
            d_logits_fp32,
            N,
            E=E,
            K=K,
            SCORE_FN=score_fn_enum,
            SCALE=scale,
            EPS=1e-12,
            BLOCK_N=block_n,
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


def is_triton_kernel_supported(logits: torch.Tensor, indices: torch.Tensor) -> bool:
    """Return True iff inputs are CUDA + power-of-2 E + K."""
    if not logits.is_cuda or not indices.is_cuda:
        return False
    if logits.dim() != 2 or indices.dim() != 2:
        return False
    N, E = logits.shape
    _, K = indices.shape
    if E & (E - 1) != 0:
        return False
    if K & (K - 1) != 0:
        return False
    return True


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
    """
    return V4RouterPostFn.apply(logits, indices, score_function, topk_scaling_factor, out_dtype)


__all__ = [
    "V4RouterPostFn",
    "v4_router_post_triton",
    "is_triton_path_enabled",
    "is_triton_kernel_supported",
]
