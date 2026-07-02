###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
FlexAttention backend — pure PyTorch, relies on torch.compile for performance.

This module provides a drop-in replacement for the SDPA/FlashAttention paths
using ``torch.nn.attention.flex_attention``.  FlexAttention compiles a
user-defined ``score_mod`` function into a fused kernel via ``torch.compile``,
giving FlashAttention-level performance without third-party dependencies.

Key capabilities over SDPA:
    - Variable-length padding masks via ``create_block_mask``
    - Sliding-window attention via ``score_mod``
    - Composable attention score modifications
"""

from __future__ import annotations

import torch

# Import flex_attention from PyTorch (available since PyTorch 2.5+).
# We keep the import at module level so that ``FLEX_ATTENTION_AVAILABLE``
# reflects the actual environment at init time.
FLEX_ATTENTION_AVAILABLE = False
_flex_attention_compiled = None
_create_block_mask = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # torch.compile is essential for FlexAttention performance — without it,
    # score_mod/block_mask run as eager Python callbacks (separate kernel per op).
    # We attempt compile lazily on first call; if Triton/Inductor fails (e.g.
    # missing system libs on ROCm), we fall back to eager with a warning.
    _flex_attention_eager = flex_attention
    _flex_attention_compiled = None  # lazy-init on first call
    _create_block_mask = create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except Exception:
    FLEX_ATTENTION_AVAILABLE = False


def _get_flex_attention():
    """
    Return compiled flex_attention, falling back to eager if compile fails.

    torch.compile is lazy: wrapping succeeds immediately, but actual Triton
    codegen happens on first invocation and can fail (e.g. missing system libs
    on ROCm).  We handle this by catching runtime errors on first call and
    permanently switching to eager mode.
    """
    global _flex_attention_compiled
    if _flex_attention_compiled is not None:
        return _flex_attention_compiled

    import logging

    logger = logging.getLogger(__name__)
    try:
        compiled = torch.compile(_flex_attention_eager)
        logger.info("flex_attention: torch.compile wrapper created (lazy — actual compile on first call)")
        _flex_attention_compiled = compiled
    except Exception as e:
        logger.warning("flex_attention: torch.compile wrapping failed (%s), using eager mode", e)
        _flex_attention_compiled = _flex_attention_eager
    return _flex_attention_compiled


def _flex_attention_with_fallback(q, k, v, **kwargs):
    """Call flex_attention with runtime fallback if compiled version fails."""
    global _flex_attention_compiled
    fn = _get_flex_attention()
    try:
        return fn(q, k, v, **kwargs)
    except Exception as e:
        if fn is _flex_attention_eager:
            raise  # already in eager mode, real error
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "flex_attention: compiled call failed (%s: %s), falling back to eager mode",
            type(e).__name__,
            e,
        )
        _flex_attention_compiled = _flex_attention_eager
        return _flex_attention_eager(q, k, v, **kwargs)


def _build_score_mod(
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softmax_scale: float | None = None,
):
    """
    Compose a ``score_mod`` function for ``flex_attention``.

    The returned function has signature ``(score, b, h, q_idx, kv_idx) -> score``
    and is compiled by ``torch.compile`` into fused operations.
    """
    mods: list = []

    if softmax_scale is not None:
        # flex_attention already applies 1/sqrt(d) by default, so we only
        # inject a custom scale if the caller wants a non-default value.
        def scale_mod(score, b, h, q_idx, kv_idx):
            return score * softmax_scale

        mods.append(scale_mod)

    if causal:

        def causal_mod(score, b, h, q_idx, kv_idx):
            return torch.where(q_idx >= kv_idx, score, float("-inf"))

        mods.append(causal_mod)

    if window_size != (-1, -1):
        left, right = window_size

        def window_mod(score, b, h, q_idx, kv_idx):
            in_window = True
            if left >= 0:
                in_window = in_window & (q_idx - kv_idx <= left)
            if right >= 0:
                in_window = in_window & (kv_idx - q_idx <= right)
            return torch.where(in_window, score, float("-inf"))

        mods.append(window_mod)

    if not mods:
        return None

    # Compose all mods into one function.
    def composed_score_mod(score, b, h, q_idx, kv_idx):
        for mod in mods:
            score = mod(score, b, h, q_idx, kv_idx)
        return score

    return composed_score_mod


def _build_block_mask(
    B: int,
    N: int,
    Lq: int,
    Lk: int,
    q_lens: torch.Tensor | None,
    k_lens: torch.Tensor | None,
    device: torch.device,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
):
    """
    Build a ``BlockMask`` for ``flex_attention`` from variable-length seqs.

    If both ``q_lens`` and ``k_lens`` are None (uniform-length), returns None
    so that ``flex_attention`` uses a full dense mask.
    """
    if q_lens is None and k_lens is None and not causal and window_size == (-1, -1):
        return None

    # Ensure length tensors are on the same device as the mask indices
    # (create_block_mask uses vmap which creates index tensors on `device`)
    if q_lens is not None:
        q_lens = q_lens.to(device)
    if k_lens is not None:
        k_lens = k_lens.to(device)

    # Build the mask function. Closures capture the lengths.
    def mask_fn(b, h, q_idx, kv_idx):
        mask = True
        if q_lens is not None:
            mask = mask & (q_idx < q_lens[b])
        if k_lens is not None:
            mask = mask & (kv_idx < k_lens[b])
        if causal:
            mask = mask & (q_idx >= kv_idx)
        if window_size != (-1, -1):
            left, right = window_size
            if left >= 0:
                mask = mask & (q_idx - kv_idx <= left)
            if right >= 0:
                mask = mask & (kv_idx - q_idx <= right)
        return mask

    return _create_block_mask(mask_fn, B=B, H=N, Q_LEN=Lq, KV_LEN=Lk, device=device)


def flex_attention_fn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    fa_version=None,
) -> torch.Tensor:
    """
    Attention using ``torch.nn.attention.flex_attention``.

    Drop-in replacement for ``attention()`` — same signature, same tensor
    layout (q/k/v: ``[B, L, N, D]``).

    Args:
        q: Query tensor  ``[B, Lq, N, D]``
        k: Key tensor    ``[B, Lk, N, D]``
        v: Value tensor  ``[B, Lk, N, D]``
        q_lens: Optional valid lengths per batch for queries ``[B]``
        k_lens: Optional valid lengths per batch for keys   ``[B]``
        dropout_p: Dropout probability (passed through)
        softmax_scale: Custom softmax scale (None = default 1/sqrt(D))
        q_scale: Pre-multiply q by this scalar
        causal: Whether to apply causal mask
        window_size: (left, right) sliding window; (-1, -1) = disabled
        deterministic: Ignored (for API compat with flash_attention)
        dtype: Target dtype for computation
        fa_version: Ignored (for API compat)
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "flex_attention backend requested but torch.nn.attention.flex_attention "
            "is not available. Requires PyTorch >= 2.5."
        )

    out_dtype = q.dtype
    B, Lq, N, D = q.shape
    Lk = k.shape[1]

    # Cast to compute dtype
    half_dtypes = (torch.float16, torch.bfloat16)
    q_ = q.to(dtype) if q.dtype not in half_dtypes else q
    k_ = k.to(dtype) if k.dtype not in half_dtypes else k
    v_ = v.to(dtype) if v.dtype not in half_dtypes else v
    q_ = q_.to(v_.dtype)
    k_ = k_.to(v_.dtype)

    if q_scale is not None:
        q_ = q_ * q_scale

    # flex_attention expects [B, N, L, D] (heads-first)
    q_ = q_.transpose(1, 2)  # [B, N, Lq, D]
    k_ = k_.transpose(1, 2)  # [B, N, Lk, D]
    v_ = v_.transpose(1, 2)  # [B, N, Lk, D]

    # Build block mask (handles padding, causal, window in the mask itself)
    block_mask = _build_block_mask(
        B=B,
        N=N,
        Lq=Lq,
        Lk=Lk,
        q_lens=q_lens,
        k_lens=k_lens,
        device=q.device,
        causal=causal,
        window_size=window_size,
    )

    # Build score_mod only for softmax_scale override
    # (causal and window are handled in block_mask for better sparsity)
    score_mod = None
    if softmax_scale is not None:
        score_mod = _build_score_mod(softmax_scale=softmax_scale)

    # Call flex_attention (with runtime compile-error fallback)
    out = _flex_attention_with_fallback(q_, k_, v_, score_mod=score_mod, block_mask=block_mask)

    # Back to [B, L, N, D]
    out = out.transpose(1, 2).contiguous()
    return out.to(out_dtype)
