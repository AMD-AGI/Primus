###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Dual-RoPE for DeepSeek-V4.

Reference: techblog §6 ("RoPE: dual base + YaRN details").

V4 layers fall into two RoPE regimes, decided per-layer by
``compress_ratios[i]``:

* ``compress_ratio == 0`` (dense / SWA layers): use the **main** RoPE base
  (``rope_theta = 10000``), **no YaRN**.
* ``compress_ratio != 0`` (CSA / HCA layers): use the **compress** RoPE
  base (``compress_rope_theta = 160000``) **with YaRN scaling**
  (``factor=16, beta_fast=32, beta_slow=1, original_max_position_embeddings=65536``).

Two important corrections over the HF Llama-style RoPE that V4 inherits
from the released weights:

1. **Interleaved RoPE** (pairs ``(2k, 2k+1)``), **not** rotate-half pairs
   ``(d, d+rd/2)``. NeMo's port did this; HF PR 45616 originally did not.
2. **Partial RoPE**: only the last ``qk_pos_emb_head_dim`` (= 64) channels of
   each head are rotated. The first ``head_dim - 64`` channels stay nope.

This module exposes:

* :class:`DualRoPE` — owns two ``RoPECache`` instances (main + compress).
  ``forward(layer_compress_ratio, x, position_ids, *, key=False)`` applies
  the right partial-RoPE.
* :class:`RoPECache` — single-base RoPE, with optional YaRN scaling.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# YaRN scaling
# ---------------------------------------------------------------------------


def _yarn_freq_scaling(
    inv_freq: torch.Tensor,
    *,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    original_max_position_embeddings: int,
) -> torch.Tensor:
    """Apply YaRN frequency-band scaling to ``inv_freq``.

    Each frequency component ``inv_freq[i] = 1 / theta**(2i/D)``. YaRN groups
    the frequency band by wavelength ``2π / inv_freq[i]``:

    * High-freq band (wavelength < ``original_seq / beta_fast``): keep as-is.
    * Low-freq band  (wavelength > ``original_seq / beta_slow``): scale
      ``inv_freq[i]`` by ``1/factor`` (i.e. lengthen wavelength).
    * Mid-freq band: linear interpolation between the two.

    Standard YaRN reference; see e.g. NeMo's ``YarnRotaryPositionEmbedding``.
    """
    if factor == 1.0 or original_max_position_embeddings <= 0:
        return inv_freq

    # Wavelengths corresponding to each inv_freq component.
    wavelens = 2.0 * math.pi / inv_freq  # same shape as inv_freq

    low_thresh = original_max_position_embeddings / beta_fast
    high_thresh = original_max_position_embeddings / beta_slow

    # Linear blend factor: 0 at high-freq side (no scaling), 1 at low-freq side (full /factor scaling).
    smooth = ((wavelens - low_thresh) / max(high_thresh - low_thresh, 1e-12)).clamp(min=0.0, max=1.0)

    # Scaled inv_freq.
    inv_freq_scaled = inv_freq / factor
    return inv_freq * (1.0 - smooth) + inv_freq_scaled * smooth


def _yarn_attn_scale(factor: float) -> float:
    """The standard YaRN attention magnitude scale ``m_scale``.

    ``m_scale = 0.1 * log(factor) + 1`` — used to scale the attention
    softmax temperature when YaRN is on.
    """
    if factor <= 1.0:
        return 1.0
    return 0.1 * math.log(factor) + 1.0


# ---------------------------------------------------------------------------
# RoPE cache (single base)
# ---------------------------------------------------------------------------


class RoPECache(nn.Module):
    """Builds ``cos`` / ``sin`` tables on demand for a single RoPE base.

    Stores ``inv_freq`` as a (non-trainable) buffer so it follows the model's
    device / dtype movements. ``cos`` / ``sin`` are computed lazily on
    ``forward``.
    """

    def __init__(
        self,
        rotary_dim: int,
        *,
        theta: float,
        yarn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        original_max_position_embeddings: int = 0,
    ) -> None:
        super().__init__()
        if rotary_dim % 2 != 0:
            raise ValueError(f"rotary_dim must be even, got {rotary_dim}")

        self.rotary_dim = rotary_dim
        self.theta = float(theta)
        self.yarn_factor = float(yarn_factor)
        self.yarn_beta_fast = float(yarn_beta_fast)
        self.yarn_beta_slow = float(yarn_beta_slow)
        self.original_max_position_embeddings = int(original_max_position_embeddings)

        # inv_freq[i] = 1 / theta**(2i/rotary_dim), i in [0, rotary_dim/2)
        i = torch.arange(0, rotary_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (i / rotary_dim))
        inv_freq = _yarn_freq_scaling(
            inv_freq,
            factor=self.yarn_factor,
            beta_fast=self.yarn_beta_fast,
            beta_slow=self.yarn_beta_slow,
            original_max_position_embeddings=self.original_max_position_embeddings,
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # YaRN's m_scale; exposed for the caller to multiply attention scale.
        self.attn_scale: float = _yarn_attn_scale(self.yarn_factor)

    def forward(self, position_ids: torch.Tensor) -> tuple:
        """Return (cos, sin) for the given positions.

        ``position_ids``: any integer/float tensor (typically ``[B, S]``
        or ``[S]``).

        Output ``(cos, sin)`` shape is ``position_ids.shape + (rotary_dim/2,)``;
        the caller broadcasts over the head/batch axes.
        """
        # outer product positions × inv_freq
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq  # [..., rotary_dim/2]
        cos = freqs.cos()
        sin = freqs.sin()
        return cos, sin


# ---------------------------------------------------------------------------
# Partial interleaved RoPE application
# ---------------------------------------------------------------------------


def apply_interleaved_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    rotary_dim: int,
) -> torch.Tensor:
    """Apply RoPE to the **last** ``rotary_dim`` channels of ``x`` using the
    **interleaved** pairing convention: pairs ``(2k, 2k+1)``.

    Args:
        x: ``[..., head_dim]``. Typical layouts are ``[B, S, H, head_dim]``
            or ``[S, B, H, head_dim]``.
        cos, sin: must broadcast to ``x[..., :rotary_dim//2]`` after the
            internal pair-reshape (i.e. shape ``[..., rotary_dim/2]`` matching
            x's leading dims with a singleton heads axis inserted as needed).
            The simplest contract: pass shape ``[..., rotary_dim/2]`` where
            the leading dims are exactly the ``position_ids`` shape; this
            function will insert a singleton "heads" dim at position ``-2``
            so it broadcasts against the ``H`` axis of ``x``.
        rotary_dim: the partial RoPE size; must be ``<= x.shape[-1]`` and
            even.

    Returns:
        ``x`` with the last ``rotary_dim`` channels rotated; first
        ``head_dim - rotary_dim`` channels untouched.
    """
    head_dim = x.shape[-1]
    if rotary_dim > head_dim or rotary_dim % 2 != 0:
        raise ValueError(f"rotary_dim must be even and <= head_dim ({head_dim}), got {rotary_dim}")
    if rotary_dim == 0:
        return x

    nope = head_dim - rotary_dim
    x_nope = x[..., :nope]
    x_rope = x[..., nope:]

    # interleaved pairs: reshape last dim to (rotary_dim/2, 2)
    x_pairs = x_rope.reshape(*x_rope.shape[:-1], rotary_dim // 2, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]

    # Always insert a singleton "heads" axis at position -2 so cos/sin
    # broadcast across H. cos starts as ``position_ids.shape + [rd/2]``;
    # after unsqueeze it becomes ``position_ids.shape + [1, rd/2]`` which
    # broadcasts naturally against ``even`` of shape ``[..., H, rd/2]``.
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)

    rot_even = even * cos - odd * sin
    rot_odd = even * sin + odd * cos

    rotated = torch.stack([rot_even, rot_odd], dim=-1).reshape(*x_rope.shape[:-1], rotary_dim)
    return torch.cat([x_nope, rotated], dim=-1)


# ---------------------------------------------------------------------------
# DualRoPE — main + compress bases together
# ---------------------------------------------------------------------------


class DualRoPE(nn.Module):
    """Holds the two RoPE caches V4 needs and routes per-layer applications.

    Args:
        rotary_dim: partial-RoPE dim ``qk_pos_emb_head_dim`` (V4 = 64).
        rope_theta: base for dense / SWA layers.
        compress_rope_theta: base for CSA / HCA layers (longer base).
        yarn_factor / yarn_beta_fast / yarn_beta_slow /
            original_max_position_embeddings: YaRN config; applied **only**
            to the compress base. Set ``yarn_factor=1.0`` to disable.
    """

    def __init__(
        self,
        *,
        rotary_dim: int,
        rope_theta: float,
        compress_rope_theta: float,
        yarn_factor: float = 1.0,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        original_max_position_embeddings: int = 0,
    ) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim

        self.main_rope = RoPECache(
            rotary_dim=rotary_dim,
            theta=rope_theta,
        )
        self.compress_rope = RoPECache(
            rotary_dim=rotary_dim,
            theta=compress_rope_theta,
            yarn_factor=yarn_factor,
            yarn_beta_fast=yarn_beta_fast,
            yarn_beta_slow=yarn_beta_slow,
            original_max_position_embeddings=original_max_position_embeddings,
        )

    def get_rope(self, *, compress_ratio: int) -> RoPECache:
        """Pick the right cache for a layer.

        ``compress_ratio == 0`` → main (dense / SWA). Anything else →
        compress (CSA / HCA).
        """
        return self.main_rope if compress_ratio == 0 else self.compress_rope

    def apply(
        self,
        x: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        compress_ratio: int,
    ) -> torch.Tensor:
        """Convenience: pick the right rope, build cos/sin, apply partial RoPE.

        ``position_ids`` shape ``[B, S]`` or ``[S]``. Cos/sin will broadcast
        over any extra heads dim of ``x``.
        """
        rope = self.get_rope(compress_ratio=compress_ratio)
        cos, sin = rope(position_ids)
        return apply_interleaved_partial_rope(x, cos, sin, rotary_dim=self.rotary_dim)

    # Convenience accessors for callers who need the YaRN m_scale (e.g. to
    # adjust attention softmax scale on compressed layers).
    def attn_scale(self, *, compress_ratio: int) -> float:
        return self.get_rope(compress_ratio=compress_ratio).attn_scale


__all__ = [
    "RoPECache",
    "DualRoPE",
    "apply_interleaved_partial_rope",
]
