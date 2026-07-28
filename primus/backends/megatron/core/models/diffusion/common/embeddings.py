# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Embedding layers for diffusion models.

Provides timestep and vector conditioning embeddings used across
Flux and other diffusion architectures.

This module implements:
    - TimeStepEmbedder: Sinusoidal timestep embedding with MLP projection
    - MLPEmbedder: Simple 2-layer MLP for vector conditioning (e.g., CLIP pooled)
    - Timesteps: Helper module for sinusoidal timestep encoding
    - get_timestep_embedding: Standalone function for sinusoidal embeddings

Reference:
    - Based on Denoising Diffusion Probabilistic Models (Ho et al., 2020)
"""

import math

import torch
import torch.nn as nn
from torch import Tensor


class TimeStepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal encoding.

    This module converts scalar timestep values into high-dimensional embeddings
    using sinusoidal position encoding (similar to transformers), followed by
    an MLP projection.

    Architecture:
        1. Sinusoidal position encoding (creates periodic features)
        2. Linear projection to hidden_dim
        3. SiLU activation
        4. Linear projection to hidden_dim

    Args:
        embedding_dim: Number of channels for sinusoidal encoding (typically 256)
        hidden_dim: Hidden dimension for MLP projection (model hidden size, e.g., 3072)
        flip_sin_to_cos: Whether to flip sine and cosine order (default: True)
        downscale_freq_shift: Controls delta between frequencies (default: 0)
        scale: Scaling factor for timesteps (default: 1.0)
        max_period: Maximum period for sinusoidal encoding (default: 10000)

    Input:
        t: Timestep scalars [B] or [B, 1], typically in range [0, 1000]

    Output:
        Timestep embeddings [B, hidden_dim]

    Example:
        >>> embedder = TimeStepEmbedder(embedding_dim=256, hidden_dim=3072)
        >>> timesteps = torch.randn(4)  # Batch of 4 timesteps
        >>> t_emb = embedder(timesteps)
        >>> assert t_emb.shape == (4, 3072)

    Reference:
        - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        - Adapted from NeMo's Flux implementation
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

        # MLP for projecting sinusoidal embeddings to hidden_dim
        self.time_proj = Timesteps(
            embedding_dim=embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift,
            scale=scale,
            max_period=max_period,
        )

        self.time_embedding = MLPEmbedder(in_dim=embedding_dim, hidden_dim=hidden_dim)

    def forward(self, t: Tensor) -> Tensor:
        """
        Forward pass: Convert timesteps to embeddings.

        Args:
            t: Timesteps [B] or [B, 1], typically in range [0, 1000]

        Returns:
            Timestep embeddings [B, hidden_dim]
        """
        # Get sinusoidal embeddings
        t_emb = self.time_proj(t)  # [B, embedding_dim]

        # Project through MLP
        t_emb = self.time_embedding(t_emb)  # [B, hidden_dim]

        return t_emb


class Timesteps(nn.Module):
    """
    Converts timesteps to sinusoidal embeddings.

    This is a helper module that creates sinusoidal position encodings
    from scalar timestep values.

    Args:
        embedding_dim: Dimension of output embeddings
        flip_sin_to_cos: Whether to order as [cos, sin] instead of [sin, cos]
        downscale_freq_shift: Frequency shift parameter
        scale: Scaling factor for embeddings
        max_period: Maximum period for sinusoidal functions

    Input:
        timesteps: Scalar timesteps [B]

    Output:
        Sinusoidal embeddings [B, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
        scale: float = 1.0,
        max_period: int = 10000,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            timesteps: Timesteps [B]

        Returns:
            Sinusoidal embeddings [B, embedding_dim]
        """
        t_emb = get_timestep_embedding(
            timesteps,
            self.embedding_dim,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
            max_period=self.max_period,
        )
        return t_emb


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    This matches the implementation in Denoising Diffusion Probabilistic Models.
    It creates position encodings using sine and cosine functions at different
    frequencies.

    Args:
        timesteps: 1-D Tensor of N indices, one per batch element [B]
        embedding_dim: Dimension of the output embeddings
        flip_sin_to_cos: Whether embedding order should be [cos, sin] (True) or [sin, cos] (False)
        downscale_freq_shift: Controls delta between frequencies
        scale: Scaling factor applied to embeddings
        max_period: Controls maximum frequency of embeddings

    Returns:
        Tensor of positional embeddings [B, embedding_dim]

    Reference:
        - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
        - Adapted from NeMo's Flux implementation
    """
    if len(timesteps.shape) != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2

    # Remember input dtype to preserve it
    input_dtype = timesteps.dtype

    # Compute frequencies
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # Scale embeddings
    emb = scale * emb

    # Concatenate sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # Flip sine and cosine embeddings if requested
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # Zero pad if embedding_dim is odd
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))

    # Convert back to input dtype
    emb = emb.to(input_dtype)

    return emb


class MLPEmbedder(nn.Module):
    """
    Simple 2-layer MLP for vector conditioning (e.g., CLIP pooled embeddings).

    This module provides a learnable projection from an input vector space
    to the model's hidden dimension space. It's commonly used for:
    - CLIP pooled text embeddings: [B, 768] -> [B, hidden_dim]
    - Other vector-level conditioning signals

    Architecture:
        1. Linear projection: in_dim -> hidden_dim
        2. SiLU activation
        3. Linear projection: hidden_dim -> hidden_dim

    Args:
        in_dim: Input dimension (e.g., 768 for CLIP-L pooled embeddings)
        hidden_dim: Hidden dimension (model hidden size, e.g., 3072)

    Input:
        x: Vector conditioning [B, in_dim]

    Output:
        Embedded conditioning [B, hidden_dim]

    Example:
        >>> # For CLIP-L pooled embeddings
        >>> embedder = MLPEmbedder(in_dim=768, hidden_dim=3072)
        >>> clip_pooled = torch.randn(4, 768)
        >>> embedded = embedder(clip_pooled)
        >>> assert embedded.shape == (4, 3072)

    Reference:
        - Adapted from NeMo's Flux implementation
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # Two-layer MLP with SiLU activation
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: Project input vectors to hidden dimension.

        Args:
            x: Input vectors [B, in_dim]

        Returns:
            Embedded vectors [B, hidden_dim]
        """
        x = self.in_layer(x)
        x = self.silu(x)
        x = self.out_layer(x)
        return x
