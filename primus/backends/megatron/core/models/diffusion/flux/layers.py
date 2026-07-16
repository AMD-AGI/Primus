# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux-specific layers and components.

This module implements components unique to the Flux architecture:
    - EmbedND: Multi-dimensional Rotary Position Embedding (3D RoPE)
    - Helper functions for position encoding

Reference:
    - Flux Paper: "Flux: A Scalable Diffusion Model"
    - RoPE Paper: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    - Adapted from NeMo's Flux layers
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    """
    Generate RoPE (Rotary Position Embedding) frequencies for given positions.

    This is adapted for Megatron-Core's attention implementation, which
    calculates sin/cos internally. We only generate the frequency matrix here.

    Args:
        pos: Position indices [..., n] where n is the number of positions
        dim: Dimension of the embedding (must be even)
        theta: Base for frequency computation (typically 10000)

    Returns:
        Frequency matrix [..., n, dim/2] for RoPE computation

    Note:
        This differs from standard RoPE implementations because Megatron
        attention applies sin/cos internally, so we only provide frequencies.

    Reference:
        - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        - Adapted from NeMo's Flux layers
    """
    if dim % 2 != 0:
        raise ValueError("The dimension must be even for RoPE.")

    # Compute scaling factors for each dimension pair
    # scale: [0, 2, 4, ..., dim-2] / dim = [0, 1/dim, 2/dim, ..., (dim-2)/dim]
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim

    # Compute base frequencies: 1 / (theta ^ scale)
    omega = 1.0 / (theta**scale)

    # Outer product of positions and frequencies
    # pos: [..., n], omega: [dim/2] -> out: [..., n, dim/2]
    out = torch.einsum("...n,d->...nd", pos, omega)

    return out.float()


class EmbedND(nn.Module):
    """
    Multi-dimensional Rotary Position Embedding (RoPE) for images.

    Flux uses 3D RoPE with three axes:
        - Axis 0: Always 0 for 2D images (reserved for video/temporal)
        - Axis 1: Height positions (y-coordinates)
        - Axis 2: Width positions (x-coordinates)

    Each axis gets independent sinusoidal frequencies, concatenated to form
    the complete position embedding.

    Args:
        dim: Model hidden dimension (stored for reference, not used in forward)
        theta: Base for frequency computation (default: 10000)
        axes_dim: Dimensions per axis, e.g., [16, 56, 56]

    Input:
        ids: Position IDs [B, seq, num_axes] -- for images: [B, H*W, 3]

    Output:
        RoPE frequency matrix [seq, B, 1, dim] for Megatron attention

    Reference:
        - Flux: 3D RoPE for spatial + channel position encoding
        - RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        - Adapted from NeMo's Flux layers
    """

    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        Compute 3D RoPE frequencies from position IDs.

        Args:
            ids: Position IDs [B, seq, num_axes] where num_axes = len(axes_dim)

        Returns:
            RoPE frequency matrix [seq, B, 1, dim] for Megatron attention
        """
        n_axes = ids.shape[-1]

        # Generate RoPE frequencies for each axis and concatenate
        # For each axis i:
        #   - ids[..., i]: positions for that axis [B, seq]
        #   - rope(): generates frequencies [B, seq, axes_dim[i]/2]
        # Concatenate along last dimension to get [B, seq, dim/2]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-1,
        )

        # Reshape for Megatron attention format:
        # 1. Add sequence dimension: [B, seq, dim/2] -> [B, 1, seq, dim/2]
        # 2. Permute to [seq, B, 1, dim/2]
        emb = emb.unsqueeze(1).permute(2, 0, 1, 3)

        # Stack [cos, sin] pairs and reshape to final format
        # torch.stack([emb, emb], dim=-1): [seq, B, 1, dim/2, 2]
        # reshape: [seq, B, 1, dim]
        return torch.stack([emb, emb], dim=-1).reshape(*emb.shape[:-1], -1)
