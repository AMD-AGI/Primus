# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Core utilities shared between Flux training and inference.

This module provides fundamental operations used by both training and inference:
- Latent packing/unpacking (2x2 spatial grouping)
- Position ID generation (3D RoPE for Flux transformer)

Design: Pure functions with no side effects, fully type-annotated.
"""


import torch
from torch import Tensor


def pack_latents(latents: Tensor) -> Tensor:
    """
    Pack latents from (B, C, H, W) to (B, H*W/4, C*4) format.

    Groups 2x2 spatial patches into sequence tokens for transformer processing.
    Used by both training and inference pipelines.

    Args:
        latents: Input tensor of shape (B, C, H, W)

    Returns:
        Packed tensor of shape (B, H*W/4, C*4)

    Example:
        >>> latents = torch.randn(2, 16, 128, 128)
        >>> packed = pack_latents(latents)
        >>> packed.shape
        torch.Size([2, 4096, 64])  # 128*128/4=4096, 16*4=64
    """
    batch_size, num_channels, height, width = latents.shape

    # Reshape to group 2x2 patches: (B, C, H, W) -> (B, C, H//2, 2, W//2, 2)
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)

    # Permute to bring patch dimensions together: (B, H//2, W//2, C, 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)

    # Flatten patches: (B, H//2*W//2, C*4)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)

    return latents


def unpack_latents(
    latents: Tensor,
    height: int,
    width: int,
    vae_scale_factor: int = 1,
) -> Tensor:
    """
    Unpack latents from (B, N, C*4) to (B, C, H, W) format.

    Reverses pack_latents operation. Supports optional VAE scaling for inference.

    Args:
        latents: Packed tensor of shape (B, N, C*4)
        height: Target spatial height
        width: Target spatial width
        vae_scale_factor: Downsampling factor (default: 1 for training, 8 for inference)

    Returns:
        Unpacked tensor of shape (B, C, H, W)

    Example:
        >>> packed = torch.randn(2, 4096, 64)
        >>> unpacked = unpack_latents(packed, 1024, 1024, vae_scale_factor=8)
        >>> unpacked.shape
        torch.Size([2, 16, 128, 128])  # 1024/8 = 128 (VAE downscaling)
    """
    batch_size, num_patches, channels = latents.shape

    # Apply VAE downsampling if specified (inference path)
    if vae_scale_factor > 1:
        height = height // vae_scale_factor
        width = width // vae_scale_factor

    # For packed latents: height and width are the dimensions BEFORE packing
    # So we need to use height//2 and width//2 for the reshaped view
    h_packed = height // 2
    w_packed = width // 2

    # Reshape to restore patch structure: (B, H//2, W//2, C//4, 2, 2)
    latents = latents.view(batch_size, h_packed, w_packed, channels // 4, 2, 2)

    # Permute to restore spatial dimensions: (B, C//4, H//2, 2, W//2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)

    # Flatten spatial dimensions: (B, C//4, H, W)
    latents = latents.reshape(batch_size, channels // 4, height, width)

    return latents


def generate_image_position_ids(
    batch_size: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate 3D position IDs for image patches (RoPE).

    Creates position encodings for Flux's 3D RoPE where:
    - Dimension 0: Always 0 (reserved for future temporal/video models)
    - Dimension 1: Row index (y-coordinate, 0 to height//2-1 after packing)
    - Dimension 2: Column index (x-coordinate, 0 to width//2-1 after packing)

    Follows NeMo conventions for Flux position encoding. For video models, dimension 0
    would encode frame indices instead of being 0.

    NOTE FOR ROPE FUSION: When using fused RoPE kernels (apply_rope_fusion=True),
    this function should be called with batch_size=1 to satisfy Transformer Engine's
    dimension constraints (freqs tensor must have shape [S, 1, 1, D]). The resulting
    [1, H*W/4, 3] position IDs will broadcast across the actual batch dimension
    during attention computation. This requires all images in the batch to have the
    same resolution (same height and width).

    Args:
        batch_size: Batch size (use 1 for RoPE fusion, actual batch size otherwise)
        height: Latent height in unpacked format (before 2x2 packing)
        width: Latent width in unpacked format (before 2x2 packing)
        device: Target device
        dtype: Target dtype

    Returns:
        Position IDs tensor of shape (batch_size, H*W/4, 3)

    Example:
        >>> ids = generate_image_position_ids(2, 128, 128, torch.device('cpu'))
        >>> ids.shape
        torch.Size([2, 4096, 3])  # 128*128/4 = 4096 packed positions
    """
    # Generate for packed latents (2x2 grouping)
    h_packed = height // 2
    w_packed = width // 2

    # Create position grid
    img_ids = torch.zeros(h_packed, w_packed, 3, device=device, dtype=dtype)

    # Fill row dimension (dim 1)
    img_ids[..., 1] = torch.arange(h_packed, device=device, dtype=dtype)[:, None]

    # Fill column dimension (dim 2)
    img_ids[..., 2] = torch.arange(w_packed, device=device, dtype=dtype)[None, :]

    # Flatten spatial dimensions and expand for batch
    img_ids = img_ids.reshape(h_packed * w_packed, 3)
    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return img_ids


def generate_text_position_ids(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Generate position IDs for text tokens.

    For Flux, text position IDs are zeros (no explicit positional encoding).

    Args:
        batch_size: Batch size
        seq_len: Text sequence length
        device: Target device
        dtype: Target dtype

    Returns:
        Position IDs tensor of shape (B, seq_len, 3) filled with zeros

    Example:
        >>> ids = generate_text_position_ids(2, 512, torch.device('cpu'))
        >>> ids.shape
        torch.Size([2, 512, 3])
        >>> ids.sum().item()
        0.0
    """
    return torch.zeros(batch_size, seq_len, 3, device=device, dtype=dtype)


__all__ = [
    "pack_latents",
    "unpack_latents",
    "generate_image_position_ids",
    "generate_text_position_ids",
]
