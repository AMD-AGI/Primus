# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Test helper utilities for diffusion model testing.

Provides utility functions for tensor validation, mock data creation,
and common test operations.
"""

from typing import Optional, Tuple

import torch

from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_LONG,
    VAE_LATENT_CHANNELS,
)


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], name: Optional[str] = None):
    """
    Assert that a tensor has the expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        name: Optional tensor name for error messages

    Raises:
        AssertionError if shapes don't match
    """
    name_str = f"{name} " if name else ""
    assert (
        tensor.shape == expected_shape
    ), f"{name_str}shape mismatch: expected {expected_shape}, got {tensor.shape}"


def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, name: Optional[str] = None):
    """
    Assert that a tensor has the expected dtype.

    Args:
        tensor: Tensor to check
        expected_dtype: Expected dtype
        name: Optional tensor name for error messages

    Raises:
        AssertionError if dtypes don't match
    """
    name_str = f"{name} " if name else ""
    assert (
        tensor.dtype == expected_dtype
    ), f"{name_str}dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"


def assert_no_nan_inf(tensor: torch.Tensor, name: Optional[str] = None):
    """
    Assert that a tensor contains no NaN or Inf values.

    Args:
        tensor: Tensor to check
        name: Optional tensor name for error messages

    Raises:
        AssertionError if NaN or Inf found
    """
    name_str = f"{name} " if name else ""
    assert not torch.isnan(tensor).any(), f"{name_str}contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name_str}contains Inf values"


def assert_tensor_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: Optional[str] = None,
):
    """
    Assert that two tensors are close within tolerance.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Optional tensor name for error messages

    Raises:
        AssertionError if tensors are not close
    """
    name_str = f"{name} " if name else ""
    assert torch.allclose(
        tensor1, tensor2, rtol=rtol, atol=atol
    ), f"{name_str}tensors are not close within rtol={rtol}, atol={atol}"


def create_mock_latents(
    batch_size: int,
    height: int,
    width: int,
    channels: int = VAE_LATENT_CHANNELS,  # VAE output channels for Flux
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Create mock latent tensors for testing.

    Args:
        batch_size: Batch size
        height: Latent height
        width: Latent width
        channels: Number of channels (default: 16 for Flux)
        device: Device to create tensors on
        dtype: Tensor dtype
        seed: Optional random seed for reproducibility

    Returns:
        Mock latents of shape (batch_size, channels, height, width)
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(
            batch_size, channels, height, width, device=device, dtype=dtype, generator=generator
        )
    else:
        return torch.randn(batch_size, channels, height, width, device=device, dtype=dtype)


def create_mock_text_embeddings(
    batch_size: int,
    seq_len: int,
    hidden_dim: int = T5_XXL_EMBEDDING_DIM,  # T5-XXL dimension
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Create mock T5 text embeddings for testing.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension (default: 4096 for T5-XXL)
        device: Device to create tensors on
        dtype: Tensor dtype
        seed: Optional random seed for reproducibility

    Returns:
        Mock text embeddings of shape (batch_size, seq_len, hidden_dim)
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype, generator=generator)
    else:
        return torch.randn(batch_size, seq_len, hidden_dim, device=device, dtype=dtype)


def create_mock_clip_embeddings(
    batch_size: int,
    hidden_dim: int = CLIP_L_EMBEDDING_DIM,  # CLIP-L dimension
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Create mock CLIP pooled embeddings for testing.

    Args:
        batch_size: Batch size
        hidden_dim: Hidden dimension (default: 768 for CLIP-L)
        device: Device to create tensors on
        dtype: Tensor dtype
        seed: Optional random seed for reproducibility

    Returns:
        Mock CLIP embeddings of shape (batch_size, hidden_dim)
    """
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(batch_size, hidden_dim, device=device, dtype=dtype, generator=generator)
    else:
        return torch.randn(batch_size, hidden_dim, device=device, dtype=dtype)


def create_mock_position_ids_3d(
    batch_size: int,
    height: int,
    width: int,
    text_seq_len: int = TEXT_SEQ_LEN_LONG,  # Default text sequence length
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create mock 3D position IDs for Flux (RoPE format).

    Args:
        batch_size: Batch size
        height: Image height (in latent space / 2)
        width: Image width (in latent space / 2)
        text_seq_len: Text sequence length
        device: Device to create tensors on
        dtype: Tensor dtype

    Returns:
        Tuple of (img_ids, txt_ids) where:
            - img_ids: (batch_size, height*width, 3)
            - txt_ids: (batch_size, text_seq_len, 3)
    """
    # Image position IDs
    img_ids = torch.zeros(batch_size, height * width, 3, device=device, dtype=dtype)

    # Generate 2D grid for spatial positions
    h_coords = torch.arange(height, device=device, dtype=dtype)
    w_coords = torch.arange(width, device=device, dtype=dtype)
    h_grid, w_grid = torch.meshgrid(h_coords, w_coords, indexing="ij")

    # Flatten and repeat for batch
    h_flat = h_grid.reshape(-1)
    w_flat = w_grid.reshape(-1)

    for b in range(batch_size):
        img_ids[b, :, 1] = h_flat
        img_ids[b, :, 2] = w_flat

    # Text position IDs (all zeros for text)
    txt_ids = torch.zeros(batch_size, text_seq_len, 3, device=device, dtype=dtype)

    return img_ids, txt_ids


__all__ = [
    "assert_tensor_shape",
    "assert_tensor_dtype",
    "assert_no_nan_inf",
    "assert_tensor_close",
    "create_mock_latents",
    "create_mock_text_embeddings",
    "create_mock_clip_embeddings",
    "create_mock_position_ids_3d",
]
