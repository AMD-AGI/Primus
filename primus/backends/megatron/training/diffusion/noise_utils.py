# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Noise application utilities for diffusion training.

This module provides pure functions for applying noise according to
different diffusion forward processes. All functions are stateless
and work with tensors of any shape.

Supported processes:
    - Flow Matching: Linear interpolation between clean and noise
    - DDPM: Variance-preserving noise schedule

These utilities encapsulate the mathematical formulas for the forward
diffusion process, making the code self-documenting and reusable
across different model architectures (2D images, 3D video, etc.).
"""

from torch import Tensor


def apply_flow_matching_noise(
    clean_latents: Tensor,
    noise: Tensor,
    sigma: Tensor,
) -> Tensor:
    """
    Apply noise using flow matching forward process.

    Flow matching uses a simple linear interpolation between clean
    latents and noise, controlled by the sigma parameter:

    Formula: noisy = (1 - sigma) * clean + sigma * noise

    This formulation ensures:
    - When sigma=0: noisy = clean (no noise)
    - When sigma=1: noisy = noise (pure noise)
    - Linear interpolation in between

    Args:
        clean_latents: Clean latents [any shape]
        noise: Sampled noise [same shape as clean_latents]
        sigma: Noise schedule values [broadcast compatible]
               Should be in range [0, 1]

    Returns:
        Noisy latents [same shape as clean_latents]

    Reference:
        Flow Matching for Generative Modeling
        https://arxiv.org/abs/2210.02747

    Example:
        >>> clean = torch.randn(2, 16, 64, 64)
        >>> noise = torch.randn(2, 16, 64, 64)
        >>> sigma = torch.tensor([0.3, 0.7]).reshape(2, 1, 1, 1)
        >>> noisy = apply_flow_matching_noise(clean, noise, sigma)
        >>> # sigma[0]=0.3 means 30% noise, 70% clean for first sample
    """
    return (1.0 - sigma) * clean_latents + sigma * noise


__all__ = [
    "apply_flow_matching_noise",
]
