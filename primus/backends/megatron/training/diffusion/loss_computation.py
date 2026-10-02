# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Portions copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Loss computation for diffusion models.

This module provides reusable loss computation logic for different
diffusion training objectives. All functions are pure (no state),
making them easy to test and reusable across different model architectures.

Supported loss types:
    - Flow Matching: Used by Flux, SD3, and video models
    - Epsilon Prediction: Used by DDPM and older models
    - V-Prediction: Alternative parameterization

All loss functions accept tensors of any shape and reduce to scalar.
"""

from typing import Optional

import torch.nn.functional as F
from torch import Tensor


def compute_flow_matching_loss(
    prediction: Tensor,
    clean_latents: Tensor,
    noise: Tensor,
    loss_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute flow matching loss with optional masking.

    Flow matching models predict the velocity field that transforms
    noise to clean latents. The training objective is:

    Formula: target = noise - clean_latents
             loss = MSE(prediction, target)

    This loss is model-agnostic and works with any tensor shape,
    making it reusable across 2D (images) and 3D (video) models.

    Args:
        prediction: Model output (predicted velocity) [any shape]
        clean_latents: Original clean latents [same shape as prediction]
        noise: Sampled noise [same shape as prediction]
        loss_mask: Optional mask for variable-length sequences [batch_size]
                   or broadcast-compatible shape. Default None (no masking).

    Returns:
        Scalar loss value (mean squared error, optionally masked). With a mask
        it is the mean over the loss elements the mask covers, so a
        ``[batch_size]`` mask of ones equals the unmasked loss, and a mask
        that keeps nothing gives 0.

    Reference:
        Flow Matching for Generative Modeling
        https://arxiv.org/abs/2210.02747

    Example (no masking):
        >>> pred = torch.randn(2, 16, 64, 64)  # Batch of 2D latents
        >>> clean = torch.randn(2, 16, 64, 64)
        >>> noise = torch.randn(2, 16, 64, 64)
        >>> loss = compute_flow_matching_loss(pred, clean, noise)
        >>> loss.backward()

    Example (with packing/masking):
        >>> pred = torch.randn(2, 16, 64, 64)
        >>> clean = torch.randn(2, 16, 64, 64)
        >>> noise = torch.randn(2, 16, 64, 64)
        >>> mask = torch.tensor([1.0, 0.0])  # Second sample is padding
        >>> loss = compute_flow_matching_loss(pred, clean, noise, mask)
    """
    target = noise - clean_latents

    if loss_mask is None:
        # Simple case: direct mean reduction
        loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
    else:
        # With masking for variable-length sequences (packing support)
        loss_per_element = F.mse_loss(prediction.float(), target.float(), reduction="none")

        # Broadcast mask to match loss_per_element shape if needed
        if loss_mask.dim() == 1 and loss_per_element.dim() > 1:
            mask_shape = [loss_mask.shape[0]] + [1] * (loss_per_element.dim() - 1)
            loss_mask = loss_mask.view(*mask_shape)
        loss_mask = loss_mask.to(dtype=loss_per_element.dtype, device=loss_per_element.device)
        loss_mask = loss_mask.expand_as(loss_per_element)

        # Count the mask after broadcasting: a [B] mask covers every latent
        # element of its sample, so its own sum would give a per-sample sum,
        # not a per-element mean. An all-zero mask yields 0 rather than NaN.
        loss = (loss_per_element * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    return loss


__all__ = [
    "compute_flow_matching_loss",
]
