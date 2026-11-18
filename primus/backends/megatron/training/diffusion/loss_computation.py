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

import torch
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
        # not a per-element mean. An all-zero mask yields 0 rather than NaN;
        # any other total, even a soft one below 1, is divided by as is.
        count = loss_mask.sum()
        loss = (loss_per_element * loss_mask).sum() / torch.where(count > 0, count, torch.ones_like(count))

    return loss


def compute_weighted_flow_matching_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    loss_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Compute flow matching loss with a per-timestep weight.

    Video flow matching (Wan, MovieGen) scales the squared error by a
    timestep-dependent weight, so that noise levels contribute unevenly to
    the gradient. Two things differ from :func:`compute_flow_matching_loss`:

    1. The target is supplied by the caller rather than derived as
       ``noise - clean_latents``, so alternative parameterizations
       (velocity, ``noise - sample``) can reuse this helper.
    2. The squared error is multiplied by ``weight`` before reduction.

    Formula: loss = mean(((target - prediction) ** 2) * weight)

    Passing ``weight=1.0`` recovers the unweighted objective, which is how
    the ``uniform`` loss weighting mode is expressed. With a ``[batch_size]``
    mask the result is the mean over the kept samples' elements; the
    unweighted helper divides that case by the sample count instead.

    Args:
        prediction: Model output [any shape]
        target: Training target from the scheduler [same shape as prediction]
        weight: Per-sample weight, broadcast-compatible with target.
                Typically [batch_size] or [batch_size, 1, 1, ...].
        loss_mask: Optional mask for variable-length sequences [batch_size]
                   or broadcast-compatible shape. Default None (no masking).

    Returns:
        Scalar loss value (weighted mean squared error, optionally masked).

    Reference:
        DiffSynth-Studio ``FlowMatchSFTLoss``; maxdiffusion
        ``wan_trainer.step_optimizer``.

    Example:
        >>> pred = torch.randn(2, 16, 8, 32, 32)  # Batch of 3D latents
        >>> target = torch.randn(2, 16, 8, 32, 32)
        >>> weight = torch.tensor([0.8, 1.2])  # Per-sample timestep weight
        >>> loss = compute_weighted_flow_matching_loss(pred, target, weight)
    """
    diff_sq = (target.float() - prediction.float()) ** 2

    if weight.dim() == 1 and diff_sq.dim() > 1:
        weight_shape = [weight.shape[0]] + [1] * (diff_sq.dim() - 1)
        weight = weight.view(*weight_shape)
    weight = weight.to(dtype=diff_sq.dtype, device=diff_sq.device)

    weighted = diff_sq * weight

    if loss_mask is None:
        return weighted.mean()

    if loss_mask.dim() == 1 and weighted.dim() > 1:
        mask_shape = [loss_mask.shape[0]] + [1] * (weighted.dim() - 1)
        loss_mask = loss_mask.view(*mask_shape)
    loss_mask = loss_mask.to(dtype=weighted.dtype, device=weighted.device).expand_as(weighted)

    # Count the mask after broadcasting: a [B] mask covers every latent element
    # of its sample, so its own sum would give a per-sample, not per-element,
    # mean. An all-zero mask yields 0 rather than NaN; any other total, even a
    # soft one below 1, is divided by as is.
    count = loss_mask.sum()
    return (weighted * loss_mask).sum() / torch.where(count > 0, count, torch.ones_like(count))


__all__ = [
    "compute_flow_matching_loss",
    "compute_weighted_flow_matching_loss",
]
