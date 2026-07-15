# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Timestep sampling strategies for diffusion training.

Training timestep sampling is a hyperparameter optimization technique
that is separate from inference denoising algorithms. Different sampling
strategies can improve training convergence and model quality.

Key Insight:
    - Training: Use logit-normal sampling (hyperparameter optimization)
    - Inference: Use linear schedule (denoising algorithm requirement)

These are fundamentally different concerns and should be separated.

Supported strategies:
    - LogitNormal: Better convergence (from SD3 paper)
    - Uniform: Baseline approach for comparison
    - Mode: Alternative from SD3 paper

Reference:
    Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206v1
    Section 3.1: Timestep sampling
"""

import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class TimestepSampler(ABC):
    """
    Abstract base class for training timestep sampling strategies.

    Different sampling strategies can improve training convergence by
    emphasizing certain timesteps where the model needs to learn more.

    All samplers must implement the `sample()` method that returns
    both timesteps and their corresponding sigma values.
    """

    @abstractmethod
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scheduler,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample timesteps and compute sigmas for training.

        Args:
            batch_size: Number of samples in the batch
            device: Target device for the tensors
            scheduler: Scheduler instance (for accessing timesteps/sigmas)

        Returns:
            Tuple of (timesteps, sigmas):
                - timesteps: Sampled timestep values [batch_size]
                - sigmas: Corresponding noise schedule values [batch_size]

        Example:
            >>> sampler = LogitNormalSampler()
            >>> timesteps, sigmas = sampler.sample(
            ...     batch_size=32,
            ...     device='cuda',
            ...     scheduler=flow_scheduler
            ... )
        """


class LogitNormalSampler(TimestepSampler):
    """
    Logit-normal timestep sampling (current approach, from SD3 paper).

    This sampling strategy uses a logit-normal distribution to sample
    timesteps. With default parameters (mean=0, std=1), the distribution
    emphasizes mid-range timesteps (around t≈500). The mean and std
    parameters control the distribution shape.

    The distribution is created by:
    1. Sample u from Normal(mean, std)
    2. Apply sigmoid: u = sigmoid(u)
    3. Map to timestep indices: indices = u * num_train_timesteps

    This approach has been shown to improve training convergence
    compared to uniform sampling.

    Args:
        mean: Mean of the normal distribution (default: 0.0)
        std: Standard deviation of the normal distribution (default: 1.0)

    Reference:
        Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206v1
        Section 3.1: "We use rf/lognorm(0.00,1.00)"

    Example:
        >>> sampler = LogitNormalSampler(mean=0.0, std=1.0)
        >>> timesteps, sigmas = sampler.sample(32, 'cuda', scheduler)
        >>> # More samples near the middle (t~500) than the extremes
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scheduler,
    ) -> Tuple[Tensor, Tensor]:
        """Sample using logit-normal distribution."""
        # Sample from Normal(mean, std)
        u = torch.normal(
            mean=self.mean,
            std=self.std,
            size=(batch_size,),
            device="cpu",  # CPU to match scheduler.timesteps device
        )

        # Apply sigmoid to get logit-normal distribution
        u = torch.sigmoid(u)

        # Map to timestep indices
        indices = (u * scheduler.num_train_timesteps).long()
        indices = torch.clamp(indices, 0, scheduler.num_train_timesteps - 1)

        # Get timesteps
        timesteps = scheduler.timesteps[indices].to(device=device)

        # Get corresponding sigmas
        sigmas = scheduler.sigmas.to(device=device)
        schedule_timesteps = scheduler.timesteps.to(device=device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()

        return timesteps, sigma


class UniformSampler(TimestepSampler):
    """
    Uniform timestep sampling (baseline approach).

    Samples timesteps uniformly from [0, num_train_timesteps), giving
    equal probability to all timesteps. This is simpler but may converge
    slower than logit-normal sampling.

    Useful as a baseline for comparison with more sophisticated
    sampling strategies.

    Example:
        >>> sampler = UniformSampler()
        >>> timesteps, sigmas = sampler.sample(32, 'cuda', scheduler)
        >>> # Equal probability for all timesteps
    """

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scheduler,
    ) -> Tuple[Tensor, Tensor]:
        """Sample uniformly from all timesteps."""
        indices = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device="cpu")

        timesteps = scheduler.timesteps[indices].to(device=device)

        sigmas = scheduler.sigmas.to(device=device)
        schedule_timesteps = scheduler.timesteps.to(device=device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()

        return timesteps, sigma


class DirectUniformSampler(TimestepSampler):
    """
    Direct uniform timestep sampling without scheduler indirection (NVIDIA MLPerf-compatible).

    Samples timesteps continuously from [0, num_train_timesteps) using torch.rand,
    bypassing the scheduler's discrete timestep/sigma lookup tables. This matches
    NVIDIA's MLPerf Flux implementation exactly:

        timesteps = torch.rand((batch_size,)) * num_train_timesteps
        sigma = timesteps / num_train_timesteps  (i.e., sigma == normalized timestep)

    Advantages over UniformSampler:
        - Continuous distribution (no 1000-bin discretization)
        - No Python loop for sigma lookup (avoids per-element .nonzero() calls)
        - Produces tensors directly on target device

    Note: Assumes shift=1.0 in the scheduler (no timestep shifting). With shift=1.0,
    sigma == normalized_timestep, which matches the scheduler's linear mapping.

    Example:
        >>> sampler = DirectUniformSampler()
        >>> timesteps, sigmas = sampler.sample(32, 'cuda', scheduler)
    """

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scheduler,
    ) -> Tuple[Tensor, Tensor]:
        """Sample uniformly from [0, num_train_timesteps) without scheduler lookup."""
        sigma = torch.rand((batch_size,), device=device)
        timesteps = sigma * scheduler.num_train_timesteps
        return timesteps, sigma


class ModeSampler(TimestepSampler):
    """
    Mode-based sampling from SD3 paper (alternative to logit-normal).

    This sampling strategy uses a mode-based distribution:

    Formula: u = 1 - u - mode_scale * (cos(π*u/2)² - 1 + u)

    Where u is initially sampled uniformly from [0, 1).

    This creates a distribution that emphasizes certain timesteps
    differently than logit-normal, potentially offering better
    performance for some models.

    Args:
        mode_scale: Scaling factor for mode distribution (default: 1.29)
                    Value from SD3 paper

    Reference:
        Stable Diffusion 3 paper: https://arxiv.org/abs/2403.03206v1
        Section 3.1: Alternative sampling strategy

    Example:
        >>> sampler = ModeSampler(mode_scale=1.29)
        >>> timesteps, sigmas = sampler.sample(32, 'cuda', scheduler)
    """

    def __init__(self, mode_scale: float = 1.29):
        self.mode_scale = mode_scale

    def sample(
        self,
        batch_size: int,
        device: torch.device,
        scheduler,
    ) -> Tuple[Tensor, Tensor]:
        """Sample using mode distribution."""
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - self.mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)

        indices = (u * scheduler.num_train_timesteps).long()
        indices = torch.clamp(indices, 0, scheduler.num_train_timesteps - 1)

        timesteps = scheduler.timesteps[indices].to(device=device)

        sigmas = scheduler.sigmas.to(device=device)
        schedule_timesteps = scheduler.timesteps.to(device=device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()

        return timesteps, sigma


def create_timestep_sampler(strategy: str = "logit_normal", **kwargs) -> TimestepSampler:
    """
    Factory function for creating timestep samplers.

    This function provides a convenient way to create samplers
    by name, allowing easy experimentation with different strategies.

    Args:
        strategy: Sampling strategy name
                  Options: "logit_normal", "uniform", "direct_uniform", "mode"
        **kwargs: Additional arguments for the sampler
                  (e.g., mean/std for LogitNormal, mode_scale for Mode)

    Returns:
        TimestepSampler instance

    Raises:
        ValueError: If strategy is unknown

    Example:
        >>> # Default logit-normal
        >>> sampler = create_timestep_sampler("logit_normal")
        >>>
        >>> # Custom parameters
        >>> sampler = create_timestep_sampler(
        ...     "logit_normal",
        ...     mean=0.5,
        ...     std=0.5
        ... )
        >>>
        >>> # Uniform baseline
        >>> sampler = create_timestep_sampler("uniform")
        >>>
        >>> # Mode sampling
        >>> sampler = create_timestep_sampler("mode", mode_scale=1.5)
    """
    if strategy == "logit_normal":
        return LogitNormalSampler(**kwargs)
    elif strategy == "uniform":
        return UniformSampler(**kwargs)
    elif strategy == "direct_uniform":
        return DirectUniformSampler(**kwargs)
    elif strategy == "mode":
        return ModeSampler(**kwargs)
    else:
        raise ValueError(
            f"Unknown sampling strategy: {strategy}. "
            f"Choose from: logit_normal, uniform, direct_uniform, mode"
        )


__all__ = [
    "TimestepSampler",
    "LogitNormalSampler",
    "UniformSampler",
    "DirectUniformSampler",
    "ModeSampler",
    "create_timestep_sampler",
]
