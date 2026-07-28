# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from NeMo's flow matching scheduler

"""
Flow Matching Euler Discrete Scheduler for Flux.

This scheduler implements the Euler discrete sampling method for flow matching
models like Flux. It supports both training and inference modes with optional
dynamic timestep shifting for variable resolution.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseScheduler


class FlowMatchEulerDiscreteScheduler(BaseScheduler):
    """
    Euler scheduler for flow matching diffusion models.

    This scheduler is used for Flux and implements:
        - Flow matching training objective
        - Euler discrete sampling for inference
        - Dynamic timestep shifting for variable resolution

    Args:
        num_train_timesteps: Number of diffusion steps for training (default: 1000)
        shift: Base shift value for timestep schedule (default: 1.0)
        use_dynamic_shifting: Whether to use dynamic shifting based on image resolution
        base_shift: Base shift for dynamic shifting (default: 0.5)
        max_shift: Maximum shift for dynamic shifting (default: 1.15)
        base_image_seq_len: Base image sequence length for dynamic shifting (default: 256)
        max_image_seq_len: Maximum image sequence length for dynamic shifting (default: 4096)

    Reference:
        - Flux paper: https://blackforestlabs.ai/flux-1-tools/
        - Adapted from NeMo's flow matching scheduler
    """

    _compatibles = []
    order = 1

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
    ):
        """Initialize Flow Matching Euler Discrete Scheduler."""
        # Generate initial timesteps
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        # Convert to sigmas (normalized timesteps in [0, 1])
        sigmas = timesteps / num_train_timesteps

        # Apply static shifting if not using dynamic shifting
        if not use_dynamic_shifting:
            # Shift formula: shift * sigma / (1 + (shift - 1) * sigma)
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        # Move sigmas to CPU to avoid too much CPU/GPU communication
        self.sigmas = sigmas.to("cpu")
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        # Store parameters
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.use_dynamic_shifting = use_dynamic_shifting
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

    @property
    def step_index(self):
        """The index counter for current timestep."""
        return self._step_index

    @property
    def begin_index(self):
        """The index for the first timestep."""
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        """
        Set the begin index for the scheduler.

        Args:
            begin_index: The begin index for the scheduler
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Add noise to sample using flow matching forward process.

        Flow matching interpolation: x_t = sigma * noise + (1 - sigma) * sample

        Args:
            sample: Clean sample tensor
            timestep: Current timestep(s)
            noise: Optional noise tensor (generated if None)

        Returns:
            Noisy sample
        """
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn_like(sample)

        # Ensure sigmas and timesteps have same device and dtype as sample
        sigmas = self.sigmas.to(device=sample.device, dtype=sample.dtype)

        # Handle MPS device (doesn't support float64)
        if sample.device.type == "mps" and torch.is_floating_point(timestep):
            schedule_timesteps = self.timesteps.to(sample.device, dtype=torch.float32)
            timestep = timestep.to(sample.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(sample.device)
            timestep = timestep.to(sample.device)

        # Get step indices for given timesteps
        # begin_index is None during training
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timestep]
        elif self.step_index is not None:
            # add_noise called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timestep.shape[0]
        else:
            # add noise called before first denoising step (img2img)
            step_indices = [self.begin_index] * timestep.shape[0]

        # Get sigma values for these indices
        sigma = sigmas[step_indices].flatten()

        # Broadcast sigma to match sample shape
        while len(sigma.shape) < len(sample.shape):
            sigma = sigma.unsqueeze(-1)

        # Flow matching interpolation
        noisy_sample = sigma * noise + (1.0 - sigma) * sample

        return noisy_sample

    def _sigma_to_t(self, sigma: float) -> float:
        """Convert sigma to timestep."""
        return sigma * self.num_train_timesteps

    def time_shift(self, mu: float, sigma: float, t: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic time shifting.

        Formula: exp(mu) / (exp(mu) + (1/t - 1)^sigma)

        Args:
            mu: Shift parameter (logarithmic)
            sigma: Exponent parameter (fixed at 1.0 for Flux)
            t: Timesteps to shift

        Returns:
            Shifted timesteps
        """
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):
        """
        Set discrete timesteps for inference.

        Args:
            num_inference_steps: Number of diffusion steps for inference
            device: Device to place timesteps on
            sigmas: Optional precomputed sigma values
            mu: Shift parameter for dynamic shifting
        """
        # Validate dynamic shifting requirements
        if self.use_dynamic_shifting and mu is None:
            raise ValueError("Must pass a value for `mu` when `use_dynamic_shifting` is True")

        # Generate sigmas if not provided
        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )
            sigmas = timesteps / self.num_train_timesteps

        # Apply shifting (dynamic or static)
        if self.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        # Convert to tensors
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        timesteps = sigmas * self.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        # Append zero sigma at the end
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(
        self, timestep: Union[float, torch.FloatTensor], schedule_timesteps: Optional[torch.Tensor] = None
    ) -> int:
        """
        Get the index for a given timestep.

        Args:
            timestep: Timestep value
            schedule_timesteps: Optional schedule to search in (uses self.timesteps if None)

        Returns:
            Index of the timestep
        """
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index taken for the very first step is always the second index
        # (or the last index if there is only 1). This ensures we don't accidentally
        # skip a sigma when starting in the middle of the schedule.
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep: Union[float, torch.FloatTensor]):
        """Initialize step index from timestep."""
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor]:
        """
        Perform one Euler denoising step.

        Args:
            model_output: Model prediction (velocity field for flow matching)
            timestep: Current timestep
            sample: Current noisy sample
            generator: Optional random number generator (unused)
            **kwargs: Additional arguments (unused)

        Returns:
            Tuple containing the denoised sample
        """
        # Validate timestep type
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError(
                "Passing integer indices as timesteps is not supported. "
                "Pass one of scheduler.timesteps as a timestep."
            )

        # Initialize step index if needed
        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues
        sample = sample.to(torch.float32)

        # Get current and next sigma
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        # Euler step: x_{t-1} = x_t + (sigma_next - sigma) * model_output
        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast back to model dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # Increment step index
        self._step_index += 1

        return (prev_sample,)

    def __len__(self):
        """Return number of training timesteps."""
        return self.num_train_timesteps


__all__ = ["FlowMatchEulerDiscreteScheduler"]
