# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Base scheduler interface for diffusion models.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch


class BaseScheduler(ABC):
    """
    Abstract base class for diffusion schedulers.

    Schedulers handle:
        1. Noise schedule generation
        2. Forward process (adding noise to samples)
        3. Reverse process (denoising step)
    """

    @abstractmethod
    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Add noise to sample (forward process).

        Args:
            sample: Clean sample
            timestep: Current timestep
            noise: Optional noise tensor (generated if None)

        Returns:
            Noisy sample
        """
        raise NotImplementedError("Subclasses must implement scale_noise()")

    @abstractmethod
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None, **kwargs):
        """
        Set discrete timesteps for inference.

        Args:
            num_inference_steps: Number of diffusion steps
            device: Device to place timesteps on
            **kwargs: Additional scheduler-specific arguments
        """
        raise NotImplementedError("Subclasses must implement set_timesteps()")

    @abstractmethod
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        Perform one denoising step (reverse process).

        Args:
            model_output: Model prediction (noise or velocity)
            timestep: Current timestep
            sample: Current noisy sample
            **kwargs: Additional scheduler-specific arguments

        Returns:
            Tuple containing denoised sample (and optionally other values)
        """
        raise NotImplementedError("Subclasses must implement step()")


__all__ = ["BaseScheduler"]
