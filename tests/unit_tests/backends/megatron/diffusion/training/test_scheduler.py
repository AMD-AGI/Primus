# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for diffusion schedulers.

These tests validate the FlowMatchEulerDiscreteScheduler and other scheduler
implementations used by Flux and future diffusion models.

The tests follow NeMo's scheduler API:
    - scale_noise(): Add noise during training (forward process)
    - set_timesteps(): Setup inference schedule
    - step(): Perform denoising step (reverse process)
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.training.diffusion.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    BATCH_SIZE_PAIR,
    BATCH_SIZE_QUAD,
    BATCH_SIZE_SINGLE,
    DEFAULT_SHIFT,
    IMG_SIZE_MICRO,
    IMG_SIZE_MINI,
    IMG_SIZE_SMALL,
    TENSOR_CHANNELS_RGB,
    TRAINING_STEPS_MODERATE,
)
from tests.utils import PrimusUT


class TestFlowMatchEulerDiscreteScheduler(PrimusUT):
    """Tests for FlowMatchEulerDiscreteScheduler."""

    # ========================================================================
    # Initialization Tests
    # ========================================================================

    def test_scheduler_initialization_with_shift(self):
        """Test scheduler with custom shift parameter."""
        scheduler = FlowMatchEulerDiscreteScheduler(shift=2.0)

        assert scheduler.shift == 2.0

        # With higher shift, sigmas should be different
        scheduler_no_shift = FlowMatchEulerDiscreteScheduler(shift=DEFAULT_SHIFT)
        assert not torch.allclose(scheduler.sigmas, scheduler_no_shift.sigmas)

    # ========================================================================
    # Sigma Schedule Tests
    # ========================================================================

    def test_sigma_range(self):
        """Test sigma values are in valid range [0, 1]."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        assert (scheduler.sigmas >= 0.0).all()
        assert (scheduler.sigmas <= 1.0).all()

        # Sigmas should be monotonically decreasing
        assert (scheduler.sigmas[:-1] >= scheduler.sigmas[1:]).all()

    def test_sigma_schedule_with_shift(self):
        """Test shift parameter affects sigma distribution."""
        scheduler_shift_1 = FlowMatchEulerDiscreteScheduler(shift=DEFAULT_SHIFT)
        scheduler_shift_2 = FlowMatchEulerDiscreteScheduler(shift=2.0)

        # Higher shift moves distribution towards higher values
        mean_shift_1 = scheduler_shift_1.sigmas.mean()
        mean_shift_2 = scheduler_shift_2.sigmas.mean()

        assert mean_shift_2 > mean_shift_1

    # ========================================================================
    # Training Tests (Forward Process - scale_noise)
    # ========================================================================

    def test_scale_noise_shape(self):
        """Test scale_noise produces correct shape."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        batch_size = BATCH_SIZE_QUAD  # Quad sample tests
        channels = 64
        height = IMG_SIZE_SMALL  # Small size for standard tests
        width = IMG_SIZE_SMALL

        sample = torch.randn(batch_size, channels, height, width)
        noise = torch.randn(batch_size, channels, height, width)
        timesteps = scheduler.timesteps[[100, 200, 300, 400]]

        noisy = scheduler.scale_noise(sample, timesteps, noise)

        assert noisy.shape == sample.shape
        assert noisy.dtype == sample.dtype

    def test_scale_noise_flow_matching_interpolation(self):
        """Test flow matching interpolation: x_t = (1-σ)*x_0 + σ*noise."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        sample = torch.ones(
            BATCH_SIZE_SINGLE,
            TENSOR_CHANNELS_RGB,
            IMG_SIZE_MICRO,
            IMG_SIZE_MICRO,
        )
        noise = torch.zeros(
            BATCH_SIZE_SINGLE,
            TENSOR_CHANNELS_RGB,
            IMG_SIZE_MICRO,
            IMG_SIZE_MICRO,
        )

        # At σ≈0 (t=0), should be mostly sample
        timesteps = scheduler.timesteps[-1:]  # Last timestep (smallest sigma)
        noisy = scheduler.scale_noise(sample, timesteps, noise)
        assert torch.allclose(noisy, sample, atol=0.1)

        # At σ≈1 (t=max), should be mostly noise
        timesteps = scheduler.timesteps[0:1]  # First timestep (largest sigma)
        noisy = scheduler.scale_noise(sample, timesteps, noise)
        assert torch.allclose(noisy, noise, atol=0.1)

    def test_scale_noise_without_provided_noise(self):
        """Test scale_noise generates noise if not provided."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        sample = torch.randn(BATCH_SIZE_PAIR, TENSOR_CHANNELS_RGB, IMG_SIZE_MINI, IMG_SIZE_MINI)
        timesteps = scheduler.timesteps[[100, 200]]

        # Should generate noise internally
        noisy = scheduler.scale_noise(sample, timesteps, noise=None)

        assert noisy.shape == sample.shape
        # Should be different from original (has noise added)
        assert not torch.allclose(noisy, sample)

    def test_scale_noise_dtype_consistency(self):
        """Test scale_noise preserves dtype."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        for dtype in [torch.float32, torch.float16]:
            sample = torch.randn(
                BATCH_SIZE_PAIR,
                TENSOR_CHANNELS_RGB,
                IMG_SIZE_MINI,
                IMG_SIZE_MINI,
                dtype=dtype,
            )
            noise = torch.randn(
                BATCH_SIZE_PAIR,
                TENSOR_CHANNELS_RGB,
                IMG_SIZE_MINI,
                IMG_SIZE_MINI,
                dtype=dtype,
            )
            timesteps = scheduler.timesteps[[100, 200]]

            noisy = scheduler.scale_noise(sample, timesteps, noise)

            assert noisy.dtype == dtype

    # ========================================================================
    # Inference Tests (Reverse Process - set_timesteps, step)
    # ========================================================================

    def test_set_timesteps_basic(self):
        """Test set_timesteps creates inference schedule."""
        scheduler = FlowMatchEulerDiscreteScheduler()

        num_steps = TRAINING_STEPS_MODERATE  # Moderate training steps
        scheduler.set_timesteps(num_inference_steps=num_steps, device="cpu")

        assert len(scheduler.timesteps) == num_steps
        assert scheduler.timesteps[0] > scheduler.timesteps[-1]  # Descending
        assert len(scheduler.sigmas) == num_steps + 1  # +1 for final zero

        # Verify sigma at end is zero
        assert scheduler.sigmas[-1] == 0.0

    def test_set_timesteps_with_dynamic_shift(self):
        """Test set_timesteps with dynamic shifting."""
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

        mu = 0.5  # Shift parameter based on image resolution
        scheduler.set_timesteps(num_inference_steps=10, device="cpu", mu=mu)

        assert len(scheduler.timesteps) == 10

        # Should raise error if mu not provided
        scheduler2 = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
        with pytest.raises(ValueError, match="Must pass a value for `mu`"):
            scheduler2.set_timesteps(num_inference_steps=10)

    def test_step_basic(self):
        """Test step performs Euler denoising step."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=5, device="cpu")

        sample = torch.randn(BATCH_SIZE_SINGLE, TENSOR_CHANNELS_RGB, IMG_SIZE_MINI, IMG_SIZE_MINI)
        model_output = torch.randn(
            BATCH_SIZE_SINGLE, TENSOR_CHANNELS_RGB, IMG_SIZE_MINI, IMG_SIZE_MINI
        )  # Velocity prediction
        timestep = scheduler.timesteps[0]

        result = scheduler.step(model_output, timestep, sample)

        # step() returns a tuple (prev_sample,)
        prev_sample = result[0]

        assert prev_sample.shape == sample.shape
        assert prev_sample.dtype == sample.dtype
        # Should be different from input
        assert not torch.equal(prev_sample, sample)

    def test_step_consistency(self):
        """Test multiple steps are consistent."""
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(num_inference_steps=5, device="cpu")

        sample = torch.randn(BATCH_SIZE_SINGLE, TENSOR_CHANNELS_RGB, IMG_SIZE_MINI, IMG_SIZE_MINI)
        model_output = torch.randn(BATCH_SIZE_SINGLE, TENSOR_CHANNELS_RGB, IMG_SIZE_MINI, IMG_SIZE_MINI)

        # Perform multiple steps
        current_sample = sample
        for timestep in scheduler.timesteps:
            result = scheduler.step(model_output, timestep, current_sample)
            # step() returns a tuple (prev_sample,)
            current_sample = result[0]

        # Final sample should be different from initial
        assert not torch.equal(current_sample, sample)
        assert current_sample.shape == sample.shape

    def test_dynamic_shifting_behavior(self):
        """Test dynamic shifting changes timesteps based on resolution."""
        scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)

        # Test with different mu values (resolution-dependent)
        mu_small = 0.3  # Smaller resolution
        mu_large = 0.8  # Larger resolution

        scheduler.set_timesteps(num_inference_steps=10, device="cpu", mu=mu_small)
        timesteps_small = scheduler.timesteps.clone()

        scheduler.set_timesteps(num_inference_steps=10, device="cpu", mu=mu_large)
        timesteps_large = scheduler.timesteps.clone()

        # Timesteps should be different for different mu
        assert not torch.allclose(timesteps_small, timesteps_large)
