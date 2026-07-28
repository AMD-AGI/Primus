# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for prepare_flux_latents function.

Tests the higher-level prepare_flux_latents function that orchestrates
latent preparation for training. Lower-level pack/unpack utilities are
tested in test_flux_utils.py.
"""

import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.training.diffusion.forward_step import (
    prepare_flux_latents,
)
from primus.backends.megatron.training.diffusion.schedulers.flow_matching import (
    FlowMatchEulerDiscreteScheduler,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    DEFAULT_NUM_TRAIN_TIMESTEPS,
    VAE_LATENT_CHANNELS,
)
from tests.unit_tests.backends.megatron.diffusion.helpers import (
    assert_tensor_shape,
    create_mock_latents,
)
from tests.utils import PrimusUT


class TestPrepareFluxLatents(PrimusUT):
    """Tests for prepare_flux_latents function."""

    def test_prepare_flux_latents_shapes(self):
        """Test that prepare_flux_latents produces correct output shapes."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        batch_size = 2
        channels = VAE_LATENT_CHANNELS  # VAE output channels
        height, width = 128, 128

        latents = create_mock_latents(batch_size, height, width, channels=channels, seed=42)
        h_ids, w_ids = height // 2, width // 2
        img_ids = torch.zeros(batch_size, h_ids * w_ids, 3)

        (
            clean_latents,
            noise,
            packed_noisy_latents,
            img_ids_out,
            guidance_vec,
            timesteps,
            sigma_1d,
        ) = prepare_flux_latents(
            latents=latents,
            scheduler=scheduler,
            img_ids=img_ids,
        )

        # Check shapes
        assert_tensor_shape(clean_latents, (batch_size, channels, height, width), "clean_latents")
        assert_tensor_shape(noise, (batch_size, channels, height, width), "noise")

        # Packed latents: (B, H*W/4, C*4)
        expected_packed_shape = (
            batch_size,
            (height // 2) * (width // 2),
            channels * 4,
        )
        assert_tensor_shape(packed_noisy_latents, expected_packed_shape, "packed_noisy_latents")

        # Timesteps: (B,)
        assert len(timesteps) == batch_size

        # Sigma: (B,) in [0, 1]
        assert_tensor_shape(sigma_1d, (batch_size,), "sigma_1d")

        # Guidance should be None when not used
        assert guidance_vec is None

    def test_prepare_flux_latents_with_guidance(self):
        """Test prepare_flux_latents with guidance embedding enabled."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        batch_size = 2
        latents = create_mock_latents(batch_size, 128, 128, seed=42)
        img_ids = torch.zeros(batch_size, 64 * 64, 3)
        guidance_scale = 3.5

        (
            clean_latents,
            noise,
            packed_noisy_latents,
            img_ids_out,
            guidance_vec,
            timesteps,
            sigma_1d,
        ) = prepare_flux_latents(
            latents=latents,
            scheduler=scheduler,
            img_ids=img_ids,
            guidance_scale=guidance_scale,
            use_guidance_embed=True,
        )

        # Guidance vector should be created
        assert guidance_vec is not None
        assert_tensor_shape(guidance_vec, (batch_size,), "guidance_vec")

        # All guidance values should be the scale
        assert torch.allclose(guidance_vec, torch.tensor(guidance_scale))

    def test_prepare_flux_latents_timestep_sampling(self):
        """Test that timesteps are sampled correctly."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        batch_size = 8
        latents = create_mock_latents(batch_size, 128, 128, seed=42)
        img_ids = torch.zeros(batch_size, 64 * 64, 3)

        (
            clean_latents,
            noise,
            packed_noisy_latents,
            img_ids_out,
            guidance_vec,
            timesteps,
            sigma_1d,
        ) = prepare_flux_latents(
            latents=latents,
            scheduler=scheduler,
            img_ids=img_ids,
        )

        # Check timesteps are valid
        assert len(timesteps) == batch_size

        # All timesteps should be in scheduler's timestep range
        for t in timesteps:
            assert t >= 0 and t <= scheduler.num_train_timesteps

    def test_prepare_flux_latents_deterministic_with_seed(self):
        """Test that results are deterministic with fixed random seed."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        latents = create_mock_latents(2, 64, 64, seed=42)
        img_ids = torch.zeros(2, 32 * 32, 3)

        # Set seed and run
        torch.manual_seed(12345)
        result1 = prepare_flux_latents(latents, scheduler, img_ids)

        # Set same seed and run again
        torch.manual_seed(12345)
        result2 = prepare_flux_latents(latents, scheduler, img_ids)

        # Results should be identical
        assert torch.allclose(result1[2], result2[2]), "Results should be deterministic with same seed"

    def test_timestep_sampling_range(self):
        """Test that timestep sampling covers full [0, 1] range with logit-normal."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        # Sample many timesteps to check distribution
        num_samples = 1000
        batch_size = 100
        latents = create_mock_latents(batch_size, 64, 64, seed=42)

        all_timesteps = []
        for i in range(num_samples // batch_size):
            torch.manual_seed(42 + i)
            result = prepare_flux_latents(
                latents=latents,
                scheduler=scheduler,
                img_ids=None,  # Test with auto-generation
            )
            # Use sigma_1d (index 6) which is already in [0, 1] range
            all_timesteps.append(result[6].float())

        all_timesteps = torch.cat(all_timesteps)

        # Check that we sample across the full range
        min_t = all_timesteps.min().item()
        max_t = all_timesteps.max().item()

        # With logit-normal, we should get values close to 0 and 1
        assert min_t < 0.1, f"Min timestep {min_t} should be < 0.1 (covers early diffusion)"
        assert max_t > 0.9, f"Max timestep {max_t} should be > 0.9 (covers late diffusion)"

        # Check distribution is reasonable (not all in one region)
        mean_t = all_timesteps.mean().item()
        assert 0.3 < mean_t < 0.7, f"Mean timestep {mean_t} should be roughly centered"

    def test_img_ids_optional_parameter(self):
        """Test that img_ids can be None and will be generated automatically."""
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=DEFAULT_NUM_TRAIN_TIMESTEPS)

        batch_size = 2
        height, width = 128, 128
        latents = create_mock_latents(batch_size, height, width, seed=42)

        # Call without img_ids (should auto-generate)
        result = prepare_flux_latents(
            latents=latents,
            scheduler=scheduler,
            img_ids=None,  # Test auto-generation
        )

        # Should succeed and return valid img_ids
        img_ids_out = result[3]
        assert img_ids_out is not None

        # Check shape is correct
        expected_shape = (batch_size, (height // 2) * (width // 2), 3)
        assert_tensor_shape(img_ids_out, expected_shape, "auto_generated_img_ids")
