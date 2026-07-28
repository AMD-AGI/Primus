# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for VAE encoder implementations.

Tests AutoencoderKL with mocked models for fast unit testing.

NOTE: Common wrapper tests (initialization, from_pretrained, device handling, etc.)
have been moved to test_encoder_wrappers_consolidated.py. This file contains only
VAE-specific tests that verify Primus's VAE integration logic (scale/shift, latent shapes).
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from primus.backends.megatron.data.diffusion.encoders.config import VAEConfig


class TestAutoencoderKL:
    """Tests for AutoencoderKL Primus-specific logic."""

    @patch("primus.backends.megatron.data.diffusion.encoders.image.autoencoder_kl.DiffusersAutoencoderKL")
    def test_encode_applies_scale_and_shift(self, mock_diffusers_vae, mock_vae_model):
        """encode() must apply z = scale_factor * (sample - shift_factor) to the VAE sample."""
        from primus.backends.megatron.data.diffusion.encoders.image import AutoencoderKL

        mock_diffusers_vae.from_pretrained.return_value = mock_vae_model

        config = VAEConfig(
            type="autoencoder_kl", model_path="/path/to/vae", scale_factor=0.5, shift_factor=0.1
        )
        vae = AutoencoderKL.from_pretrained("/path/to/vae", config=config)

        assert vae.config.scale_factor == 0.5
        assert vae.config.shift_factor == 0.1

        # Pin the VAE's sampled latent so we can assert the exact scale/shift math
        # (the shared fixture returns a fresh random sample each call).
        known_sample = torch.randn(1, 16, 64, 64)
        latent_dist = MagicMock()
        latent_dist.sample = MagicMock(return_value=known_sample)
        vae.vae.encode = MagicMock(return_value=MagicMock(latent_dist=latent_dist))

        images = torch.randn(1, 3, 256, 256)
        latents = vae.encode(images)

        expected = config.scale_factor * (
            known_sample.to(latents.device, latents.dtype) - config.shift_factor
        )
        assert latents.shape == known_sample.shape
        torch.testing.assert_close(latents, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
