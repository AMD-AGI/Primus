# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Tests for base encoder classes.

Tests abstract base classes for encoders, ensuring proper methods and behavior.
"""

from unittest.mock import Mock

import pytest
import torch

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseEncoder,
    BaseTextEncoder,
    BaseVAE,
)
from primus.backends.megatron.data.diffusion.encoders.config import (
    EncoderConfig,
    T5XXLConfig,
    VAEConfig,
)
from tests.utils import PrimusUT


# Concrete implementations for testing abstract classes
class ConcreteEncoder(BaseEncoder):
    """Concrete encoder for testing BaseEncoder."""

    def encode(self, *args, **kwargs):
        return torch.randn(1, 10)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        return cls(config or EncoderConfig(type="test", model_path=model_path))


class ConcreteVAE(BaseVAE):
    """Concrete VAE for testing BaseVAE."""

    def encode(self, images):
        batch_size = images.shape[0]
        latent_h = images.shape[2] // self.latent_downsample_factor
        latent_w = images.shape[3] // self.latent_downsample_factor
        return torch.randn(batch_size, self.out_channels, latent_h, latent_w)

    def decode(self, latents):
        batch_size = latents.shape[0]
        img_h = latents.shape[2] * self.latent_downsample_factor
        img_w = latents.shape[3] * self.latent_downsample_factor
        return torch.randn(batch_size, self.in_channels, img_h, img_w)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        return cls(config or VAEConfig(model_path=model_path))


class ConcreteTextEncoder(BaseTextEncoder):
    """Concrete text encoder for testing BaseTextEncoder."""

    def __init__(self, config):
        super().__init__(config)
        self._tokenizer = Mock()  # Mock tokenizer

    def encode(self, texts, **kwargs):
        texts_list = self._prepare_texts(texts)
        batch_size = len(texts_list)
        return torch.randn(batch_size, self.max_length, self.embedding_dim)

    @classmethod
    def from_pretrained(cls, model_path, config=None):
        return cls(config or T5XXLConfig(model_path=model_path))


class TestBaseEncoder(PrimusUT):
    """Tests for BaseEncoder abstract class."""

    def test_get_dtype_invalid_defaults_to_bf16(self):
        """Test that invalid precision defaults to bfloat16."""
        config = EncoderConfig(type="test", model_path="/path")
        encoder = ConcreteEncoder(config)

        assert encoder._get_dtype("unknown") == torch.bfloat16


class TestBaseVAE(PrimusUT):
    """Tests for BaseVAE abstract class."""

    def test_get_latent_shape_various_sizes(self):
        """Test latent shape calculation with various image sizes."""
        config = VAEConfig(type="autoencoder_kl", model_path="/path/to/vae")
        vae = ConcreteVAE(config)

        test_cases = [
            ((256, 256), (16, 32, 32)),
            ((512, 512), (16, 64, 64)),
            ((1024, 1024), (16, 128, 128)),
            ((512, 1024), (16, 64, 128)),
        ]

        for (h, w), expected_shape in test_cases:
            assert vae.get_latent_shape(h, w) == expected_shape

    def test_get_latent_shape_custom_downsample(self):
        """Test latent shape with custom downsample factor."""
        config = VAEConfig(type="autoencoder_kl", model_path="/path/to/vae", latent_downsample_factor=16)
        vae = ConcreteVAE(config)

        latent_shape = vae.get_latent_shape(512, 512)

        assert latent_shape == (16, 32, 32)  # 512/16 = 32


class TestBaseTextEncoder(PrimusUT):
    """Tests for BaseTextEncoder abstract class."""

    def test_tokenizer_not_initialized_error(self):
        """Test that accessing tokenizer before initialization raises error."""
        config = T5XXLConfig(type="t5_xxl", model_path="/path/to/t5")
        encoder = ConcreteTextEncoder(config)
        encoder._tokenizer = None

        with pytest.raises(ValueError, match="Tokenizer not initialized"):
            _ = encoder.tokenizer


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
