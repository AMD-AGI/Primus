# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for diffusion model embedding layers.

Tests TimeStepEmbedder and MLPEmbedder to ensure correct shapes and behavior.
"""

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.common.embeddings import (
    get_timestep_embedding,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    TIMESTEP_EMBEDDING_DIM,
)
from tests.utils import PrimusUT


class TestGetTimestepEmbedding(PrimusUT):
    """Tests for get_timestep_embedding function (Primus implementation of sinusoidal embeddings)."""

    def test_odd_embedding_dim(self):
        """Test that odd embedding dimensions are handled correctly by Primus implementation."""
        timesteps = torch.randn(4)
        embedding_dim = 257  # Odd number

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (4, embedding_dim), f"Expected shape (4, {embedding_dim}), got {emb.shape}"

    def test_sinusoidal_properties(self):
        """Test that Primus sinusoidal embeddings have correct mathematical properties."""
        timesteps = torch.linspace(0, 100, 10)
        embedding_dim = TIMESTEP_EMBEDDING_DIM

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.abs().max() < 100, "Embeddings should be in reasonable range"
        assert (emb > 0).any() and (emb < 0).any(), "Should have both positive and negative values"

        # For timestep=0: cos(0)=1 for all frequencies (first half),
        # sin(0)=0 for all frequencies (second half)
        emb_zero = get_timestep_embedding(torch.tensor([0.0]), embedding_dim)
        half = embedding_dim // 2
        assert torch.allclose(
            emb_zero[0, :half], torch.ones(half), atol=1e-6
        ), "cos(0) should be 1 for all frequencies"
        assert torch.allclose(
            emb_zero[0, half : 2 * half], torch.zeros(half), atol=1e-6
        ), "sin(0) should be 0 for all frequencies"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
