# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for shared Flux utilities (pack/unpack, position IDs).

Tests canonical implementations in flux/utils.py that are shared between
training and inference code paths.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
    unpack_latents,
)
from tests.utils import PrimusUT


class TestPackUnpackLatents(PrimusUT):
    """Tests for latent packing/unpacking operations."""

    def test_pack_unpack_roundtrip(self):
        """Test that pack -> unpack is reversible."""
        batch_size, channels, height, width = 2, 16, 128, 128
        original = torch.randn(batch_size, channels, height, width)

        # Pack and unpack
        packed = pack_latents(original)
        unpacked = unpack_latents(packed, height, width, vae_scale_factor=1)

        # Should match original
        assert unpacked.shape == original.shape
        torch.testing.assert_close(unpacked, original)


class TestPositionIDs(PrimusUT):
    """Tests for position ID generation."""

    def test_generate_image_position_ids_values(self):
        """Test image position IDs have correct structure."""
        batch_size, height, width = 2, 8, 8  # Small for inspection
        device = torch.device("cpu")

        img_ids = generate_image_position_ids(batch_size, height, width, device)

        # Dimension 0 should be all zeros
        assert (img_ids[:, :, 0] == 0).all()

        # Dimension 1 (row) should range from 0 to height//2-1
        max_row = img_ids[:, :, 1].max()
        assert max_row == height // 2 - 1

        # Dimension 2 (col) should range from 0 to width//2-1
        max_col = img_ids[:, :, 2].max()
        assert max_col == width // 2 - 1


class TestEdgeCases(PrimusUT):
    """Tests for edge cases and error conditions."""

    def test_pack_latents_minimum_size(self):
        """Test pack_latents with minimum size (2x2)."""
        batch_size, channels = 1, 16
        latents = torch.randn(batch_size, channels, 2, 2)

        packed = pack_latents(latents)

        # 2x2 packed is 1 token
        assert packed.shape == (batch_size, 1, channels * 4)

    def test_unpack_latents_minimum_size(self):
        """Test unpack_latents with minimum size."""
        batch_size, channels = 1, 64
        packed = torch.randn(batch_size, 1, channels)  # 1 packed token

        # Unpack to 2x2 latent (after VAE scale of 1)
        unpacked = unpack_latents(packed, 2, 2, vae_scale_factor=1)

        # Output should be (B, C//4, 2, 2)
        assert unpacked.shape == (batch_size, channels // 4, 2, 2)


class TestCompatibility(PrimusUT):
    """Tests for backward compatibility with existing code."""

    def test_unpack_latents_signature_compatibility(self):
        """Test unpack_latents works with both training and inference signatures."""
        packed = torch.randn(2, 4096, 64)

        # Training path (no VAE scaling)
        unpacked_train = unpack_latents(packed, 128, 128, vae_scale_factor=1)
        assert unpacked_train.shape == (2, 16, 128, 128)

        # Inference path (VAE scaling)
        unpacked_infer = unpack_latents(packed, 1024, 1024, vae_scale_factor=8)
        assert unpacked_infer.shape == (2, 16, 128, 128)  # Same result: 1024/8 = 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
