# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux-specific layers.

Tests EmbedND (3D RoPE) and related functions.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.layers import EmbedND
from tests.unit_tests.backends.megatron.diffusion.constants import (
    BATCH_SIZE_PAIR,
    HIDDEN_DIM_FLUX,
    ROPE_THETA_DEFAULT,
    SEQ_LEN_TINY,
)
from tests.utils import PrimusUT


class TestEmbedND(PrimusUT):
    """Tests for EmbedND class."""

    def test_forward_output_shape(self):
        """Test that forward produces correct output shape.

        Output size is determined by sum(axes_dim), not dim parameter.
        """
        dim = HIDDEN_DIM_FLUX  # This is just stored, not used for output size
        axes_dim = [32, 48, 48]  # Sum = 128
        batch_size = BATCH_SIZE_PAIR  # Paired sample tests
        seq_len = SEQ_LEN_TINY  # Small sequence for basic tests
        num_axes = 3

        embed_nd = EmbedND(dim=dim, theta=ROPE_THETA_DEFAULT, axes_dim=axes_dim)
        ids = torch.randn(batch_size, seq_len, num_axes)

        output = embed_nd(ids)

        # Output size is based on sum(axes_dim) = 128, not dim = 3072
        # Shape: [seq, B, 1, sum(axes_dim)] after permute and reshape
        output_dim = sum(axes_dim)
        expected_shape = (seq_len, batch_size, 1, output_dim)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
