# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux and base diffusion configurations.

Tests FluxConfig and BaseDiffusionConfig validation, preset configurations.
"""

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.common import BaseDiffusionConfig
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from tests.utils import PrimusUT

# ========================================================================
# Base Diffusion Configuration Tests
# ========================================================================


class TestBaseDiffusionConfig(PrimusUT):
    """Tests for BaseDiffusionConfig."""

    def test_base_config_validation_invalid_channels(self):
        """Test validation catches invalid channel counts."""
        with self.assertRaises(ValueError) as cm:
            config = BaseDiffusionConfig(
                in_channels=0,
                num_attention_heads=8,
                num_layers=1,
            )
            config.validate()
        self.assertIn("in_channels must be positive", str(cm.exception))


class TestFluxConfig(PrimusUT):
    """Tests for FluxConfig class."""

    def test_validation_positive_joint_layers(self):
        """Test validation fails for non-positive num_joint_layers."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(num_joint_layers=0)
            config.validate()
        self.assertIn("num_joint_layers must be positive", str(cm.exception))

    def test_validation_positive_single_layers(self):
        """Test validation fails for non-positive num_single_layers."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(num_single_layers=-1)
            config.validate()
        self.assertIn("num_single_layers must be positive", str(cm.exception))

    def test_validation_positive_context_dim(self):
        """Test validation fails for non-positive context_dim."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(context_dim=0)
            config.validate()
        self.assertIn("context_dim must be positive", str(cm.exception))

    def test_validation_positive_vec_in_dim(self):
        """Test validation fails for non-positive vec_in_dim."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(vec_in_dim=-768)
            config.validate()
        self.assertIn("vec_in_dim must be positive", str(cm.exception))

    def test_validation_positive_theta(self):
        """Test validation fails for non-positive theta."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(theta=0)
            config.validate()
        self.assertIn("theta must be positive", str(cm.exception))

    def test_validation_axes_dim_length(self):
        """Test validation fails for axes_dim with wrong length."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(axes_dim=(16, 56))  # Only 2 elements
            config.validate()
        self.assertIn("axes_dim must have 3 elements", str(cm.exception))

    def test_validation_axes_dim_positive_values(self):
        """Test validation fails for non-positive axes_dim values."""
        with self.assertRaises(ValueError) as cm:
            config = FluxConfig(axes_dim=(16, 0, 56))
            config.validate()
        self.assertIn("All axes_dim values must be positive", str(cm.exception))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
