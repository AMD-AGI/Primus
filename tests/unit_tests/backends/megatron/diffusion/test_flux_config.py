# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux and base diffusion configurations.

Tests FluxConfig and BaseDiffusionConfig validation, preset configurations.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.common import BaseDiffusionConfig
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from tests.utils import PrimusUT


def graduated_config_kwargs(**overrides):
    kwargs = {
        "transformer_impl": "local",
        "fp4": "mxfp4",
        "fp4_recipe": "mxfp4",
        "bf16": True,
        "fp16": False,
        "params_dtype": torch.bfloat16,
        "sensitive_layers_enabled": True,
        "sensitive_layers_start": 4,
        "sensitive_layers_end": 4,
        "sensitive_layer_precision": "tw_fp8",
        "outer_sensitive_layers_start": 1,
        "outer_sensitive_layers_end": 1,
        "outer_sensitive_layer_precision": "bf16",
    }
    kwargs.update(overrides)
    return kwargs


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

    def test_sensitive_layer_counts_require_sensitive_routing(self):
        """Boundary counts are invalid when heterogeneous routing is disabled."""
        for field_name in ("sensitive_layers_start", "outer_sensitive_layers_start"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    "sensitive layer counts require sensitive_layers_enabled=True",
                ):
                    FluxConfig.flux_12b(**{field_name: 1})

    def test_outer_sensitive_layers_must_fit_inside_boundary(self):
        """The outer override cannot extend into the MXFP4 middle."""
        with self.assertRaisesRegex(
            ValueError, "outer_sensitive_layers_start .* exceeds sensitive_layers_start"
        ):
            FluxConfig.flux_12b(
                sensitive_layers_enabled=True,
                sensitive_layers_start=4,
                sensitive_layers_end=4,
                outer_sensitive_layers_start=5,
            )

    def test_outer_sensitive_layer_counts_must_be_non_negative(self):
        with self.assertRaisesRegex(ValueError, "outer_sensitive_layers_end must be non-negative"):
            FluxConfig.flux_12b(outer_sensitive_layers_end=-1)

    def test_sensitive_layer_counts_must_be_integers(self):
        cases = [
            ("sensitive_layers_start", 0.5),
            ("sensitive_layers_end", True),
            ("outer_sensitive_layers_start", "1"),
            ("outer_sensitive_layers_end", 1.0),
        ]
        for field_name, value in cases:
            with self.subTest(field_name=field_name, value=value):
                with self.assertRaisesRegex(ValueError, f"{field_name} must be an integer"):
                    FluxConfig.flux_12b(**{field_name: value})

    def test_outer_sensitive_end_must_fit_inside_boundary(self):
        with self.assertRaisesRegex(ValueError, "outer_sensitive_layers_end .* exceeds sensitive_layers_end"):
            FluxConfig.flux_12b(
                sensitive_layers_enabled=True,
                sensitive_layers_start=4,
                sensitive_layers_end=4,
                outer_sensitive_layers_end=5,
            )

    def test_sensitive_routing_defaults_are_backward_compatible(self):
        config = FluxConfig.flux_12b()
        assert config.sensitive_layers_enabled is False
        assert config.outer_sensitive_layers_start == 0
        assert config.outer_sensitive_layers_end == 0

    def test_graduated_routing_normalizes_tensorwise_fp8(self):
        config = FluxConfig.flux_12b(**graduated_config_kwargs())
        assert config.fp8 == "e4m3"
        assert config.fp8_recipe == "tensorwise"
        assert config.bf16 is True
        assert config.params_dtype == torch.bfloat16

    def test_graduated_routing_accepts_explicit_normalized_fp8(self):
        config = FluxConfig.flux_12b(
            **graduated_config_kwargs(
                fp8="e4m3",
                fp8_recipe="tensorwise",
            )
        )
        assert config.fp8 == "e4m3"
        assert config.fp8_recipe == "tensorwise"

    def test_sensitive_routing_requires_local_transformer(self):
        with self.assertRaisesRegex(ValueError, "sensitive layer routing requires transformer_impl='local'"):
            FluxConfig.flux_12b(**graduated_config_kwargs(transformer_impl="transformer_engine"))

    def test_sensitive_routing_requires_local_mxfp4(self):
        cases = [
            {"fp4": None},
            {"fp4_recipe": None},
            {"fp4": "nvfp4", "fp4_recipe": "nvfp4"},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    ValueError,
                    "sensitive layer routing requires fp4='mxfp4' and fp4_recipe='mxfp4'",
                ):
                    FluxConfig.flux_12b(**graduated_config_kwargs(**overrides))

    def test_tw_fp8_sensitive_layers_require_tensorwise_recipe(self):
        for fp8_recipe in ("blockwise", "mxfp8", "custom"):
            with self.subTest(fp8_recipe=fp8_recipe):
                with self.assertRaisesRegex(
                    ValueError,
                    "tw_fp8 sensitive layers require fp8_recipe='tensorwise'",
                ):
                    FluxConfig.flux_12b(**graduated_config_kwargs(fp8_recipe=fp8_recipe))

    def test_tw_fp8_sensitive_layers_require_e4m3_format(self):
        with self.assertRaisesRegex(ValueError, "tw_fp8 sensitive layers require fp8=None or fp8='e4m3'"):
            FluxConfig.flux_12b(**graduated_config_kwargs(fp8="hybrid"))

    def test_bf16_sensitive_layers_require_bf16_model_dtype(self):
        cases = [
            {"bf16": False},
            {"bf16": False, "fp16": True, "params_dtype": torch.float16},
            {"params_dtype": torch.float32},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, "bf16 sensitive layers require bf16=True"):
                    FluxConfig.flux_12b(**graduated_config_kwargs(**overrides))

    def test_sensitive_routing_rejects_unknown_precision(self):
        for field_name in (
            "sensitive_layer_precision",
            "outer_sensitive_layer_precision",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError,
                    "sensitive layer precision must be 'bf16' or 'tw_fp8'",
                ):
                    FluxConfig.flux_12b(**graduated_config_kwargs(**{field_name: "mxfp8"}))

    def test_graduated_routing_rejects_precision_role_inversion(self):
        with self.assertRaisesRegex(
            ValueError,
            "requires inner precision 'tw_fp8' and outer precision 'bf16'",
        ):
            FluxConfig.flux_12b(
                **graduated_config_kwargs(
                    sensitive_layer_precision="bf16",
                    outer_sensitive_layer_precision="tw_fp8",
                )
            )

    def test_graduated_routing_requires_inner_layer_on_each_outer_side(self):
        cases = [
            {"sensitive_layers_start": 1},
            {"sensitive_layers_end": 1},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(
                    ValueError,
                    "requires at least one inner sensitive layer on the same side",
                ):
                    FluxConfig.flux_12b(**graduated_config_kwargs(**overrides))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
