# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux layer spec backend selection.

Tests that get_flux_layer_spec() correctly selects backend based on transformer_impl.
This ensures alignment between backend selection and FSDP2 wrapping decisions.
"""

import pytest

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    get_flux_layer_spec,
)
from tests.utils import PrimusUT


class TestFluxLayerSpecBackendSelection(PrimusUT):
    """Tests for backend selection in get_flux_layer_spec()."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for layer spec tests."""

    def test_backend_selection_fp8_local_spec(self):
        """Test that local + fp8 selects PrimusTurboFloat8LocalSpecProvider."""
        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            Float8ColumnParallelLinear,
        )

        config = FluxConfig.flux_535m(
            transformer_impl="local",
            fp8="e4m3",
            fp8_recipe="tensorwise",
        )

        block_submodules = get_flux_layer_spec(config, backend=None)

        # At least one linear spec should reference Float8ColumnParallelLinear
        found_fp8 = False
        for layer_spec in block_submodules.layer_specs:
            attn_spec = layer_spec.submodules.self_attention
            if hasattr(attn_spec, "submodules") and hasattr(attn_spec.submodules, "linear_qkv"):
                if attn_spec.submodules.linear_qkv == Float8ColumnParallelLinear:
                    found_fp8 = True
                    break
        assert found_fp8, "Expected Float8ColumnParallelLinear in layer specs for local+fp8"

    def test_backend_selection_local_no_fp8_uses_native_linear(self):
        """Test that local without fp8 uses native ColumnParallelLinear."""

        from megatron.core.tensor_parallel import ColumnParallelLinear

        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            Float8ColumnParallelLinear,
        )

        config = FluxConfig.flux_535m(transformer_impl="local", fp8=None)

        block_submodules = get_flux_layer_spec(config, backend=None)

        found_any = False
        for layer_spec in block_submodules.layer_specs:
            attn_spec = layer_spec.submodules.self_attention
            if hasattr(attn_spec, "submodules") and hasattr(attn_spec.submodules, "linear_qkv"):
                found_any = True
                assert (
                    attn_spec.submodules.linear_qkv != Float8ColumnParallelLinear
                ), "Should NOT use Float8ColumnParallelLinear when fp8=None"
                assert (
                    attn_spec.submodules.linear_qkv == ColumnParallelLinear
                ), f"Expected native ColumnParallelLinear when fp8=None, got {attn_spec.submodules.linear_qkv}"
        assert found_any, "No attention linear_qkv specs found to validate backend selection"
