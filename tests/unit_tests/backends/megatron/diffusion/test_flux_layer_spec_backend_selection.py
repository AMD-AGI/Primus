# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux layer spec backend selection.

Tests that get_flux_layer_spec() correctly selects backend based on transformer_impl.
This ensures alignment between backend selection and FSDP2 wrapping decisions.
"""

from collections import Counter
from unittest.mock import patch

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    get_flux_layer_spec,
)
from tests.utils import PrimusUT


def graduated_config():
    return FluxConfig.flux_12b(
        transformer_impl="local",
        fp4="mxfp4",
        fp4_recipe="mxfp4",
        mxfp4_backward_precision="mxfp4",
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        sensitive_layers_enabled=True,
        sensitive_layers_start=4,
        sensitive_layers_end=4,
        sensitive_layer_precision="tw_fp8",
        outer_sensitive_layers_start=1,
        outer_sensitive_layers_end=1,
        outer_sensitive_layer_precision="bf16",
    )


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

    def test_graduated_boundary_routes_outer_bf16_inner_fp8_middle_mxfp4(self):
        """Every declared Flux linear follows the exact three-precision layout."""
        from megatron.core.tensor_parallel import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            Float8ColumnParallelLinear,
            Float8RowParallelLinear,
        )
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
            MXFP4RowParallelLinear,
        )

        expected_classes = {
            "bf16": (ColumnParallelLinear, RowParallelLinear),
            "fp8": (Float8ColumnParallelLinear, Float8RowParallelLinear),
            "mxfp4": (MXFP4ColumnParallelLinear, MXFP4RowParallelLinear),
        }

        block_submodules = get_flux_layer_spec(graduated_config(), backend=None)
        precision_counts = Counter()
        observed_slots = 0
        for index, layer_spec in enumerate(block_submodules.layer_specs):
            expected_precision = (
                "bf16"
                if index in (0, 56)
                else "fp8"
                if index in (1, 2, 3, 53, 54, 55)
                else "mxfp4"
            )
            column_class, row_class = expected_classes[expected_precision]
            attention = layer_spec.submodules.self_attention.submodules
            mlp = layer_spec.submodules.mlp.submodules
            slots = {
                "linear_qkv": (attention.linear_qkv, column_class),
                "linear_proj": (attention.linear_proj, row_class),
                "linear_fc1": (mlp.linear_fc1, column_class),
                "linear_fc2": (mlp.linear_fc2, row_class),
            }
            if index < 19:
                slots["added_linear_qkv"] = (
                    attention.added_linear_qkv,
                    column_class,
                )

            for slot, (actual_class, expected_class) in slots.items():
                assert actual_class is expected_class, (
                    f"block {index} {slot}: expected {expected_class}, "
                    f"got {actual_class}"
                )
                precision_counts[expected_precision] += 1
                observed_slots += 1

        assert observed_slots == 247
        assert precision_counts == {"bf16": 9, "fp8": 27, "mxfp4": 211}

    def test_graduated_routing_rejects_explicit_backend(self):
        from primus.backends.megatron.core.extensions.primus_turbo_local_spec import (
            PrimusTurboMXFP4LocalSpecProvider,
        )

        with self.assertRaisesRegex(
            ValueError, "does not support an explicit backend"
        ):
            get_flux_layer_spec(
                graduated_config(),
                backend=PrimusTurboMXFP4LocalSpecProvider(),
            )

    def test_graduated_routing_rejects_missing_mxfp4_provider(self):
        with patch(
            "primus.backends.megatron.core.models.diffusion.flux.layer_spec."
            "PrimusTurboMXFP4LocalSpecProvider",
            None,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires the MXFP4 local provider"
            ):
                get_flux_layer_spec(graduated_config(), backend=None)

    def test_graduated_routing_rejects_missing_fp8_provider(self):
        with patch(
            "primus.backends.megatron.core.models.diffusion.flux.layer_spec."
            "PrimusTurboFloat8LocalSpecProvider",
            None,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires the FP8 local provider"
            ):
                get_flux_layer_spec(graduated_config(), backend=None)

    def test_graduated_routing_rejects_missing_native_provider(self):
        with patch(
            "primus.backends.megatron.core.models.diffusion.flux.layer_spec."
            "PrimusTurboLocalSpecProvider",
            None,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires the native local provider"
            ):
                get_flux_layer_spec(graduated_config(), backend=None)

    def test_graduated_routing_revalidates_mutated_config(self):
        cases = [
            (
                "fp8_recipe",
                "blockwise",
                "require normalized fp8='e4m3'",
            ),
            (
                "outer_sensitive_layer_precision",
                "tw_fp8",
                "requires inner precision 'tw_fp8' and outer precision 'bf16'",
            ),
            (
                "outer_sensitive_layers_start",
                4,
                "requires at least one inner sensitive layer on the same side",
            ),
            (
                "outer_sensitive_layers_start",
                5,
                "outer_sensitive_layers_start cannot exceed sensitive_layers_start",
            ),
            (
                "outer_sensitive_layers_end",
                5,
                "outer_sensitive_layers_end cannot exceed sensitive_layers_end",
            ),
            (
                "sensitive_layers_start",
                "4",
                "sensitive_layers_start must be an integer",
            ),
            (
                "sensitive_layers_end",
                -1,
                "sensitive_layers_end must be non-negative",
            ),
            (
                "sensitive_layers_start",
                54,
                "sensitive layer counts exceed the number of Flux layers",
            ),
            (
                "params_dtype",
                torch.float32,
                "bf16 sensitive layers require bf16=True",
            ),
        ]
        for field_name, value, message in cases:
            with self.subTest(field_name=field_name, value=value):
                config = graduated_config()
                setattr(config, field_name, value)
                with self.assertRaisesRegex(ValueError, message):
                    get_flux_layer_spec(config, backend=None)

        config = graduated_config()
        config.sensitive_layers_start = 0
        config.sensitive_layers_end = 0
        config.outer_sensitive_layers_start = 0
        config.outer_sensitive_layers_end = 0
        with self.assertRaisesRegex(ValueError, "requires a non-empty boundary"):
            get_flux_layer_spec(config, backend=None)
