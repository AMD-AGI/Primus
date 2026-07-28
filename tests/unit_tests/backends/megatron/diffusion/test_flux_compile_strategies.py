# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux compile_model() strategy validation.

Tests that invalid configuration combinations raise the expected errors
and that fallback behavior works correctly.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from tests.utils import PrimusUT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestFluxCompileStrategies(PrimusUT):
    """Tests for compile_model() error/warning behavior."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    def test_whole_model_rejects_overlap_param_gather(self):
        """whole_model strategy must reject overlap_param_gather."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "whole_model"
        model = Flux(config)

        with patch(
            "megatron.training.get_args",
            return_value=SimpleNamespace(overlap_param_gather=True),
        ):
            with pytest.raises(ValueError, match="overlap_param_gather"):
                model.compile_model()

    def test_stack_rejects_recompute_granularity(self):
        """stack strategy must reject recompute_granularity='full'."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "stack"
        config.recompute_granularity = "full"
        model = Flux(config)

        with pytest.raises(ValueError, match="recompute_granularity"):
            model.compile_model()

    def test_double_stack_rejects_recompute_granularity(self):
        """double_stack strategy must reject recompute_granularity='full'."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "double_stack"
        config.recompute_granularity = "full"
        model = Flux(config)

        with pytest.raises(ValueError, match="recompute_granularity"):
            model.compile_model()

    def test_local_spec_stack_falls_back_to_per_block(self):
        """Local spec (non-TE) with stack strategy should fallback to per_block."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "stack"
        config.transformer_impl = "local"
        model = Flux(config)

        compile_calls = []

        def mock_compile(fn, **kwargs):
            compile_calls.append(fn)
            return fn

        with patch("torch.compile", side_effect=mock_compile):
            model.compile_model()

        assert len(compile_calls) == len(model.transformer.layers), (
            f"Expected {len(model.transformer.layers)} per-block compile calls, " f"got {len(compile_calls)}"
        )

    def test_per_block_with_cuda_graph_warns(self):
        """per_block + enable_cuda_graph should emit a warning."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "per_block"
        # Build without cuda_graph (avoids CudaGraphManager compat issues),
        # then set the flag before compile_model() which only reads config.
        model = Flux(config)
        model.config.enable_cuda_graph = True

        compile_calls = []

        def mock_compile(fn, **kwargs):
            compile_calls.append(fn)
            return fn

        output_lines = []

        def capture_log(msg):
            output_lines.append(msg)

        with patch("torch.compile", side_effect=mock_compile), patch(
            "primus.core.utils.module_utils.log_rank_0", side_effect=capture_log
        ):
            model.compile_model()

        output_text = "\n".join(output_lines)
        assert "CUDA graph" in output_text, f"Expected CUDA graph warning in output, got: {output_text[:200]}"

    def test_invalid_strategy_raises_value_error(self):
        """Invalid strategy name must raise ValueError."""
        config = FluxConfig.flux_535m()
        config.enable_torch_compile = True
        config.torch_compile_strategy = "nonexistent"
        model = Flux(config)

        with pytest.raises(ValueError, match="Invalid torch_compile_strategy"):
            model.compile_model()
