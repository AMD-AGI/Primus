###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for Primus Turbo linear layer CPU initialization.

Tests verify that when use_cpu_initialization=True:
1. Weights are properly initialized using _initialize_affine_weight_cpu
2. Bias is zeroed correctly
3. Bias allreduce attribute is set for LayerNormColumnParallelLinear (our fix)
"""

import functools

import pytest
import torch
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboColumnParallelLinear,
    PrimusTurboLayerNormColumnParallelLinear,
    PrimusTurboRowParallelLinear,
)
from tests.utils import PrimusUT


def create_dummy_args():
    """Create dummy args namespace for Megatron global_vars."""
    from types import SimpleNamespace

    return SimpleNamespace(
        rank=0,
        world_size=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        offload=False,
        offload_ops=[],
        patch_primus_pipeline=False,
        pp_algorithm=None,
        patch_zero_bubble=False,
        enable_zero_bubble=False,
        rampup_batch_size=None,
        global_batch_size=1,
        micro_batch_size=1,
        data_parallel_size=1,
        decrease_batch_size_if_needed=False,
    )


def init_method_xavier():
    """Create a simple Xavier uniform init method for testing."""
    return functools.partial(torch.nn.init.xavier_uniform_)


class TestPrimusTurboCPUInit(PrimusUT):
    """Test CPU initialization for all Primus Turbo linear layer classes."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state, monkeypatch):
        """Initialize parallel state for model tests."""
        dummy_args = create_dummy_args()
        import megatron.training.global_vars as global_vars_module

        monkeypatch.setattr(global_vars_module, "_GLOBAL_ARGS", dummy_args)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
    def test_row_parallel_weights_initialized_with_cpu_init(self):
        """Test that RowParallel weights are initialized (not zeros) when CPU init enabled."""
        config = ModelParallelConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        setattr(config, "symmetric_ar_type", "none")
        setattr(config, "disable_parameter_transpose_cache", False)
        setattr(config, "init_model_with_meta_device", False)

        layer = PrimusTurboRowParallelLinear(
            input_size=64,
            output_size=128,
            config=config,
            init_method=init_method_xavier(),
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        weight = layer.weight
        assert weight is not None, "Weight should exist"
        assert not torch.allclose(
            weight, torch.zeros_like(weight)
        ), "Weights should be initialized, not all zeros"
        assert weight.std() > 0.01, "Weights should have non-trivial variance after initialization"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
    def test_column_parallel_weights_initialized_with_cpu_init(self):
        """Test that ColumnParallel weights are initialized (not zeros) when CPU init enabled."""
        config = ModelParallelConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        setattr(config, "symmetric_ar_type", "none")
        setattr(config, "disable_parameter_transpose_cache", False)
        setattr(config, "init_model_with_meta_device", False)

        layer = PrimusTurboColumnParallelLinear(
            input_size=64,
            output_size=128,
            config=config,
            init_method=init_method_xavier(),
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        weight = layer.weight
        assert weight is not None, "Weight should exist"
        assert not torch.allclose(
            weight, torch.zeros_like(weight)
        ), "Weights should be initialized, not all zeros"
        assert weight.std() > 0.01, "Weights should have non-trivial variance after initialization"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
    def test_layer_norm_column_parallel_weights_initialized_with_cpu_init(self):
        """Test that LayerNormColumnParallel weights are initialized (not zeros) when CPU init enabled."""
        transformer_config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=8,
            num_layers=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )

        layer = PrimusTurboLayerNormColumnParallelLinear(
            input_size=64,
            output_size=128,
            config=transformer_config,
            init_method=init_method_xavier(),
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        weight = layer.weight
        assert weight is not None, "Weight should exist"
        assert not torch.allclose(
            weight, torch.zeros_like(weight)
        ), "Weights should be initialized, not all zeros"
        assert weight.std() > 0.01, "Weights should have non-trivial variance after initialization"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
    def test_row_parallel_bias_initialized_to_zero_with_cpu_init(self):
        """Test that RowParallel bias is initialized to zero when CPU init enabled."""
        config = ModelParallelConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )
        setattr(config, "symmetric_ar_type", "none")
        setattr(config, "disable_parameter_transpose_cache", False)
        setattr(config, "init_model_with_meta_device", False)

        layer = PrimusTurboRowParallelLinear(
            input_size=64,
            output_size=128,
            config=config,
            init_method=init_method_xavier(),
            bias=True,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        bias = torch.cat([getattr(layer, name) for name in layer.bias_names])
        assert torch.allclose(bias, torch.zeros_like(bias)), "Bias should be initialized to zero"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA/Transformer Engine")
    def test_layer_norm_bias_allreduce_attribute_set(self):
        """
        Test that bias has allreduce=True attribute for LayerNormColumnParallelLinear.

        This tests our explicit fix: we set allreduce=True on bias when CPU init is enabled
        for PrimusTurboLayerNormColumnParallelLinear (unlike Row/Column parallel, where
        the parent TELinear class sets it automatically).
        """
        transformer_config = TransformerConfig(
            hidden_size=64,
            num_attention_heads=8,
            num_layers=1,
            use_cpu_initialization=True,
            params_dtype=torch.float32,
        )

        layer = PrimusTurboLayerNormColumnParallelLinear(
            input_size=64,
            output_size=128,
            config=transformer_config,
            init_method=init_method_xavier(),
            bias=True,
            gather_output=False,
            skip_bias_add=False,
            is_expert=False,
        ).cuda()

        # Check that bias has allreduce attribute set to True (our fix)
        for bias_name in layer.bias_names:
            bias_param = getattr(layer, bias_name)
            assert hasattr(bias_param, "allreduce"), f"Bias {bias_name} should have 'allreduce' attribute"
            assert (
                getattr(bias_param, "allreduce") is True
            ), f"Bias {bias_name} should have allreduce=True (set by our CPU init code)"
