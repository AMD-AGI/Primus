# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux checkpoint converter.

Tests QKV fusion, key mapping, and checkpoint conversion logic.
"""

import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux import FluxConfig
from tests.utils import PrimusUT

# Note: We use lazy imports for checkpoint_converter functions to avoid
# triggering parent package imports that require Megatron. Functions are
# imported inside test methods when needed.


class TestQKVWeightFusion(PrimusUT):
    """Test QKV weight fusion for GQA."""

    def test_qkv_weight_fusion_shape(self):
        """Test that QKV fusion produces correct output shape."""
        # Lazy import to avoid parent package Megatron dependency
        from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
            _fuse_qkv_weights,
        )

        # Use Flux 12B config: hidden_size=3072, num_heads=24, head_size=128
        config = FluxConfig.flux_12b()

        hidden_size = config.hidden_size
        q_weight = torch.randn(hidden_size, hidden_size)
        k_weight = torch.randn(hidden_size, hidden_size)
        v_weight = torch.randn(hidden_size, hidden_size)

        # Fuse
        qkv_weight = _fuse_qkv_weights(config, q_weight, k_weight, v_weight)

        # Expected shape: [head_size * (num_heads + 2*num_query_groups), hidden_size]
        # For Flux: num_query_groups = num_heads = 24
        # So: [128 * (24 + 2*24), 3072] = [128 * 72, 3072] = [9216, 3072]
        num_heads = config.num_attention_heads
        num_query_groups = getattr(config, "num_query_groups", num_heads)
        head_size = hidden_size // num_heads
        expected_shape = (head_size * (num_heads + 2 * num_query_groups), hidden_size)

        assert qkv_weight.shape == expected_shape, f"Expected {expected_shape}, got {qkv_weight.shape}"

    def test_qkv_weight_fusion_interleaving(self):
        """Test that QKV weights are interleaved correctly per group."""
        # Lazy import to avoid parent package Megatron dependency
        from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
            _fuse_qkv_weights,
        )

        # Use Flux 535M config with simple values for easier validation
        # Note: Flux 535M has hidden_size=3072, num_heads=24, but we adjust expectations
        # for the interleaving test which needs smaller values
        config = FluxConfig.flux_535m()
        # Override for test simplicity - fusion algorithm works with any config
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.num_query_groups = 4

        hidden_size = 64
        head_size = 16

        # Create identifiable weights
        q_weight = torch.ones(hidden_size, hidden_size) * 1.0
        k_weight = torch.ones(hidden_size, hidden_size) * 2.0
        v_weight = torch.ones(hidden_size, hidden_size) * 3.0

        qkv_weight = _fuse_qkv_weights(config, q_weight, k_weight, v_weight)

        # Reshape to verify interleaving: [heads_per_group + 2, num_groups, head_size, hidden_size]
        # For this config: heads_per_group=1, num_groups=4
        # So we expect pattern: [Q0, K0, V0, Q1, K1, V1, Q2, K2, V2, Q3, K3, V3]
        qkv_reshaped = qkv_weight.reshape(4, 3, head_size, hidden_size)

        # Check first group
        assert torch.allclose(qkv_reshaped[0, 0], torch.ones(head_size, hidden_size) * 1.0), "Q0 mismatch"
        assert torch.allclose(qkv_reshaped[0, 1], torch.ones(head_size, hidden_size) * 2.0), "K0 mismatch"
        assert torch.allclose(qkv_reshaped[0, 2], torch.ones(head_size, hidden_size) * 3.0), "V0 mismatch"


class TestQKVBiasFusion(PrimusUT):
    """Test QKV bias fusion for GQA."""

    def test_qkv_bias_fusion_shape(self):
        """Test that QKV bias fusion produces correct output shape."""
        # Lazy import to avoid parent package Megatron dependency
        from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
            _fuse_qkv_bias,
        )

        # Use Flux 12B config
        config = FluxConfig.flux_12b()

        hidden_size = config.hidden_size
        q_bias = torch.randn(hidden_size)
        k_bias = torch.randn(hidden_size)
        v_bias = torch.randn(hidden_size)

        qkv_bias = _fuse_qkv_bias(config, q_bias, k_bias, v_bias)

        # Expected shape: [head_size * (num_heads + 2*num_query_groups)]
        num_heads = config.num_attention_heads
        num_query_groups = getattr(config, "num_query_groups", num_heads)
        head_size = hidden_size // num_heads
        expected_shape = (head_size * (num_heads + 2 * num_query_groups),)

        assert qkv_bias.shape == expected_shape, f"Expected {expected_shape}, got {qkv_bias.shape}"

    def test_qkv_bias_fusion_interleaving(self):
        """Test that QKV biases are interleaved correctly."""
        # Lazy import to avoid parent package Megatron dependency
        from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
            _fuse_qkv_bias,
        )

        # Use Flux 535M config with simple values for easier validation
        config = FluxConfig.flux_535m()
        # Override for test simplicity - fusion algorithm works with any config
        config.hidden_size = 64
        config.num_attention_heads = 4
        config.num_query_groups = 4

        # Create identifiable biases
        q_bias = torch.ones(64) * 1.0
        k_bias = torch.ones(64) * 2.0
        v_bias = torch.ones(64) * 3.0

        qkv_bias = _fuse_qkv_bias(config, q_bias, k_bias, v_bias)

        # Reshape to verify: [num_groups, heads_per_group + 2, head_size]
        qkv_reshaped = qkv_bias.reshape(4, 3, 16)

        # Check first group
        assert torch.allclose(qkv_reshaped[0, 0], torch.ones(16) * 1.0)
        assert torch.allclose(qkv_reshaped[0, 1], torch.ones(16) * 2.0)
        assert torch.allclose(qkv_reshaped[0, 2], torch.ones(16) * 3.0)


class TestCheckpointConversion:
    """Test end-to-end checkpoint conversion (with mock data)."""

    def test_mock_conversion(self, tmp_path):
        """Test conversion with mock HF checkpoint."""
        from safetensors.torch import save_file as save_safetensors

        from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
            convert_hf_checkpoint,
        )

        # Create minimal mock HF checkpoint for Flux 535M (1 joint + 1 single layer)
        config = FluxConfig.flux_535m()
        hidden_size = config.hidden_size

        mock_state_dict = {}

        # Double block 0
        mock_state_dict["transformer_blocks.0.norm1.linear.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.norm1.linear.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_q.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_q.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_k.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_k.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_v.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_v.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_out.0.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_out.0.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.norm_q.weight"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.norm_k.weight"] = torch.randn(hidden_size)

        # Added attention
        mock_state_dict["transformer_blocks.0.attn.add_q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.add_q_proj.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.add_k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.add_k_proj.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.add_v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.add_v_proj.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_add_out.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["transformer_blocks.0.attn.to_add_out.bias"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.norm_added_q.weight"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.attn.norm_added_k.weight"] = torch.randn(hidden_size)
        mock_state_dict["transformer_blocks.0.norm1_context.linear.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["transformer_blocks.0.norm1_context.linear.bias"] = torch.randn(hidden_size)

        # MLP
        mock_state_dict["transformer_blocks.0.ff.net.0.proj.weight"] = torch.randn(
            hidden_size * 4, hidden_size
        )
        mock_state_dict["transformer_blocks.0.ff.net.0.proj.bias"] = torch.randn(hidden_size * 4)
        mock_state_dict["transformer_blocks.0.ff.net.2.weight"] = torch.randn(hidden_size, hidden_size * 4)
        mock_state_dict["transformer_blocks.0.ff.net.2.bias"] = torch.randn(hidden_size)

        # Context MLP
        mock_state_dict["transformer_blocks.0.ff_context.net.0.proj.weight"] = torch.randn(
            hidden_size * 4, hidden_size
        )
        mock_state_dict["transformer_blocks.0.ff_context.net.0.proj.bias"] = torch.randn(hidden_size * 4)
        mock_state_dict["transformer_blocks.0.ff_context.net.2.weight"] = torch.randn(
            hidden_size, hidden_size * 4
        )
        mock_state_dict["transformer_blocks.0.ff_context.net.2.bias"] = torch.randn(hidden_size)

        # Single block 0
        mock_state_dict["single_transformer_blocks.0.norm.linear.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["single_transformer_blocks.0.norm.linear.bias"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.attn.to_q.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["single_transformer_blocks.0.attn.to_q.bias"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.attn.to_k.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["single_transformer_blocks.0.attn.to_k.bias"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.attn.to_v.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["single_transformer_blocks.0.attn.to_v.bias"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.attn.norm_q.weight"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.attn.norm_k.weight"] = torch.randn(hidden_size)
        mock_state_dict["single_transformer_blocks.0.proj_mlp.weight"] = torch.randn(
            hidden_size * 4, hidden_size
        )
        mock_state_dict["single_transformer_blocks.0.proj_mlp.bias"] = torch.randn(hidden_size * 4)
        mock_state_dict["single_transformer_blocks.0.proj_out.weight"] = torch.randn(
            hidden_size, hidden_size * 2
        )
        mock_state_dict["single_transformer_blocks.0.proj_out.bias"] = torch.randn(hidden_size)

        # Root level
        mock_state_dict["x_embedder.weight"] = torch.randn(hidden_size, 64)
        mock_state_dict["x_embedder.bias"] = torch.randn(hidden_size)
        mock_state_dict["context_embedder.weight"] = torch.randn(hidden_size, 4096)
        mock_state_dict["context_embedder.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.timestep_embedder.linear_1.weight"] = torch.randn(hidden_size, 256)
        mock_state_dict["time_text_embed.timestep_embedder.linear_1.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.timestep_embedder.linear_2.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["time_text_embed.timestep_embedder.linear_2.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.text_embedder.linear_1.weight"] = torch.randn(hidden_size, 768)
        mock_state_dict["time_text_embed.text_embedder.linear_1.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.text_embedder.linear_2.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["time_text_embed.text_embedder.linear_2.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.guidance_embedder.linear_1.weight"] = torch.randn(hidden_size, 256)
        mock_state_dict["time_text_embed.guidance_embedder.linear_1.bias"] = torch.randn(hidden_size)
        mock_state_dict["time_text_embed.guidance_embedder.linear_2.weight"] = torch.randn(
            hidden_size, hidden_size
        )
        mock_state_dict["time_text_embed.guidance_embedder.linear_2.bias"] = torch.randn(hidden_size)
        mock_state_dict["norm_out.linear.weight"] = torch.randn(hidden_size, hidden_size)
        mock_state_dict["norm_out.linear.bias"] = torch.randn(hidden_size)
        mock_state_dict["proj_out.weight"] = torch.randn(64, hidden_size)
        mock_state_dict["proj_out.bias"] = torch.randn(64)

        # Save mock checkpoint
        checkpoint_path = tmp_path / "mock_flux.safetensors"
        save_safetensors(mock_state_dict, str(checkpoint_path))

        # Convert
        primus_state_dict = convert_hf_checkpoint(
            checkpoint_path,
            flux_config=config,
            save_to=None,
        )

        # Verify conversion
        assert len(primus_state_dict) > 0, "Converted state dict is empty"

        # Check that QKV was fused (with new TransformerBlock naming)
        assert "transformer.layers.0.self_attention.linear_qkv.weight" in primus_state_dict
        assert "transformer.layers.0.self_attention.linear_qkv.bias" in primus_state_dict
        assert "transformer.layers.1.self_attention.linear_qkv.weight" in primus_state_dict

        # Check that proj_out was split for single blocks
        assert "transformer.layers.1.self_attention.linear_proj.weight" in primus_state_dict
        assert "transformer.layers.1.mlp.linear_fc2.weight" in primus_state_dict

        # Verify key mapping worked (with new TransformerBlock naming)
        assert "transformer.layers.0.adaln.adaLN_modulation.1.weight" in primus_state_dict
        # Note: Layer 1 is FluxSingleTransformerBlock which has different adaln structure

        # norm_out scale/shift swap (the converter's only non-trivial root-level
        # math): HF Diffusers stores the modulation as [SCALE; SHIFT] but
        # Primus/BFL native expects [SHIFT; SCALE], so the two halves must be
        # swapped along dim 0 -- a plain key-rename/passthrough would be a bug.
        orig_w = mock_state_dict["norm_out.linear.weight"]
        orig_b = mock_state_dict["norm_out.linear.bias"]
        half_w = orig_w.shape[0] // 2
        half_b = orig_b.shape[0] // 2
        expected_w = torch.cat([orig_w[half_w:, :], orig_w[:half_w, :]], dim=0)
        expected_b = torch.cat([orig_b[half_b:], orig_b[:half_b]], dim=0)

        conv_w = primus_state_dict["norm_out.adaLN_modulation.1.weight"]
        conv_b = primus_state_dict["norm_out.adaLN_modulation.1.bias"]
        assert torch.equal(conv_w, expected_w), "norm_out weight halves were not swapped"
        assert torch.equal(conv_b, expected_b), "norm_out bias halves were not swapped"
        # Positive control: the swap must actually reorder, not pass through.
        assert not torch.equal(conv_w, orig_w), "norm_out weight passed through without the scale/shift swap"
        assert not torch.equal(conv_b, orig_b), "norm_out bias passed through without the scale/shift swap"
