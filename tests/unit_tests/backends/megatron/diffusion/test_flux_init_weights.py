# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for Flux model weight initialization.

Validates the NeMo-aligned init_weights() implementation:
- AdaLN modulation layers are zero-initialized
- Embeddings receive Xavier/Normal initialization
- proj_out and norm_out are zero-initialized
- Initialization is deterministic given fixed RNG state
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from tests.utils import PrimusUT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestFluxInitWeights(PrimusUT):
    """Tests for Flux init_weights() correctness."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    def _create_model(self):
        config = FluxConfig.flux_535m()
        return Flux(config)

    def test_adaln_modulation_zero_init_single_blocks(self):
        """Single-block AdaLN modulation last layer must be zero."""
        model = self._create_model()
        num_joint = model.config.num_joint_layers

        for i, layer in enumerate(model.transformer.layers[num_joint:]):
            w = layer.adaln.adaLN_modulation[-1].weight
            b = layer.adaln.adaLN_modulation[-1].bias
            assert torch.all(w == 0), f"Single block {i}: adaLN weight not zero"
            assert torch.all(b == 0), f"Single block {i}: adaLN bias not zero"

    def test_adaln_modulation_zero_init_joint_blocks(self):
        """Joint-block AdaLN modulation last layers must be zero."""
        model = self._create_model()
        num_joint = model.config.num_joint_layers

        for i, layer in enumerate(model.transformer.layers[:num_joint]):
            w = layer.adaln.adaLN_modulation[-1].weight
            b = layer.adaln.adaLN_modulation[-1].bias
            assert torch.all(w == 0), f"Joint block {i}: adaln weight not zero"
            assert torch.all(b == 0), f"Joint block {i}: adaln bias not zero"

            w_ctx = layer.adaln_context.adaLN_modulation[-1].weight
            b_ctx = layer.adaln_context.adaLN_modulation[-1].bias
            assert torch.all(w_ctx == 0), f"Joint block {i}: adaln_context weight not zero"
            assert torch.all(b_ctx == 0), f"Joint block {i}: adaln_context bias not zero"

    def test_proj_out_zero_init(self):
        """proj_out weight and bias must be zero."""
        model = self._create_model()
        assert torch.all(model.proj_out.weight == 0)
        assert torch.all(model.proj_out.bias == 0)

    def test_norm_out_adaln_zero_init(self):
        """norm_out.adaLN_modulation last layer must be zero."""
        model = self._create_model()
        w = model.norm_out.adaLN_modulation[-1].weight
        b = model.norm_out.adaLN_modulation[-1].bias
        assert torch.all(w == 0)
        assert torch.all(b == 0)

    def test_img_embed_xavier_distribution(self):
        """img_embed should match Xavier-uniform statistics.

        xavier_uniform_ produces U(-bound, bound) with bound = sqrt(6/(in+out));
        std of that distribution is bound/sqrt(3) = sqrt(2/(in+out)).
        """
        import math

        model = self._create_model()
        w = model.img_embed.weight
        out_dim, in_dim = w.shape
        expected_std = math.sqrt(2.0 / (in_dim + out_dim))

        actual_std = w.float().std().item()
        actual_mean = w.float().mean().item()

        rel_err = abs(actual_std - expected_std) / expected_std
        assert rel_err < 0.15, (
            f"img_embed std={actual_std:.4f} deviates from Xavier expected "
            f"{expected_std:.4f} by {rel_err:.1%} (limit 15%)"
        )
        assert abs(actual_mean) < 0.005, f"img_embed mean={actual_mean:.4f} should be ~0 for Xavier"

    def test_txt_embed_xavier_distribution(self):
        """txt_embed should match Xavier-uniform statistics."""
        import math

        model = self._create_model()
        w = model.txt_embed.weight
        out_dim, in_dim = w.shape
        expected_std = math.sqrt(2.0 / (in_dim + out_dim))

        actual_std = w.float().std().item()
        actual_mean = w.float().mean().item()

        rel_err = abs(actual_std - expected_std) / expected_std
        assert rel_err < 0.15, (
            f"txt_embed std={actual_std:.4f} deviates from Xavier expected "
            f"{expected_std:.4f} by {rel_err:.1%} (limit 15%)"
        )
        assert abs(actual_mean) < 0.005, f"txt_embed mean={actual_mean:.4f} should be ~0 for Xavier"

    def test_timestep_embedding_normal_init(self):
        """Timestep embedding MLP should match Normal(std=0.02) statistics."""
        model = self._create_model()
        w = model.timestep_embedding.time_embedding.in_layer.weight
        expected_std = 0.02

        actual_std = w.float().std().item()
        actual_mean = w.float().mean().item()

        rel_err = abs(actual_std - expected_std) / expected_std
        assert rel_err < 0.15, (
            f"timestep_embedding std={actual_std:.4f} deviates from Normal(0,0.02) "
            f"expected std {expected_std} by {rel_err:.1%} (limit 15%)"
        )
        assert (
            abs(actual_mean) < 0.005
        ), f"timestep_embedding mean={actual_mean:.4f} should be ~0 for Normal init"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestFluxInitWeightsDeterminism(PrimusUT):
    """Tests that init_weights produces deterministic results given fixed RNG."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    def test_deterministic_construction(self):
        """Two models built with the same RNG state must be bitwise identical."""
        from megatron.core.tensor_parallel import random as tp_random

        config = FluxConfig.flux_535m()
        seed = 123

        # Build model 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        tp_random.model_parallel_cuda_manual_seed(seed)
        model1 = Flux(config)

        # Build model 2 with identical RNG state
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        tp_random.model_parallel_cuda_manual_seed(seed)
        model2 = Flux(config)

        for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert name1 == name2
            assert torch.equal(p1, p2), (
                f"Parameter {name1} differs between constructions "
                f"(max diff: {(p1 - p2).abs().max().item():.2e})"
            )
