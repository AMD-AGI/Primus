# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for NeMo chimera initialization behavior.

Validates that contaminating the default CUDA generator with per-DP-rank seeds
causes non-parallel layers (img_embed, txt_embed, MLPEmbedder) to diverge
across ranks, while ColumnParallelLinear layers (using model-parallel tracker)
remain identical.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from tests.utils import PrimusUT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestFluxChimeraInit(PrimusUT):
    """Tests that chimera init creates intended per-rank diversity."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    def _build_model_with_rank_seed(self, base_seed, dp_rank):
        """Build a Flux model simulating chimera init for a given DP rank."""
        from megatron.core.tensor_parallel import random as tp_random

        per_rank_seed = base_seed + 100 * dp_rank
        torch.manual_seed(per_rank_seed)
        torch.cuda.manual_seed(per_rank_seed)
        # Model-parallel tracker uses the SAME seed across all DP ranks
        tp_random.model_parallel_cuda_manual_seed(42)

        config = FluxConfig.flux_535m()
        return Flux(config)

    def test_img_embed_differs_across_ranks(self):
        """img_embed (nn.Linear) should differ when default generator seed differs."""
        model_rank0 = self._build_model_with_rank_seed(1234, dp_rank=0)
        model_rank1 = self._build_model_with_rank_seed(1234, dp_rank=1)

        assert not torch.equal(
            model_rank0.img_embed.weight, model_rank1.img_embed.weight
        ), "img_embed.weight should differ between DP ranks"

    def test_txt_embed_differs_across_ranks(self):
        """txt_embed (nn.Linear) should differ when default generator seed differs."""
        model_rank0 = self._build_model_with_rank_seed(1234, dp_rank=0)
        model_rank1 = self._build_model_with_rank_seed(1234, dp_rank=1)

        assert not torch.equal(
            model_rank0.txt_embed.weight, model_rank1.txt_embed.weight
        ), "txt_embed.weight should differ between DP ranks"

    def test_timestep_embedding_differs_across_ranks(self):
        """MLPEmbedder layers (nn.Linear) should differ across DP ranks."""
        model_rank0 = self._build_model_with_rank_seed(1234, dp_rank=0)
        model_rank1 = self._build_model_with_rank_seed(1234, dp_rank=1)

        w0 = model_rank0.timestep_embedding.time_embedding.in_layer.weight
        w1 = model_rank1.timestep_embedding.time_embedding.in_layer.weight
        assert not torch.equal(w0, w1), "timestep_embedding in_layer should differ between DP ranks"

    def test_adaln_zero_init_identical_regardless_of_seed(self):
        """AdaLN zero-init is deterministic — identical on both ranks."""
        model_rank0 = self._build_model_with_rank_seed(1234, dp_rank=0)
        model_rank1 = self._build_model_with_rank_seed(1234, dp_rank=1)

        for i, (layer0, layer1) in enumerate(
            zip(model_rank0.transformer.layers, model_rank1.transformer.layers)
        ):
            w0 = layer0.adaln.adaLN_modulation[-1].weight
            w1 = layer1.adaln.adaLN_modulation[-1].weight
            assert torch.equal(w0, w1), f"Layer {i} adaLN weight should be identical (zero)"
            assert torch.all(w0 == 0), f"Layer {i} adaLN weight should be zero"
