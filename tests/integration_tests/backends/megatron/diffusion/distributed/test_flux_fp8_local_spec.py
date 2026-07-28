# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Integration tests for Flux model with Float8 local spec.

Tests that a real (small) Flux model constructs, uses correct linear types,
and produces valid output when using PrimusTurboFloat8LocalSpecProvider.
"""

import pytest
import torch

from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
    Float8ColumnParallelLinear,
    Float8RowParallelLinear,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    IMG_SIZE_TINY,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_SHORT,
    VAE_LATENT_CHANNELS,
)
from tests.utils import PrimusUT


class TestFluxFP8LocalSpec(PrimusUT):
    """Integration tests for Flux with Float8 local spec provider."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for FP8 local spec integration tests."""

    def _make_fp8_config(self):
        return FluxConfig.flux_535m(
            transformer_impl="local",
            fp8="e4m3",
            fp8_recipe="tensorwise",
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_535m_fp8_local_spec_constructs(self):
        """Test that Flux 535M with FP8 local spec constructs without error."""
        config = self._make_fp8_config()
        model = Flux(config)
        assert model is not None
        assert isinstance(model, Flux)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_535m_fp8_local_spec_linear_types(self):
        """Test that spec-provided parallel linears are Float8 variants after construction.

        Only checks linears inside attention (linear_qkv, linear_proj) and MLP
        (linear_fc1, linear_fc2) submodules. AdaLN modulation layers are created
        directly and intentionally use standard ColumnParallelLinear.
        """
        config = self._make_fp8_config()
        model = Flux(config)

        spec_linear_names = {"linear_qkv", "added_linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"}

        from megatron.core.tensor_parallel.layers import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        for name, module in model.named_modules():
            leaf_name = name.rsplit(".", 1)[-1] if "." in name else name
            if leaf_name not in spec_linear_names:
                continue
            if isinstance(module, ColumnParallelLinear):
                assert isinstance(
                    module, Float8ColumnParallelLinear
                ), f"{name}: expected Float8ColumnParallelLinear, got {type(module).__name__}"
            if isinstance(module, RowParallelLinear):
                assert isinstance(
                    module, Float8RowParallelLinear
                ), f"{name}: expected Float8RowParallelLinear, got {type(module).__name__}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_535m_fp8_local_spec_forward(self):
        """Test forward pass produces valid (non-NaN/Inf) output with correct shape."""
        config = self._make_fp8_config()
        model = Flux(config).cuda().to(torch.bfloat16)
        model.eval()

        batch_size = 2
        height, width = IMG_SIZE_TINY, IMG_SIZE_TINY
        channels = VAE_LATENT_CHANNELS
        txt_seq_len = TEXT_SEQ_LEN_SHORT

        img = torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16).cuda()
        txt = torch.randn(batch_size, txt_seq_len, T5_XXL_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        timesteps = torch.rand(batch_size, dtype=torch.bfloat16).cuda()

        packed_img = pack_latents(img)
        packed_img = packed_img.transpose(0, 1)
        txt_t = txt.transpose(0, 1)

        img_ids = generate_image_position_ids(batch_size, height, width, device="cuda")
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3).cuda()

        with torch.no_grad():
            output = model(packed_img, txt_t, y, timesteps, img_ids, txt_ids)

        assert output is not None
        assert len(output.shape) == 3
        assert output.shape[1] == batch_size
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
