# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Basic unit tests for Flux model.

Tests the Flux model initialization and basic forward pass.
"""

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    pack_latents,
    unpack_latents,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    T5_XXL_EMBEDDING_DIM,
    TEXT_SEQ_LEN_SHORT,
    VAE_LATENT_CHANNELS,
)
from tests.utils import PrimusUT


class TestFluxModel(PrimusUT):
    """Core tests for Flux model initialization and basic operations."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    def test_forward_pass_small(self):
        """Test forward pass with small inputs."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        config = FluxConfig.flux_535m()
        model = Flux(config).cuda()
        model.eval()

        batch_size = 2
        height, width = 16, 16
        channels = VAE_LATENT_CHANNELS
        txt_seq_len = TEXT_SEQ_LEN_SHORT

        # Prepare inputs
        img = torch.randn(batch_size, channels, height, width).cuda()
        txt = torch.randn(batch_size, txt_seq_len, T5_XXL_EMBEDDING_DIM).cuda()
        y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM).cuda()
        timesteps = torch.rand(batch_size).cuda()

        # Pack latents
        packed_img = pack_latents(img)
        packed_img = packed_img.transpose(0, 1)
        txt_t = txt.transpose(0, 1)

        # Generate position IDs
        img_ids = generate_image_position_ids(batch_size, height, width, device="cuda")
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3).cuda()

        # Forward pass
        with torch.no_grad():
            output = model(packed_img, txt_t, y, timesteps, img_ids, txt_ids)

        # Unpack output
        output = output.transpose(0, 1)
        output = unpack_latents(output, height, width, vae_scale_factor=1)

        # Check output shape
        assert output.shape == img.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
