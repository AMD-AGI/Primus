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
from tests.unit_tests.backends.megatron.conftest import requires_mxfp4
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

    @pytest.fixture(autouse=True)
    def pin_fp4_aiter(self, monkeypatch):
        """Pin the backend required by MXFP4 module construction."""
        import collections
        import os

        from primus_turbo.pytorch.core.backend import (
            BackendType,
            GlobalBackendManager,
            PrecisionType,
        )

        if os.environ.get("PRIMUS_TURBO_GEMM_BACKEND", None) == "":
            monkeypatch.delenv("PRIMUS_TURBO_GEMM_BACKEND", raising=False)
        pinned = collections.defaultdict(lambda: None)
        if GlobalBackendManager._gemm_backend:
            pinned.update(GlobalBackendManager._gemm_backend)
        pinned[PrecisionType.FP4] = BackendType.AITER
        monkeypatch.setattr(GlobalBackendManager, "_gemm_backend", pinned)
        monkeypatch.setattr(GlobalBackendManager, "_auto_tune", False)

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

    @requires_mxfp4
    def test_build_graduated_precision_model(self):
        """Build a small model that instantiates BF16, FP8, and MXFP4 blocks."""
        from megatron.core.tensor_parallel import ColumnParallelLinear

        from primus.backends.megatron.core.extensions.primus_turbo_float8_local import (
            Float8ColumnParallelLinear,
        )
        from primus.backends.megatron.core.extensions.primus_turbo_mxfp4_local import (
            MXFP4ColumnParallelLinear,
        )

        config = FluxConfig(
            num_joint_layers=2,
            num_single_layers=3,
            hidden_size=128,
            num_attention_heads=4,
            ffn_hidden_size=256,
            context_dim=128,
            vec_in_dim=64,
            model_channels=64,
            axes_dim=(4, 14, 14),
            transformer_impl="local",
            fp4="mxfp4",
            fp4_recipe="mxfp4",
            bf16=True,
            fp16=False,
            params_dtype=torch.bfloat16,
            sensitive_layers_enabled=True,
            sensitive_layers_start=2,
            sensitive_layers_end=2,
            sensitive_layer_precision="tw_fp8",
            outer_sensitive_layers_start=1,
            outer_sensitive_layers_end=1,
            outer_sensitive_layer_precision="bf16",
        )

        model = Flux(config).cuda()
        layers = model.transformer.layers

        assert len(layers) == 5
        assert type(layers[0].self_attention.linear_qkv) is ColumnParallelLinear
        assert type(layers[1].self_attention.linear_qkv) is Float8ColumnParallelLinear
        assert type(layers[2].self_attention.linear_qkv) is MXFP4ColumnParallelLinear
        assert type(layers[3].self_attention.linear_qkv) is Float8ColumnParallelLinear
        assert type(layers[4].self_attention.linear_qkv) is ColumnParallelLinear


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
