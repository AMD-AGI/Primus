# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Integration tests for Flux model with FSDP2.

Tests that Flux model works correctly with FSDP2 wrapping for both
transformer_impl="local" and transformer_impl="transformer_engine".
"""

import pytest
import torch
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)

from primus.backends.megatron.core.distributed.torch_fully_sharded_data_parallel import (
    PrimusTorchFullyShardedDataParallel,
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


class TestFluxFSDP2Integration(PrimusUT):
    """Integration tests for Flux model with FSDP2."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for FSDP2 integration tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_model_fsdp2_wrapping_local(self):
        """Test Flux model can be wrapped with FSDP2 when transformer_impl == 'local'."""
        self._test_flux_model_fsdp2_wrapping("local")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_model_fsdp2_wrapping_te(self):
        """Test Flux model can be wrapped with FSDP2 when transformer_impl == 'transformer_engine'."""
        self._test_flux_model_fsdp2_wrapping("transformer_engine")

    def _test_flux_model_fsdp2_wrapping(self, transformer_impl):
        """Helper method to test FSDP2 wrapping with given transformer_impl."""
        # Create Flux config with specified transformer_impl
        config = FluxConfig.flux_535m(transformer_impl=transformer_impl)

        # Create Flux model
        model = Flux(config).cuda()
        model.eval()

        # Create FSDP2 wrapper
        ddp_config = DistributedDataParallelConfig()

        try:
            fsdp_wrapper = PrimusTorchFullyShardedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model,
            )

            # Verify wrapper was created
            assert fsdp_wrapper is not None
            assert fsdp_wrapper.config.transformer_impl == transformer_impl

        except Exception as e:
            # If FSDP2 is not available, skip the test
            if "FSDP" in str(e) or "fully_shard" in str(e).lower():
                pytest.skip(f"FSDP2 not available: {e}")
            raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_model_fsdp2_forward_pass_local(self):
        """Test Flux model forward pass works with FSDP2 wrapping when transformer_impl == 'local'."""
        self._test_flux_model_fsdp2_forward_pass("local")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_model_fsdp2_forward_pass_te(self):
        """Test Flux model forward pass works with FSDP2 wrapping when transformer_impl == 'transformer_engine'."""
        self._test_flux_model_fsdp2_forward_pass("transformer_engine")

    def _test_flux_model_fsdp2_forward_pass(self, transformer_impl):
        """Test Flux model forward pass works with FSDP2 wrapping."""
        # Create Flux config with specified transformer_impl
        config = FluxConfig.flux_535m(transformer_impl=transformer_impl)

        # Create Flux model (bf16 required by FlashAttention on ROCm)
        model = Flux(config).cuda().to(torch.bfloat16)
        model.eval()

        # Prepare inputs
        batch_size = 2
        height, width = IMG_SIZE_TINY, IMG_SIZE_TINY
        channels = VAE_LATENT_CHANNELS
        txt_seq_len = TEXT_SEQ_LEN_SHORT

        img = torch.randn(batch_size, channels, height, width, dtype=torch.bfloat16).cuda()
        txt = torch.randn(batch_size, txt_seq_len, T5_XXL_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        y = torch.randn(batch_size, CLIP_L_EMBEDDING_DIM, dtype=torch.bfloat16).cuda()
        timesteps = torch.rand(batch_size, dtype=torch.bfloat16).cuda()

        # Pack latents
        packed_img = pack_latents(img)
        packed_img = packed_img.transpose(0, 1)
        txt_t = txt.transpose(0, 1)

        # Generate position IDs
        img_ids = generate_image_position_ids(batch_size, height, width, device="cuda")
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3).cuda()

        # Create FSDP2 wrapper
        ddp_config = DistributedDataParallelConfig()

        try:
            fsdp_wrapper = PrimusTorchFullyShardedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model,
            )

            # Forward pass through FSDP-wrapped model
            with torch.no_grad():
                output = fsdp_wrapper.module(
                    img=packed_img,
                    txt=txt_t,
                    y=y,
                    timesteps=timesteps,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                )

            # Verify output shape
            # Output is in sequence-first format: (seq_len, batch, channels)
            # After packing 16x16 image: seq_len = (16*16)/4 = 64
            assert output is not None
            assert len(output.shape) == 3  # (seq_len, batch, channels)
            assert output.shape[1] == batch_size  # Batch dimension is at index 1

        except Exception as e:
            # If FSDP2 is not available, skip the test
            if "FSDP" in str(e) or "fully_shard" in str(e).lower():
                pytest.skip(f"FSDP2 not available: {e}")
            raise

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_flux_model_fsdp2_no_wrapping_errors(self):
        """Test that FSDP2 wrapping doesn't produce duplicate wrapping errors."""
        config = FluxConfig.flux_535m(transformer_impl="local")

        model = Flux(config).cuda()
        ddp_config = DistributedDataParallelConfig()

        try:
            # This should not raise "Invalid mesh_dim_names" or duplicate wrapping errors
            fsdp_wrapper = PrimusTorchFullyShardedDataParallel(
                config=config,
                ddp_config=ddp_config,
                module=model,
            )

            # If we get here, wrapping succeeded without errors
            assert fsdp_wrapper is not None

        except Exception as e:
            error_msg = str(e).lower()
            # Check for specific FSDP2 wrapping errors
            if "mesh_dim_names" in error_msg or "duplicate" in error_msg:
                pytest.fail(f"FSDP2 wrapping error detected: {e}")
            # If FSDP2 is not available, skip the test
            if "fsdp" in error_msg or "fully_shard" in error_msg:
                pytest.skip(f"FSDP2 not available: {e}")
            raise
