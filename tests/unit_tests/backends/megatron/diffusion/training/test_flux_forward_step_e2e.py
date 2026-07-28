# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
End-to-end tests for flux_forward_step_func.

Tests the full forward step with a real Flux 535M model on CUDA,
covering presampled, resample, validation, and CFG dropout paths.
"""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.utils import skip_if_no_cuda

skip_if_no_cuda()

from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.model import Flux
from primus.backends.megatron.training.diffusion.forward_step import (
    flux_forward_step_func,
)
from primus.backends.megatron.training.diffusion.schedulers.flow_matching import (
    FlowMatchEulerDiscreteScheduler,
)
from tests.unit_tests.backends.megatron.diffusion.constants import (
    CLIP_L_EMBEDDING_DIM,
    T5_XXL_EMBEDDING_DIM,
    VAE_LATENT_CHANNELS,
)


def _patch_parallel_state():
    """Return a stack of mock.patch context managers for parallel state."""
    return [
        patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "megatron.core.parallel_state.get_pipeline_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "megatron.core.parallel_state.get_data_parallel_rank",
            return_value=0,
        ),
        patch(
            "megatron.core.parallel_state.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "megatron.training.get_args",
            return_value=SimpleNamespace(seed=42),
        ),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestFluxForwardStepE2E:
    """End-to-end tests for flux_forward_step_func with real model."""

    @pytest.fixture(autouse=True)
    def setup_parallel(self, init_parallel_state):
        """Initialize parallel state for model tests."""

    @pytest.fixture
    def model(self):
        config = FluxConfig.flux_535m()
        m = Flux(config).cuda()
        m.train()
        return m

    @pytest.fixture
    def scheduler(self):
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)

    def _make_presampled_batch(self, batch_size=2, height=16, width=16):
        return {
            "latents": torch.randn(batch_size, VAE_LATENT_CHANNELS, height, width),
            "prompt_embeds": torch.randn(batch_size, 32, T5_XXL_EMBEDDING_DIM),
            "pooled_prompt_embeds": torch.randn(batch_size, CLIP_L_EMBEDDING_DIM),
        }

    def _make_resample_batch(self, batch_size=2, height=16, width=16):
        return {
            "mean": torch.randn(batch_size, VAE_LATENT_CHANNELS, height, width),
            "logvar": torch.randn(batch_size, VAE_LATENT_CHANNELS, height, width),
            "prompt_embeds": torch.randn(batch_size, 32, T5_XXL_EMBEDDING_DIM),
            "pooled_prompt_embeds": torch.randn(batch_size, CLIP_L_EMBEDDING_DIM),
        }

    def test_presampled_path(self, model, scheduler):
        """Test forward step with pre-encoded latents."""
        batch = self._make_presampled_batch()
        data_iterator = iter([batch])

        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)
            result = flux_forward_step_func(
                data_iterator,
                model,
                scheduler=scheduler,
                step_count=1,
            )

        noise_pred, clean_latents, noise, loss_mask, metrics, is_validation = result
        assert noise_pred.shape == (2, VAE_LATENT_CHANNELS, 16, 16)
        assert clean_latents.shape == noise_pred.shape
        assert noise.shape == noise_pred.shape
        assert loss_mask is None
        assert not is_validation
        assert metrics["batch_size"] == 2
        assert metrics["latent_channels"] == VAE_LATENT_CHANNELS
        assert metrics["image_height"] == 16 * 8
        assert metrics["image_width"] == 16 * 8

    def test_resample_path(self, model, scheduler):
        """Test forward step with VAE resample mode."""
        batch = self._make_resample_batch()
        # Snapshot mean before the call (the forward step casts and moves it).
        original_mean = batch["mean"].clone()
        data_iterator = iter([batch])

        vae_scale = 0.3611
        vae_shift = 0.1159

        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)
            result = flux_forward_step_func(
                data_iterator,
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=vae_scale,
                vae_shift=vae_shift,
                step_count=1,
            )

        noise_pred, clean_latents, _, _, _, is_validation = result
        assert noise_pred.shape == (2, VAE_LATENT_CHANNELS, 16, 16)
        assert not is_validation

        # Verify reparameterization actually fired: clean_latents must NOT
        # equal the deterministic scale/shift on `mean` alone (which would
        # be the result if vae_eps had been dropped or zeroed).
        expected_no_eps = vae_scale * (original_mean.cuda() - vae_shift)
        assert not torch.allclose(clean_latents.float(), expected_no_eps.float(), atol=1e-3), (
            "clean_latents matches mean*(scale-shift) — reparameterization "
            "noise term `eps * std` appears to have been dropped"
        )

    def test_validation_with_timestep_key(self, model, scheduler):
        """Batch with 'timestep' key triggers validation mode."""
        batch = self._make_presampled_batch()
        batch["timestep"] = torch.arange(2)
        data_iterator = iter([batch])

        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)
            result = flux_forward_step_func(
                data_iterator,
                model,
                scheduler=scheduler,
                step_count=1,
            )

        _, _, _, _, _, is_validation = result
        assert is_validation is True
        # The forward step writes derived timesteps (timestep / 8.0) into the batch.
        assert "timesteps" in batch
        assert torch.equal(
            batch["timesteps"].float().cpu(),
            torch.arange(2).float() / 8.0,
        )

    def test_validation_equidistant_injection(self, model, scheduler):
        """model.eval() without timestep key injects equidistant timesteps."""
        model.eval()
        batch_size = 8
        batch = self._make_presampled_batch(batch_size=batch_size)
        data_iterator = iter([batch])

        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)
            result = flux_forward_step_func(
                data_iterator,
                model,
                scheduler=scheduler,
                step_count=1,
            )

        _, _, _, _, _, is_validation = result
        assert is_validation is True
        # Equidistant injection: batch["timestep"] = arange(B) % 8,
        # batch["timesteps"] = that / 8.0.
        assert "timestep" in batch
        assert "timesteps" in batch
        expected_timestep = torch.arange(batch_size, device="cuda") % 8
        assert torch.equal(batch["timestep"], expected_timestep), (
            f"batch['timestep'] expected {expected_timestep.tolist()}, " f"got {batch['timestep'].tolist()}"
        )
        assert torch.allclose(
            batch["timesteps"].float(),
            expected_timestep.float() / 8.0,
        )

    def test_cfg_dropout_full(self, model, scheduler):
        """cfg_dropout_prob=1.0 replaces all text embeddings with empty encodings."""
        batch_size = 2
        seq_len = 32
        batch = self._make_presampled_batch(batch_size=batch_size)
        data_iterator = iter([batch])

        # Use distinguishable empty encodings (not zeros) so we can verify
        # the dropout actually substituted them rather than incidentally
        # matching the original prompt_embeds.
        empty_t5_value = 7.5
        empty_clip_value = -3.25
        empty_t5 = torch.full((seq_len, 1, T5_XXL_EMBEDDING_DIM), empty_t5_value, dtype=torch.float32)
        empty_clip = torch.full((CLIP_L_EMBEDDING_DIM,), empty_clip_value, dtype=torch.float32)

        # Capture the inputs the model receives by patching its forward.
        captured = {}
        original_forward = model.forward

        def capturing_forward(*args, **kwargs):
            captured["txt"] = kwargs.get("txt").detach().clone()
            captured["y"] = kwargs.get("y").detach().clone()
            return original_forward(*args, **kwargs)

        model.forward = capturing_forward

        try:
            with contextlib.ExitStack() as stack:
                for p in _patch_parallel_state():
                    stack.enter_context(p)
                result = flux_forward_step_func(
                    data_iterator,
                    model,
                    scheduler=scheduler,
                    cfg_dropout_prob=1.0,
                    empty_t5_encodings=empty_t5,
                    empty_clip_encodings=empty_clip,
                    step_count=1,
                )
        finally:
            model.forward = original_forward

        noise_pred, _, _, _, _, is_validation = result
        assert noise_pred.shape == (batch_size, VAE_LATENT_CHANNELS, 16, 16)
        assert not is_validation

        # txt is in (S, B, C) layout after the transpose.
        # With prob=1.0 every row must be the empty encoding.
        txt = captured["txt"].float()
        assert torch.allclose(txt, torch.full_like(txt, empty_t5_value), atol=1e-2), (
            f"prompt_embeds was not replaced by empty_t5 with cfg_dropout_prob=1.0; "
            f"sample value={txt.flatten()[0].item():.3f}, expected {empty_t5_value}"
        )

        y = captured["y"].float()
        assert torch.allclose(y, torch.full_like(y, empty_clip_value), atol=1e-2), (
            f"pooled_prompt_embeds was not replaced by empty_clip with prob=1.0; "
            f"sample value={y.flatten()[0].item():.3f}, expected {empty_clip_value}"
        )
