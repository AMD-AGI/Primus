# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Unit tests for per-step RNG isolation reproducibility.

Validates that:
- Same step_count + same seed produces bitwise-identical outputs
- Different step_count produces different outputs (RNG isolation works)
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


def _patch_parallel_state(seed=42):
    """Return a list of mock.patch context managers for parallel state."""
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
            return_value=SimpleNamespace(seed=seed),
        ),
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestVaeResampleReproducibility:
    """Tests per-step RNG isolation via observable output reproducibility."""

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

    def _make_resample_batch(self, seed=99):
        """Create a deterministic resample batch."""
        g = torch.Generator().manual_seed(seed)
        return {
            "mean": torch.randn(2, VAE_LATENT_CHANNELS, 16, 16, generator=g),
            "logvar": torch.randn(2, VAE_LATENT_CHANNELS, 16, 16, generator=g),
            "prompt_embeds": torch.randn(2, 32, T5_XXL_EMBEDDING_DIM, generator=g),
            "pooled_prompt_embeds": torch.randn(2, CLIP_L_EMBEDDING_DIM, generator=g),
        }

    def test_same_step_count_produces_identical_output(self, model, scheduler):
        """Identical seed + step_count must produce bitwise-identical noise_pred."""
        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)

            batch1 = self._make_resample_batch(seed=99)
            result1 = flux_forward_step_func(
                iter([batch1]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=5,
            )
            noise_pred_1 = result1[0].detach().clone()

            batch2 = self._make_resample_batch(seed=99)
            result2 = flux_forward_step_func(
                iter([batch2]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=5,
            )
            noise_pred_2 = result2[0].detach().clone()

        assert torch.equal(noise_pred_1, noise_pred_2), (
            "Same step_count should produce identical output "
            f"(max diff: {(noise_pred_1 - noise_pred_2).abs().max().item():.2e})"
        )

    def test_different_step_count_produces_different_latents(self, model, scheduler):
        """Different step_count values should produce different clean_latents.

        Note: We compare clean_latents (index 1) rather than noise_pred (index 0)
        because proj_out is zero-initialized (NeMo-aligned init), making noise_pred
        always zero until training updates the projection layer.
        """
        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state():
                stack.enter_context(p)

            batch1 = self._make_resample_batch(seed=99)
            result1 = flux_forward_step_func(
                iter([batch1]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=5,
            )
            latents_1 = result1[1].detach().clone()

            batch2 = self._make_resample_batch(seed=99)
            result2 = flux_forward_step_func(
                iter([batch2]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=6,
            )
            latents_2 = result2[1].detach().clone()

        assert not torch.equal(latents_1, latents_2), (
            "Different step_count values should produce different latents "
            "(per-step RNG isolation not working)"
        )

    def test_different_base_seed_produces_different_output(self, model, scheduler):
        """Different args.seed (same step_count) must produce different latents.

        Validates the `_per_rank_seed = seed + 100*dp_rank` term in the step
        seed formula at forward_step.py:237. A regression that drops `seed`
        and uses only `step_count` would cause both runs to produce identical
        output (since step_count is fixed at 5).
        """
        # Run 1: seed=42
        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state(seed=42):
                stack.enter_context(p)
            batch1 = self._make_resample_batch(seed=99)
            result1 = flux_forward_step_func(
                iter([batch1]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=5,
            )
            latents_42 = result1[1].detach().clone()

        # Run 2: seed=43, identical step_count and identical batch contents.
        with contextlib.ExitStack() as stack:
            for p in _patch_parallel_state(seed=43):
                stack.enter_context(p)
            batch2 = self._make_resample_batch(seed=99)
            result2 = flux_forward_step_func(
                iter([batch2]),
                model,
                scheduler=scheduler,
                vae_latent_mode="resample",
                vae_scale=0.3611,
                vae_shift=0.1159,
                step_count=5,
            )
            latents_43 = result2[1].detach().clone()

        assert not torch.equal(latents_42, latents_43), (
            "Different args.seed values produced identical output — "
            "the per-rank seed term appears to be missing from the step seed formula"
        )
