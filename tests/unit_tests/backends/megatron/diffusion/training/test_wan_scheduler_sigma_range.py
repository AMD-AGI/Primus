# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
``scheduler_sigma_min`` / ``scheduler_sigma_max`` have to bound the noise
training actually applies, not only the inference schedule table.
"""

import pytest
import torch

from primus.backends.megatron.training.diffusion.schedulers import WanFlowMatchScheduler

N = 1000


def test_default_range_leaves_sigma_unclamped():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=3.0)
    t = torch.arange(N)

    base = t.float() / N
    expected = 3.0 * base / (1.0 + 2.0 * base)
    assert torch.equal(scheduler.sigma_from_timestep(t), expected)


def test_configured_range_bounds_noise_and_conditioning():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=3.0, sigma_min=0.1, sigma_max=0.9)
    t = torch.arange(N)

    sigma = scheduler.sigma_from_timestep(t)
    assert sigma.min().item() == pytest.approx(0.1)
    assert sigma.max().item() == pytest.approx(0.9)
    assert scheduler.conditioning_timestep(t).max().item() == pytest.approx(0.9 * N)

    latents = torch.zeros(N, 4)
    noise = torch.ones(N, 4)
    noised = scheduler.add_noise(latents, noise, t)
    assert torch.allclose(noised[:, 0], sigma)


def test_inference_schedule_spans_the_configured_range():
    """``set_timesteps`` shifts its endpoints, so the range must not reach it already shifted."""
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0, sigma_min=0.1, sigma_max=0.9)
    scheduler.set_timesteps(num_inference_steps=10, device="cpu")

    assert scheduler.sigmas[0].item() == pytest.approx(0.9, abs=1e-6)
    assert scheduler.sigmas[-2].item() == pytest.approx(0.1, abs=1e-6)


def test_inference_schedule_spans_the_default_training_schedule():
    """The default range takes the same pre-shift endpoints, so its low end is not shifted twice."""
    training_sigmas = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0).sigmas
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    scheduler.set_timesteps(num_inference_steps=10, device="cpu")

    assert scheduler.sigmas[0].item() == pytest.approx(training_sigmas[0].item(), abs=1e-6)
    assert scheduler.sigmas[-2].item() == pytest.approx(training_sigmas[-1].item(), abs=1e-6)


def test_a_configured_range_with_dynamic_shifting_is_rejected():
    with pytest.raises(ValueError, match="static shift"):
        WanFlowMatchScheduler(num_train_timesteps=N, sigma_min=0.1, sigma_max=0.9, use_dynamic_shifting=True)


def test_noise_is_mixed_at_the_fp32_sigma_the_model_is_conditioned_on():
    """bf16 inputs are mixed in fp32 and only the result is rounded.

    Rounding sigma to bf16 first moves the applied noise level by up to about
    two timestep units for sigma >= 0.5, away from the fp32 conditioning value.
    """
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    t = torch.arange(1, N + 1)
    torch.manual_seed(0)
    latents = torch.randn(N, 64).bfloat16()
    noise = torch.randn(N, 64).bfloat16()

    noised = scheduler.add_noise(latents, noise, t)

    sigma = scheduler.sigma_from_timestep(t).unsqueeze(-1)
    expected = (sigma * noise.float() + (1.0 - sigma) * latents.float()).bfloat16()
    assert noised.dtype == torch.bfloat16
    assert torch.equal(noised, expected)


@pytest.mark.parametrize("sigma_min, sigma_max", [(0.5, 0.5), (0.9, 0.1), (-0.1, 1.0), (0.0, 1.5)])
def test_invalid_range_is_rejected(sigma_min, sigma_max):
    with pytest.raises(ValueError, match="sigma_min"):
        WanFlowMatchScheduler(num_train_timesteps=N, sigma_min=sigma_min, sigma_max=sigma_max)
