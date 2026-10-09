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


@pytest.mark.parametrize("sigma_min, sigma_max", [(0.5, 0.5), (0.9, 0.1), (-0.1, 1.0), (0.0, 1.5)])
def test_invalid_range_is_rejected(sigma_min, sigma_max):
    with pytest.raises(ValueError, match="sigma_min"):
        WanFlowMatchScheduler(num_train_timesteps=N, sigma_min=sigma_min, sigma_max=sigma_max)
