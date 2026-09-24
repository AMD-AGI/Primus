# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN 2.2 per-expert training must cover exactly the samples routing gives it.

The router sends a sample to the high-noise expert when its conditioning
timestep, ``sigma * N`` after the static shift, is at least
``boundary_ratio * N``. A per-expert job's window is derived from the same
``boundary_ratio``, so it has to be read on that noise-level axis too; read as a
fraction of the raw index it hands the low-noise expert everything up to sigma
~0.972 at shift 5 and leaves the high-noise expert only the band above it.
"""

import math

import pytest
import torch

from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig
from primus.backends.megatron.training.diffusion.schedulers import WanFlowMatchScheduler

BOUNDARY = 0.875
N = 1000


def _routes_high(scheduler, timesteps):
    conditioned = scheduler.conditioning_timestep(timesteps)
    return conditioned >= torch.full_like(conditioned, float(BOUNDARY * N))


@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0])
def test_stage_windows_match_routing(shift):
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=shift)

    low = scheduler.sample_training_timesteps(8192, torch.device("cpu"), (0.0, BOUNDARY))
    high = scheduler.sample_training_timesteps(8192, torch.device("cpu"), (BOUNDARY, 1.0))

    assert not _routes_high(scheduler, low).any()
    assert _routes_high(scheduler, high).all()


@pytest.mark.parametrize("shift", [1.0, 3.0, 5.0])
def test_window_edge_is_the_routing_cut(shift):
    """The two windows meet at the first timestep routing sends high."""
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=shift)
    cut = scheduler.first_timestep_at_noise_level(BOUNDARY)

    routes_high = _routes_high(scheduler, torch.arange(N))
    assert not routes_high[:cut].any()
    assert routes_high[cut:].all()


def test_edge_inverts_the_static_shift():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    expected = math.ceil(N * BOUNDARY / (5.0 - 4.0 * BOUNDARY))
    assert scheduler.first_timestep_at_noise_level(BOUNDARY) == expected


def test_full_window_is_the_whole_schedule():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    assert scheduler.first_timestep_at_noise_level(0.0) == 0
    assert scheduler.first_timestep_at_noise_level(1.0) == N


@pytest.mark.parametrize("stage", ["high_noise", "low_noise"])
def test_per_expert_stage_rejects_dual_transformer(stage):
    """Both experts would load the stage's weights from backbone_subfolder."""
    with pytest.raises(ValueError, match="num_transformers: 1"):
        WanConfig.wan2_2_t2v_a14b(stage=stage).validate()


@pytest.mark.parametrize(
    "stage, window, subfolder",
    [
        ("high_noise", (BOUNDARY, 1.0), "transformer"),
        ("low_noise", (0.0, BOUNDARY), "transformer_2"),
    ],
)
def test_per_expert_stage_on_one_transformer(stage, window, subfolder):
    config = WanConfig.wan2_2_t2v_a14b(stage=stage, num_transformers=1)
    config.validate()
    assert config.timestep_window == window
    assert config.backbone_subfolder == subfolder
