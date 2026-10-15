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

    timesteps = torch.arange(1, N + 1)
    routes_high = _routes_high(scheduler, timesteps)
    assert not routes_high[timesteps < cut].any()
    assert routes_high[timesteps >= cut].all()


def test_edge_inverts_the_static_shift():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    expected = math.ceil(N * BOUNDARY / (5.0 - 4.0 * BOUNDARY))
    assert scheduler.first_timestep_at_noise_level(BOUNDARY) == expected


def test_full_window_is_the_whole_schedule():
    """Timesteps run 1..N; sigma 1 (t = N) is inside a window that ends at 1."""
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    assert scheduler.first_timestep_at_noise_level(0.0) == 1
    assert scheduler.first_timestep_at_noise_level(1.0) == N
    assert scheduler.timestep_bounds((0.0, 1.0)) == (1, N + 1)
    assert scheduler.timestep_bounds((BOUNDARY, 1.0))[1] == N + 1


def _expected_bounds(scheduler, window):
    lo, hi = window
    upper = N + 1 if hi >= 1.0 else scheduler.first_timestep_at_noise_level(hi)
    return scheduler.first_timestep_at_noise_level(lo), upper


@pytest.mark.parametrize("window", [(0.0, BOUNDARY), (BOUNDARY, 1.0), (0.2, 0.6)])
def test_memoized_bounds_match_the_edges(window, monkeypatch):
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    expected = _expected_bounds(scheduler, window)

    assert scheduler.timestep_bounds(window) == expected

    def recomputed(_):
        raise AssertionError("the bounds were recomputed")

    monkeypatch.setattr(scheduler, "first_timestep_at_noise_level", recomputed)
    assert scheduler.timestep_bounds(window) == expected
    assert scheduler.sample_training_timesteps(64, torch.device("cpu"), window).min() >= expected[0]


def test_memoized_bounds_follow_a_shift_change():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    at_five = scheduler.timestep_bounds((0.0, BOUNDARY))

    scheduler.shift = 3.0

    assert scheduler.timestep_bounds((0.0, BOUNDARY)) == (
        1,
        scheduler.first_timestep_at_noise_level(BOUNDARY),
    )
    assert scheduler.timestep_bounds((0.0, BOUNDARY)) != at_five


def test_an_empty_window_is_rejected():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)

    with pytest.raises(ValueError, match="holds no integer timestep"):
        scheduler.timestep_bounds((0.5, 0.5))


def test_validate_rejects_a_window_without_a_timestep():
    """Caught at config time, not on the first training step.

    At shift 5 the sigma gap between t = 999 and t = 1000 is ~2e-4, so a
    window inside it is valid as a range but holds no integer timestep.
    """
    with pytest.raises(ValueError, match="holds no integer timestep"):
        WanConfig.wan2_1_t2v_1_3b(timestep_window_min=0.99995, timestep_window_max=0.99999)


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


def test_unwindowed_draws_cover_one_to_n():
    """Base sigma {1/N, ..., 1}: never a clean input, and pure noise is trained."""
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    torch.manual_seed(0)
    draws = scheduler.sample_training_timesteps(200_000, torch.device("cpu"))

    assert draws.min() == 1
    assert draws.max() == N
    assert scheduler.sigma_from_timestep(draws.max()).item() == 1.0


def test_high_noise_expert_trains_pure_noise():
    scheduler = WanFlowMatchScheduler(num_train_timesteps=N, shift=5.0)
    torch.manual_seed(0)
    high = scheduler.sample_training_timesteps(200_000, torch.device("cpu"), (BOUNDARY, 1.0))

    assert high.max() == N
