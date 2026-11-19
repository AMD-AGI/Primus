# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Unit tests for the per-DP-rank diffusion training seed."""

import torch

from primus.backends.megatron.training.diffusion.rng import (
    seed_training_rng_per_dp_rank,
)


def _draws(seed, dp_rank, iteration):
    seed_training_rng_per_dp_rank(seed, dp_rank, iteration)
    return torch.rand(8)


class TestSeedTrainingRngPerDpRank:
    def test_fresh_run_keeps_per_rank_seed(self):
        assert seed_training_rng_per_dp_rank(1234, 3) == 1234 + 300
        assert seed_training_rng_per_dp_rank(1234, 3, iteration=0) == 1234 + 300

    def test_same_inputs_give_same_draws(self):
        assert torch.equal(_draws(1234, 1, 500), _draws(1234, 1, 500))

    def test_resume_does_not_replay_step0_draws(self):
        assert not torch.equal(_draws(1234, 0, 0), _draws(1234, 0, 500))

    def test_ranks_and_iterations_do_not_collide(self):
        seeds = {
            seed_training_rng_per_dp_rank(1234, dp_rank, iteration)
            for dp_rank in range(256)
            for iteration in (0, 1, 500, 10_000)
        }
        assert len(seeds) == 256 * 4
