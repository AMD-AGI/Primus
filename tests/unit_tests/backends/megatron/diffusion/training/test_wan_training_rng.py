# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Each data-parallel rank has to draw its own training noise and timesteps.

Megatron seeds every data-parallel rank alike unless ``data_parallel_random_init``
is set, and the WAN forward step draws both from the ambient RNG, so without a
per-rank reseed every rank trains on the same timesteps.
"""

from types import SimpleNamespace

import torch

import primus.backends.megatron.training.diffusion.wan_forward_step as wan_forward_step
import primus.backends.megatron.wan_pretrain_trainer as trainer_module


def test_data_parallel_ranks_draw_different_noise():
    draws = []
    with torch.random.fork_rng():
        for dp_rank in (0, 1):
            trainer_module.seed_training_rng_per_dp_rank(1234, dp_rank)
            draws.append(torch.rand(8))

    assert not torch.equal(draws[0], draws[1])


def test_the_same_rank_draws_the_same_noise():
    with torch.random.fork_rng():
        trainer_module.seed_training_rng_per_dp_rank(1234, 3)
        first = torch.rand(8)
        trainer_module.seed_training_rng_per_dp_rank(1234, 3)
        second = torch.rand(8)

    assert torch.equal(first, second)


def test_the_offset_matches_flux():
    with torch.random.fork_rng():
        assert trainer_module.seed_training_rng_per_dp_rank(1234, 2) == 1434


def test_forward_step_seeds_once_from_the_data_parallel_rank(monkeypatch):
    import megatron.training
    from megatron.core import parallel_state

    seeds = []
    monkeypatch.setattr(
        trainer_module,
        "seed_training_rng_per_dp_rank",
        lambda seed, dp_rank: seeds.append((seed, dp_rank)) or seed,
    )
    monkeypatch.setattr(parallel_state, "get_data_parallel_rank", lambda: 3)
    monkeypatch.setattr(megatron.training, "get_args", lambda: SimpleNamespace(seed=1234))
    monkeypatch.setattr(wan_forward_step, "wan_forward_step_func", lambda *a, **k: (None,) * 6 + ({},))

    trainer = SimpleNamespace(
        _training_rng_seeded=False,
        _forward_step_count=0,
        scheduler=None,
        timestep_window=None,
        boundary_timestep=None,
        loss_weighting="uniform",
        runtime_state=None,
    )
    model = SimpleNamespace(training=True)

    trainer_module.WanPretrainTrainer.forward_step(trainer, iter(()), model)
    trainer_module.WanPretrainTrainer.forward_step(trainer, iter(()), model)

    assert seeds == [(1234, 3)]
    assert trainer._training_rng_seeded is True
