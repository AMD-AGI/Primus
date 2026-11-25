# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Training RNG seeding shared by the Megatron diffusion trainers."""

import torch

# Larger than any 100 * dp_rank offset, so no two (dp_rank, iteration) pairs share a seed.
_ITERATION_SEED_STRIDE = 1_000_003


def seed_training_rng_per_dp_rank(seed: int, dp_rank: int, iteration: int = 0) -> int:
    """Seed the CPU and CUDA generators so each data-parallel rank draws its own noise and timesteps.

    ``iteration`` is the step training resumes from. A fresh run (``iteration == 0``)
    keeps the ``seed + 100 * dp_rank`` seed; a resumed run gets a different one
    instead of replaying the step-0 draws.
    """
    per_rank_seed = seed + 100 * dp_rank + _ITERATION_SEED_STRIDE * iteration
    torch.manual_seed(per_rank_seed)
    torch.cuda.manual_seed(per_rank_seed)
    return per_rank_seed
