###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Native equivalents of AMDiffusionBenchmark's timestep sampling helpers."""

from __future__ import annotations

import torch


def _shift(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)


def amdb_flow_sigma_grid(
    *,
    shift: float = 3.0,
    num_train_timesteps: int = 1000,
) -> torch.Tensor:
    """Return AMDB's ``self.sigmas`` table, i.e. shifted ``scheduler.timesteps / N``.

    FlowMatchEulerDiscreteScheduler's training grid is ``linspace(1, N, N)[::-1] / N``.
    AMDB's ``init_sigmas`` then applies ``--shift``.
    """
    n = num_train_timesteps
    base = torch.arange(n, 0, -1, dtype=torch.float32) / n
    return _shift(base, shift)


def sample_shifted_logit_normal_sigmas(
    batch_size: int,
    *,
    device: torch.device,
    sigma_grid: torch.Tensor | None = None,
    shift: float = 3.0,
    num_train_timesteps: int = 1000,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
) -> torch.Tensor:
    """Match AMDB's ``compute_density_based_timestep_sampling`` without Diffusers.

    ``u = sigmoid(N(logit_mean, logit_std))`` is drawn on the CPU, as Diffusers'
    ``compute_density_for_timestep_sampling`` does, and indexes the FP32 sigma table.
    The returned FP32 sigma is both the noise level and AMDB's normalized timestep.
    """
    if sigma_grid is None:
        sigma_grid = amdb_flow_sigma_grid(shift=shift, num_train_timesteps=num_train_timesteps)
    u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu").sigmoid_()
    indices = (u * len(sigma_grid)).long().clamp_(max=len(sigma_grid) - 1)
    # Pageable source: a non_blocking HIP copy can read the buffer after it is freed.
    return sigma_grid[indices].to(device=device)
