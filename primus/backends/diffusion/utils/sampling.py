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
    scheduler_shift: float | None = None,
    zero_terminal: bool = False,
) -> torch.Tensor:
    """Return AMDB's ``self.sigmas`` table, i.e. shifted ``scheduler.timesteps / N``.

    FlowMatchEulerDiscreteScheduler's training grid is ``linspace(1, N, N)[::-1] / N``;
    with ``use_dynamic_shifting=False`` it is pre-shifted by the scheduler's own
    ``shift`` (HunyuanVideo: 7.0) before AMDB's ``init_sigmas`` applies ``--shift``.
    UniPCMultistepScheduler (Wan2.1) starts from ``linspace(0, N - 1, N)[::-1] / N``
    (``zero_terminal=True``), like SDXL's EulerDiscreteScheduler.
    """
    n = num_train_timesteps
    start = n - 1 if zero_terminal else n
    base = torch.arange(start, start - n, -1, dtype=torch.float32) / n
    if scheduler_shift is not None:
        base = _shift(base, scheduler_shift)
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


def sample_amdb_sdxl_sigmas(
    batch_size: int,
    *,
    device: torch.device,
    shift: float = 3.0,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match AMDB SDXL's uniform timestep ids and shifted EulerDiscrete sigma table."""
    n = num_train_timesteps
    grid = amdb_flow_sigma_grid(shift=shift, num_train_timesteps=n, zero_terminal=True).to(device)
    timestep_ids = torch.randint(0, n, (batch_size,), device=device)
    return timestep_ids, grid[timestep_ids]
