# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Wan-template flow matching scheduler.

The Wan recipe (shared by WAN 2.1 and WAN 2.2) has the same form as
:class:`FlowMatchEulerDiscreteScheduler` but adds three pieces the training
loop needs and the Flux scheduler does not expose:

- ``training_target(latents, noise, timesteps)``: the per-sample target
  velocity, following the maxdiffusion convention ``target = noise - sample``.
- ``training_weight(timesteps)``: a per-timestep loss weight (bell-shaped,
  normalized to mean 1).
- ``sample_training_timesteps(...)``: timestep draw, optionally restricted to
  a window so the two WAN 2.2 experts can be trained separately.

Forward and inference behavior is inherited unchanged (Euler step plus the
``shift * sigma / (1 + (shift - 1) * sigma)`` schedule).

Reference:
    DiffSynth-Studio ``diffsynth/schedulers/flow_match.py`` (Wan template).
    maxdiffusion ``src/maxdiffusion/trainers/wan_trainer.py``.
"""

from typing import Dict, Optional, Tuple, Union

import torch

from primus.backends.megatron.training.diffusion.schedulers.flow_matching import (
    FlowMatchEulerDiscreteScheduler,
)


class WanFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):
    """Wan template scheduler.

    Args:
        num_train_timesteps: Discrete schedule length (Wan default 1000).
        shift: Static schedule shift. Wan default is 5.0.
        sigma_min, sigma_max: Sigma range (Wan default ``[0, 1]``).
        use_dynamic_shifting, base_shift, max_shift, base_image_seq_len,
        max_image_seq_len: Inference / dynamic-shift hooks, forwarded to the
            base scheduler.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
        sigma_min: float = 0.0,
        sigma_max: float = 1.0,
        use_dynamic_shifting: bool = False,
        base_shift: Optional[float] = 0.5,
        max_shift: Optional[float] = 1.15,
        base_image_seq_len: Optional[int] = 256,
        max_image_seq_len: Optional[int] = 4096,
    ):
        super().__init__(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            base_shift=base_shift,
            max_shift=max_shift,
            base_image_seq_len=base_image_seq_len,
            max_image_seq_len=max_image_seq_len,
        )

        if not 0.0 <= sigma_min < sigma_max <= 1.0:
            raise ValueError(f"need 0 <= sigma_min < sigma_max <= 1, got [{sigma_min}, {sigma_max}]")

        # The range bounds the post-shift sigma. The Wan defaults [0, 1] leave
        # everything as the base class built it. ``sigma_range`` is what
        # training reads; the base class's ``sigma_min`` / ``sigma_max`` are the
        # inference endpoints, which ``set_timesteps`` shifts again, so they
        # get the pre-shift values that land on the range.
        self.sigma_range: Optional[Tuple[float, float]] = None
        if (sigma_min, sigma_max) != (0.0, 1.0):
            if use_dynamic_shifting:
                raise ValueError(
                    "a sigma range other than [0, 1] needs the static shift; "
                    "use_dynamic_shifting shifts by a per-call mu that cannot be inverted here"
                )
            self.sigmas = self.sigmas.clamp(min=sigma_min, max=sigma_max)
            self.sigma_min = self._unshift(sigma_min)
            self.sigma_max = self._unshift(sigma_max)
            self.sigma_range = (float(sigma_min), float(sigma_max))

        self.training_weights = self._build_training_weights()
        self._timestep_bounds: Dict[Tuple[float, float, float], Tuple[int, int]] = {}

    def _unshift(self, sigma: float) -> float:
        """Base sigma whose static shift is ``sigma``: ``s / (shift - (shift - 1) * s)``."""
        return float(sigma) / (self.shift - (self.shift - 1.0) * float(sigma))

    def _build_training_weights(self) -> torch.Tensor:
        """Precompute the per-timestep loss weight table.

        DiffSynth builds the table over the (post-shift) training schedule::

            y = exp(-2 * ((t - N/2) / N) ** 2)
            y = y - y.min()
            w = y * (N / y.sum())

        The subtraction makes the weight vanish at the schedule endpoint that
        is furthest from the center, and the final scaling normalizes the table
        to mean 1. Normalization is what keeps the weighted objective on the
        same scale as the unweighted one, so enabling weighting does not shift
        the loss magnitude or require re-tuning the learning rate.
        """
        t = self.timesteps.to(dtype=torch.float32)
        n = float(self.num_train_timesteps)

        y = torch.exp(-2.0 * ((t - n / 2.0) / n) ** 2)
        y = y - y.min()

        return y * (n / y.sum())

    # ------------------------------------------------------------------
    # Training-only helpers
    # ------------------------------------------------------------------

    def sigma_from_timestep(
        self,
        timesteps: Union[int, float, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Map a sampled integer timestep index to its post-shift sigma.

        Uses the same static shift formula as the base scheduler:

            base_sigma = t / num_train_timesteps
            sigma      = shift * base_sigma / (1 + (shift - 1) * base_sigma)

        then clamped to ``[sigma_min, sigma_max]`` when a non-default range
        was configured, so the range reaches :meth:`add_noise` and
        :meth:`conditioning_timestep` alike.

        Returns a 1-D float32 tensor (no broadcasting); callers reshape as
        needed.
        """
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=device)
        t = timesteps.to(device=device or timesteps.device, dtype=torch.float32)
        base_sigma = t / float(self.num_train_timesteps)
        sigma = self.shift * base_sigma / (1.0 + (self.shift - 1.0) * base_sigma)
        if self.sigma_range is not None:
            sigma = sigma.clamp(*self.sigma_range)
        return sigma

    def conditioning_timestep(
        self,
        timesteps: Union[int, float, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Timestep value to condition the transformer on (``sigma * N``).

        This is the inference-aligned timestep. At inference the schedule feeds
        the transformer ``sigma_shifted * num_train_timesteps``, so training
        must condition on the same quantity rather than on the raw sampled
        index ``t``, whose noise level is ``sigma_shifted(t)`` and not ``t / N``.

        Conditioning on the raw index would teach the model a timestep->noise
        mapping the sampler never reproduces, which lets training loss fall
        while sample quality stalls.
        """
        return self.sigma_from_timestep(timesteps, device) * float(self.num_train_timesteps)

    def add_noise(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        """Forward process: ``x_t = sigma * noise + (1 - sigma) * latents``.

        Wan training samples integer timesteps uniformly over
        ``{1, ..., num_train_timesteps}``. Those integers are not present in
        ``self.timesteps`` (which holds the post-shift float schedule used at
        inference), so :meth:`scale_noise`'s schedule lookup cannot be reused.
        Sigma is instead computed directly from the timestep, which matches
        DiffSynth's ``FlowMatchScheduler.add_noise`` and works for any ``shift``
        at any sampled timestep rather than only those that happen to coincide
        with the inference schedule.

        The mix runs in fp32 and only the result takes the latents' dtype: the
        model is conditioned on the fp32 sigma, and a bf16 sigma would put the
        actual noise level up to about two timestep units away from it.
        """
        sigmas = self.sigma_from_timestep(timesteps, device=latents.device)
        while sigmas.dim() < latents.dim():
            sigmas = sigmas.unsqueeze(-1)

        return (sigmas * noise.float() + (1.0 - sigmas) * latents.float()).to(latents.dtype)

    def training_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Velocity target: ``noise - latents``.

        Matches maxdiffusion's ``wan_trainer.step_optimizer``. The
        ``timesteps`` argument is accepted for interface uniformity; the Wan
        target does not depend on it.
        """
        del timesteps  # Wan's target is timestep-independent.
        return noise - latents

    def training_weight(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Look up the per-sample loss weight for ``timesteps``.

        ``timesteps`` are the conditioning timesteps (``sigma * N``), matching
        the schedule the weight table is built over. Each is resolved to the
        nearest schedule entry, as DiffSynth does.

        Returns a float32 ``[B]`` tensor with values in roughly ``[0, 1.6]``
        and mean 1 over the full schedule.
        """
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps)

        t = timesteps.to(dtype=torch.float32).flatten()
        schedule = self.timesteps.to(device=t.device, dtype=torch.float32)
        weights = self.training_weights.to(device=t.device)

        nearest = (t.unsqueeze(1) - schedule.unsqueeze(0)).abs().argmin(dim=1)

        return weights[nearest]

    # ------------------------------------------------------------------
    # Timestep sampling helper
    # ------------------------------------------------------------------

    def first_timestep_at_noise_level(self, noise_level: float) -> int:
        """Smallest timestep in ``{1, ..., N}`` whose post-shift sigma is ``>= noise_level``.

        Analytically ``max(1, ceil(N * s / (shift - (shift - 1) * s)))``, the
        inverse of the static shift, or ``N + 1`` when no timestep reaches the
        level. It is evaluated through :meth:`conditioning_timestep` and
        compared the way the WAN 2.2 router compares, ``timestep >=
        boundary_ratio * N``, so a window edge at ``boundary_ratio`` splits the
        integer timesteps exactly where routing does.
        """
        n = self.num_train_timesteps
        conditioned = self.conditioning_timestep(torch.arange(1, n + 1, dtype=torch.long))
        edge = torch.tensor(float(noise_level) * float(n), dtype=conditioned.dtype)
        return 1 + int((conditioned < edge).sum())

    def timestep_bounds(self, timestep_window: Tuple[float, float]) -> Tuple[int, int]:
        """``[lo, hi)``: the integer timesteps whose post-shift sigma is in the window.

        The window is half-open like the router's split, except that an upper
        edge of 1 is closed: ``t = N`` is pure noise, the first inference step,
        and belongs to the full range and to the high-noise expert.

        Memoized, since each edge evaluates the whole schedule and the window
        is fixed for a run while this is asked for every step. ``shift`` is a
        plain attribute that a caller can reassign, so it is part of the key.
        """
        key = (float(timestep_window[0]), float(timestep_window[1]), float(self.shift))
        bounds = self._timestep_bounds.get(key)
        if bounds is None:
            lo = self.first_timestep_at_noise_level(key[0])
            hi = self.num_train_timesteps + 1 if key[1] >= 1.0 else self.first_timestep_at_noise_level(key[1])
            if hi <= lo:
                raise ValueError(
                    f"timestep_window {tuple(timestep_window)} holds no integer timestep "
                    f"at shift={self.shift}, num_train_timesteps={self.num_train_timesteps}"
                )
            bounds = self._timestep_bounds[key] = (lo, hi)
        return bounds

    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        timestep_window: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Sample ``[B]`` integer timesteps for training.

        Without ``timestep_window``: uniform over ``{1, ..., num_train_timesteps}``,
        i.e. base sigma ``{1/N, ..., 1}`` as in DiffSynth and the loss-weight
        table. Sigma 0 would be a clean input with an unpredictable target, and
        sigma 1 is the first inference step.

        With ``timestep_window=(lo, hi)``: uniform over the integer timesteps
        whose post-shift sigma lies in ``[lo, hi)``. The DiffSynth-style
        per-expert recipe uses this to train ``transformer`` and
        ``transformer_2`` in separate jobs.

        The window is a noise level, the axis ``boundary_ratio`` and the WAN
        2.2 router work on, not a fraction of the raw index: under shift 5 the
        raw index ``0.875 * N`` already has sigma ~0.972, so windowing the raw
        index would train the low-noise expert on a band that routing sends to
        the high-noise one.
        """
        if timestep_window is None:
            lo, hi = 1, self.num_train_timesteps + 1
        else:
            lo, hi = self.timestep_bounds(timestep_window)

        return torch.randint(
            low=lo,
            high=hi,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )


__all__ = ["WanFlowMatchScheduler"]
