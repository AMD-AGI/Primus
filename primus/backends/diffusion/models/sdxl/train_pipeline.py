###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Epsilon-prediction min-SNR training for the official SDXL UNet."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from primus.backends.diffusion.models.sdxl.util import timestep_embedding
from primus.backends.diffusion.utils.pixels import images_to_tensor
from primus.backends.diffusion.utils.sampling import (
    amdb_flow_sigma_grid,
    sample_amdb_sdxl_sigmas,
)

SDXL_LATENT_SCALE = 0.13025
_NUM_TRAIN_TIMESTEPS = 1000


def _scaled_linear_alphas(
    num_steps: int = _NUM_TRAIN_TIMESTEPS, beta_start: float = 0.00085, beta_end: float = 0.012
):
    betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float32) ** 2
    return torch.cumprod(1.0 - betas, dim=0)


class SDXLTrainPipeline:
    def __init__(
        self, *, resolution: int = 1024, snr_gamma: float = 5.0, scale_factor: float = SDXL_LATENT_SCALE
    ):
        self.resolution = int(resolution)
        self.snr_gamma = float(snr_gamma)
        self.scale_factor = float(scale_factor)
        self._alphas = _scaled_linear_alphas()
        self._cached_alphas: dict[torch.device, torch.Tensor] = {}
        self._cached_sigma_grid: dict[torch.device, torch.Tensor] = {}
        self._time_id_row: torch.Tensor | None = None

    def _alphas_on(self, device: torch.device) -> torch.Tensor:
        cached = self._cached_alphas.get(device)
        if cached is None:
            cached = self._alphas.to(device=device, dtype=torch.float32)
            self._cached_alphas[device] = cached
        return cached

    def _sigma_grid_on(self, device: torch.device) -> torch.Tensor:
        cached = self._cached_sigma_grid.get(device)
        if cached is None:
            cached = amdb_flow_sigma_grid(
                shift=3.0, num_train_timesteps=_NUM_TRAIN_TIMESTEPS, zero_terminal=True
            ).to(device)
            self._cached_sigma_grid[device] = cached
        return cached

    def _added_vector(self, pooled: torch.Tensor) -> torch.Tensor:
        row = self._time_id_row
        if row is None or row.device != pooled.device or row.dtype != pooled.dtype:
            row = pooled.new_tensor(
                [self.resolution, self.resolution, 0, 0, self.resolution, self.resolution]
            )
            self._time_id_row = row
        time_ids = row.view(1, 6).expand(pooled.shape[0], 6)
        embeds = [timestep_embedding(time_ids[:, index], 256) for index in range(6)]
        return torch.cat([pooled, torch.cat(embeds, dim=-1)], dim=-1)

    def encode_batch(
        self,
        *,
        autoencoder,
        text_encoder,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Frozen VAE and CLIP encodings for one raw batch."""
        if "images" in batch:
            pixels = images_to_tensor(batch["images"], height=self.resolution, width=self.resolution)
        elif isinstance(batch.get("pixel_values"), torch.Tensor):
            pixels = batch["pixel_values"]
        else:
            raise ValueError("SDXL training batch must contain 'images' or 'pixel_values'.")
        pixels = pixels.to(device=device, dtype=dtype)
        prompts = list(batch.get("prompts") or [])
        if not prompts:
            raise ValueError("SDXL training batch must contain prompts.")

        text_encoder.eval()
        autoencoder.eval()
        with torch.no_grad():
            context, pooled = text_encoder(prompts)
            latents = autoencoder.encode(pixels)
        return {
            "latents": latents.to(device=device, dtype=dtype),
            "context": context.to(device=device, dtype=dtype),
            "pooled": pooled.to(device=device, dtype=dtype),
        }

    def compute_loss(
        self, *, dit, autoencoder, text_encoder, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        if not isinstance(batch.get("latents"), torch.Tensor):
            batch = self.encode_batch(
                autoencoder=autoencoder, text_encoder=text_encoder, batch=batch, device=device, dtype=dtype
            )
        latents = batch["latents"].to(device=device, dtype=dtype)
        context = batch["context"].to(device=device, dtype=dtype)
        pooled = batch["pooled"].to(device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        # Timestep, applied noise, and min-SNR weight share the scaled-linear
        # schedule. Index 0 is clean; index 999 is pure noise.
        timesteps, _ = sample_amdb_sdxl_sigmas(
            latents.shape[0],
            device=device,
            num_train_timesteps=_NUM_TRAIN_TIMESTEPS,
            sigma_grid=self._sigma_grid_on(device),
        )
        alpha = self._alphas_on(device)[timesteps].view(-1, 1, 1, 1)
        noisy = (alpha.sqrt() * latents.float() + (1.0 - alpha).sqrt() * noise.float()).to(latents.dtype)
        added = self._added_vector(pooled).to(dtype=dtype)
        prediction = dit(noisy, timesteps, context=context, y=added)

        snr = (alpha / (1.0 - alpha).clamp(min=1e-8)).view(-1)
        weights = torch.minimum(snr, torch.full_like(snr, self.snr_gamma)) / snr
        loss = F.mse_loss(prediction.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=(1, 2, 3)) * weights
        return {"loss": loss.mean()}
