###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Flow-matching loss for the official Mochi AsymmDiTJoint."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from primus.backends.diffusion.models.mochi.vae_models import add_fourier_features
from primus.backends.diffusion.models.mochi.vae_stats import STATS
from primus.backends.diffusion.utils.pixels import videos_to_tensor
from primus.backends.diffusion.utils.sampling import sample_shifted_logit_normal_sigmas


class MochiTrainPipeline:
    def __init__(
        self,
        *,
        width: int = 960,
        height: int = 544,
        num_frames: int = 9,
        sigma_grid: torch.Tensor | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)
        self.sigma_grid = sigma_grid

    def encode_batch(
        self,
        *,
        encoder,
        text_encoder,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Frozen VAE and T5 encodings for one raw batch."""
        if "videos" in batch:
            pixels = videos_to_tensor(
                batch["videos"], num_frames=self.num_frames, height=self.height, width=self.width
            )
        elif isinstance(batch.get("video"), torch.Tensor):
            pixels = batch["video"]
        else:
            raise ValueError("Mochi training batch must contain 'videos' or 'video'.")
        prompts = list(batch.get("prompts") or [])
        if not prompts:
            raise ValueError("Mochi training batch must contain prompts.")
        pixels = pixels.to(device=device)
        encoder.eval()
        text_encoder.eval()
        encoder_dtype = next((p.dtype for p in encoder.parameters()), torch.float32)
        with torch.no_grad():
            # Fourier features need FP32 inputs; the encoder then runs in its own
            # dtype, as the Diffusers Mochi VAE does.
            features = add_fourier_features(pixels.float()).to(dtype=encoder_dtype)
            latents = encoder(features).sample().to(dtype=dtype)
            text_hidden, text_mask = text_encoder(prompts)
        return {
            "latents": latents,
            "text_hidden": text_hidden.to(device=device, dtype=dtype),
            "text_mask": text_mask.to(device=device),
        }

    def compute_loss(self, *, dit, encoder, text_encoder, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        if not isinstance(batch.get("latents"), torch.Tensor):
            batch = self.encode_batch(
                encoder=encoder, text_encoder=text_encoder, batch=batch, device=device, dtype=dtype
            )
        vae_latents = batch["latents"].to(device=device, dtype=dtype)
        text_hidden = batch["text_hidden"].to(device=device, dtype=dtype)
        text_mask = batch["text_mask"].to(device=device)

        mean = STATS["mean"].to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = STATS["std"].to(device=device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        latents = ((vae_latents.float() - mean) / std).to(dtype=dtype)
        noise = torch.randn_like(latents)
        sigma = sample_shifted_logit_normal_sigmas(
            latents.shape[0], device=device, sigma_grid=self.sigma_grid
        )
        sigma_view = sigma.view(-1, 1, 1, 1, 1)
        noisy = (sigma_view * noise + (1.0 - sigma_view) * latents).to(latents.dtype)
        sigma = sigma.to(latents.dtype)
        batch_size, _, frames, height, width = latents.shape
        visual_tokens = frames * (height // 2) * (width // 2)
        visual_mask = torch.ones((batch_size, visual_tokens), device=device, dtype=torch.bool)
        valid_tokens = torch.cat((visual_mask, text_mask.bool()), dim=1)
        sequence_lengths = valid_tokens.sum(dim=1, dtype=torch.int32)
        packed_indices = {
            "cu_seqlens_kv": torch.cat(
                (torch.zeros(1, device=device, dtype=torch.int32), sequence_lengths.cumsum(dim=0))
            ),
            "max_seqlen_in_batch_kv": int(sequence_lengths.max().item()),
            "valid_token_indices_kv": valid_tokens.flatten().nonzero(as_tuple=False).flatten(),
        }
        prediction = dit(
            noisy,
            # AsymmDiTJoint embeds `(1 - sigma) * timestep_scale` with
            # timestep_scale=1000. That is the same value AMDiffusionBenchmark
            # passes into Diffusers. Passing `1000 * (1 - sigma)` here applies
            # the conversion twice and the pretrained denoiser misses the target.
            sigma,
            y_feat=[text_hidden],
            y_mask=[text_mask],
            packed_indices=packed_indices,
        )
        target = (latents - noise).float()
        loss = F.mse_loss(prediction.float(), target)
        return {"loss": loss}
