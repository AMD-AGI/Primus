###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Flow-matching loss for the official HunyuanVideo transformer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from primus.backends.diffusion.models.hunyuan.posemb_layers import (
    get_nd_rotary_pos_embed,
)
from primus.backends.diffusion.utils.sampling import (
    sample_shifted_logit_normal_sigmas,
)


class HunyuanTrainPipeline:
    def __init__(
        self,
        *,
        guidance: float = 1000.0,
        sigma_grid: torch.Tensor | None = None,
        width: int = 960,
        height: int = 544,
        num_frames: int = 9,
    ):
        self.guidance = float(guidance)
        self.sigma_grid = sigma_grid
        self.width = int(width)
        self.height = int(height)
        self.num_frames = int(num_frames)

    def _encode_video(
        self, vae: torch.nn.Module, batch: dict[str, Any], *, device: torch.device, dtype: torch.dtype
    ):
        """Match AMDiffusionBenchmark: causal-3D VAE sample, then ``scaling_factor``."""
        from primus.backends.diffusion.utils.pixels import videos_to_tensor

        if "videos" in batch:
            pixels = videos_to_tensor(
                batch["videos"], num_frames=self.num_frames, height=self.height, width=self.width
            )
        elif isinstance(batch.get("video"), torch.Tensor):
            pixels = batch["video"]
        else:
            raise ValueError(
                "HunyuanVideo training requires batch['latents'] or a raw 'videos'/'video' batch "
                "with the official causal 3D VAE."
            )
        vae_dtype = next(vae.parameters()).dtype
        if next(vae.parameters()).device != device or vae_dtype != dtype:
            vae.to(device=device, dtype=dtype)
            vae_dtype = dtype
        vae.eval()
        pixels = pixels.to(device=device, dtype=vae_dtype)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample()
        scale = float(getattr(vae.config, "scaling_factor", 1.0))
        return (latents * scale).to(dtype=dtype)

    def encode_batch(
        self,
        *,
        text_encoder,
        vae: torch.nn.Module | None,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        """Frozen causal-3D VAE and LLaMA/CLIP encodings for one raw batch."""
        latents = batch.get("latents")
        if not isinstance(latents, torch.Tensor):
            if vae is None:
                raise ValueError("HunyuanVideo raw video training requires the official causal 3D VAE.")
            latents = self._encode_video(vae, batch, device=device, dtype=dtype)
        prompts = list(batch.get("prompts") or [])
        if not prompts:
            raise ValueError("HunyuanVideo training batch must contain prompts.")
        text_encoder.eval()
        with torch.no_grad():
            text_states, text_mask, text_states_2 = text_encoder(prompts)
        return {
            "latents": latents.to(device=device, dtype=dtype),
            "text_states": text_states.to(device=device, dtype=dtype),
            "text_mask": text_mask.to(device=device),
            "text_states_2": text_states_2.to(device=device, dtype=dtype),
        }

    def compute_loss(
        self, *, dit, text_encoder, vae: torch.nn.Module | None = None, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        if not isinstance(batch.get("text_states"), torch.Tensor):
            batch = self.encode_batch(
                text_encoder=text_encoder, vae=vae, batch=batch, device=device, dtype=dtype
            )
        latents = batch["latents"].to(device=device, dtype=dtype)
        text_states = batch["text_states"].to(device=device, dtype=dtype)
        text_mask = batch["text_mask"].to(device=device)
        text_states_2 = batch["text_states_2"].to(device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        sigma = sample_shifted_logit_normal_sigmas(
            latents.shape[0], device=device, sigma_grid=self.sigma_grid
        )
        sigma_view = sigma.view(-1, 1, 1, 1, 1)
        noisy = (sigma_view * noise + (1.0 - sigma_view) * latents).to(latents.dtype)
        timesteps = (sigma * 1000.0).long()
        _, _, _, height, width = noisy.shape
        patch_t, patch_h, patch_w = dit.patch_size
        grid = (noisy.shape[2] // patch_t, height // patch_h, width // patch_w)
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            dit.rope_dim_list, grid, theta=256, use_real=True, theta_rescale_factor=1
        )
        freqs_cos = freqs_cos.to(device=device, dtype=dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dtype)
        guidance = torch.full((latents.shape[0],), self.guidance, device=device, dtype=dtype)
        prediction = dit(
            noisy,
            timesteps,
            text_states=text_states,
            text_mask=text_mask,
            text_states_2=text_states_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance,
            return_dict=False,
        )
        target = noise - latents
        loss = F.mse_loss(prediction.float(), target.float())
        return {"loss": loss}
