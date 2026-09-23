###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""WorldPlay autoregressive flow-matching SFT recipe."""

from __future__ import annotations

from typing import Any

import torch

from .distributed import set_sp_group


class WorldPlayARTrainPipeline:
    def __init__(
        self,
        *,
        train_time_shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
    ):
        self.train_time_shift = float(train_time_shift)
        self.logit_mean = float(logit_mean)
        self.logit_std = float(logit_std)

    def _sample_sigmas(
        self,
        latents: torch.Tensor,
        memory_sample: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, frames, _, _ = latents.shape
        if frames % 4:
            raise ValueError(f"WorldPlay latent frame count must be divisible by 4, got {frames}")
        chunks = frames // 4
        logits = torch.randn(
            (batch, chunks),
            device=latents.device,
            dtype=torch.float32,
        )
        logits = logits * self.logit_std + self.logit_mean
        u = torch.sigmoid(logits)
        shifted = self.train_time_shift * u / (
            1.0 + (self.train_time_shift - 1.0) * u
        )
        # Native WorldPlay replaces historical-memory chunk timesteps with
        # scheduler indices in [500, 985), keeping only the final prediction
        # chunk on the ordinary logit-normal draw.
        for batch_index in torch.nonzero(memory_sample, as_tuple=False).flatten():
            history_index = torch.randint(
                500,
                985,
                (max(chunks - 1, 0),),
                device=latents.device,
            )
            base_sigma = 1.0 - history_index.float() / 1000.0
            shifted[batch_index, :-1] = (
                self.train_time_shift
                * base_sigma
                / (1.0 + (self.train_time_shift - 1.0) * base_sigma)
            )
        per_frame = shifted.repeat_interleave(4, dim=1)
        sigma = per_frame[:, None, :, None, None].to(dtype=latents.dtype)
        timestep = (per_frame * 1000.0).reshape(-1).to(dtype=latents.dtype)
        return sigma, timestep

    @staticmethod
    def _condition_latents(image_cond: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        condition = image_cond.repeat(1, 1, latents.shape[2], 1, 1)
        condition[:, :, 1:] = 0
        mask = torch.zeros(
            (latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )
        mask[:, :, 0] = 1
        return torch.cat((condition, mask), dim=1)

    def compute_loss(
        self,
        *,
        dit: torch.nn.Module,
        batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        set_sp_group(batch.get("sp_group"))
        latents = batch["latent"]
        noise = torch.randn_like(latents)
        memory_sample = batch["memory_sample"].bool().reshape(-1)
        sigmas, timesteps = self._sample_sigmas(latents, memory_sample)
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        hidden_states = torch.cat(
            (noisy, self._condition_latents(batch["image_cond"], latents)),
            dim=1,
        )

        prediction = dit(
            hidden_states=hidden_states,
            timestep=timesteps,
            timestep_txt=torch.zeros(
                (1,), device=latents.device, dtype=latents.dtype
            ),
            text_states=batch["prompt_embed"],
            text_states_2=None,
            encoder_attention_mask=batch["prompt_mask"],
            timestep_r=None,
            vision_states=batch["vision_states"],
            mask_type="i2v",
            guidance=None,
            extra_kwargs={
                "byt5_text_states": batch["byt5_text_states"],
                "byt5_text_mask": batch["byt5_text_mask"],
            },
            viewmats=batch["w2c"],
            Ks=batch["intrinsic"],
            action=batch["action"].reshape(-1).to(dtype=latents.dtype),
            return_dict=False,
        )[0]

        target = noise - latents
        loss_mask = batch["i2v_mask"].clone()
        loss_mask[memory_sample, :, :-4] = 0
        squared_error = (prediction.float() - target.float()).square() * loss_mask.float()
        loss = squared_error.sum() / loss_mask.sum().clamp_min(1)
        return {
            "loss": loss,
            "log_metrics": {
                "worldplay/memory_sample_rate": memory_sample.float().mean().detach(),
                "worldplay/sigma_mean": sigmas.float().mean().detach(),
            },
        }
