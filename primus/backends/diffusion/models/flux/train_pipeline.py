###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from primus.backends.diffusion.models.flux.utils import (
    create_position_encoding_for_latents,
    generate_latent_from_mean_logvar,
    pack_latents,
)


@dataclass
class FluxFlowMatchTrainPipelineConfig:
    autoencoder_scale_factor: float = 0.3611
    autoencoder_shift_factor: float = 0.1159
    guidance: float | None = None


class FluxFlowMatchTrainPipeline:
    """Flow-matching loss for FLUX using precomputed or online encodings."""

    def __init__(self, cfg: FluxFlowMatchTrainPipelineConfig | None = None):
        self.cfg = cfg or FluxFlowMatchTrainPipelineConfig()

    @staticmethod
    def _require_module(module: torch.nn.Module | None, name: str) -> torch.nn.Module:
        if module is None:
            raise ValueError(f"FLUX raw image-text training requires `{name}` to be configured.")
        return module

    @staticmethod
    def _align_module_dtype(module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype) -> None:
        """Move a frozen encoder to the target device/dtype only when it differs.

        `nn.Module.to(dtype=...)` casts floating-point params/buffers only, leaving
        integer buffers (e.g. token position ids) untouched. We inspect the first
        floating-point parameter to avoid re-casting on every step.
        """
        current = next((p for p in module.parameters() if p.is_floating_point()), None)
        if current is None:
            module.to(device=device)
            return
        if current.dtype != dtype or current.device != device:
            module.to(device=device, dtype=dtype)

    def _prepare_precomputed(
        self,
        *,
        batch: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        model_config: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t5_encodings = batch["t5_encodings"].to(device=device, dtype=dtype, non_blocking=True)
        clip_encodings = batch["clip_encodings"].to(device=device, dtype=dtype, non_blocking=True)
        mean = batch["mean"].to(device=device, dtype=dtype, non_blocking=True)
        logvar = batch["logvar"].to(device=device, dtype=dtype, non_blocking=True)

        latents = generate_latent_from_mean_logvar(mean, logvar)
        scale = float(getattr(model_config, "autoencoder_scale_factor", self.cfg.autoencoder_scale_factor))
        shift = float(getattr(model_config, "autoencoder_shift_factor", self.cfg.autoencoder_shift_factor))
        labels = (latents - shift) * scale
        return labels, t5_encodings, clip_encodings

    def _prepare_raw(
        self,
        *,
        batch: dict[str, Any],
        autoencoder: torch.nn.Module | None,
        t5_encoder: torch.nn.Module | None,
        clip_encoder: torch.nn.Module | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ae = self._require_module(autoencoder, "model.config.encoder.autoencoder")
        t5 = self._require_module(t5_encoder, "model.config.encoder.t5_encoder")
        clip = self._require_module(clip_encoder, "model.config.encoder.clip_encoder")
        image = batch["image"].to(device=device, dtype=dtype, non_blocking=True)
        prompts = batch.get("prompts")
        if not isinstance(prompts, list):
            raise ValueError("FLUX raw batch requires `prompts` as list[str].")

        # The frozen encoders are built in bf16 regardless of the training compute
        # dtype; align them with the DiT dtype so running them on `image`/inputs of
        # `dtype` does not raise a dtype-mismatch error (e.g. for fp32 runs).
        self._align_module_dtype(ae, device=device, dtype=dtype)
        self._align_module_dtype(t5, device=device, dtype=dtype)
        self._align_module_dtype(clip, device=device, dtype=dtype)
        ae.eval()
        t5.eval()
        clip.eval()
        with torch.no_grad():
            labels = ae.encode(image).to(device=device, dtype=dtype)
            t5_encodings = t5(prompts).to(device=device, dtype=dtype)
            clip_encodings = clip(prompts).to(device=device, dtype=dtype)
        return labels, t5_encodings, clip_encodings

    def compute_loss(
        self,
        *,
        dit: torch.nn.Module,
        batch: dict[str, Any],
        model_config: Any,
        autoencoder: torch.nn.Module | None = None,
        t5_encoder: torch.nn.Module | None = None,
        clip_encoder: torch.nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        if batch.get("sp_group") is not None:
            raise ValueError("FLUX diffusion training currently requires `parallelism.sp_size: 1`.")

        device = next(dit.parameters()).device
        dtype = next(dit.parameters()).dtype
        if "image" in batch:
            labels, t5_encodings, clip_encodings = self._prepare_raw(
                batch=batch,
                autoencoder=autoencoder,
                t5_encoder=t5_encoder,
                clip_encoder=clip_encoder,
                device=device,
                dtype=dtype,
            )
        else:
            required = ("t5_encodings", "clip_encodings", "mean", "logvar")
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"FLUX precomputed batch is missing required keys: {missing}")
            labels, t5_encodings, clip_encodings = self._prepare_precomputed(
                batch=batch,
                device=device,
                dtype=dtype,
                model_config=model_config,
            )

        bsz = labels.shape[0]
        noise = torch.randn_like(labels)
        timesteps = torch.rand((bsz,), device=device, dtype=dtype)
        sigmas = timesteps.view(-1, 1, 1, 1)
        noisy_latents = (1 - sigmas) * labels + sigmas * noise
        target = noise - labels

        _, _, latent_height, latent_width = noisy_latents.shape
        # Position ids are integer grid indices consumed by RoPE; build them in
        # float32 (independent of the model compute dtype) so that indices remain
        # exactly representable. bf16 only represents integers up to 256 exactly,
        # which would silently corrupt positions for larger latent grids.
        img_ids = create_position_encoding_for_latents(
            bsz,
            latent_height,
            latent_width,
            position_dim=3,
            device=device,
            dtype=torch.float32,
        )
        txt_ids = torch.zeros(bsz, t5_encodings.shape[1], 3, device=device, dtype=torch.float32)
        noisy_latents = pack_latents(noisy_latents)
        target = pack_latents(target)

        guidance_value = getattr(model_config, "guidance", self.cfg.guidance)
        guidance = (
            None
            if guidance_value is None
            else torch.full((bsz,), float(guidance_value), device=device, dtype=dtype)
        )
        pred = dit(
            img=noisy_latents,
            img_ids=img_ids,
            txt=t5_encodings,
            txt_ids=txt_ids,
            y=clip_encodings,
            timesteps=timesteps,
            guidance=guidance,
        )
        loss = F.mse_loss(pred.float(), target.float(), reduction="sum") / target.numel()
        return {"loss": loss, "log_metrics": {"mse": loss.detach()}}
