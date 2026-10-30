###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .components import WanComponents


@dataclass
class WanFlowMatchTrainPipelineConfig:
    """
    Minimal training-pipeline config.

    We intentionally keep this small and derive most behavior from the model config
    to maximize YAML reuse.
    """

    # For Wan VAEs, temporal downsample is typically (4, 1): 4n+1 frames.
    time_division_factor: int = 4
    time_division_remainder: int = 1
    # AMDiffusionBenchmark logit-normal sampling over its shifted sigma table (cached
    # latents always use it); None uses the FlowMatchEuler grid.
    amdb_sampling: bool = False
    sigma_grid: Optional[torch.Tensor] = None


class WanFlowMatchTrainPipeline:
    """
    Flow-Matching training pipeline for Wan-style DiT models.

    Contract:
      compute_loss(components, batch, scheduler) -> {"loss": Tensor, ...}
    """

    def __init__(self, cfg: Optional[WanFlowMatchTrainPipelineConfig] = None):
        self.cfg = cfg or WanFlowMatchTrainPipelineConfig()

    @staticmethod
    def _encode_prompt(
        text_encoder: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        # Match existing wan_new behavior: zero-out padded embeddings explicitly.
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = text_encoder(input_ids, attention_mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        return prompt_emb

    @staticmethod
    def _get_seed_from_env_or_batch(batch: Dict[str, Any]) -> Optional[int]:
        seed = batch.get("seed", None)
        if seed is None:
            return None
        if isinstance(seed, torch.Tensor):
            flat_seed = seed.detach().reshape(-1).cpu()
            if flat_seed.numel() == 0:
                return None
            if flat_seed.numel() > 1 and not bool((flat_seed == flat_seed[0]).all()):
                raise ValueError(
                    "Per-sample dataset seeds are not supported; got different seeds in one batch."
                )
            seed = flat_seed[0].item()
        elif isinstance(seed, (list, tuple)):
            if not seed:
                return None
            if len(seed) > 1 and any(value != seed[0] for value in seed):
                raise ValueError(
                    "Per-sample dataset seeds are not supported; got different seeds in one batch."
                )
            seed = seed[0]
        return int(seed)

    @staticmethod
    def _randn_like_on_cpu_then_to(
        x: torch.Tensor,
        *,
        seed: Optional[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
        noise = torch.randn(x.shape, generator=generator, device="cpu", dtype=torch.float32)
        return noise.to(device=device, dtype=dtype)

    @staticmethod
    def prepare_host_batch(batch: Dict[str, Any], model_config: Any) -> Dict[str, Any]:
        if batch.get("video") is not None or "videos" not in batch:
            return batch
        from primus.backends.diffusion.utils.pixels import decode_host_pixels

        return decode_host_pixels(
            batch,
            height=int(getattr(model_config, "height", 544)),
            width=int(getattr(model_config, "width", 960)),
            num_frames=int(getattr(model_config, "num_frames", 9)),
        )

    def _maybe_prepare_raw_batch(self, components: WanComponents, batch: Dict[str, Any], model_config: Any):
        raw = "videos" in batch or batch.get("prompts") is not None
        if not raw or isinstance(batch.get("latents"), torch.Tensor):
            return batch
        prepared = self.prepare_host_batch(batch, model_config)
        if prepared.get("input_ids") is None:
            tokenizer = components.tokenizer
            prompts = list(batch.get("prompts") or [])
            if tokenizer is None:
                raise ValueError("Wan prompt batches require model.encoder.tokenizer.")
            text_len = int(getattr(model_config, "text_len", 512))
            tokens = tokenizer(
                prompts,
                padding="max_length",
                truncation=True,
                max_length=text_len,
                return_tensors="pt",
            )
            prepared["input_ids"] = tokens["input_ids"]
            prepared["attention_mask"] = tokens["attention_mask"]
        return prepared

    def _i2v_condition(self, components: WanComponents, video: torch.Tensor, latents: torch.Tensor):
        if components.image_encoder is None:
            raise ValueError("Wan I2V requires model.encoder.image_encoder (official CLIP checkpoint).")
        condition_video = video.clone()
        condition_video[:, :, 1:] = 0
        with torch.no_grad():
            condition_latents = self._vae_encode(components.vae, condition_video).to(
                device=latents.device, dtype=latents.dtype
            )
            clip_fea = components.image_encoder.visual(video[:, :, 0])
        batch, _, frames, height, width = condition_latents.shape
        pixel_frames = video.shape[2]
        mask = torch.ones(batch, pixel_frames, height, width, device=latents.device, dtype=latents.dtype)
        mask[:, 1:] = 0
        mask = torch.cat([mask[:, :1].repeat(1, 4, 1, 1), mask[:, 1:]], dim=1)
        mask = mask.view(batch, mask.shape[1] // 4, 4, height, width).transpose(1, 2)
        if mask.shape[2] != frames:
            raise ValueError(
                f"Wan I2V mask has {mask.shape[2]} latent frames but the VAE produced {frames}. "
                "Use a frame count of 4n+1."
            )
        condition = torch.cat([mask, condition_latents], dim=1)
        return [condition[index] for index in range(batch)], clip_fea

    @staticmethod
    def _vae_encode(vae: torch.nn.Module, videos_bcthw: torch.Tensor) -> torch.Tensor:
        # Wan VAEs in this repo use list-of-tensors interface: List[[C,T,H,W]] -> List[[C,F,H',W']]
        videos_list = [videos_bcthw[i] for i in range(videos_bcthw.shape[0])]
        latents_list = vae.encode(videos_list)
        return torch.stack(latents_list)

    @staticmethod
    def _amdb_training_latents(vae: Any, latents: torch.Tensor) -> torch.Tensor:
        """Match AMDiffusionBenchmark's training latent.

        The official Wan VAE returns `(mu - mean) / std`. AMDB's Diffusers VAE
        sample is the raw posterior, and `prepare_latents` normalizes only the
        I2V condition. Undo the scale on the sample that enters the flow-matching
        loss so the MSE is on the same tensor AMDB trains against.
        """
        mean = getattr(vae, "mean", None)
        std = getattr(vae, "std", None)
        if mean is None or std is None:
            return latents
        mean = mean.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        std = std.to(device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        return latents * std + mean

    @staticmethod
    def _get_dit_patch_size(model_config: Any) -> tuple[int, int, int]:
        patch_size = tuple(getattr(model_config, "dit_patch_size", (1, 2, 2)))
        if len(patch_size) != 3:
            raise ValueError(f"Expected 3D dit_patch_size, got {patch_size!r}")
        return patch_size

    @staticmethod
    def _pad_latents_for_dit(
        latents: torch.Tensor, *, patch_size: tuple[int, int, int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, _, height, width = latents.shape
        _, patch_h, patch_w = patch_size
        pad_h = (-height) % patch_h
        pad_w = (-width) % patch_w
        if pad_h == 0 and pad_w == 0:
            return latents, (height, width)

        # Pad only the latent-space bottom/right edges so DiT patchify works on any
        # VAE output shape. We crop predictions back before computing loss.
        latents = F.pad(latents, (0, pad_w, 0, pad_h))
        return latents, (height, width)

    @staticmethod
    def _crop_latents(latents: torch.Tensor, *, spatial_size: tuple[int, int]) -> torch.Tensor:
        height, width = spatial_size
        return latents[..., :height, :width]

    @staticmethod
    def _select_timestep(scheduler: Any, device: torch.device) -> torch.Tensor:
        """Sample one training timestep uniformly."""
        # Draw the index on CPU. ``scheduler.timesteps`` lives on CPU, and a GPU
        # index would sync the device on every micro-batch inside the timed step.
        timestep_id = torch.randint(0, int(scheduler.num_train_timesteps), (1,))
        timestep = scheduler.timesteps[timestep_id].float()
        return timestep.to(device=device)

    @staticmethod
    def _maybe_expand_separated_timestep(
        *,
        timestep: torch.Tensor,
        x_list: list[torch.Tensor],
        patch_size: tuple[int, int, int],
        enabled: bool,
    ) -> torch.Tensor:
        """
        Match `wan_new.forward_dit` separated-timestep behavior (only used when enabled).
        For each sample, build per-token timestep [L] and set first-frame patches to 0.
        Returns:
          - t: [B] if not enabled
          - t: [B, L] if enabled
        """
        if not enabled:
            if timestep.ndim == 0:
                return timestep.unsqueeze(0).repeat(len(x_list))
            if timestep.ndim == 1 and timestep.numel() == 1 and len(x_list) > 1:
                return timestep.repeat(len(x_list))
            if timestep.ndim == 1 and timestep.shape[0] != len(x_list):
                return timestep.repeat(len(x_list))
            return timestep

        d_f, d_h, d_w = patch_size
        if timestep.ndim == 0:
            timestep_b = timestep.unsqueeze(0).repeat(len(x_list))
        elif timestep.ndim == 1 and timestep.shape[0] != len(x_list):
            timestep_b = timestep.repeat(len(x_list))
        else:
            timestep_b = timestep

        t_expand_list: list[torch.Tensor] = []
        for i, x in enumerate(x_list):
            # x: [C, F, H, W] in latent space
            f, h, w = x.shape[1], x.shape[2], x.shape[3]
            seq_len = (f // d_f) * (h // d_h) * (w // d_w)
            t_seq = torch.full(
                (seq_len,),
                timestep_b[i],
                device=timestep_b.device,
                dtype=timestep_b.dtype,
            )
            spatial_patches = (h // d_h) * (w // d_w)
            t_seq[:spatial_patches] = 0
            t_expand_list.append(t_seq)
        return torch.stack(t_expand_list)

    def encode_batch(
        self,
        *,
        components: WanComponents,
        batch: Dict[str, Any],
        model_config: Any,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Encode a raw I2V batch into the cached tensors consumed by `compute_loss`.

        Returns None when the cached path cannot reproduce the online path
        (non-AMDB sampling or non-I2V models), so callers keep encoding online.
        """
        if not self.cfg.amdb_sampling or getattr(model_config, "model_type", None) != "i2v":
            return None
        batch = self._maybe_prepare_raw_batch(components, batch, model_config)
        device = next(components.dit.parameters()).device
        dtype = next(components.dit.parameters()).dtype
        video = batch["video"].to(device=device, dtype=dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device=device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device=device, non_blocking=True)
        components.text_encoder.eval()
        with torch.no_grad():
            latents = self._amdb_training_latents(
                components.vae, self._vae_encode(components.vae, video).to(device=device, dtype=dtype)
            )
            context = self._encode_prompt(components.text_encoder, input_ids, attention_mask)
            y_list, clip_fea = self._i2v_condition(components, video, latents)
        return {
            "latents": latents,
            "context": context.to(dtype),
            "y": torch.stack(y_list).to(dtype),
            "clip_fea": clip_fea.to(dtype),
        }

    def compute_loss(
        self,
        *,
        components: WanComponents,
        batch: Dict[str, Any],
        scheduler: Any,
        model_config: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        batch requirements (from current dataset/processor):
          - video: Tensor [B, C, T, H, W]
          - input_ids: Tensor [B, L]
          - attention_mask: Tensor [B, L]
        """
        batch = self._maybe_prepare_raw_batch(components, batch, model_config)
        if isinstance(batch.get("latents"), torch.Tensor):
            device = next(components.dit.parameters()).device
            dtype = next(components.dit.parameters()).dtype
            input_latents = batch["latents"].to(device=device, dtype=dtype)
            context = batch["context"].to(device=device, dtype=dtype)
            cached_y = batch["y"].to(device=device, dtype=dtype)
            clip_fea = batch["clip_fea"].to(device=device, dtype=dtype)
            patch_size = self._get_dit_patch_size(model_config)
            input_latents, original_latent_spatial_size = self._pad_latents_for_dit(
                input_latents, patch_size=patch_size
            )
            from primus.backends.diffusion.utils.sampling import (
                sample_shifted_logit_normal_sigmas,
            )

            timestep = sample_shifted_logit_normal_sigmas(
                input_latents.shape[0], device=device, sigma_grid=self.cfg.sigma_grid
            )
            noise = torch.randn_like(input_latents)
            sigma = timestep.view(-1, 1, 1, 1, 1)
            target = noise - input_latents
            noisy_latents = (sigma * noise + (1.0 - sigma) * input_latents).to(input_latents.dtype)
            timestep = timestep.to(dtype)
            x_list = [noisy_latents[i] for i in range(noisy_latents.shape[0])]
            context_list = [context[i] for i in range(context.shape[0])]
            d_f, d_h, d_w = patch_size
            max_seq_len = 0
            for sample in x_list:
                seq_len = (sample.shape[1] // d_f) * (sample.shape[2] // d_h) * (sample.shape[3] // d_w)
                max_seq_len = max(max_seq_len, seq_len)
            y_list = [cached_y[i] for i in range(cached_y.shape[0])]
            noise_pred_list = components.dit(
                x=x_list,
                t=timestep,
                context=context_list,
                seq_len=max_seq_len,
                y=y_list,
                clip_fea=clip_fea,
                sp_group=batch.get("sp_group", None),
            )
            noise_pred = torch.stack(noise_pred_list)
            noise_pred = self._crop_latents(noise_pred, spatial_size=original_latent_spatial_size)
            target = self._crop_latents(target, spatial_size=original_latent_spatial_size)
            loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
            return {"loss": loss}
        video = batch.get("video")
        if video is None:
            raise ValueError("Batch must contain 'video'")
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError(
                f"Expected batch['video'] as 5D tensor [B,C,T,H,W], got {type(video)} shape={getattr(video, 'shape', None)}"
            )

        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        if input_ids is None or attention_mask is None:
            raise ValueError("Batch must contain 'input_ids' and 'attention_mask'")

        device = next(components.dit.parameters()).device
        dtype = next(components.dit.parameters()).dtype

        video = video.to(device=device, dtype=dtype, non_blocking=True)
        input_ids = input_ids.to(device=device, non_blocking=True)
        attention_mask = attention_mask.to(device=device, non_blocking=True)

        # 1) Encode to latents
        # Keep VAE/text encoder in eval() by default (consistent with wan_new practice).
        components.text_encoder.eval()
        try:
            components.vae.eval()
        except AttributeError:
            # Some VAE compatibility wrappers may not expose eval(); real eval
            # failures should still surface.
            pass

        with torch.no_grad():
            input_latents = self._vae_encode(components.vae, video).to(device=device, dtype=dtype)
            if self.cfg.amdb_sampling:
                input_latents = self._amdb_training_latents(components.vae, input_latents)
        patch_size = self._get_dit_patch_size(model_config)
        input_latents, original_latent_spatial_size = self._pad_latents_for_dit(
            input_latents, patch_size=patch_size
        )

        if self.cfg.amdb_sampling:
            from primus.backends.diffusion.utils.sampling import (
                sample_shifted_logit_normal_sigmas,
            )

            # 2) AMDiffusionBenchmark flow-matching noise schedule.
            timestep = sample_shifted_logit_normal_sigmas(
                input_latents.shape[0], device=device, sigma_grid=self.cfg.sigma_grid
            )
            noise = torch.randn_like(input_latents)
            sigma = timestep.view(-1, 1, 1, 1, 1)

            # 3) AMDiffusionBenchmark velocity target and noise injection.
            target = noise - input_latents
            noisy_latents = (sigma * noise + (1.0 - sigma) * input_latents).to(input_latents.dtype)
            timestep = timestep.to(dtype)
        else:
            # 2) Sample timestep + noise (match `wan_new`: noise on CPU, timestep from scheduler.timesteps)
            timestep = self._select_timestep(scheduler, device=device)  # [1] (float)
            seed = self._get_seed_from_env_or_batch(batch)
            noise = self._randn_like_on_cpu_then_to(input_latents, seed=seed, dtype=dtype, device=device)

            # 3) Diffusion target + noise injection (match `wan_new`)
            # Note: in this repo's FlowMatchScheduler, `training_target(sample, noise, t)` uses `noise - sample`.
            # `wan_new` passes `input_latents` (not noisy latents).
            target = scheduler.training_target(input_latents, noise, timestep)
            noisy_latents = scheduler.add_noise(input_latents, noise, timestep=timestep)

        # 4) Text embeddings
        with torch.no_grad():
            context = self._encode_prompt(components.text_encoder, input_ids, attention_mask)

        # 5) DiT forward (official interface: List[Tensor] per-sample)
        x_list = [noisy_latents[i] for i in range(noisy_latents.shape[0])]
        context_list = [context[i] for i in range(context.shape[0])]

        # seq_len matches DiT Conv3d patchification over [F, H, W].
        d_f, d_h, d_w = patch_size
        max_seq_len = 0
        for x in x_list:
            seq_len = (x.shape[1] // d_f) * (x.shape[2] // d_h) * (x.shape[3] // d_w)
            max_seq_len = max(max_seq_len, seq_len)

        separated = bool(getattr(model_config, "separated_timestep", False))
        fuse_flag = bool(getattr(model_config, "fuse_vae_embedding_in_latents", False))
        t = self._maybe_expand_separated_timestep(
            timestep=timestep,
            x_list=x_list,
            patch_size=(d_f, d_h, d_w),
            enabled=bool(separated and fuse_flag),
        )

        sp_group = batch.get("sp_group", None)
        y_list = None
        clip_fea = None
        if getattr(model_config, "model_type", None) == "i2v":
            y_list, clip_fea = self._i2v_condition(components, video, input_latents)
        noise_pred_list = components.dit(
            x=x_list,
            t=t,
            context=context_list,
            seq_len=max_seq_len,
            y=y_list,
            clip_fea=clip_fea,
            sp_group=sp_group,
        )
        noise_pred = torch.stack(noise_pred_list)
        noise_pred = self._crop_latents(noise_pred, spatial_size=original_latent_spatial_size)
        target = self._crop_latents(target, spatial_size=original_latent_spatial_size)

        # 6) Loss
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
        if not self.cfg.amdb_sampling:
            # `wan_new` uses scalar `timestep` for weighting (not expanded per-token)
            loss = loss * scheduler.training_weight(timestep)
        return {"loss": loss}
