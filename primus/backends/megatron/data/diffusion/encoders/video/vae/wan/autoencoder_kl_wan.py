# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Wan VAE (AutoencoderKLWan) wrapper.

Provides a Primus-shaped wrapper around the diffusers ``AutoencoderKLWan``
3D video VAE used by both WAN 2.1 and WAN 2.2. The wrapper handles:

- Loading from a HuggingFace repo / local path (matching the standard Wan
  layout, e.g. ``Wan-AI/Wan2.1-T2V-14B-Diffusers/vae``).
- Encoding pixel videos ``(B, C, T, H, W)`` to latents
  ``(B, latent_channels, T', H', W')``.
- Decoding latents back to videos.

The diffusers ``AutoencoderKLWan`` exposes per-channel mean / std for the
latent normalization, so we do not need the Flux-style global
``scale_factor`` / ``shift_factor``. Per-channel normalization is applied by
the diffusers model itself; this wrapper just forwards encode/decode calls.

Reference:
    https://huggingface.co/docs/diffusers/en/api/models/autoencoder_kl_wan
"""

import logging
from typing import Optional

import torch

try:
    from diffusers import AutoencoderKLWan as DiffusersAutoencoderKLWan
except ImportError:
    DiffusersAutoencoderKLWan = None

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseVAE,
    get_torch_dtype,
    load_pretrained_with_subfolder_fallback,
)
from primus.backends.megatron.data.diffusion.encoders.config import WanVAEConfig

logger = logging.getLogger(__name__)


class AutoencoderKLWan(BaseVAE):
    """
    Wan 3D video VAE for WAN 2.1 / WAN 2.2 diffusion models.

    Inputs / outputs (encode):
        videos: (B, C, T, H, W) pixel videos in range [-1, 1]
        latents: (B, latent_channels, T_lat, H_lat, W_lat)

    Inputs / outputs (decode):
        latents: (B, latent_channels, T_lat, H_lat, W_lat)
        videos: (B, C, T, H, W) pixel videos

    Spatial downsampling is 8x (Wan 2.1) and temporal downsampling is 4x.
    """

    def __init__(self, config: WanVAEConfig):
        """
        Initialize AutoencoderKLWan encoder.

        Args:
            config: WanVAEConfig with model_path, subfolder, precision, etc.
        """
        super().__init__(config)

        if DiffusersAutoencoderKLWan is None:
            raise ImportError(
                "diffusers library is required for AutoencoderKLWan. "
                "Install with: pip install -U diffusers"
            )

        self.temporal_downsample_factor = getattr(config, "temporal_downsample_factor", 4)
        self.vae = None  # Will be loaded in from_pretrained

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[WanVAEConfig] = None,
        subfolder: Optional[str] = None,
    ) -> "AutoencoderKLWan":
        """
        Load AutoencoderKLWan from pretrained weights.

        Args:
            model_path: Path to pretrained model (local path or HuggingFace repo)
            config: Optional WanVAEConfig. If None, uses defaults.
            subfolder: Subfolder for VAE weights. Priority: param > config.subfolder

        Returns:
            Loaded AutoencoderKLWan instance

        Examples:
            >>> config = WanVAEConfig(
            ...     model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            ...     subfolder="vae",
            ... )
            >>> vae = AutoencoderKLWan.from_pretrained(
            ...     "Wan-AI/Wan2.1-T2V-14B-Diffusers", config=config
            ... )
        """
        if config is None:
            config = WanVAEConfig(
                type="autoencoder_kl_wan",
                model_path=model_path,
                precision="bf16",
            )

        instance = cls(config)

        # Prepare kwargs for from_pretrained calls
        pretrained_kwargs = {}
        if config.cache_dir:
            pretrained_kwargs["cache_dir"] = config.cache_dir
            logger.info(f"Using cache directory: {config.cache_dir}")

        # Resolve model subfolder with priority: param > config.subfolder > error
        model_subfolder = subfolder if subfolder is not None else getattr(config, "subfolder", None)

        if model_subfolder is None and not hasattr(config, "subfolder"):
            raise ValueError(
                f"subfolder must be specified for AutoencoderKLWan with model_path='{model_path}'. "
                f"For standard Wan diffusers repos (e.g., Wan-AI/Wan2.1-T2V-14B-Diffusers), "
                f"use subfolder='vae'. Set it via config.subfolder or the subfolder parameter."
            )

        # Load VAE from diffusers
        torch_dtype = get_torch_dtype(config.precision)

        logger.info(f"Loading AutoencoderKLWan from {model_path} (subfolder={model_subfolder})")
        instance.vae = load_pretrained_with_subfolder_fallback(
            DiffusersAutoencoderKLWan,
            model_path,
            subfolder=model_subfolder,
            torch_dtype=torch_dtype,
            **pretrained_kwargs,
        )

        instance.vae.to(instance.device)

        if config.freeze_weights:
            instance.freeze()
            instance.vae.eval()

        logger.info(
            f"Loaded AutoencoderKLWan: in_channels={instance.in_channels}, "
            f"out_channels={instance.out_channels}, "
            f"spatial_downsample={instance.latent_downsample_factor}x, "
            f"temporal_downsample={instance.temporal_downsample_factor}x"
        )

        return instance

    @torch.no_grad()
    def encode(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel videos to latent representations.

        Args:
            videos: Input videos tensor of shape (B, C, T, H, W)
                    Values should be in range [-1, 1] (normalized)

        Returns:
            Latent representations of shape (B, latent_channels, T_lat, H_lat, W_lat)
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call from_pretrained() first.")

        videos = videos.to(device=self.device, dtype=self.dtype)

        latent_dist = self.vae.encode(videos).latent_dist
        latents = latent_dist.sample()

        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to pixel videos.

        Args:
            latents: Latent representations of shape (B, latent_channels, T_lat, H_lat, W_lat)

        Returns:
            Reconstructed videos of shape (B, C, T, H, W)
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call from_pretrained() first.")

        latents = latents.to(device=self.device, dtype=self.dtype)

        videos = self.vae.decode(latents, return_dict=False)[0]

        return videos

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode (reconstruction).

        Args:
            videos: Input videos tensor of shape (B, C, T, H, W)

        Returns:
            Reconstructed videos of shape (B, C, T, H, W)
        """
        latents = self.encode(videos)
        reconstructed = self.decode(latents)
        return reconstructed


# Register encoder in registry
from primus.backends.megatron.data.diffusion.encoders import register_encoder

register_encoder("autoencoder_kl_wan", AutoencoderKLWan)
