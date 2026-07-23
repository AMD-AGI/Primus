# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.
#
# Adapted from NeMo's VAE implementation

"""
AutoencoderKL implementation for Flux VAE.

This module provides a wrapper around the diffusers AutoencoderKL model
for use with Flux diffusion models. It handles encoding images to latent
representations and decoding latents back to images.
"""

import logging
from typing import Optional, Tuple

import torch

try:
    from diffusers import AutoencoderKL as DiffusersAutoencoderKL
except ImportError:
    DiffusersAutoencoderKL = None

from primus.backends.megatron.data.diffusion.encoders.base import (
    BaseVAE,
    get_torch_dtype,
    load_pretrained_with_subfolder_fallback,
)
from primus.backends.megatron.data.diffusion.encoders.config import VAEConfig

logger = logging.getLogger(__name__)


class AutoencoderKL(BaseVAE):
    """
    AutoencoderKL VAE for Flux diffusion models.

    This encoder wraps the diffusers AutoencoderKL model and applies
    Flux-specific scale and shift factors to the latent representations.

    For Flux, the default configuration is:
        - scale_factor: 0.3611
        - shift_factor: 0.1159
        - in_channels: 3 (RGB images)
        - out_channels: 16 (latent channels)
        - latent_downsample_factor: 8 (spatial downsampling)

    This means a 1024x1024 image becomes a 16x128x128 latent.
    """

    def __init__(self, config: VAEConfig):
        """
        Initialize AutoencoderKL encoder.

        Args:
            config: VAEConfig with model_path, scale_factor, shift_factor, etc.
        """
        super().__init__(config)

        if DiffusersAutoencoderKL is None:
            raise ImportError(
                "diffusers library is required for AutoencoderKL. " "Install with: pip install diffusers"
            )

        self.vae = None  # Will be loaded in from_pretrained

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        config: Optional[VAEConfig] = None,
        subfolder: Optional[str] = None,
    ) -> "AutoencoderKL":
        """
        Load AutoencoderKL from pretrained weights.

        Args:
            model_path: Path to pretrained model (local path or HuggingFace repo)
            config: Optional VAEConfig. If None, uses defaults.
            subfolder: Subfolder for VAE weights. Priority: param > config.subfolder

        Returns:
            Loaded AutoencoderKL instance

        Examples:
            >>> # Using config (recommended)
            >>> config = VAEConfig(
            ...     model_path="black-forest-labs/FLUX.1-dev",
            ...     subfolder="vae"
            ... )
            >>> vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", config=config)

            >>> # Using method parameter (backward compatible)
            >>> vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="vae")
        """
        if config is None:
            config = VAEConfig(
                type="autoencoder_kl",
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

        # Require explicit configuration for model subfolder
        if model_subfolder is None and not hasattr(config, "subfolder"):
            raise ValueError(
                f"subfolder must be specified for AutoencoderKL with model_path='{model_path}'. "
                f"For FLUX models (e.g., black-forest-labs/FLUX.1-dev), use subfolder='vae'. "
                f"Set it via config.subfolder or the subfolder parameter."
            )

        # Load VAE from diffusers
        torch_dtype = get_torch_dtype(config.precision)

        logger.info(f"Loading AutoencoderKL from {model_path} (subfolder={model_subfolder})")
        instance.vae = load_pretrained_with_subfolder_fallback(
            DiffusersAutoencoderKL,
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
            f"Loaded AutoencoderKL: in_channels={instance.in_channels}, "
            f"out_channels={instance.out_channels}, scale={instance.scale_factor:.4f}, "
            f"shift={instance.shift_factor:.4f}"
        )

        return instance

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations.

        Applies the Flux scale and shift factors:
            latents = scale_factor * (vae_encode(images) - shift_factor)

        Args:
            images: Input images tensor of shape (B, C, H, W)
                    Values should be in range [-1, 1] (normalized)

        Returns:
            Latent representations of shape (B, 16, H/8, W/8)

        Example:
            >>> images = torch.randn(2, 3, 1024, 1024)  # B=2, 1024x1024 RGB
            >>> latents = vae.encode(images)
            >>> latents.shape
            torch.Size([2, 16, 128, 128])
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call from_pretrained() first.")

        # Ensure images are on correct device and dtype
        images = images.to(device=self.device, dtype=self.dtype)

        # Encode using diffusers VAE
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample()

        # Apply Flux scale and shift factors
        # Formula: z = scale_factor * (encoded - shift_factor)
        latents = self.scale_factor * (latents - self.shift_factor)

        return latents

    @torch.no_grad()
    def encode_for_resample(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode images returning posterior parameters for training-time resampling.

        Unlike ``encode()`` which draws a single stochastic sample, this method
        returns the raw posterior parameters (mean, logvar) so the training loop
        can re-draw latents via reparameterization at every step.

        The returned ``latents`` use the deterministic posterior mode (mean)
        with scale/shift applied, for backward compatibility and debugging.

        Args:
            images: Input images tensor (B, C, H, W) in range [-1, 1]

        Returns:
            Tuple of (latents, mean, logvar):
                - latents: scale * (mode - shift), shape (B, 16, H/8, W/8)
                - mean: raw posterior mean, shape (B, 16, H/8, W/8)
                - logvar: raw posterior log-variance, shape (B, 16, H/8, W/8)
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call from_pretrained() first.")

        images = images.to(device=self.device, dtype=self.dtype)

        latent_dist = self.vae.encode(images).latent_dist
        latents = self.scale_factor * (latent_dist.mode() - self.shift_factor)

        return latents, latent_dist.mean, latent_dist.logvar

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.

        Reverses the Flux scale and shift factors before decoding:
            vae_decode(latents / scale_factor + shift_factor)

        Args:
            latents: Latent representations of shape (B, 16, H/8, W/8)

        Returns:
            Reconstructed images of shape (B, C, H, W)
            Values in range [-1, 1]

        Example:
            >>> latents = torch.randn(2, 16, 128, 128)
            >>> images = vae.decode(latents)
            >>> images.shape
            torch.Size([2, 3, 1024, 1024])
        """
        if self.vae is None:
            raise RuntimeError("VAE not loaded. Call from_pretrained() first.")

        # Ensure latents are on correct device and dtype
        latents = latents.to(device=self.device, dtype=self.dtype)

        # Reverse Flux scale and shift factors
        # Formula: decoded_input = z / scale_factor + shift_factor
        latents = latents / self.scale_factor + self.shift_factor

        # Decode using diffusers VAE
        images = self.vae.decode(latents, return_dict=False)[0]

        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode (reconstruction).

        Args:
            images: Input images tensor of shape (B, C, H, W)

        Returns:
            Reconstructed images of shape (B, C, H, W)
        """
        latents = self.encode(images)
        reconstructed = self.decode(latents)
        return reconstructed


# Register encoder in registry
from primus.backends.megatron.data.diffusion.encoders import register_encoder

register_encoder("autoencoder_kl", AutoencoderKL)
