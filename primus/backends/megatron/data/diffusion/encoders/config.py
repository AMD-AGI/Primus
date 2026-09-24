# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Configuration dataclasses for diffusion encoders.

This module defines configuration classes for various encoder types used in diffusion models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EncoderType(Enum):
    """Categories of encoders for diffusion models."""

    IMAGE = "image"
    TEXT = "text"


@dataclass
class EncoderConfig:
    """Base configuration for encoders."""

    model_path: str  # Path to pretrained model weights (REQUIRED)
    type: str = None  # Encoder type identifier (e.g., 'autoencoder_kl', 't5_xxl', 'clip_l')
    encoder_type: Optional[EncoderType] = None  # Category: IMAGE or TEXT
    precision: str = "bf16"  # Model precision: 'bf16', 'fp32', 'fp16'
    device: str = "cuda"  # Device to load model on
    use_cached: bool = True  # Whether to use cached/pre-encoded data
    freeze_weights: bool = True  # Whether to freeze encoder weights during training
    cache_dir: Optional[str] = None  # Directory to cache downloaded models (if None, uses HF default)
    subfolder: Optional[str] = None  # Subfolder containing model weights (e.g., 'vae', 'text_encoder_2')
    # Opt-in only: allow executing custom modeling code from the HF repo. Defaults to False
    # so a malicious/unexpected repo cannot run arbitrary code unless explicitly enabled.
    trust_remote_code: bool = False

    # Optional parameters for specific encoders
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.type is None:
            raise ValueError("'type' must be specified in encoder config")
        valid_precisions = ["bf16", "fp32", "fp16"]
        if self.precision not in valid_precisions:
            raise ValueError(f"precision must be one of {valid_precisions}, got {self.precision}")


@dataclass
class VAEConfig(EncoderConfig):
    """Configuration for VAE encoders."""

    type: str = "autoencoder_kl"  # VAE type identifier
    scale_factor: float = 0.3611  # Flux default scale factor
    shift_factor: float = 0.1159  # Flux default shift factor
    in_channels: int = 3  # Input image channels
    out_channels: int = 16  # Latent channels
    latent_downsample_factor: int = 8  # Spatial downsampling factor (H/8, W/8)

    def __post_init__(self):
        """Set encoder type and validate."""
        if self.encoder_type is None:
            self.encoder_type = EncoderType.IMAGE
        super().__post_init__()


@dataclass
class TextEncoderConfig(EncoderConfig):
    """Configuration for text encoders."""

    max_length: int = 512  # Maximum sequence length
    embedding_dim: int = 4096  # Output embedding dimension
    return_pooled: bool = False  # Whether to return pooled embeddings
    tokenizer_path: Optional[str] = None  # Optional separate tokenizer path
    tokenizer_subfolder: Optional[str] = (
        None  # Subfolder containing tokenizer files (e.g., 'tokenizer', 'tokenizer_2')
    )

    def __post_init__(self):
        """Validate configuration."""
        if not self.tokenizer_path:
            self.tokenizer_path = self.model_path  # Default to same as model path
        if self.encoder_type is None:
            self.encoder_type = EncoderType.TEXT
        super().__post_init__()


@dataclass
class T5XXLConfig(TextEncoderConfig):
    """Configuration for T5-XXL encoder."""

    type: str = "t5_xxl"  # T5-XXL type identifier
    max_length: int = 512  # Flux uses 512 for T5-XXL
    embedding_dim: int = 4096  # T5-XXL hidden size

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()


@dataclass
class CLIPLConfig(TextEncoderConfig):
    """Configuration for CLIP-L encoder."""

    type: str = "clip_l"  # CLIP-L type identifier
    max_length: int = 77  # CLIP uses 77 max length
    embedding_dim: int = 768  # CLIP-L hidden size
    return_pooled: bool = True  # CLIP returns both sequence and pooled embeddings
    pooled_dim: int = 768  # CLIP-L pooled embedding dimension

    def __post_init__(self):
        """Validate configuration."""
        super().__post_init__()


@dataclass
class FluxEncoderConfig:
    """Configuration for all Flux encoders (VAE + T5-XXL + CLIP-L)."""

    vae: VAEConfig
    t5: T5XXLConfig
    clip: CLIPLConfig
    use_preencoded: bool = True  # Whether to use pre-encoded data

    @classmethod
    def from_pretrained_flux(
        cls,
        model_path: str = "black-forest-labs/FLUX.1-dev",
        precision: str = "bf16",
        device: str = "cuda",
        use_preencoded: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """Create FluxEncoderConfig from pretrained Flux model path."""
        return cls(
            vae=VAEConfig(
                type="autoencoder_kl",
                model_path=f"{model_path}",
                precision=precision,
                device=device,
                cache_dir=cache_dir,
            ),
            t5=T5XXLConfig(
                type="t5_xxl",
                model_path=f"{model_path}",
                precision=precision,
                device=device,
                cache_dir=cache_dir,
            ),
            clip=CLIPLConfig(
                type="clip_l",
                model_path=f"{model_path}",
                precision=precision,
                device=device,
                cache_dir=cache_dir,
            ),
            use_preencoded=use_preencoded,
        )
