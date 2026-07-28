# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Base encoder classes for diffusion models.

This module defines abstract base classes for different encoder types.
All concrete encoder implementations should inherit from these base classes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Utility Functions
# ============================================================================


def get_torch_dtype(precision: str) -> torch.dtype:
    """
    Convert precision string to torch dtype.

    Args:
        precision: One of 'bf16', 'fp16', 'fp32'

    Returns:
        Corresponding torch dtype

    Example:
        >>> dtype = get_torch_dtype('bf16')
        >>> dtype
        torch.bfloat16
    """
    DTYPE_MAP = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return DTYPE_MAP.get(precision, torch.bfloat16)


def load_pretrained_with_subfolder_fallback(
    loader_class: Any, model_path: str, subfolder: Optional[str] = None, **kwargs
) -> Any:
    """
    Load a pretrained model with automatic subfolder fallback.

    Tries to load from subfolder first, falls back to root path if that fails.
    This handles both HuggingFace repos (with subfolders) and local paths.

    Args:
        loader_class: Model class with from_pretrained method
        model_path: Path to model (HuggingFace repo or local path)
        subfolder: Optional subfolder within model_path
        **kwargs: Additional arguments passed to from_pretrained
                  (e.g., torch_dtype, cache_dir, token)

    Returns:
        Loaded model instance

    Example:
        >>> from transformers import T5EncoderModel
        >>> model = load_pretrained_with_subfolder_fallback(
        ...     T5EncoderModel,
        ...     "black-forest-labs/FLUX.1-dev",
        ...     subfolder="text_encoder_2",
        ...     torch_dtype=torch.bfloat16,
        ...     cache_dir="/custom/cache/path"
        ... )
    """
    # Add token from environment if available and not already in kwargs
    import os

    if "token" not in kwargs and "HF_TOKEN" in os.environ:
        kwargs["token"] = os.environ["HF_TOKEN"]

    if subfolder:
        try:
            return loader_class.from_pretrained(model_path, subfolder=subfolder, **kwargs)
        except Exception as e:
            logger.debug(f"Failed loading from subfolder '{subfolder}': {e}")
            logger.debug("Retrying without subfolder...")

    return loader_class.from_pretrained(model_path, **kwargs)


# ============================================================================
# Base Encoder Classes
# ============================================================================


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for all encoders."""

    def __init__(self, config):
        """
        Initialize encoder.

        Args:
            config: Encoder configuration object
        """
        super().__init__()
        self.config = config
        self._device = config.device if hasattr(config, "device") else "cuda"
        self._dtype = get_torch_dtype(config.precision if hasattr(config, "precision") else "bf16")

    @property
    def device(self) -> torch.device:
        """Get encoder device."""
        if isinstance(self._device, str):
            return torch.device(self._device)
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        """Get encoder dtype."""
        return self._dtype

    def _get_dtype(self, precision: str) -> torch.dtype:
        """
        Convert precision string to torch dtype.

        Args:
            precision: One of 'bf16', 'fp16', 'fp32'

        Returns:
            Corresponding torch dtype
        """
        return get_torch_dtype(precision)

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Encode input data.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement encode()")

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_path: str, config=None):
        """
        Load encoder from pretrained weights.

        Args:
            model_path: Path to pretrained model
            config: Optional encoder configuration

        Returns:
            Loaded encoder instance
        """
        raise NotImplementedError("Subclasses must implement from_pretrained()")

    def freeze(self):
        """Freeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all encoder parameters."""
        for param in self.parameters():
            param.requires_grad = True


class BaseVAE(BaseEncoder):
    """Abstract base class for VAE encoders."""

    def __init__(self, config):
        """
        Initialize VAE encoder.

        Args:
            config: VAEConfig object
        """
        super().__init__(config)
        self.scale_factor = config.scale_factor if hasattr(config, "scale_factor") else 1.0
        self.shift_factor = config.shift_factor if hasattr(config, "shift_factor") else 0.0
        self.in_channels = config.in_channels if hasattr(config, "in_channels") else 3
        self.out_channels = config.out_channels if hasattr(config, "out_channels") else 16
        self.latent_downsample_factor = (
            config.latent_downsample_factor if hasattr(config, "latent_downsample_factor") else 8
        )

    @abstractmethod
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations.

        Args:
            images: Input images tensor of shape (B, C, H, W)

        Returns:
            Latent representations of shape (B, latent_channels, H/downsample, W/downsample)
        """
        raise NotImplementedError("Subclasses must implement encode()")

    @abstractmethod
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.

        Args:
            latents: Latent representations of shape (B, latent_channels, H/downsample, W/downsample)

        Returns:
            Reconstructed images of shape (B, C, H, W)
        """
        raise NotImplementedError("Subclasses must implement decode()")

    def get_latent_shape(self, image_height: int, image_width: int) -> Tuple[int, int, int]:
        """
        Calculate latent shape from image dimensions.

        Args:
            image_height: Input image height
            image_width: Input image width

        Returns:
            Tuple of (channels, latent_height, latent_width)
        """
        latent_h = image_height // self.latent_downsample_factor
        latent_w = image_width // self.latent_downsample_factor
        return (self.out_channels, latent_h, latent_w)


class BaseTextEncoder(BaseEncoder):
    """Abstract base class for text encoders."""

    def __init__(self, config):
        """
        Initialize text encoder.

        Args:
            config: TextEncoderConfig object
        """
        super().__init__(config)
        self.max_length = config.max_length if hasattr(config, "max_length") else 512
        self.embedding_dim = config.embedding_dim if hasattr(config, "embedding_dim") else 4096
        self.return_pooled = config.return_pooled if hasattr(config, "return_pooled") else False
        self._tokenizer = None

    @abstractmethod
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode text(s) to embeddings.

        Args:
            texts: Single text string or list of text strings
            **kwargs: Additional encoding arguments

        Returns:
            Text embeddings tensor(s)
            - For standard encoders: (B, seq_len, embedding_dim)
            - For pooled encoders: tuple of (sequence_embeddings, pooled_embeddings)
        """
        raise NotImplementedError("Subclasses must implement encode()")

    @property
    def tokenizer(self):
        """Get tokenizer instance."""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call from_pretrained() first.")
        return self._tokenizer

    def _prepare_texts(self, texts: Union[str, List[str]]) -> List[str]:
        """
        Prepare text inputs for encoding.

        Args:
            texts: Single text string or list of text strings

        Returns:
            List of text strings
        """
        if isinstance(texts, str):
            return [texts]
        return texts
