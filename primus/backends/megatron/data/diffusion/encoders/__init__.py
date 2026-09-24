# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Encoder implementations for diffusion models.

This module contains encoders for various modalities:
    - Image encoders (VAE, etc.)
    - Text encoders (T5, CLIP, etc.)
    - Encoder registry for config-driven selection
    - Base encoder classes

Hierarchical structure:
    - encoders/image/autoencoder_kl.py - VAE (AutoencoderKL)
    - encoders/text/t5_xxl.py - T5-XXL text encoder
    - encoders/text/clip_l.py - CLIP-L text encoder

Each encoder type can have multiple implementations registered
in the ENCODER_REGISTRY for flexible configuration.
"""

import importlib
import logging
from typing import Dict, Optional, Type

from .base import BaseEncoder, BaseTextEncoder, BaseVAE
from .config import (
    CLIPLConfig,
    EncoderConfig,
    FluxEncoderConfig,
    T5XXLConfig,
    TextEncoderConfig,
    VAEConfig,
)

logger = logging.getLogger(__name__)

# Global encoder registry
ENCODER_REGISTRY: Dict[str, Type[BaseEncoder]] = {}


def register_encoder(name: str, encoder_class: Type[BaseEncoder]):
    """
    Register an encoder class in the global registry.

    Args:
        name: Unique identifier for the encoder (e.g., 'autoencoder_kl', 't5_xxl')
        encoder_class: Encoder class to register

    Example:
        >>> register_encoder('my_encoder', MyEncoder)
    """
    if name in ENCODER_REGISTRY:
        logger.warning(f"Encoder '{name}' already registered. Overwriting with {encoder_class}")
    ENCODER_REGISTRY[name] = encoder_class
    logger.debug(f"Registered encoder: {name} -> {encoder_class.__name__}")


def get_encoder(config: EncoderConfig) -> BaseEncoder:
    """
    Factory function to get an encoder instance from config.

    Args:
        config: Encoder configuration object with 'type' field

    Returns:
        Instantiated encoder instance

    Raises:
        ValueError: If encoder type not found in registry

    Example:
        >>> config = VAEConfig(type='autoencoder_kl', model_path='...')
        >>> encoder = get_encoder(config)
    """
    _ensure_encoders_discovered()

    encoder_type = config.type

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Encoder type '{encoder_type}' not found in registry. "
            f"Available encoders: {list(ENCODER_REGISTRY.keys())}"
        )

    encoder_class = ENCODER_REGISTRY[encoder_type]
    logger.info(f"Loading encoder: {encoder_type} from {config.model_path}")

    # Use from_pretrained if available, otherwise use constructor
    if hasattr(encoder_class, "from_pretrained") and callable(encoder_class.from_pretrained):
        return encoder_class.from_pretrained(config.model_path, config=config)
    else:
        return encoder_class(config)


def list_encoders() -> Dict[str, Type[BaseEncoder]]:
    """
    List all registered encoders.

    Returns:
        Dictionary mapping encoder names to encoder classes
    """
    _ensure_encoders_discovered()
    return ENCODER_REGISTRY.copy()


_ENCODERS_DISCOVERED = False


def _auto_discover_encoders():
    """
    Auto-discover and register all encoder implementations.

    This function imports encoder modules which triggers their registration
    through decorators or explicit register_encoder() calls.
    """
    encoder_modules = [
        "primus.backends.megatron.data.diffusion.encoders.image.autoencoder_kl",
        "primus.backends.megatron.data.diffusion.encoders.text.t5_xxl",
        "primus.backends.megatron.data.diffusion.encoders.text.clip_l",
    ]

    for module_path in encoder_modules:
        try:
            importlib.import_module(module_path)
            logger.debug(f"Successfully imported encoder module: {module_path}")
        except ImportError as e:
            logger.debug(f"Could not import encoder module {module_path}: {e}")
        except Exception as e:
            logger.warning(f"Error importing encoder module {module_path}: {e}")


def _ensure_encoders_discovered():
    """Run encoder discovery once, lazily (on first get_encoder/list_encoders).

    Avoids importing all encoder backends (and their heavy deps) at package
    import time.
    """
    global _ENCODERS_DISCOVERED
    if not _ENCODERS_DISCOVERED:
        _auto_discover_encoders()
        _ENCODERS_DISCOVERED = True


__all__ = [
    # Base classes
    "BaseEncoder",
    "BaseVAE",
    "BaseTextEncoder",
    # Config classes
    "EncoderConfig",
    "VAEConfig",
    "TextEncoderConfig",
    "T5XXLConfig",
    "CLIPLConfig",
    "FluxEncoderConfig",
    # Registry functions
    "register_encoder",
    "get_encoder",
    "list_encoders",
    "ENCODER_REGISTRY",
]
