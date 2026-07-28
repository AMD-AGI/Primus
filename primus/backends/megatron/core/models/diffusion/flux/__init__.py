# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Flux diffusion model components.

This module provides all components needed for the Flux architecture:
- Model: Flux (main model class)
- Configuration: FluxConfig
- Layers: EmbedND (3D RoPE position embedding)
- Attention: JointSelfAttention, FluxSingleAttention
- Layer specs: MMDiTLayer, FluxSingleTransformerBlock

Quick Start - Model Creation:
    >>> from primus.backends.megatron.core.models.diffusion.flux import Flux, FluxConfig
    >>>
    >>> # Create and configure model
    >>> config = FluxConfig.flux_12b()
    >>> model = Flux(config=config)
    >>>
    >>> # Load checkpoint (native Primus format)
    >>> model.load_checkpoint("flux_12b.safetensors")

Checkpoint Conversion:
    >>> from primus.backends.megatron.core.models.diffusion.flux import convert_hf_checkpoint, FluxConfig
    >>>
    >>> # Convert HuggingFace checkpoint to Primus format
    >>> config = FluxConfig.flux_12b()
    >>> primus_sd = convert_hf_checkpoint(
    ...     "black-forest-labs/FLUX.1-dev/transformer",
    ...     flux_config=config,
    ...     save_to="primus_flux_12b.safetensors"
    ... )
"""

from primus.backends.megatron.core.models.diffusion.flux.attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)
from primus.backends.megatron.core.models.diffusion.flux.checkpoint_converter import (
    convert_hf_checkpoint,
)
from primus.backends.megatron.core.models.diffusion.flux.config import FluxConfig
from primus.backends.megatron.core.models.diffusion.flux.layer_spec import (
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_spec_for_backend,
    get_flux_layer_spec,
    get_flux_single_transformer_spec_for_backend,
)
from primus.backends.megatron.core.models.diffusion.flux.layers import EmbedND, rope
from primus.backends.megatron.core.models.diffusion.flux.utils import (
    generate_image_position_ids,
    generate_text_position_ids,
    pack_latents,
    unpack_latents,
)


# LAZY IMPORT: Don't import Flux here to avoid early TransformerBlock import
def __getattr__(name):
    """Lazy import for Flux to avoid early TransformerBlock import."""
    if name == "Flux":
        from primus.backends.megatron.core.models.diffusion.flux.model import Flux

        return Flux
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Model
    "Flux",
    # Configuration
    "FluxConfig",
    # Layers
    "EmbedND",
    "rope",
    # Attention
    "JointSelfAttention",
    "FluxSingleAttention",
    "JointSelfAttentionSubmodules",
    # Layer specs
    "MMDiTLayer",
    "FluxSingleTransformerBlock",
    "get_flux_double_transformer_spec_for_backend",
    "get_flux_single_transformer_spec_for_backend",
    "get_flux_layer_spec",
    # Utils
    "pack_latents",
    "unpack_latents",
    "generate_image_position_ids",
    "generate_text_position_ids",
    # Checkpoint conversion
    "convert_hf_checkpoint",
]
