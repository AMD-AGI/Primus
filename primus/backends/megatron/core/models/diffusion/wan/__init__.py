# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN video diffusion model components.

This module provides all components needed for the WAN 2.1 / 2.2 architecture:
- Models: Wan (single transformer), Wan2_2 (dual expert)
- Configuration: WanConfig
- Layers: WanRotaryPosEmbed (3D RoPE), WanConditionEmbedder
- Attention: WanSelfAttention, WanCrossAttention
- Layer specs: get_wan_layer_spec

Quick Start - Model Creation:
    >>> from primus.backends.megatron.core.models.diffusion.wan import Wan, WanConfig
    >>>
    >>> config = WanConfig.wan2_1_t2v_1_3b()
    >>> model = Wan(config=config)

WAN 2.2 dual expert:
    >>> from primus.backends.megatron.core.models.diffusion.wan import Wan2_2, WanConfig
    >>>
    >>> config = WanConfig.wan2_2_t2v_a14b()
    >>> model = Wan2_2(config=config)

Per-expert training selects one side of the timestep boundary:
    >>> config = WanConfig.wan2_2_t2v_a14b(stage="high_noise")
    >>> config.timestep_window
    (0.875, 1.0)
"""

from typing import TYPE_CHECKING

from primus.backends.megatron.core.models.diffusion.wan.attention import (
    WanCrossAttention,
    WanCrossAttentionSubmodules,
    WanSelfAttention,
    WanSelfAttentionSubmodules,
)
from primus.backends.megatron.core.models.diffusion.wan.config import WanConfig
from primus.backends.megatron.core.models.diffusion.wan.layer_spec import (
    WanLayerSpec,
    get_wan_layer_spec,
    get_wan_transformer_spec_for_backend,
)
from primus.backends.megatron.core.models.diffusion.wan.layers import (
    WanConditionEmbedder,
    WanRotaryPosEmbed,
)
from primus.backends.megatron.core.models.diffusion.wan.utils import (
    fold_patches_into_batch,
    patch_grid,
    unpatchify,
)

# The names below are resolved at runtime by __getattr__ below, which static
# analysis cannot see; importing them under TYPE_CHECKING declares them without
# binding anything at module scope, which would shadow __getattr__ entirely.
if TYPE_CHECKING:
    from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
        convert_hf_checkpoint,
    )
    from primus.backends.megatron.core.models.diffusion.wan.model import (
        Wan,
        Wan2_2,
        WanTransformer3D,
        build_wan_backbone,
    )

_LAZY_MODEL_EXPORTS = ("Wan", "Wan2_2", "WanTransformer3D", "build_wan_backbone")


# LAZY IMPORT: keep the model classes out of import time so the package can be
# imported (for configs and specs) without pulling in the full backbone stack.
def __getattr__(name):
    """Lazy import for the WAN model classes."""
    if name in _LAZY_MODEL_EXPORTS:
        from primus.backends.megatron.core.models.diffusion.wan import model

        return getattr(model, name)
    if name == "convert_hf_checkpoint":
        from primus.backends.megatron.core.models.diffusion.wan.checkpoint_converter import (
            convert_hf_checkpoint,
        )

        return convert_hf_checkpoint
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    # Models
    "Wan",
    "Wan2_2",
    "WanTransformer3D",
    "build_wan_backbone",
    # Configuration
    "WanConfig",
    # Layers
    "WanRotaryPosEmbed",
    "WanConditionEmbedder",
    # Attention
    "WanSelfAttention",
    "WanCrossAttention",
    "WanSelfAttentionSubmodules",
    "WanCrossAttentionSubmodules",
    # Layer specs
    "WanLayerSpec",
    "get_wan_layer_spec",
    "get_wan_transformer_spec_for_backend",
    # Utils
    "patch_grid",
    "fold_patches_into_batch",
    "unpatchify",
    # Checkpoint conversion
    "convert_hf_checkpoint",
]
