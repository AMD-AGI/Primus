# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Common components for diffusion models in Primus.

This module provides shared components used across different diffusion model
architectures (Flux, DiT, MovieGen, etc.).
"""

from primus.backends.megatron.core.models.diffusion.common.config import (
    BaseDiffusionConfig,
)
from primus.backends.megatron.core.models.diffusion.common.embeddings import (
    MLPEmbedder,
    TimeStepEmbedder,
    Timesteps,
    get_timestep_embedding,
)
from primus.backends.megatron.core.models.diffusion.common.normalization import (
    AdaLN,
    AdaLNContinuous,
    RMSNorm,
)

__all__ = [
    # Base classes
    "BaseDiffusionConfig",
    # Embeddings
    "TimeStepEmbedder",
    "MLPEmbedder",
    "Timesteps",
    "get_timestep_embedding",
    # Normalization
    "RMSNorm",
    "AdaLN",
    "AdaLNContinuous",
]
