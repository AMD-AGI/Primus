# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Diffusion models for Primus-Megatron.

This module contains implementations of various diffusion model architectures
following Megatron-Core conventions.

Supported models:
    - Flux: Flow-based diffusion with MMDiT architecture
    - (Future) DiT: Diffusion Transformer
    - (Future) MovieGen: Video diffusion
"""


# Lazy import for Flux to avoid early dependencies. Import common components and
# other models from their submodules (e.g. ``...diffusion.common``,
# ``...diffusion.flux``) directly.
def __getattr__(name):
    """Lazy import for the Flux model and config."""
    if name == "Flux":
        from primus.backends.megatron.core.models.diffusion.flux import Flux

        return Flux
    elif name == "FluxConfig":
        from primus.backends.megatron.core.models.diffusion.flux import FluxConfig

        return FluxConfig
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "Flux",
    "FluxConfig",
]
