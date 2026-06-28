# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Base diffusion module with Megatron-Core integration.
"""

from primus.backends.megatron.core.models.common.diffusion_module.diffusion_module import (
    DiffusionModule,
)

__all__ = ["DiffusionModule"]
