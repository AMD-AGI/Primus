# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Video encoder implementations for diffusion models.
"""

from primus.backends.megatron.data.diffusion.encoders.base import BaseVAE

from .vae.wan import AutoencoderKLWan

__all__ = [
    "BaseVAE",
    "AutoencoderKLWan",
]
