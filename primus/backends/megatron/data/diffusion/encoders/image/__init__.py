# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Image encoder implementations for diffusion models.
"""

from primus.backends.megatron.data.diffusion.encoders.base import BaseVAE

from .autoencoder_kl import AutoencoderKL

__all__ = [
    "BaseVAE",
    "AutoencoderKL",
]
