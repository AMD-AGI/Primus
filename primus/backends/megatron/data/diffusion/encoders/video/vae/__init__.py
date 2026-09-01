# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Video VAE implementations for diffusion models.
"""

from .wan import AutoencoderKLWan

__all__ = [
    "AutoencoderKLWan",
]
