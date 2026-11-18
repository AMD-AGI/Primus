###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# SDXL autoencoder wrapper around the official generative-models Encoder.
# Weights match the original sd_xl VAE (encoder / quant_conv), not a Diffusers
# pipeline.

from __future__ import annotations

import torch
import torch.nn as nn

from primus.backends.diffusion.models.sdxl.autoencoder import Encoder
from primus.backends.diffusion.models.sdxl.train_pipeline import SDXL_LATENT_SCALE


class SDXLAutoencoderKL(nn.Module):
    def __init__(self, scale_factor: float = SDXL_LATENT_SCALE):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = Encoder(
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            attn_resolutions=(),
            dropout=0.0,
            resamp_with_conv=True,
            in_channels=3,
            resolution=256,
            z_channels=4,
            double_z=True,
        )
        self.quant_conv = nn.Conv2d(8, 8, 1)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        # The frozen VAE is loaded in FP32. Training pixels are BF16, and a Conv
        # bias left in FP32 raises before the first SDXL step.
        if any(parameter.dtype != pixels.dtype for parameter in self.parameters()):
            self.to(dtype=pixels.dtype)
        moments = self.quant_conv(self.encoder(pixels))
        mean, logvar = moments.chunk(2, dim=1)
        std = torch.exp(0.5 * logvar.clamp(max=20))
        latents = mean + std * torch.randn_like(std)
        return latents * self.scale_factor
