###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# Adapted from the Genmo Mochi official implementation.
# https://github.com/genmoai/mochi

import torch


def modulated_rmsnorm(x, scale, eps=1e-6):
    dtype = x.dtype
    x = x.float()

    # Compute RMS
    mean_square = x.pow(2).mean(-1, keepdim=True)
    inv_rms = torch.rsqrt(mean_square + eps)

    # Normalize and modulate
    x_normed = x * inv_rms
    x_modulated = x_normed * (1 + scale.unsqueeze(1).float())
    return x_modulated.to(dtype)
