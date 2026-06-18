from __future__ import annotations

import torch
from torch import Tensor

PATCH_HEIGHT = 2
PATCH_WIDTH = 2
LATENT_CHANNELS = 16
IMAGE_LATENT_SIZE_RATIO = 8


def generate_latent_from_mean_logvar(mean: Tensor, logvar: Tensor) -> Tensor:
    return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)


def create_position_encoding_for_latents(
    bsz: int,
    latent_height: int,
    latent_width: int,
    position_dim: int = 3,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH
    position_encoding = torch.zeros(height, width, position_dim, device=device, dtype=dtype)
    position_encoding[:, :, 1] = torch.arange(height, device=device, dtype=dtype).unsqueeze(1)
    position_encoding[:, :, 2] = torch.arange(width, device=device, dtype=dtype).unsqueeze(0)
    return position_encoding.view(1, height * width, position_dim).repeat(bsz, 1, 1)


def pack_latents(x: Tensor) -> Tensor:
    bsz, channels, latent_height, latent_width = x.shape
    if latent_height % PATCH_HEIGHT != 0 or latent_width % PATCH_WIDTH != 0:
        raise ValueError(
            "FLUX latents must have height and width divisible by 2, "
            f"got shape={tuple(x.shape)}"
        )
    height = latent_height // PATCH_HEIGHT
    width = latent_width // PATCH_WIDTH
    x = x.unfold(2, PATCH_HEIGHT, PATCH_HEIGHT).unfold(3, PATCH_WIDTH, PATCH_WIDTH)
    x = x.permute(0, 2, 3, 1, 4, 5)
    return x.reshape(bsz, height * width, channels * PATCH_HEIGHT * PATCH_WIDTH)
