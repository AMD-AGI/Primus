# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
WAN latent packing helpers.

Flux's ``flux/utils.py`` packs 2D image latents into a token sequence; WAN packs
a 5D video volume ``[B, C, T, H, W]`` into ``[B, S, C*p_t*p_h*p_w]`` over a 3D
patch grid, so none of it carries over.
"""

from typing import Tuple

from torch import Tensor


def patch_grid(num_frames: int, height: int, width: int, patch_size_3d) -> Tuple[int, int, int]:
    """Patch-grid extent ``(ppf, pph, ppw)`` for a latent volume."""
    p_t, p_h, p_w = patch_size_3d
    for name, size, patch in (
        ("num_frames", num_frames, p_t),
        ("height", height, p_h),
        ("width", width, p_w),
    ):
        if size % patch != 0:
            raise ValueError(f"{name}={size} is not divisible by its patch size {patch}")
    return num_frames // p_t, height // p_h, width // p_w


def fold_patches_into_batch(latents: Tensor, in_channels: int, patch_size_3d) -> Tensor:
    """Reorder ``[B, C, T, H, W]`` into ``[B*S, C, p_t, p_h, p_w]``.

    Each non-overlapping patch becomes its own batch element so a 1x1x1 Conv3d
    produces the patch embedding. This is numerically identical to one strided
    Conv3d over the whole volume, but the backward reduces over the batch
    dimension the way Megatron-Bridge does -- under deterministic conv both
    stacks then select the same MIOpen backward algorithm instead of splitting
    between the strided-volume and batched implicit-GEMM paths.
    """
    batch_size = latents.shape[0]
    p_t, p_h, p_w = patch_size_3d
    ppf, pph, ppw = patch_grid(latents.shape[2], latents.shape[3], latents.shape[4], patch_size_3d)

    x = latents.reshape(batch_size, in_channels, ppf, p_t, pph, p_h, ppw, p_w)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.reshape(-1, in_channels, p_t, p_h, p_w)


def unpatchify(
    hidden_states: Tensor,
    batch_size: int,
    grid: Tuple[int, int, int],
    patch_size_3d,
) -> Tensor:
    """Reassemble ``[B, S, out*p_t*p_h*p_w]`` back into ``[B, C_out, T, H, W]``."""
    ppf, pph, ppw = grid
    p_t, p_h, p_w = patch_size_3d

    x = hidden_states.reshape(batch_size, ppf, pph, ppw, p_t, p_h, p_w, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return x.flatten(6, 7).flatten(4, 5).flatten(2, 3)


__all__ = ["patch_grid", "fold_patches_into_batch", "unpatchify"]
