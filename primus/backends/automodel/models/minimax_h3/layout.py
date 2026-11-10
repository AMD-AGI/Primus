###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Packed-sequence layout and latent packing for MiniMax-H3 training.

WHY THIS FILE EXISTS:
  MiniMax-H3 runs one stack of blocks over ONE packed 1-D sequence
  ``[text | keyframe conditions | target audio | target video]`` with full
  self-attention and no cross-attention. The transformer does not build that
  sequence; its caller does, and tells it, per row, the modality, the rotary
  ``(t, h, w)`` position and which timestep the row sits at. Diffusers builds it
  for inference in ``MiniMaxH3PrepareLayoutStep.build_packed_sequence``. Training
  calls that function rather than copying it, so a diffusers change to the layout
  reaches training too, and a test pins its output.

  Everything that depends only on the clip shape is built once and cached on the
  device, because the shape repeats: the layout of a 768x1344, 5.2 s clip is about
  38K rows of FP64 positions and index tensors.

THE BATCH SHARES ONE LAYOUT:
  The structural tensors describe a single packed layout that every batch item
  shares, and the timestep table is shared with them. So one forward pass cannot
  hold items at different noise levels, and training runs at micro-batch 1.

TIMESTEPS:
  The reference reduces the per-row timesteps with ``torch.unique``. Training
  needs only two values, one for the video (and text) rows and one for the audio
  rows, so the table is ``[t_video, t_audio]`` and the row index is a function of
  the layout alone. That is the same mapping from row to timestep value, without a
  host round trip for the noise level.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class H3Constants:
    """What the layout depends on, as fixed by the released model and its VAEs.

    The defaults are the released model's values. ``constants`` replaces the ones
    diffusers exports with diffusers' own, and a test checks the two agree, so a
    drift shows up as a failure rather than a mislaid layout.
    """

    video_tag: int = 0
    text_tag: int = 1
    audio_tag: int = 2
    # Stereo; the soundtrack is packed channel-major over the channels.
    audio_channels: int = 2
    # Channels of one audio latent row, and of one video latent frame.
    audio_latent_channels: int = 32
    video_latent_channels: int = 24
    # The transformer's (t, h, w) patch. Only a temporal patch of 1 is supported:
    # the reference layout counts latent frames as rows of frames.
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    # Pixel frames the video VAE encodes per chunk, and the latent frames a chunk
    # keeps: a clip of 17 * n + 5 frames has 5 * n + 2 latent frames.
    frames_per_chunk: int = 17
    latents_per_chunk: int = 5
    vae_spatial_compression: int = 16
    fps: int = 24
    audio_latents_per_second: int = 40


_CONSTANTS: H3Constants | None = None


def constants() -> H3Constants:
    """The released model's constants, with diffusers' own values where it has them."""
    global _CONSTANTS
    if _CONSTANTS is None:
        values = {}
        try:
            from diffusers.modular_pipelines.minimax_h3 import modular_pipeline as ref

            values = {
                "video_tag": ref.MINIMAX_H3_VIDEO_TAG,
                "text_tag": ref.MINIMAX_H3_TEXT_TAG,
                "audio_tag": ref.MINIMAX_H3_AUDIO_TAG,
                "audio_channels": ref.MINIMAX_H3_AUDIO_CHANNELS,
                "fps": ref.MINIMAX_H3_FPS,
                "audio_latents_per_second": ref.MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
            }
        except ImportError:
            # Diffusers older than the release that has H3, or without the optional
            # packages its modular pipelines import: the defaults are the same values.
            pass
        _CONSTANTS = H3Constants(**values)
    return _CONSTANTS


@dataclass(frozen=True)
class H3Geometry:
    """Latent shape of one clip, derived from its size in pixels and frames."""

    num_frames: int
    latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int


def latent_geometry(height: int, width: int, num_frames: int, c: H3Constants | None = None) -> H3Geometry:
    """Latent shape of a ``height`` x ``width`` clip of ``num_frames`` frames.

    Strict on purpose: a frame count the video VAE cannot encode (not of the form
    17 * n + 5) or a side that is not a multiple of 32 is a configuration error in
    a training bucket, where inference would round it.
    """
    c = c or constants()
    multiple = c.vae_spatial_compression * c.patch_size[2]
    if height % multiple or width % multiple:
        raise ValueError(
            f"MiniMax-H3 clip sides must be multiples of {multiple} pixels, got {height}x{width}."
        )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        audio_latent_num_frames,
        video_latent_num_frames,
    )

    # Raises for a frame count the VAE cannot encode.
    latent_frames = video_latent_num_frames(num_frames, c.frames_per_chunk, c.latents_per_chunk)
    return H3Geometry(
        num_frames=num_frames,
        latent_frames=latent_frames,
        latent_height=height // c.vae_spatial_compression,
        latent_width=width // c.vae_spatial_compression,
        num_audio_latents=audio_latent_num_frames(num_frames, c.fps, c.audio_latents_per_second),
    )


@dataclass(frozen=True)
class H3Layout:
    """The packed layout of one clip shape, on one device."""

    position_ids: torch.Tensor  # [S, 3] float64, the (t, h, w) rotary coordinates
    token_tags: torch.Tensor  # [S] int64, the modality of every row
    timestep_indices: torch.Tensor  # [S] int64, 0 for text and video rows, 1 for audio rows
    video_indices: torch.Tensor
    audio_indices: torch.Tensor
    text_indices: torch.Tensor

    @property
    def num_video_rows(self) -> int:
        return int(self.video_indices.numel())

    @property
    def num_audio_rows(self) -> int:
        return int(self.audio_indices.numel())

    def model_kwargs(self) -> Dict[str, torch.Tensor]:
        """The structural arguments of the transformer's forward."""
        return {
            "timestep_indices": self.timestep_indices,
            "token_tags": self.token_tags,
            "position_ids": self.position_ids,
            "video_indices": self.video_indices,
            "audio_indices": self.audio_indices,
            "text_indices": self.text_indices,
        }


def build_layout(
    num_text_tokens: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    device: torch.device | str = "cpu",
    c: H3Constants | None = None,
) -> H3Layout:
    """Build the ``[text | audio | video]`` layout of a text-to-audio-video clip."""
    c = c or constants()
    if c.patch_size[0] != 1:
        raise ValueError(f"only a temporal patch of 1 is supported, got patch_size={c.patch_size}")
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
    )

    text_token_tags = torch.full((num_text_tokens,), c.text_tag, dtype=torch.long)
    position_ids, token_tags, video_indices, audio_indices, text_indices, _, _ = (
        MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags,
            latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            c.patch_size,
            c.audio_channels,
            c.audio_tag,
            c.video_tag,
        )
    )
    timestep_indices = (token_tags == c.audio_tag).to(torch.long)
    return H3Layout(
        position_ids=position_ids.to(device),
        token_tags=token_tags.to(device),
        timestep_indices=timestep_indices.to(device),
        video_indices=video_indices.to(device),
        audio_indices=audio_indices.to(device),
        text_indices=text_indices.to(device),
    )


# A run uses a handful of clip shapes, each a few MB on the device. The bound stops
# a loader that varies the text length per sample from growing it without limit.
_CACHE_MAX_ENTRIES = 16
_layout_cache: "OrderedDict[tuple, H3Layout]" = OrderedDict()


def get_layout(
    num_text_tokens: int,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    device: torch.device | str,
) -> H3Layout:
    """``build_layout``, built once per clip shape and device."""
    key = (str(device), num_text_tokens, latent_frames, latent_height, latent_width, num_audio_latents)
    layout = _layout_cache.get(key)
    if layout is not None:
        _layout_cache.move_to_end(key)
        return layout
    layout = build_layout(
        num_text_tokens, latent_frames, latent_height, latent_width, num_audio_latents, device=device
    )
    _layout_cache[key] = layout
    while len(_layout_cache) > _CACHE_MAX_ENTRIES:
        _layout_cache.popitem(last=False)
    return layout


def clear_layout_cache() -> None:
    _layout_cache.clear()


def patchify_video(latents: torch.Tensor, patch_size: Tuple[int, int, int]) -> torch.Tensor:
    """Video latents ``[B, C, F, H, W]`` to transformer rows ``[B, N, C * prod(patch)]``.

    Rows are frame-major, then row-major, and each row holds its patch channel-major,
    which is the order of diffusers' ``patchify_video_latents`` (a test checks it).
    """
    pt, ph, pw = patch_size
    b, c, f, h, w = latents.shape
    if f % pt or h % ph or w % pw:
        raise ValueError(
            f"latents of shape {tuple(latents.shape)} are not divisible by the patch {patch_size}"
        )
    x = latents.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(b, (f // pt) * (h // ph) * (w // pw), c * pt * ph * pw)


def unpatchify_video(
    rows: torch.Tensor, patch_size: Tuple[int, int, int], channels: int, frames: int, height: int, width: int
) -> torch.Tensor:
    """Inverse of ``patchify_video``: rows ``[B, N, C * prod(patch)]`` to ``[B, C, F, H, W]``."""
    pt, ph, pw = patch_size
    b = rows.shape[0]
    x = rows.reshape(b, frames // pt, height // ph, width // pw, channels, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(b, channels, frames, height, width)


def pack_audio(latents: torch.Tensor) -> torch.Tensor:
    """Audio latents ``[B, S, A, L]`` (stereo channel, latent channels, length) to rows ``[B, S * L, A]``.

    Channel-major over the stereo channels, as the reference packs the soundtrack.
    """
    b, s, a, length = latents.shape
    return latents.permute(0, 1, 3, 2).reshape(b, s * length, a)


def unpack_audio(rows: torch.Tensor, stereo_channels: int, length: int) -> torch.Tensor:
    """Inverse of ``pack_audio``: rows ``[B, S * L, A]`` to ``[B, S, A, L]``."""
    b, _, a = rows.shape
    return rows.reshape(b, stereo_channels, length, a).permute(0, 1, 3, 2)
