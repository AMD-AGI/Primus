###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Turn raw image and video samples into NCHW / NCTHW tensors in [-1, 1]."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _frame_tensor(frame: Any) -> torch.Tensor:
    if isinstance(frame, torch.Tensor):
        tensor = frame.detach().float()
    else:
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")
        tensor = torch.from_numpy(np.asarray(frame)).float()
    if tensor.ndim == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"Expected an HWC or CHW frame, got shape {tuple(tensor.shape)}")
    if tensor.max() > 1.5:
        tensor = tensor / 127.5 - 1.0
    elif tensor.min() >= 0:
        tensor = tensor * 2.0 - 1.0
    return tensor


def _resize(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    return F.interpolate(frames, size=(height, width), mode="bilinear", align_corners=False)


def images_to_tensor(images: list[Any], *, height: int, width: int) -> torch.Tensor:
    frames = torch.stack(
        [_resize(_frame_tensor(image).unsqueeze(0), height, width).squeeze(0) for image in images]
    )
    return frames


def decode_host_pixels(
    batch: dict[str, Any],
    *,
    height: int,
    width: int,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Replace raw ``images`` / ``videos`` with CPU pixel tensors.

    AMDiffusionBenchmark decodes and resizes samples in the dataset transform,
    before the step's CUDA start event. Calling this when the batch is fetched
    keeps CPU decoding out of ``step_time_gpu`` here too.
    """
    prepared = dict(batch)
    if "images" in prepared:
        prepared["pixel_values"] = images_to_tensor(prepared.pop("images"), height=height, width=width)
    if "videos" in prepared:
        if num_frames is None:
            raise ValueError("Decoding videos requires num_frames.")
        prepared["video"] = videos_to_tensor(
            prepared.pop("videos"), num_frames=num_frames, height=height, width=width
        )
    return prepared


def videos_to_tensor(videos: list[Any], *, num_frames: int, height: int, width: int) -> torch.Tensor:
    clips = []
    for video in videos:
        if isinstance(video, str):
            from primus.backends.diffusion.data.benchmark import open_video_reader

            video = open_video_reader(video)
        if isinstance(video, torch.Tensor) and video.ndim == 4:
            frames = video.float()
            if frames.shape[-1] in (1, 3) and frames.shape[1] not in (1, 3):
                frames = frames.permute(0, 3, 1, 2)
        elif hasattr(video, "get_batch"):
            length = len(video)
            count = min(num_frames, length)
            batch = video.get_batch(list(range(count)))
            array = batch.asnumpy() if hasattr(batch, "asnumpy") else batch
            frames = torch.as_tensor(array).float().permute(0, 3, 1, 2)
        else:
            frame_list = []
            for item in list(video)[:num_frames]:
                data = item["data"] if isinstance(item, dict) and "data" in item else item
                frame_list.append(_frame_tensor(data))
            frames = torch.stack(frame_list)
        if frames.max() > 1.5:
            frames = frames / 127.5 - 1.0
        elif frames.min() >= 0:
            frames = frames * 2.0 - 1.0
        if frames.shape[0] < num_frames:
            pad = frames[-1:].repeat(num_frames - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, pad], dim=0)
        frames = frames[:num_frames]
        frames = _resize(frames, height, width)
        clips.append(frames.permute(1, 0, 2, 3))
    return torch.stack(clips)
