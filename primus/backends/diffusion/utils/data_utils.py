###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################


import math

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def smart_nframes(
    total_frames: int,
    video_fps: int | float,
    fps: int | float,
) -> int:
    """calculate the number of frames for video used for model inputs.

    Args:
        total_frames (int): the original total number of frames of the video.
        video_fps (int | float): the original fps of the video.
        fps (int | float): the target sampling fps for model inputs.

    Raises:
        ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

    Returns:
        int: the number of frames for video used for model inputs.
    """
    min_frames = ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
    max_frames = floor_by_factor(min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR)
    nframes = total_frames / video_fps * fps
    nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
    nframes = floor_by_factor(nframes, FRAME_FACTOR)
    if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
        raise ValueError(f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}.")
    return nframes


def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor
