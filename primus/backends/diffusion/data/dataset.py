###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Dataset class for loading video data from JSONL/CSV files.
"""

import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from primus.backends.diffusion.utils.data_utils import smart_nframes

from .collator import RawBatchCollator


def _get_decord_vr(
    video_path: str,
    *,
    num_threads: int,
):
    """Construct a decord.VideoReader."""
    from decord import VideoReader, cpu

    return VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)


class BaseDataset(Dataset):
    """Base dataset with minimal shared interface."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.samples = []

    def __len__(self):
        return len(self.samples)


class WanVideoDataset(BaseDataset):
    """Dataset for WanVideo training from JSONL/CSV."""

    def __init__(self, processor, config={}):
        """
        Initialize WanVideo dataset.

        Args:
            processor: WanVideoDataProcessor instance
            video_backend: Backend for video loading ('qwen_vl_utils' or 'decord')
        """
        super().__init__(config)
        self.config = config
        self.data_path = Path(self.config.dataset_path)
        self.processor = processor

        # Load metadata
        self.samples = self._load_metadata()
        self._sync_processor_video_limits()

    def _sync_processor_video_limits(self):
        image_processor = None
        if hasattr(self.processor, "processor") and hasattr(self.processor.processor, "image_processor"):
            image_processor = self.processor.processor.image_processor
        if image_processor is None:
            return
        if (
            getattr(image_processor, "max_pixels", None) is None
            and getattr(self.config, "video_max_pixels", None) is not None
        ):
            image_processor.max_pixels = self.config.video_max_pixels

    def _load_metadata(self) -> list[dict]:
        """Load metadata from JSONL or CSV file."""
        samples = []
        # TODO: add dummy data for debugging
        if self.data_path.name == "dummy.jsonl":
            return samples

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    samples.append(json.loads(line.strip()))
        elif self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                samples = json.load(f)
        elif self.data_path.suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(self.data_path)
            samples = df.to_dict("records")
        else:
            raise ValueError(f"Unsupported file format: {self.data_path=}")

        return samples

    def _load_video_frames(self, video_path: str, data_folder=None, fps: int = 1) -> tuple[np.ndarray, float]:
        """Load video frames using the specified backend."""
        if self.config.data_folder is not None:
            video_path = os.path.join(self.config.data_folder, video_path)

        if self.config.video_backend == "decord":
            return self.load_video_decord(video_path, fps)
        elif self.config.video_backend == "qwen_vl_utils":
            return self.load_video_qwen_vl_utils(video_path, fps)
        elif self.config.video_backend == "imageio":
            return self.load_video_imageio(video_path, fps)
        else:
            raise ValueError(f"Unsupported video backend: {self.config.video_backend}")

    def load_video_imageio(self, video_path, fps):
        import imageio

        reader = imageio.get_reader(video_path)

        # Sampling Strategy
        total_frames = reader.count_frames()
        total_frames = int(total_frames)

        if self.config.video_sampling_strategy == "frame_num":
            nframes = self.config.frame_num
            # Enforce VAE divisibility: (n - 1) % 4 == 0
            actual_nframes = min(nframes, total_frames)

            valid_nframes = (actual_nframes - 1) // 4 * 4 + 1 if actual_nframes > 1 else 1

            # DiffSynth Sequential Reading
            frames = []
            for i, frame in enumerate(reader):
                if i >= valid_nframes:
                    break
                frames.append(frame)

            # Stack to numpy (T, H, W, C)
            frames = np.array(frames)
            sample_fps = fps  # Simplification

            reader.close()
        else:
            reader.close()
            raise NotImplementedError("Only frame_num strategy implemented for imageio backend")

        return frames, sample_fps

    def load_video_decord(
        self,
        video_path: str | list[str] | BytesIO,
        fps: int,
    ) -> tuple[np.ndarray, float]:
        """
        Load video using Decord backend.

        Args:
            video_path: Path to video file or BytesIO object
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        # Keep dataset logic simple: use a fixed thread count here.
        from decord import VideoReader, cpu

        num_threads = 4
        align = os.getenv("ALIGN_WITH_DIFFSYNTH") == "1"

        if isinstance(video_path, BytesIO):
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=num_threads)
        elif isinstance(video_path, list):
            vr = _get_decord_vr(video_path[0], num_threads=num_threads)
        elif isinstance(video_path, str):
            vr = _get_decord_vr(video_path, num_threads=num_threads)
        else:
            raise ValueError(f"Unsupported video path type: {type(video_path)}")

        total_frames, video_fps = len(vr), vr.get_avg_fps()
        if self.config.video_sampling_strategy == "fps":
            nframes = smart_nframes(total_frames, video_fps=video_fps, fps=fps)
            # Maintain uniform sampling for FPS strategy
            uniform_sampled_frames = np.linspace(0, total_frames - 1, nframes, dtype=int)
        elif self.config.video_sampling_strategy == "frame_num":
            nframes = self.config.frame_num
            # Enforce VAE divisibility: (n - 1) % 4 == 0
            actual_nframes = min(nframes, total_frames)

            valid_nframes = (actual_nframes - 1) // 4 * 4 + 1 if actual_nframes > 1 else 1

            if align:
                # sequential sampling align with diffsynth
                uniform_sampled_frames = np.arange(valid_nframes, dtype=int)
            else:
                # uniform sampling
                uniform_sampled_frames = np.linspace(0, total_frames - 1, valid_nframes, dtype=int)
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")

        frame_idx = uniform_sampled_frames.tolist()
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # spare_frames = torch.tensor(spare_frames).permute(0, 3, 1, 2)  # Convert to TCHW format

        # Calculate sample_fps
        sample_fps = nframes / max(total_frames, 1e-6) * video_fps

        # Return HWC numpy array to match processor expectations
        return spare_frames, sample_fps  # (frames, height, width, channels)

    def load_video_qwen_vl_utils(
        self,
        video_path: str,
        fps: int,
    ) -> tuple[np.ndarray, float]:
        """
        Load video using Qwen VL utils.

        Args:
            video_path: Path to video file
            fps: Target frames per second

        Returns:
            Tuple of (video frames, sample fps)
        """
        from primus.backends.diffusion.utils.vision_process import fetch_video

        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "min_frames": 1,
            "max_pixels": self.config.video_max_pixels,
            "max_frames": self.config.video_max_frames,
            "min_pixels": self.config.video_min_pixels,
        }

        if self.config.video_sampling_strategy == "frame_num":
            is_even = self.config.frame_num % 2 == 0
            n_frames = self.config.frame_num if is_even else self.config.frame_num + 1
            video_dict["nframes"] = n_frames

            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()

            # if is_even:
            #     return frames, sample_fps
            # else:
            #     return frames[:-1], sample_fps

            # Enforce VAE divisibility constraint
            actual_n = len(frames)
            if actual_n > 1:
                valid_n = ((actual_n - 1) // 4) * 4 + 1
                frames = frames[:valid_n]
            # else: keep 1 frame (or handle error)

            return frames, sample_fps
        elif self.config.video_sampling_strategy == "fps":
            video_dict["fps"] = fps
            frames, sample_fps = fetch_video(video_dict, return_video_sample_fps=True)
            frames = frames.numpy()

            # Also enforce for fps strategy
            actual_n = len(frames)
            if actual_n > 1:
                valid_n = ((actual_n - 1) // 4) * 4 + 1
                frames = frames[:valid_n]

            return frames, sample_fps
        else:
            raise ValueError(f"Invalid video sampling strategy: {self.config.video_sampling_strategy}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load video frames
        video_path = sample["video"]
        video_frames, fps = self._load_video_frames(video_path)

        # Get prompt
        prompt = sample.get("prompt", "")
        # Return raw sample.
        return {
            "video_frames": video_frames,  # np.ndarray, typically (T, H, W, C)
            "prompt": prompt,
            "fps": fps,
            "num_frames": int(getattr(self.config, "frame_num", 0) or 0),
            "video_path": str(video_path),
        }

    def get_collator(self):
        # Prefer raw collation; model-specific processing should happen in processor.prepare_batch.
        return RawBatchCollator()
