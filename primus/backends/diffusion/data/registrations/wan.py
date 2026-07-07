###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from transformers import AutoTokenizer

from primus.backends.diffusion.utils.vision_process import fetch_video


class WanVideoProcessor:
    """Tokenize prompts and normalize video tensors for Wan training."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_text_length = int(config.get("max_text_length", 512))
        tokenizer_path = config.get("text_tokenizer")
        if not tokenizer_path:
            raise ValueError("Wan dataset processor requires `text_tokenizer`.")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        extra_kwargs = config.get("extra_kwargs", {}) or {}
        size = extra_kwargs.get("size", {}) or {}
        self.height = int(size.get("height", 480))
        self.width = int(size.get("width", 832))
        self.image_mean = torch.tensor(extra_kwargs.get("image_mean", [0.5, 0.5, 0.5])).view(3, 1, 1, 1)
        self.image_std = torch.tensor(extra_kwargs.get("image_std", [0.5, 0.5, 0.5])).view(3, 1, 1, 1)

    def tokenize(self, prompt: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0).long(),
            "attention_mask": encoded["attention_mask"].squeeze(0).long(),
        }

    def normalize_video(self, video_tchw: torch.Tensor) -> torch.Tensor:
        if video_tchw.ndim != 4:
            raise ValueError(f"Expected video tensor [T,C,H,W], got shape={tuple(video_tchw.shape)}")
        video_tchw = video_tchw[:, :3].float()
        if video_tchw.max() > 2:
            video_tchw = video_tchw / 255.0
        video_tchw = F.resize(video_tchw, [self.height, self.width], antialias=True)
        video_cthw = video_tchw.permute(1, 0, 2, 3).contiguous()
        return (video_cthw - self.image_mean) / self.image_std

    def prepare_batch(self, *, batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        return batch


class WanVideoDataset(Dataset):
    def __init__(self, config: dict[str, Any], processor: WanVideoProcessor):
        self.config = config
        self.processor = processor
        self.dataset_path = Path(config.get("dataset_path", ""))
        self.data_folder = Path(config.get("data_folder", ""))
        self.frame_num = int(config.get("frame_num", 81))
        self.video_backend = str(config.get("video_backend", "imageio")).lower()

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Wan dataset metadata not found: {self.dataset_path}")
        with self.dataset_path.open(encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        if not self.samples:
            raise ValueError(f"Wan dataset metadata is empty: {self.dataset_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_video_path(self, value: str) -> str:
        if value.startswith(("http://", "https://", "file://")):
            return value
        path = Path(value)
        if not path.is_absolute() and self.data_folder:
            path = self.data_folder / path
        return str(path)

    def _sample_video(self, video_tchw: torch.Tensor) -> torch.Tensor:
        total_frames = int(video_tchw.shape[0])
        if total_frames <= 0:
            raise ValueError("Video contains no frames.")
        idx = torch.linspace(0, total_frames - 1, self.frame_num).round().long()
        return video_tchw[idx]

    def _read_video_imageio(self, path: str) -> torch.Tensor:
        import imageio.v3 as iio

        frames = []
        for frame in iio.imiter(path):
            image = Image.fromarray(frame).convert("RGB")
            frames.append(torch.from_numpy(np.asarray(image).copy()))
        if not frames:
            raise ValueError(f"Video contains no frames: {path}")
        return torch.stack(frames).permute(0, 3, 1, 2)

    def _read_video_decord(self, path: str) -> torch.Tensor:
        import decord

        vr = decord.VideoReader(path)
        total_frames = len(vr)
        idx = torch.linspace(0, total_frames - 1, self.frame_num).round().long().tolist()
        return torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)

    def _read_video(self, path: str) -> torch.Tensor:
        if self.video_backend == "imageio":
            video = self._sample_video(self._read_video_imageio(path))
        elif self.video_backend == "decord":
            video = self._read_video_decord(path)
        else:
            video = fetch_video(
                {
                    "video": path,
                    "nframes": self.frame_num,
                    "resized_height": self.processor.height,
                    "resized_width": self.processor.width,
                }
            )
        return self.processor.normalize_video(video)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        prompt = sample.get("prompt") or sample.get("text") or sample.get("caption") or ""
        video_key = sample.get("video") or sample.get("video_path")
        if not video_key:
            raise ValueError(f"Wan dataset sample missing `video`: index={index}")

        item = self.processor.tokenize(str(prompt))
        item["video"] = self._read_video(self._resolve_video_path(str(video_key)))
        if "seed" in sample:
            item["seed"] = int(sample["seed"])
        return item

    @staticmethod
    def get_collator():
        def collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
            batch = {
                "video": torch.stack([sample["video"] for sample in samples]),
                "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
                "attention_mask": torch.stack([sample["attention_mask"] for sample in samples]),
            }
            if any("seed" in sample for sample in samples):
                batch["seed"] = torch.tensor([sample.get("seed", 0) for sample in samples], dtype=torch.long)
            return batch

        return collate


def build_wan_dataset(config: dict[str, Any]):
    processor = WanVideoProcessor(config.get("processor_config", {}) or {})
    dataset = WanVideoDataset(config, processor)
    return dataset, processor
