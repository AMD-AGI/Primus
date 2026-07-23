###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real (pre-encoded) Ideogram-4 cache dataloader (no-fork, Primus-side).

Reads the flat per-sample cache produced by ``scripts/ideogram4_preprocess.py``
(via :class:`Ideogram4Processor`) and emits the exact batch the
:class:`Ideogram4Adapter` + ``FlowMatchingPipeline`` consume — the same contract as
``SyntheticIdeogram4DataloaderConfig``, but with REAL Flux-2 VAE latents + Qwen3-VL
features:

  - ``image_latents``  ``[B, 128, gh, gw]``  packed+BN latents (x0)
  - ``llm_features``   ``[B, Tmax, 53248]``  LEFT-padded per-batch Qwen3-VL feats
  - ``text_lengths``   ``[B]``               real (non-pad) token count per sample
  - ``data_type``      ``"image"``

Left-padding matches the adapter/pipeline ``[left-pad][text][image]`` layout: the
real ``n`` tokens occupy the LAST ``n`` rows (positions ``[Tmax-n : Tmax]``), which is
exactly the region ``_prepare_ids`` marks as text (``offset = Tmax - n``).

Cache layout (``cache_dir``):
  - ``metadata.json``: ``{"grid_h","grid_w","llm_features_dim","in_channels",
    "samples":[{"cache_file","text_length","prompt"}, ...]}``
  - ``samples/<i>.pt``: ``{image_latents [128,gh,gw], llm_features [n,53248],
    text_length, ...}``
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

logger = logging.getLogger(__name__)


class Ideogram4CacheDataset(Dataset):
    """Reads pre-encoded Ideogram-4 samples ({image_latents, llm_features, text_length})."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Ideogram-4 cache metadata not found: {meta_path}")
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.samples: List[Dict] = self.meta["samples"]
        if not self.samples:
            raise ValueError(f"Ideogram-4 cache is empty: {self.cache_dir}")
        self.grid_h = int(self.meta.get("grid_h", 0))
        self.grid_w = int(self.meta.get("grid_w", 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        cache_file = (self.cache_dir / item["cache_file"]).resolve()
        # Contain path traversal: cache files must live under cache_dir.
        try:
            cache_file.relative_to(self.cache_dir)
        except ValueError as e:  # pragma: no cover
            raise ValueError(f"cache file {cache_file} outside {self.cache_dir}") from e
        data = torch.load(cache_file, map_location="cpu", weights_only=True)
        return {
            "image_latents": data["image_latents"].to(torch.float32),  # [128, gh, gw]
            "llm_features": data["llm_features"],  # [n, 53248] (fp16)
            "text_length": int(data["text_length"]),
        }


def _collate_ideogram4_cache(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack latents; LEFT-pad variable-length llm_features to the per-batch max."""
    image_latents = torch.stack([b["image_latents"] for b in batch], dim=0)  # [B,128,gh,gw]

    feats = [b["llm_features"] for b in batch]
    dim = feats[0].shape[-1]
    t_max = max(int(f.shape[0]) for f in feats)
    padded = feats[0].new_zeros(len(batch), t_max, dim)
    text_lengths = torch.empty(len(batch), dtype=torch.long)
    for i, f in enumerate(feats):
        n = int(f.shape[0])
        padded[i, t_max - n :] = f  # left-pad: real tokens in the LAST n rows
        text_lengths[i] = n

    return {
        "image_latents": image_latents,
        "llm_features": padded,
        "text_lengths": text_lengths,
        "data_type": "image",
    }


@dataclass
class Ideogram4CacheDataloaderConfig:
    """Construction-time config for the real (pre-encoded) Ideogram-4 dataloader.

    Selected in YAML via::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.ideogram_cache_data.Ideogram4CacheDataloaderConfig
            cache_dir: /mnt/m2m_nobackup/datasets/pcam_ideogram4_256

    Every field must be a plain YAML scalar; runtime ``dp_rank`` / ``dp_world_size`` /
    ``batch_size`` are passed to :meth:`build`.
    """

    cache_dir: str
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 2
    pin_memory: bool = True

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        dataset = Ideogram4CacheDataset(self.cache_dir)

        sampler = None
        if dp_world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_ideogram4_cache,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[Ideogram4Cache] %d samples from %s (grid=%dx%d, dp_rank=%d/%d, bs=%d, %d batches/rank)",
            len(dataset),
            self.cache_dir,
            dataset.grid_h,
            dataset.grid_w,
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
