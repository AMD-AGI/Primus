###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
"""Synthetic Wan 2.2 T2V dataloader for throughput measurement.

Mirrors ``ideogram4/data/synthetic.py``: a Primus-side ``@dataclass`` with ``build()``
that the YAML ``data.dataloader._target_`` can point at without editing Automodel.

Batch keys match ``collate_fn_video`` / ``FlowMatchingPipeline.step``:
  - ``video_latents``    [B, 16, T', H/8, W/8]  fp16/bf16 latents (x0)
  - ``text_embeddings``  [B, 226, 4096]         UMT5-like bf16 conditioning
  - ``data_type``        ``"video"``

Fixed shape for this study: 512×512×81 → latent (16, 21, 64, 64), 21,504 patch tokens.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.collate_fns import collate_fn_video
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

logger = logging.getLogger(__name__)

# 512×512×81 after Wan VAE (8× spatial, 4× temporal) and before 2×2 patchify.
DEFAULT_FRAMES = 81
DEFAULT_LATENT_T = 1 + (DEFAULT_FRAMES - 1) // 4  # 21
DEFAULT_LATENT_H = 512 // 8  # 64
DEFAULT_LATENT_W = 512 // 8  # 64
DEFAULT_IN_CHANNELS = 16
DEFAULT_TEXT_SEQ = 226  # UMT5 max for Wan (trim + re-pad)
DEFAULT_TEXT_DIM = 4096  # UMT5-XXL hidden size


class SyntheticWanDataset(Dataset):
    """Deterministic random Wan video samples."""

    def __init__(
        self,
        *,
        num_samples: int,
        in_channels: int = DEFAULT_IN_CHANNELS,
        latent_t: int = DEFAULT_LATENT_T,
        latent_h: int = DEFAULT_LATENT_H,
        latent_w: int = DEFAULT_LATENT_W,
        text_seq_len: int = DEFAULT_TEXT_SEQ,
        text_dim: int = DEFAULT_TEXT_DIM,
        latent_scale: float = 1.0,
        text_scale: float = 0.02,
        seed: int = 42,
        cache_in_memory: bool = False,
        share_text_features: bool = False,
    ) -> None:
        self.num_samples = max(int(num_samples), 1)
        self.in_channels = int(in_channels)
        self.latent_t = int(latent_t)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.text_seq_len = int(text_seq_len)
        self.text_dim = int(text_dim)
        self.latent_scale = float(latent_scale)
        self.text_scale = float(text_scale)
        self.seed = int(seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.share_text_features = bool(share_text_features)
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._shared_text: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        g = torch.Generator().manual_seed(self.seed + idx)
        video_latents = torch.randn(
            1, self.in_channels, self.latent_t, self.latent_h, self.latent_w,
            generator=g,
        ) * self.latent_scale

        if self.share_text_features:
            if self._shared_text is None:
                self._shared_text = torch.randn(
                    1, self.text_seq_len, self.text_dim, generator=g,
                ) * self.text_scale
            text_embeddings = self._shared_text.clone()
        else:
            text_embeddings = torch.randn(
                1, self.text_seq_len, self.text_dim, generator=g,
            ) * self.text_scale

        sample = {
            "video_latents": video_latents.to(torch.float16),
            "text_embeddings": text_embeddings.to(torch.bfloat16),
            "bucket_resolution": torch.tensor([512, 512]),
            "aspect_ratio": 1.0,
        }
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


@dataclass
class SyntheticWanDataloaderConfig:
    """Construction-time config for the synthetic Wan dataloader.

    YAML example::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.models.wan.data.synthetic.SyntheticWanDataloaderConfig
            num_samples: 512
            share_text_features: true
    """

    num_samples: int = 512
    in_channels: int = DEFAULT_IN_CHANNELS
    latent_t: int = DEFAULT_LATENT_T
    latent_h: int = DEFAULT_LATENT_H
    latent_w: int = DEFAULT_LATENT_W
    text_seq_len: int = DEFAULT_TEXT_SEQ
    text_dim: int = DEFAULT_TEXT_DIM
    latent_scale: float = 1.0
    text_scale: float = 0.02
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    cache_in_memory: bool = False
    share_text_features: bool = True

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        dataset = SyntheticWanDataset(
            num_samples=self.num_samples,
            in_channels=self.in_channels,
            latent_t=self.latent_t,
            latent_h=self.latent_h,
            latent_w=self.latent_w,
            text_seq_len=self.text_seq_len,
            text_dim=self.text_dim,
            latent_scale=self.latent_scale,
            text_scale=self.text_scale,
            seed=self.seed,
            cache_in_memory=self.cache_in_memory,
            share_text_features=self.share_text_features,
        )
        sampler = None
        if dp_world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dp_world_size,
                rank=dp_rank,
                shuffle=self.shuffle,
                drop_last=self.drop_last,
            )
        collate = lambda batch: collate_fn_video(batch, model_type="wan")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[SyntheticWan] %d samples, latent=%dx%dx%dx%d text=%dx%d "
            "(dp_rank=%d/%d, batch_size=%d, %d batches/rank)",
            len(dataset),
            self.in_channels,
            self.latent_t,
            self.latent_h,
            self.latent_w,
            self.text_seq_len,
            self.text_dim,
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
