###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic Wan 2.2 T2V dataloader.

A Primus-side ``@dataclass`` with a ``build()`` method that the YAML
``data.dataloader._target_`` can point at, so a run needs no dataset on disk and
no edit to AutoModel. Useful for bring-up and for measuring the model path
without a data pipeline in front of it.

Sample keys match what ``collate_fn_video(model_type="wan")`` expects:
  - ``video_latents``     [1, C, T', H/8, W/8]  fp16 latents
  - ``text_embeddings``   [1, S, D]             bf16 conditioning
  - ``bucket_resolution`` / ``aspect_ratio``    (W, H) in pixels and W/H, as the
    real cache records them

Defaults describe 512x512x81 video after the Wan VAE (8x spatial, 4x temporal),
i.e. a (16, 21, 64, 64) latent; override any of them from YAML.

Samples are deterministic given ``seed``, so two runs see identical data and a
change in step time is attributable to the code rather than the input.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
from nemo_automodel.components.datasets.diffusion.collate_fns import collate_fn_video
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# 512x512x81 after the Wan VAE (8x spatial, 4x temporal), before 2x2 patchify.
DEFAULT_FRAMES = 81
DEFAULT_LATENT_T = 1 + (DEFAULT_FRAMES - 1) // 4  # 21
DEFAULT_LATENT_H = 512 // 8  # 64
DEFAULT_LATENT_W = 512 // 8  # 64
DEFAULT_IN_CHANNELS = 16
DEFAULT_TEXT_SEQ = 226  # UMT5 max for Wan (trim + re-pad)
DEFAULT_TEXT_DIM = 4096  # UMT5-XXL hidden size
# Wan patchifies the latent 1x2x2, so the latent height and width must be even.
_SPATIAL_PATCH = 2
_VAE_SPATIAL_FACTOR = 8


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
        for name, value in (
            ("num_samples", num_samples),
            ("in_channels", in_channels),
            ("latent_t", latent_t),
            ("latent_h", latent_h),
            ("latent_w", latent_w),
            ("text_seq_len", text_seq_len),
            ("text_dim", text_dim),
        ):
            if int(value) < 1:
                raise ValueError(f"SyntheticWanDataset: {name} must be positive, got {value}")
        for name, value in (("latent_h", latent_h), ("latent_w", latent_w)):
            if int(value) % _SPATIAL_PATCH:
                raise ValueError(
                    f"SyntheticWanDataset: {name}={value} must be even, since Wan patchifies the "
                    f"latent {_SPATIAL_PATCH}x{_SPATIAL_PATCH} spatially"
                )
        self.num_samples = int(num_samples)
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
        self._cache: Dict[int, Dict[str, Any]] = {}
        # Drawn here from the seed alone, so every worker, rank and epoch gets the
        # same tensor whichever index it reads first.
        self._shared_text: torch.Tensor | None = None
        if self.share_text_features:
            self._shared_text = (
                torch.randn(
                    1,
                    self.text_seq_len,
                    self.text_dim,
                    generator=torch.Generator().manual_seed(self.seed),
                )
                * self.text_scale
            ).to(torch.bfloat16)
        # The real cache records (width, height) in pixels and width / height.
        self._bucket_resolution = (
            self.latent_w * _VAE_SPATIAL_FACTOR,
            self.latent_h * _VAE_SPATIAL_FACTOR,
        )
        self._aspect_ratio = self.latent_w / self.latent_h

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        # Seeded per index, not per epoch, so sample i is the same tensor on every
        # rank and in every run.
        g = torch.Generator().manual_seed(self.seed + idx)
        video_latents = (
            torch.randn(
                1,
                self.in_channels,
                self.latent_t,
                self.latent_h,
                self.latent_w,
                generator=g,
            )
            * self.latent_scale
        )

        if self._shared_text is not None:
            # The collate concatenates into a new tensor, so sharing one is safe.
            text_embeddings = self._shared_text
        else:
            text_embeddings = (
                torch.randn(1, self.text_seq_len, self.text_dim, generator=g) * self.text_scale
            ).to(torch.bfloat16)

        sample = {
            "video_latents": video_latents.to(torch.float16),
            "text_embeddings": text_embeddings,
            "bucket_resolution": torch.tensor(self._bucket_resolution),
            "aspect_ratio": self._aspect_ratio,
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
            _target_: primus.backends.automodel.models.wan.data.synthetic.SyntheticWanDataloaderConfig
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

        def collate(batch):
            return collate_fn_video(batch, model_type="wan")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            # Sampler and shuffle are mutually exclusive in DataLoader.
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=collate,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        if len(dataloader) == 0:
            raise ValueError(
                f"SyntheticWanDataloaderConfig: num_samples={self.num_samples} gives no batch of "
                f"{batch_size} on each of {dp_world_size} rank(s) with drop_last={self.drop_last}"
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
