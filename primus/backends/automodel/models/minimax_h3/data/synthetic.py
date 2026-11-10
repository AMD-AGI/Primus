###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Synthetic MiniMax-H3 dataloader.

WHY A DATACLASS AND NOT A REGISTRY ENTRY:
  The AutoModel diffusion recipe accepts a ``data.dataloader._target_`` that is a
  dataclass exposing ``build()``, so this module ships exactly that and needs no
  edit to the submodule. Every field is a plain YAML scalar, because the recipe
  validates the config keys against the dataclass fields. That is why the clip
  shapes are one string (``"768x1344x124,768x1344x243"``) and not a list.

WHAT IT IS FOR:
  Training the model end to end with no dataset, no encoder weights and no network:
  the weight license excludes the US, EU, UK and South Korea, so Q4 trains on
  synthetic data. Per step it costs the same as real latents of the same shape, so
  throughput, memory and scaling carry over; convergence and quality do not.

WHY THE DATASET IS FIXED RATHER THAN RANDOM PER STEP:
  Each index generates the same tensors every epoch, from a per-index seed, with its
  own distinct conditioning. The pipeline draws fresh noise every step, so the
  per-step loss is noisy, but the velocity target is recoverable from the fixed,
  distinct conditioning: a correctly wired model memorizes each sample and the loss
  trends down. A dataset random per step would sit flat at the variance of the noise,
  and a working model and a broken one would look identical.

CLIP SHAPES:
  Each sample is one clip of ``HxWxFRAMES`` pixels (a side a multiple of 32, a frame
  count of the form 17 * n + 5), cycled over the list by index. The batch shares one
  packed layout, so ``build`` refuses a batch size other than 1.

Sample keys, as ``adapter.py`` consumes them:
  video_latents    ``[1, 24, F, H, W]``  fp16
  audio_latents    ``[1, 2, 32, L]``     fp16, stereo channel, latent channels, length
  text_embeddings  ``[1, T, D]``         bf16, a stand-in for Qwen3-VL-32B hidden states
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from primus.backends.automodel.models.minimax_h3 import layout as layout_lib

logger = logging.getLogger(__name__)

# Width of Qwen3-VL-32B's hidden states, which is what the released model's text
# projection reads.
DEFAULT_TEXT_DIM = 5120
DEFAULT_TEXT_TOKENS = 256
# 768 x 1344 for 124 frames (5.2 s): the model's default canvas and its shortest clip.
DEFAULT_BUCKETS = "768x1344x124"


def parse_buckets(spec: str) -> List[Tuple[int, int, int]]:
    """``"768x1344x124,768x1344x243"`` to ``[(768, 1344, 124), (768, 1344, 243)]``.

    Each is validated through ``latent_geometry``, so a side that is not a multiple
    of 32 or a frame count the video VAE cannot encode fails here, at construction,
    and not at the first forward.
    """
    buckets = []
    for part in str(spec).split(","):
        fields = part.strip().lower().split("x")
        if len(fields) != 3 or not all(f.isdigit() for f in fields):
            raise ValueError(f"a clip shape is HEIGHTxWIDTHxFRAMES, got {part!r} in buckets={spec!r}")
        height, width, frames = (int(f) for f in fields)
        layout_lib.latent_geometry(height, width, frames)
        buckets.append((height, width, frames))
    if not buckets:
        raise ValueError("buckets must list at least one clip shape")
    return buckets


class SyntheticMiniMaxH3Dataset(Dataset):
    """Deterministic random MiniMax-H3 clips."""

    def __init__(
        self,
        *,
        num_samples: int,
        buckets: str = DEFAULT_BUCKETS,
        text_tokens: int = DEFAULT_TEXT_TOKENS,
        text_dim: int = DEFAULT_TEXT_DIM,
        latent_scale: float = 1.0,
        audio_scale: float = 1.0,
        text_scale: float = 0.1,
        seed: int = 42,
    ) -> None:
        for name, value in (
            ("num_samples", num_samples),
            ("text_tokens", text_tokens),
            ("text_dim", text_dim),
        ):
            if int(value) < 1:
                raise ValueError(f"SyntheticMiniMaxH3Dataset: {name} must be positive, got {value}")
        self.num_samples = int(num_samples)
        self.buckets = parse_buckets(buckets)
        self.text_tokens = int(text_tokens)
        self.text_dim = int(text_dim)
        self.latent_scale = float(latent_scale)
        self.audio_scale = float(audio_scale)
        self.text_scale = float(text_scale)
        self.seed = int(seed)
        self._constants = layout_lib.constants()

    def __len__(self) -> int:
        return self.num_samples

    def bucket_of(self, idx: int) -> Tuple[int, int, int]:
        return self.buckets[idx % len(self.buckets)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        c = self._constants
        height, width, frames = self.bucket_of(idx)
        geometry = layout_lib.latent_geometry(height, width, frames, c)
        # Seeded per index, not per epoch, so sample i is the same tensor on every
        # rank and in every run.
        g = torch.Generator().manual_seed(self.seed + idx)
        video = torch.randn(
            1,
            c.video_latent_channels,
            geometry.latent_frames,
            geometry.latent_height,
            geometry.latent_width,
            generator=g,
        )
        audio = torch.randn(
            1, c.audio_channels, c.audio_latent_channels, geometry.num_audio_latents, generator=g
        )
        text = torch.randn(1, self.text_tokens, self.text_dim, generator=g)
        return {
            "video_latents": (video * self.latent_scale).to(torch.float16),
            "audio_latents": (audio * self.audio_scale).to(torch.float16),
            "text_embeddings": (text * self.text_scale).to(torch.bfloat16),
        }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {key: torch.cat([item[key] for item in batch], dim=0) for key in batch[0]}
    out["data_type"] = "video"
    return out


@dataclass
class SyntheticMiniMaxH3DataloaderConfig:
    """Construction-time config for the synthetic MiniMax-H3 dataloader.

    YAML example::

        data:
          dataloader:
            _target_: primus.backends.automodel.models.minimax_h3.data.synthetic.SyntheticMiniMaxH3DataloaderConfig
            num_samples: 64
            buckets: "768x1344x124"
    """

    num_samples: int = 64
    buckets: str = DEFAULT_BUCKETS
    text_tokens: int = DEFAULT_TEXT_TOKENS
    text_dim: int = DEFAULT_TEXT_DIM
    latent_scale: float = 1.0
    audio_scale: float = 1.0
    text_scale: float = 0.1
    seed: int = 42
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 2
    pin_memory: bool = True

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        if batch_size != 1:
            raise ValueError(
                "MiniMax-H3 trains at micro-batch 1: the batch shares one packed layout and one "
                f"timestep table. Got batch_size={batch_size}; set step_scheduler.local_batch_size: 1 "
                "and raise global_batch_size for gradient accumulation."
            )
        dataset = SyntheticMiniMaxH3Dataset(
            num_samples=self.num_samples,
            buckets=self.buckets,
            text_tokens=self.text_tokens,
            text_dim=self.text_dim,
            latent_scale=self.latent_scale,
            audio_scale=self.audio_scale,
            text_scale=self.text_scale,
            seed=self.seed,
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
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            # Sampler and shuffle are mutually exclusive in DataLoader.
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        if len(dataloader) == 0:
            raise ValueError(
                f"SyntheticMiniMaxH3DataloaderConfig: num_samples={self.num_samples} gives no batch on "
                f"each of {dp_world_size} rank(s) with drop_last={self.drop_last}"
            )
        logger.info(
            "[SyntheticMiniMaxH3] %d samples, clip shapes %s, text=%dx%d (dp_rank=%d/%d, %d batches/rank)",
            len(dataset),
            ", ".join("x".join(map(str, b)) for b in dataset.buckets),
            self.text_tokens,
            self.text_dim,
            dp_rank,
            dp_world_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
