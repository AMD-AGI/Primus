###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic Ideogram-4 dataloader.

WHY A DATACLASS AND NOT A REGISTRY ENTRY:
  The AutoModel diffusion recipe resolves ``data.dataloader`` through a closed
  registry keyed by the builder's dotted path. Anything not in that registry falls
  through to the target itself, and is accepted if and only if it is a dataclass
  exposing ``build()`` with the expected signature. So this module ships exactly
  that: a dataclass the YAML can point at directly, needing no edit to the
  submodule. Every field has to be a plain YAML scalar, because the recipe
  validates the config keys against the dataclass fields.

WHAT IT IS FOR:
  Training the model end to end with no dataset, no encoder weights and no
  network. That makes it the thing to reach for when the question is "is the
  training path wired up correctly", separately from "is the data any good".

WHY THE DATASET IS FIXED RATHER THAN RANDOM PER STEP:
  Each index generates the same tensors every epoch, from a per-index seed, and
  each sample gets its own distinct conditioning. That turns the loader into an
  overfit test with a signal worth reading: the pipeline draws fresh noise every
  step, so the per-step loss is noisy, but the velocity target is recoverable from
  the (fixed, distinct) conditioning, so a correctly wired model memorizes each
  sample and the loss trends down. A dataset that were random per step would give
  a loss that sits flat at the variance of the noise, and both a working model and
  a broken one would look identical.

  This is why ``share_text_features`` defaults off. Sharing one conditioning
  tensor across samples makes the targets mutually contradictory, there is nothing
  left to memorize, and the loss-decrease signal disappears.

The feature scale matters and is not cosmetic: real tapped hidden states are
roughly unit-norm per dimension, so a raw unit-variance vector across the full
feature width would be enormous by comparison and would saturate the input
projection before anything else could be observed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)


class SyntheticIdeogram4Dataset(Dataset):
    """Deterministic synthetic samples: image latents plus text features.

    Per-index seeded, so the dataset is identical across epochs, workers and
    ranks. That is what makes the loss-decrease signal mean anything.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        max_text_tokens: int,
        min_text_tokens: int,
        llm_features_dim: int,
        feature_scale: float,
        latent_scale: float,
        seed: int,
        cache_in_memory: bool = False,
        share_text_features: bool = False,
    ) -> None:
        self.num_samples = max(int(num_samples), 1)
        self.in_channels = int(in_channels)
        self.grid_h = int(grid_h)
        self.grid_w = int(grid_w)
        self.max_text_tokens = int(max_text_tokens)
        self.min_text_tokens = max(1, min(int(min_text_tokens), self.max_text_tokens))
        self.llm_features_dim = int(llm_features_dim)
        self.feature_scale = float(feature_scale)
        self.latent_scale = float(latent_scale)
        self.seed = int(seed)
        self.cache_in_memory = bool(cache_in_memory)
        self.share_text_features = bool(share_text_features)
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._shared_features: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]

        # Per-index generator, so index i yields the same sample forever.
        generator = torch.Generator().manual_seed(self.seed + int(idx))

        image_latents = self.latent_scale * torch.randn(
            self.in_channels,
            self.grid_h,
            self.grid_w,
            dtype=torch.float32,
            generator=generator,
        )

        # The full text width is generated. Positions in the padding region are
        # excluded by the adapter's segment ids, so only the real-text region is
        # ever attended; generating them keeps the shape contract simple.
        if self.share_text_features:
            if self._shared_features is None:
                shared = torch.Generator().manual_seed(self.seed)
                self._shared_features = self.feature_scale * torch.randn(
                    self.max_text_tokens,
                    self.llm_features_dim,
                    dtype=torch.float32,
                    generator=shared,
                )
            llm_features = self._shared_features
        else:
            llm_features = self.feature_scale * torch.randn(
                self.max_text_tokens,
                self.llm_features_dim,
                dtype=torch.float32,
                generator=generator,
            )

        # Deterministic per-sample length, cycling through the configured range, so
        # a batch is ragged in the way real captions are.
        span = self.max_text_tokens - self.min_text_tokens + 1
        text_length = self.min_text_tokens + (int(idx) % span)

        sample = {
            "image_latents": image_latents,
            "llm_features": llm_features,
            "text_lengths": torch.tensor(text_length, dtype=torch.long),
        }
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    return {
        "image_latents": torch.stack([b["image_latents"] for b in batch], dim=0),
        "llm_features": torch.stack([b["llm_features"] for b in batch], dim=0),
        "text_lengths": torch.stack([b["text_lengths"] for b in batch], dim=0),
        "data_type": "image",
    }


@dataclass
class SyntheticIdeogram4DataloaderConfig:
    """Construction-time config, selected from YAML by dotted path::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.models.ideogram4.data.synthetic.SyntheticIdeogram4DataloaderConfig
            num_samples: 64
            grid_h: 16
            grid_w: 16
            max_text_tokens: 32

    The runtime values -- data-parallel rank, world size and batch size -- are
    arguments to :meth:`build`, not fields, because the recipe supplies them.
    """

    num_samples: int = 64
    in_channels: int = 128
    grid_h: int = 16
    grid_w: int = 16
    max_text_tokens: int = 32
    min_text_tokens: int = 28
    llm_features_dim: int = 53248
    feature_scale: float = 0.1
    latent_scale: float = 1.0
    seed: int = 1234
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 0
    pin_memory: bool = True

    # Generate each sample once and hand back the same object afterwards. Only
    # useful with num_workers=0, since a worker process cannot share its cache.
    cache_in_memory: bool = False

    # Reuse one text-feature buffer for every sample. This exists because at a
    # realistic feature width, generating a fresh one per sample costs more CPU
    # than the training step costs GPU, so a run meant to exercise the model ends
    # up measuring the loader instead. It destroys the overfit signal (see the
    # module docstring), so it is off by default and belongs only in runs that are
    # not reading the loss.
    share_text_features: bool = False

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        dataset = SyntheticIdeogram4Dataset(
            num_samples=self.num_samples,
            in_channels=self.in_channels,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            max_text_tokens=self.max_text_tokens,
            min_text_tokens=self.min_text_tokens,
            llm_features_dim=self.llm_features_dim,
            feature_scale=self.feature_scale,
            latent_scale=self.latent_scale,
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

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_collate,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[SyntheticIdeogram4] %d samples, channels=%d grid=%dx%d text=%d-%d "
            "features=%d (rank %d/%d, batch %d, %d batches per rank)",
            len(dataset),
            self.in_channels,
            self.grid_h,
            self.grid_w,
            self.min_text_tokens,
            self.max_text_tokens,
            self.llm_features_dim,
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        if self.share_text_features:
            logger.warning(
                "[SyntheticIdeogram4] share_text_features is on, so every sample has "
                "identical conditioning. The overfit loss-decrease signal is not "
                "meaningful in this configuration."
            )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
