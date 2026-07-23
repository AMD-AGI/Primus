###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic Ideogram-4 diffusion dataloader (no-fork, Primus-side).

WHY (no Automodel/diffusers fork):
  The AutoModel diffusion recipe resolves ``data.dataloader`` through a *closed*
  registry keyed by the builder's dotted path
  (``RecipeConfig.resolve_diffusion_dataloader``: text-to-image / text-to-video /
  meta-files / mock). Any OTHER ``_target_`` falls back to the resolved target
  itself and is accepted **iff it is a ``@dataclass`` exposing ``build()``** with
  the :class:`DiffusionDataloaderConfig` signature. So this module ships exactly
  that — a dataclass the YAML can point at directly — and needs no submodule edit.

WHAT it produces (the batch the :class:`Ideogram4Adapter` consumes, matching the
eventual Flux-2 VAE + Qwen3-VL preprocessor cache):
  - ``image_latents``   ``[B, in_channels, grid_h, grid_w]``  clean packed latents (x0)
  - ``llm_features``    ``[B, max_text_tokens, llm_features_dim]``  Qwen3-VL-like feats
  - ``text_lengths``    ``[B]`` int  real (non-pad) text token count per sample
  - ``data_type``       ``"image"``

  The clean latents are what ``FlowMatchingPipeline.step`` noises (it reads
  ``image_latents`` as x0, samples sigma, forms ``x_t=(1-σ)x0+σε`` and target
  ``v=ε-x0`` in this 128-dim packed space); the adapter packs/prepends the
  text region and builds the ``[pad][text][image]`` position/segment/indicator
  ids internally, so the loader only emits these raw tensors.

FIXED (deterministic) dataset for an overfit smoke:
  Each index generates the SAME tensors every epoch (per-index seeded RNG), and
  each sample gets a DISTINCT ``llm_features`` fingerprint. With fresh noise per
  step the per-step loss is noisy, but because ``v=(x_t-x0)/σ`` is recoverable
  from the (fixed, distinct) conditioning the model memorises x0 per sample and
  the training loss trends clearly DOWN — the signal Phase B.5 checks. Features
  are scaled by ``feature_scale`` (real tapped hidden states are ~unit-norm
  per-dim, not a raw 53248-wide unit-variance vector that blows up the input
  projection).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

logger = logging.getLogger(__name__)


class SyntheticIdeogram4Dataset(Dataset):
    """Deterministic random Ideogram-4 samples (image latents + LLM features).

    Per-index seeded so the dataset is FIXED across epochs/workers/ranks, which is
    what makes the overfit loss-decrease signal meaningful.
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
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Perf mode: generate each sample once and reuse (so throughput runs measure
        # the model step, not per-__getitem__ randn of the big 53248-dim features).
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        # Per-index generator -> identical sample every epoch (fixed dataset).
        gen = torch.Generator().manual_seed(self.seed + int(idx))

        image_latents = self.latent_scale * torch.randn(
            self.in_channels, self.grid_h, self.grid_w, dtype=torch.float32, generator=gen
        )
        # Full [max_text, dim] features; pad-region positions are masked out by the
        # adapter's segment_ids, so only the real-text region is attended.
        llm_features = self.feature_scale * torch.randn(
            self.max_text_tokens, self.llm_features_dim, dtype=torch.float32, generator=gen
        )
        # Deterministic per-sample real-text length in [min_text, max_text].
        span = self.max_text_tokens - self.min_text_tokens + 1
        text_len = self.min_text_tokens + (int(idx) % span)

        sample = {
            "image_latents": image_latents,
            "llm_features": llm_features,
            "text_lengths": torch.tensor(text_len, dtype=torch.long),
        }
        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


def _collate_synthetic_ideogram4(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack per-sample tensors into the FlowMatchingPipeline batch format."""
    return {
        "image_latents": torch.stack([b["image_latents"] for b in batch], dim=0),
        "llm_features": torch.stack([b["llm_features"] for b in batch], dim=0),
        "text_lengths": torch.stack([b["text_lengths"] for b in batch], dim=0),
        "data_type": "image",
    }


@dataclass
class SyntheticIdeogram4DataloaderConfig:
    """Construction-time config for the synthetic Ideogram-4 dataloader.

    Selected in YAML via::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.ideogram_synthetic_data.SyntheticIdeogram4DataloaderConfig
            num_samples: 64
            grid_h: 16
            grid_w: 16
            max_text_tokens: 32

    Every field here must be a plain YAML scalar (the recipe validates the YAML
    keys against these dataclass fields). The runtime ``dp_rank`` /
    ``dp_world_size`` / ``batch_size`` are passed to :meth:`build`, not fields.
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
    # Perf: cache generated samples in-process (use with num_workers=0 so the cache
    # persists across epochs) to isolate model throughput from feature generation.
    cache_in_memory: bool = False

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        """Build the synthetic dataset, per-rank sampler, and dataloader."""
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
            collate_fn=_collate_synthetic_ideogram4,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[SyntheticIdeogram4] %d samples, in_ch=%d grid=%dx%d max_text=%d feat_dim=%d "
            "(dp_rank=%d/%d, batch_size=%d, %d batches/rank)",
            len(dataset),
            self.in_channels,
            self.grid_h,
            self.grid_w,
            self.max_text_tokens,
            self.llm_features_dim,
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
