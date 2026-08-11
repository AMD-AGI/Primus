###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real (pre-encoded) Ideogram-4 cache dataloader (no-fork, Primus-side).

Reads the flat per-sample cache produced by :class:`Ideogram4Processor`
(``processor.py``) and emits the exact batch the
:class:`Ideogram4Adapter` + ``FlowMatchingPipeline`` consume — the same contract as
``SyntheticIdeogram4DataloaderConfig``, but with REAL Flux-2 VAE latents + Qwen3-VL
features:

  - ``image_latents``  ``[B, 128, gh, gw]``  packed+BN latents (x0)
  - ``llm_features``   ``[B, Tmax, 53248]``  LEFT-padded Qwen3-VL feats
  - ``text_lengths``   ``[B]``               real (non-pad) token count per sample
  - ``data_type``      ``"image"``

Left-padding matches the adapter/pipeline ``[left-pad][text][image]`` layout: the
real ``n`` tokens occupy the LAST ``n`` rows (positions ``[Tmax-n : Tmax]``), which is
exactly the region ``_prepare_ids`` marks as text (``offset = Tmax - n``).

``Tmax`` is a **dataset-level constant** by default (the longest caption in the cache,
read from ``metadata.json``), NOT the per-batch maximum. The packed sequence length
``S = Tmax + gh*gw`` therefore does not move between steps. That is a requirement of
per-layer ``torch.compile``, which keys its compiled graphs on input shapes: padding to
the per-batch maximum makes ``S`` jump whenever a batch's longest caption differs, which
recompiles the graph and blocks compile on real data no matter how ``cu_seqlens`` is
built. Configure via ``max_text_tokens`` (0 = derive, -1 = legacy per-batch).

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
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

logger = logging.getLogger(__name__)

# Collate runs in worker processes, so this only suppresses repeats within a worker --
# enough to keep the log readable without cross-process coordination.
_WARNED_KEYS: set = set()


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
        # Longest caption in the WHOLE cache, from metadata alone (no sample loads). Every
        # rank reads the same metadata.json, so this is identical across ranks -- which
        # matters, because a per-rank text width would desync the sharded shapes.
        self.max_text_length = max(int(s["text_length"]) for s in self.samples)

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


def _collate_ideogram4_cache(
    batch: List[Dict[str, torch.Tensor]], text_width: int = 0
) -> Dict[str, object]:
    """Stack latents; LEFT-pad variable-length ``llm_features`` to a fixed width.

    ``text_width`` is the padded text region size. Pass a **dataset-level constant** (the
    longest caption in the cache) so the packed sequence length ``S = text_width + n_img``
    is identical on every step: ``torch.compile`` keys its graphs on input shapes, so
    padding to the per-BATCH maximum instead (``text_width=0``) recompiles whenever a batch
    happens to contain a different longest caption. That alone blocks compile on real data,
    regardless of how ``cu_seqlens`` is produced.

    ``text_width=0`` preserves the original per-batch behaviour for callers that do not
    care about compile.

    Captions longer than ``text_width`` are truncated to their FIRST ``text_width`` tokens
    and warned about once -- reaching that branch means the width was set too small by hand,
    since the derived default cannot be exceeded.
    """
    image_latents = torch.stack([b["image_latents"] for b in batch], dim=0)  # [B,128,gh,gw]

    feats = [b["llm_features"] for b in batch]
    dim = feats[0].shape[-1]
    t_max = int(text_width) if text_width else max(int(f.shape[0]) for f in feats)

    padded = feats[0].new_zeros(len(batch), t_max, dim)
    text_lengths = torch.empty(len(batch), dtype=torch.long)
    for i, f in enumerate(feats):
        n = int(f.shape[0])
        if n > t_max:
            if "truncated" not in _WARNED_KEYS:
                _WARNED_KEYS.add("truncated")
                logger.warning(
                    "[Ideogram4Cache] caption with %d tokens exceeds text_width=%d; truncating "
                    "to the first %d tokens. Raise max_text_tokens to keep full captions.",
                    n,
                    t_max,
                    t_max,
                )
            f = f[:t_max]
            n = t_max
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
            _target_: primus.backends.nemo_automodel.models.ideogram4.data.cache.Ideogram4CacheDataloaderConfig
            cache_dir: /dataset/pcam_ideogram4_256

    Every field must be a plain YAML scalar; runtime ``dp_rank`` / ``dp_world_size`` /
    ``batch_size`` are passed to :meth:`build`.
    """

    cache_dir: str
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 2
    pin_memory: bool = True
    # Padded width of the text region. 0 (default) derives it from the cache: the longest
    # caption present, which is a dataset-level constant and so never truncates. Setting it
    # explicitly caps -- and therefore truncates -- longer captions. Any fixed value keeps
    # the packed sequence length S constant across steps, which is what torch.compile needs;
    # -1 restores the old per-batch padding (variable S, recompiles).
    max_text_tokens: int = 0

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        dataset = Ideogram4CacheDataset(self.cache_dir)

        if self.max_text_tokens < 0:
            text_width = 0  # per-batch max (legacy, shape-unstable)
        elif self.max_text_tokens == 0:
            text_width = dataset.max_text_length
        else:
            text_width = int(self.max_text_tokens)

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
            # partial (not a lambda/closure) so the collate stays picklable for the worker
            # processes when num_workers > 0.
            collate_fn=partial(_collate_ideogram4_cache, text_width=text_width),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[Ideogram4Cache] %d samples from %s (grid=%dx%d, text_width=%s, dp_rank=%d/%d, "
            "bs=%d, %d batches/rank)",
            len(dataset),
            self.cache_dir,
            dataset.grid_h,
            dataset.grid_w,
            text_width if text_width else "per-batch",
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
