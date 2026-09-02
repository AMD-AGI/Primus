###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Ideogram-4 dataloader for the pre-encoded cache.

Reads the flat per-sample cache built offline by ``primus data automodel-cache``
and emits the same batch contract as the synthetic loader, with real latents and
real text features. Because the encoders ran offline, training needs neither
their weights nor their memory.

LEFT-PADDING, which is silent when wrong:
  The real ``n`` tokens occupy the LAST ``n`` rows of the text region, positions
  ``[width - n, width)``. That is exactly the region the adapter marks as text,
  since it computes its offset as ``width - n``. Padding on the right instead
  would put real features where the adapter expects padding and padding where it
  expects text. Nothing errors: the model just trains on conditioning that does
  not line up with its own position ids.

THE TEXT WIDTH, which decides whether the model can be compiled:
  By default the width is a DATASET-level constant -- the longest caption in the
  cache, read from the metadata without loading a single sample. The packed
  sequence length is then the same on every step.

  That matters because ``torch.compile`` keys its compiled graphs on input shapes.
  Padding to the per-BATCH maximum instead makes the sequence length jump whenever
  a batch's longest caption differs from the last one's, and each new length is a
  fresh compilation. On real captions that is most batches, which blocks
  compilation outright no matter how carefully the var-len packing is built.

  Every rank derives the width from the same metadata, so they agree. A per-rank
  width would give the ranks different sharded shapes, which is a much worse
  failure than a slow one.

Cache layout, as written by the builder:

    <cache_dir>/metadata.json     an index, plus the grid and feature dimensions
    <cache_dir>/samples/<i>.pt    one sample: latents, features, text length
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# Collate runs in the worker processes, so this suppresses repeats within a worker
# only. That is enough to keep the log readable without coordinating across
# processes for a warning.
_WARNED: set = set()


class Ideogram4CacheDataset(Dataset):
    """Reads pre-encoded samples from a cache directory."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"no metadata.json in {self.cache_dir}. Build the cache with "
                "'primus data automodel-cache --model ideogram4' and point cache_dir at "
                "its output directory."
            )
        with open(metadata_path, "r") as handle:
            self.metadata = json.load(handle)

        self.samples: List[Dict] = self.metadata["samples"]
        if not self.samples:
            raise ValueError(
                f"the cache at {self.cache_dir} has no samples. Its metadata was written, "
                "so the build ran but encoded nothing."
            )
        self.grid_h = int(self.metadata.get("grid_h", 0))
        self.grid_w = int(self.metadata.get("grid_w", 0))

        # The longest caption in the whole cache, from the metadata alone. Reading
        # it here rather than per batch is what makes the text width a constant.
        self.max_text_length = max(int(s["text_length"]) for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.samples[idx]
        sample_path = (self.cache_dir / entry["cache_file"]).resolve()
        # The metadata is data, and a cache directory can come from anywhere, so a
        # relative path in it must not be able to reach outside the cache.
        try:
            sample_path.relative_to(self.cache_dir)
        except ValueError as exc:
            raise ValueError(
                f"the cache entry for sample {idx} points at {sample_path}, which is "
                f"outside {self.cache_dir}"
            ) from exc

        # weights_only, because a cache file is untrusted input in exactly the way a
        # checkpoint from elsewhere is.
        data = torch.load(sample_path, map_location="cpu", weights_only=True)
        return {
            "image_latents": data["image_latents"].to(torch.float32),
            "llm_features": data["llm_features"],
            "text_length": int(data["text_length"]),
        }


def _collate(batch: List[Dict[str, torch.Tensor]], text_width: int = 0) -> Dict[str, object]:
    """Stack the latents and LEFT-pad the variable-length features to a fixed width.

    ``text_width`` of 0 means pad to the per-batch maximum, which makes the packed
    sequence length vary between steps. See the module docstring for why that
    blocks compilation; the config's default avoids it.

    A caption longer than an explicitly configured width is truncated to its first
    ``text_width`` tokens, with one warning. Reaching that branch means the width
    was set by hand and set too small, since the derived default is by construction
    large enough for every caption in the cache.
    """
    image_latents = torch.stack([b["image_latents"] for b in batch], dim=0)

    features = [b["llm_features"] for b in batch]
    feature_dim = features[0].shape[-1]
    width = int(text_width) if text_width else max(int(f.shape[0]) for f in features)

    padded = features[0].new_zeros(len(batch), width, feature_dim)
    text_lengths = torch.empty(len(batch), dtype=torch.long)
    for row, feature in enumerate(features):
        length = int(feature.shape[0])
        if length > width:
            if "truncated" not in _WARNED:
                _WARNED.add("truncated")
                logger.warning(
                    "[Ideogram4Cache] a caption has %d tokens but max_text_tokens is set "
                    "to %d, so it is truncated to its first %d. Raise max_text_tokens, or "
                    "leave it at 0 to derive a width that fits every caption.",
                    length,
                    width,
                    width,
                )
            feature = feature[:width]
            length = width
        # Left-pad: the real tokens go in the LAST `length` rows.
        padded[row, width - length :] = feature
        text_lengths[row] = length

    return {
        "image_latents": image_latents,
        "llm_features": padded,
        "text_lengths": text_lengths,
        "data_type": "image",
    }


@dataclass
class Ideogram4CacheDataloaderConfig:
    """Construction-time config, selected from YAML by dotted path::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.models.ideogram4.data.cache.Ideogram4CacheDataloaderConfig
            cache_dir: /path/to/the/cache

    Every field has to be a plain YAML scalar, because the recipe validates the
    config keys against these fields. The runtime values are arguments to
    :meth:`build`.
    """

    cache_dir: str
    shuffle: bool = True
    drop_last: bool = True
    num_workers: int = 2
    pin_memory: bool = True

    # The padded width of the text region, which is what fixes the packed sequence
    # length. Three meanings:
    #   0   derive it from the cache: the longest caption present. A dataset-level
    #       constant, so the sequence length never moves and nothing is ever
    #       truncated. This is what you want.
    #   > 0 use this width, truncating any longer caption. Also gives a constant
    #       sequence length, and is the way to trade caption tail for sequence
    #       length deliberately.
    #   -1  pad to the per-batch maximum. Less padding on batches of short
    #       captions, at the cost of a sequence length that changes between steps
    #       -- which means the model cannot be compiled. An explicit choice, not a
    #       default, because the cost is invisible until compilation is switched on
    #       and then only shows up as a run that never stops recompiling.
    max_text_tokens: int = 0

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        dataset = Ideogram4CacheDataset(self.cache_dir)

        if self.max_text_tokens < 0:
            text_width = 0
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
            # partial rather than a closure, so the collate stays picklable for the
            # worker processes.
            collate_fn=partial(_collate, text_width=text_width),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        logger.info(
            "[Ideogram4Cache] %d samples from %s (grid=%dx%d, text width %s, rank %d/%d, "
            "batch %d, %d batches per rank)",
            len(dataset),
            self.cache_dir,
            dataset.grid_h,
            dataset.grid_w,
            text_width if text_width else "per batch",
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        if not text_width:
            logger.warning(
                "[Ideogram4Cache] max_text_tokens=-1 pads to the per-batch maximum, so "
                "the packed sequence length changes between steps. The model cannot be "
                "compiled in this configuration; set max_text_tokens to 0 to derive a "
                "constant width from the cache."
            )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
