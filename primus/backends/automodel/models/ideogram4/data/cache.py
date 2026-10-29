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
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# What the loader needs from metadata.json, beyond the sample list.
_REQUIRED_METADATA = ("grid_h", "grid_w", "in_channels", "llm_features_dim")


def _dataloader_build(**kwargs):
    """AutoModel's build result, imported here so the module loads without AutoModel."""
    from nemo_automodel.components.datasets.diffusion.loader import (
        DiffusionDataloaderBuild,
    )

    return DiffusionDataloaderBuild(**kwargs)


class Ideogram4CacheDataset(Dataset):
    """Reads pre-encoded samples from a cache directory."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        metadata_path = self.cache_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"no metadata.json in {self.cache_dir}. Build the cache with "
                "'bash ./runner/primus-cli direct --single -- data automodel-cache "
                "--model ideogram4' and point cache_dir at "
                "its output directory."
            )
        with open(metadata_path, "r") as handle:
            self.metadata = json.load(handle)

        if not isinstance(self.metadata, dict) or not isinstance(self.metadata.get("samples"), list):
            raise ValueError(f"{metadata_path} has no 'samples' list, so it is not a cache index")
        self.samples: List[Dict] = self.metadata["samples"]
        if not self.samples:
            raise ValueError(
                f"the cache at {self.cache_dir} has no samples. Its metadata was written, "
                "so the build ran but encoded nothing."
            )
        model_type = self.metadata.get("model_type", "ideogram4")
        if model_type != "ideogram4":
            raise ValueError(f"the cache at {self.cache_dir} was built for {model_type!r}, not ideogram4")
        missing = [key for key in _REQUIRED_METADATA if key not in self.metadata]
        if missing:
            raise ValueError(f"{metadata_path} is missing {', '.join(missing)}; rebuild the cache")
        self.grid_h = int(self.metadata["grid_h"])
        self.grid_w = int(self.metadata["grid_w"])
        self.in_channels = int(self.metadata["in_channels"])
        self.llm_features_dim = int(self.metadata["llm_features_dim"])

        # The longest caption in the whole cache, from the metadata alone. Reading
        # it here rather than per batch is what makes the text width a constant;
        # each sample is checked against its entry when it is read.
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
        latents, features = data["image_latents"], data["llm_features"]
        # The text width is derived from the metadata, so a sample that disagrees
        # with its entry would be cut or misplaced rather than fail. A hand-edited
        # index, or files from another build, are the usual causes.
        expected = {
            "latent shape": ((self.in_channels, self.grid_h, self.grid_w), tuple(latents.shape)),
            "text rows": (int(entry["text_length"]), int(features.shape[0])),
            "stored text_length": (int(entry["text_length"]), int(data["text_length"])),
            "feature width": (self.llm_features_dim, int(features.shape[-1])),
        }
        for what, (want, got) in expected.items():
            if want != got:
                raise ValueError(
                    f"sample {idx} ({sample_path.name}) does not match the cache index: "
                    f"{what} is {got}, the index says {want}. Rebuild the cache."
                )
        return {
            "image_latents": latents.to(torch.float32),
            "llm_features": features,
            "text_length": int(data["text_length"]),
        }


def _collate(batch: List[Dict[str, torch.Tensor]], text_width: int = 0) -> Dict[str, object]:
    """Stack the latents and LEFT-pad the variable-length features to a fixed width.

    ``text_width`` of 0 means pad to the per-batch maximum, which makes the packed
    sequence length vary between steps. See the module docstring for why that
    blocks compilation; the config's default avoids it.

    A caption is never truncated. ``build`` refuses a width below the longest
    caption in the cache, and each sample is checked against the index, so a
    caption longer than the width means the two disagree.
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
            raise ValueError(
                f"a caption has {length} tokens but the text width is {width}; the cache "
                "index and its samples disagree, so rebuild the cache"
            )
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
            _target_: primus.backends.automodel.models.ideogram4.data.cache.Ideogram4CacheDataloaderConfig
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
    #       constant, so the sequence length never moves. This is what you want.
    #   > 0 use this width, which must be at least the longest caption in the
    #       cache; a caption is never truncated. A constant sequence length too,
    #       for matching a width used elsewhere.
    #   -1  pad to the per-batch maximum. Less padding on batches of short
    #       captions, at the cost of a sequence length that changes between steps
    #       -- which means the model cannot be compiled. An explicit choice, not a
    #       default, because the cost is invisible until compilation is switched on
    #       and then only shows up as a run that never stops recompiling.
    max_text_tokens: int = 0

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int):
        if self.max_text_tokens < -1:
            raise ValueError(
                f"max_text_tokens must be 0 (derive from the cache), a positive width, or "
                f"-1 (per batch); got {self.max_text_tokens}"
            )
        dataset = Ideogram4CacheDataset(self.cache_dir)

        if self.max_text_tokens == -1:
            text_width = 0
        elif self.max_text_tokens == 0:
            text_width = dataset.max_text_length
        else:
            text_width = int(self.max_text_tokens)
            if text_width < dataset.max_text_length:
                # Truncating would keep the head of the chat-templated sequence and
                # drop the end of the prompt and the generation-prompt tokens every
                # sample ends with, which is not what the model sees at inference.
                raise ValueError(
                    f"max_text_tokens={text_width} is below the longest caption in the "
                    f"cache ({dataset.max_text_length} tokens). Captions are never "
                    "truncated: raise it, leave it at 0 to derive the width, or rebuild "
                    "the cache with a smaller --max-text-tokens to drop the long ones."
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
            # partial rather than a closure, so the collate stays picklable for the
            # worker processes.
            collate_fn=partial(_collate, text_width=text_width),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )
        if len(dataloader) == 0:
            raise ValueError(
                f"the cache at {self.cache_dir} gives rank {dp_rank} no batch: "
                f"{len(dataset)} samples over {dp_world_size} rank(s) at batch size "
                f"{batch_size} with drop_last={self.drop_last}. It needs at least "
                f"{dp_world_size * batch_size} samples."
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
        return _dataloader_build(dataloader=dataloader, sampler=sampler)
