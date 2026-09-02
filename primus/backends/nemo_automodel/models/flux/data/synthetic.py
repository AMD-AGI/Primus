###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic FLUX.1 diffusion dataloader (no fork, Primus-side).

WHY THIS WORKS WITHOUT A SUBMODULE EDIT:
  The AutoModel diffusion recipe resolves ``data.dataloader`` through a *closed*
  registry keyed by the builder's dotted path
  (``RecipeConfig.resolve_diffusion_dataloader``: text-to-image, text-to-video,
  meta-files, mock). Any other ``_target_`` falls back to the resolved target
  itself and is accepted **iff it is a dataclass exposing ``build()``** with the
  ``DiffusionDataloaderConfig`` signature. This module ships exactly that.

WHY IT EXISTS:
  It removes the input pipeline from the picture. The FLUX cache carries a large
  T5 embedding per sample, so at a large micro-batch a run can end up bound by the
  loader rather than by the GPU. Generating in-process makes a comparison between
  two configurations a comparison of the model rather than of two filesystems.

WHAT IT PRODUCES -- deliberately the POST-COLLATE batch of the real cache path, so
that a synthetic run and a real-cache run hand the model the same thing at matched
shape:

  - ``image_latents``        ``[B, 16, latent_h, latent_w]``  fp16 VAE latents (x0)
  - ``text_embeddings``      ``[B, text_seq_len, 4096]``      bf16 T5 sequence
  - ``pooled_prompt_embeds`` ``[B, 768]``                     bf16 CLIP pooled
  - ``data_type``            ``"image"``

  Those names and dtypes are what ``collate_fn_text_to_image`` emits for FLUX (the
  per-sample cache keys ``prompt_embeds`` / ``pooled_prompt_embeds`` are remapped
  to ``text_embeddings`` / ``pooled_prompt_embeds``), and are what the FLUX adapter
  reads. ``FlowMatchingPipeline.step`` noises ``image_latents`` as x0, and the
  adapter packs 2x2 patches and builds the position ids itself, so this emits raw
  tensors only.

TWO GEOMETRY FACTS THAT ARE EASY TO CONFLATE:
  The cached latent lives at **res/8** (the VAE downsample) with 16 channels. The
  model then packs 2x2, giving 64 channels on a **res/16** grid, and it is that
  res/16 grid which sets the image TOKEN count. So 1024x1024 gives a
  ``[16, 128, 128]`` latent and a 64x64 = 4096-token image sequence. ``latent_h``
  and ``latent_w`` below are res/8, **not** the token grid.

WHY THERE IS NO ``text_lengths`` KEY:
  Some var-len attention paths pack ``[pad][text][image]`` on one axis and skip
  padded text positions, so the real caption length changes the work done and the
  loader has to emit it. FLUX is different in two ways that cancel the question.
  The FLUX cache pads every caption to a fixed ``t5_tokenizer.model_max_length``
  and truncates rather than skipping over-long ones; and the FLUX processor encodes
  without an attention mask, so pad positions carry real, non-zero T5 embeddings
  which the dense joint attention then attends to. The caption-length distribution
  is therefore computationally inert here, and only ``text_seq_len`` matters.

DETERMINISM:
  Each index generates the same tensors every epoch, worker and rank (per-index
  seeded RNG). A synthetic sample carries no learnable signal, so this is for
  shape-and-speed work only -- convergence has to be shown on a real cache.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild
from torch.utils.data import DataLoader, Dataset, DistributedSampler

logger = logging.getLogger(__name__)

# FLUX conditioning widths, fixed by the architecture (transformer/config.json:
# joint_attention_dim, pooled_projection_dim) and by the VAE's latent channel count.
T5_FEATURES_DIM = 4096
CLIP_POOLED_DIM = 768
VAE_LATENT_CHANNELS = 16

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class SyntheticFluxDataset(Dataset):
    """Deterministic random FLUX samples (VAE latents plus T5/CLIP conditioning)."""

    def __init__(
        self,
        *,
        num_samples: int,
        in_channels: int,
        latent_h: int,
        latent_w: int,
        text_seq_len: int,
        t5_features_dim: int,
        pooled_dim: int,
        feature_scale: float,
        latent_scale: float,
        seed: int,
        latent_dtype: torch.dtype,
        text_dtype: torch.dtype,
        cache_in_memory: bool = False,
        share_text_features: bool = False,
        emit_clip_hidden: bool = False,
        clip_hidden_len: int = 77,
    ) -> None:
        self.num_samples = max(int(num_samples), 1)
        self.in_channels = int(in_channels)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.text_seq_len = int(text_seq_len)
        self.t5_features_dim = int(t5_features_dim)
        self.pooled_dim = int(pooled_dim)
        self.feature_scale = float(feature_scale)
        self.latent_scale = float(latent_scale)
        self.seed = int(seed)
        self.latent_dtype = latent_dtype
        self.text_dtype = text_dtype
        self.cache_in_memory = bool(cache_in_memory)
        self.share_text_features = bool(share_text_features)
        self.emit_clip_hidden = bool(emit_clip_hidden)
        self.clip_hidden_len = int(clip_hidden_len)
        self._cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._shared_text: torch.Tensor | None = None
        self._shared_clip_hidden: torch.Tensor | None = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory and idx in self._cache:
            return self._cache[idx]
        # Per-index generator, so the dataset is fixed across epochs and ranks.
        gen = torch.Generator().manual_seed(self.seed + int(idx))

        image_latents = (
            self.latent_scale
            * torch.randn(self.in_channels, self.latent_h, self.latent_w, dtype=torch.float32, generator=gen)
        ).to(self.latent_dtype)

        if self.share_text_features:
            # One [text_seq_len, 4096] buffer for every index. At FLUX's T5 width
            # this tensor is large enough that generating a distinct one per sample
            # dominates the micro-batch on the CPU side, which would make the run
            # measure the dataloader rather than the model. Nothing about speed
            # depends on the samples holding *different* noise, only on the shapes.
            # Never use this for a convergence run.
            if self._shared_text is None:
                shared_gen = torch.Generator().manual_seed(self.seed)
                self._shared_text = (
                    self.feature_scale
                    * torch.randn(
                        self.text_seq_len,
                        self.t5_features_dim,
                        dtype=torch.float32,
                        generator=shared_gen,
                    )
                ).to(self.text_dtype)
            text_embeddings = self._shared_text
        else:
            text_embeddings = (
                self.feature_scale
                * torch.randn(self.text_seq_len, self.t5_features_dim, dtype=torch.float32, generator=gen)
            ).to(self.text_dtype)

        # Pooled CLIP is 768 floats: cheap enough to always generate per sample.
        pooled = (self.feature_scale * torch.randn(self.pooled_dim, dtype=torch.float32, generator=gen)).to(
            self.text_dtype
        )

        sample = {
            "image_latents": image_latents,
            "text_embeddings": text_embeddings,
            "pooled_prompt_embeds": pooled,
        }
        if self.emit_clip_hidden:
            # The real cache also carries clip_hidden [77, 768] and the stock
            # collate passes it through, but the FLUX adapter ignores it. Off by
            # default; available for a synthetic-versus-real control where matching
            # the loader's byte volume is the point.
            if self.share_text_features:
                if self._shared_clip_hidden is None:
                    shared_gen = torch.Generator().manual_seed(self.seed + 1)
                    self._shared_clip_hidden = (
                        self.feature_scale
                        * torch.randn(
                            self.clip_hidden_len,
                            self.pooled_dim,
                            dtype=torch.float32,
                            generator=shared_gen,
                        )
                    ).to(self.text_dtype)
                sample["clip_hidden"] = self._shared_clip_hidden
            else:
                sample["clip_hidden"] = (
                    self.feature_scale
                    * torch.randn(self.clip_hidden_len, self.pooled_dim, dtype=torch.float32, generator=gen)
                ).to(self.text_dtype)

        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


def _collate_synthetic_flux(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack per-sample tensors into the FlowMatchingPipeline batch format.

    Mirrors the key names ``collate_fn_text_to_image`` produces for FLUX, so a
    synthetic run and a real-cache run hand the model the same thing.
    """
    out: Dict[str, object] = {
        "image_latents": torch.stack([b["image_latents"] for b in batch], dim=0),
        "text_embeddings": torch.stack([b["text_embeddings"] for b in batch], dim=0),
        "pooled_prompt_embeds": torch.stack([b["pooled_prompt_embeds"] for b in batch], dim=0),
        "data_type": "image",
    }
    if "clip_hidden" in batch[0]:
        out["clip_hidden"] = torch.stack([b["clip_hidden"] for b in batch], dim=0)
    return out


@dataclass
class SyntheticFluxDataloaderConfig:
    """Construction-time config for the synthetic FLUX dataloader.

    Selected in YAML via::

        data:
          dataloader:
            _target_: primus.backends.nemo_automodel.models.flux.data.synthetic.SyntheticFluxDataloaderConfig
            num_samples: 2048
            latent_h: 128           # res/8 for 1024**2, NOT the res/16 token grid
            latent_w: 128
            text_seq_len: 512
            share_text_features: true

    Every field must be a plain YAML scalar: the recipe validates YAML keys against
    these dataclass fields and an unknown key is a TypeError. The runtime
    ``dp_rank`` / ``dp_world_size`` / ``batch_size`` are arguments to
    :meth:`build`, not fields.
    """

    num_samples: int = 2048
    in_channels: int = VAE_LATENT_CHANNELS
    # res/8: 32 / 64 / 128 / 256 for 256** / 512** / 1024** / 2048**.
    latent_h: int = 128
    latent_w: int = 128
    # The real cache pads every caption to exactly the T5 cap, so a synthetic run
    # must use the same number or the two are not comparable.
    text_seq_len: int = 512
    t5_features_dim: int = T5_FEATURES_DIM
    pooled_dim: int = CLIP_POOLED_DIM
    feature_scale: float = 0.1
    latent_scale: float = 1.0
    seed: int = 1234
    # Match what the upstream FLUX processor writes: fp16 latents, bf16 text.
    # Keeping these equal to the cache's dtypes is what makes a synthetic-versus-
    # real comparison a comparison of the model rather than of a cast.
    latent_dtype: str = "float16"
    text_dtype: str = "bfloat16"
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    cache_in_memory: bool = False
    # See SyntheticFluxDataset: never set for a convergence run.
    share_text_features: bool = False
    emit_clip_hidden: bool = False
    clip_hidden_len: int = 77

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        """Build the synthetic dataset, per-rank sampler, and dataloader."""
        for name in ("latent_dtype", "text_dtype"):
            if getattr(self, name) not in _DTYPES:
                raise ValueError(f"{name}={getattr(self, name)!r} not one of {sorted(_DTYPES)}")

        dataset = SyntheticFluxDataset(
            num_samples=self.num_samples,
            in_channels=self.in_channels,
            latent_h=self.latent_h,
            latent_w=self.latent_w,
            text_seq_len=self.text_seq_len,
            t5_features_dim=self.t5_features_dim,
            pooled_dim=self.pooled_dim,
            feature_scale=self.feature_scale,
            latent_scale=self.latent_scale,
            seed=self.seed,
            latent_dtype=_DTYPES[self.latent_dtype],
            text_dtype=_DTYPES[self.text_dtype],
            cache_in_memory=self.cache_in_memory,
            share_text_features=self.share_text_features,
            emit_clip_hidden=self.emit_clip_hidden,
            clip_hidden_len=self.clip_hidden_len,
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
            collate_fn=_collate_synthetic_flux,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            **({"prefetch_factor": 2} if self.num_workers > 0 else {}),
        )
        # The run's evidence of what the loader actually built. The token grid is
        # derived (latent/2) rather than configured, so a mismatch between the
        # intended resolution and the emitted shapes surfaces here instead of
        # silently changing the sequence length.
        logger.info(
            "[SyntheticFlux] %d samples, latent=%dx%dx%d (token grid %dx%d = %d img tokens) "
            "text=%dx%d pooled=%d latent_dtype=%s text_dtype=%s share_text=%s "
            "(dp_rank=%d/%d, batch_size=%d, %d batches/rank)",
            len(dataset),
            self.in_channels,
            self.latent_h,
            self.latent_w,
            self.latent_h // 2,
            self.latent_w // 2,
            (self.latent_h // 2) * (self.latent_w // 2),
            self.text_seq_len,
            self.t5_features_dim,
            self.pooled_dim,
            self.latent_dtype,
            self.text_dtype,
            self.share_text_features,
            dp_rank,
            dp_world_size,
            batch_size,
            len(dataloader),
        )
        return DiffusionDataloaderBuild(dataloader=dataloader, sampler=sampler)
