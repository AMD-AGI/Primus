###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Synthetic FLUX.1-dev diffusion dataloader (no-fork, Primus-side).

WHY (no Automodel/diffusers fork):
  The AutoModel diffusion recipe resolves ``data.dataloader`` through a *closed*
  registry keyed by the builder's dotted path
  (``RecipeConfig.resolve_diffusion_dataloader``: text-to-image / text-to-video /
  meta-files / mock). Any OTHER ``_target_`` falls back to the resolved target itself
  and is accepted **iff it is a ``@dataclass`` exposing ``build()``** with the
  :class:`DiffusionDataloaderConfig` signature. So this module ships exactly that — a
  dataclass the YAML can point at directly — and needs no submodule edit. Same
  mechanism the Ideogram-4 synthetic loader uses.

WHY IT EXISTS AT ALL
  Throughput, peak memory and MFU are measured on synthetic data precisely to isolate
  the GPU from the input pipeline. The FLUX cache costs ~4 MB of T5 embedding per
  sample, so at a large micro-batch an end-to-end run can be loader-bound rather than
  GPU-bound (the Ideogram study measured exactly that at 256²). Generating in-process
  removes the disk and lets the search stages compare configurations rather than
  filesystems.

WHAT IT PRODUCES — deliberately the POST-COLLATE batch of the real cache path, so a
synthetic arm and a real arm are comparable at matched shape:

  - ``image_latents``        ``[B, 16, latent_h, latent_w]``  fp16, clean VAE latents (x0)
  - ``text_embeddings``      ``[B, text_seq_len, 4096]``      bf16, T5-XXL sequence
  - ``pooled_prompt_embeds`` ``[B, 768]``                     bf16, CLIP pooled
  - ``data_type``            ``"image"``

  Those key names and dtypes are the ones ``collate_fn_text_to_image`` emits (Automodel
  ``datasets/diffusion/collate_fns.py``: the per-sample cache keys ``prompt_embeds`` /
  ``pooled_prompt_embeds`` are remapped to ``text_embeddings`` / ``pooled_prompt_embeds``),
  and they are what :class:`FluxAdapter` reads. ``FlowMatchingPipeline.step`` noises
  ``image_latents`` as x0; the adapter packs 2x2 patches and builds the img/txt position
  ids itself, so the loader emits raw tensors only.

TWO GEOMETRY FACTS THAT ARE EASY TO CONFLATE
  The cached latent lives at **res/8** (the VAE downsample) with 16 channels. The model
  then packs 2x2, giving 64 channels on a **res/16** grid, and it is that res/16 grid
  which sets the image TOKEN count. So for 1024²: latent [16, 128, 128], 64x64 = 4,096
  image tokens. ``latent_h`` / ``latent_w`` below are res/8, NOT the token grid. This
  lands FLUX on the same token grid as Ideogram-4 at every resolution, which is what
  makes ``img-tok/s/GPU`` comparable between the two studies.

WHY THERE IS NO ``text_lengths`` KEY (unlike the Ideogram loader)
  Ideogram packs ``[pad][text][image]`` on one axis and its var-len attention path skips
  the padded text positions, so the real per-sample caption length changes the work done
  and the loader must emit it. FLUX is different in two ways that cancel the question:
  the FLUX cache pads every caption to a FIXED ``t5_tokenizer.model_max_length`` (512 for
  FLUX.1-dev) and TRUNCATES rather than skipping over-long ones; and the FLUX processor
  encodes without an attention mask, so pad positions carry real, non-zero T5 embeddings
  which the dense joint attention then attends to. The caption-length distribution is
  therefore computationally INERT on this path — only ``text_seq_len`` matters. That is a
  real asymmetry against the Ideogram study rather than a detail, and it belongs in the
  results caveats, not hidden in a loader.

FIXED (deterministic) dataset:
  Each index generates the SAME tensors every epoch (per-index seeded RNG). Note that a
  synthetic sample carries no learnable signal, so this loader is for throughput and
  memory only — convergence must be shown on the real cache.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nemo_automodel.components.datasets.diffusion.loader import DiffusionDataloaderBuild

logger = logging.getLogger(__name__)

# FLUX.1-dev conditioning widths, fixed by the architecture (transformer/config.json:
# joint_attention_dim / pooled_projection_dim) and by the VAE's latent channel count.
T5_FEATURES_DIM = 4096
CLIP_POOLED_DIM = 768
VAE_LATENT_CHANNELS = 16


class SyntheticFluxDataset(Dataset):
    """Deterministic random FLUX.1-dev samples (VAE latents + T5/CLIP conditioning).

    Per-index seeded so the dataset is FIXED across epochs, workers and ranks.
    """

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
        # Per-index generator -> identical sample every epoch (fixed dataset).
        gen = torch.Generator().manual_seed(self.seed + int(idx))

        image_latents = (
            self.latent_scale
            * torch.randn(
                self.in_channels, self.latent_h, self.latent_w, dtype=torch.float32, generator=gen
            )
        ).to(self.latent_dtype)

        if self.share_text_features:
            # One [text_seq_len, 4096] buffer for every index. At FLUX.1-dev's 512-token
            # T5 width this tensor is 4 MB in bf16, so generating a distinct one per
            # sample dominates the micro-batch on the CPU side and makes a throughput run
            # measure the dataloader rather than the model. Nothing about throughput
            # depends on the samples holding different noise, only on their shapes.
            # PERF ONLY: never use this for a convergence run.
            if self._shared_text is None:
                shared_gen = torch.Generator().manual_seed(self.seed)
                self._shared_text = (
                    self.feature_scale
                    * torch.randn(
                        self.text_seq_len, self.t5_features_dim, dtype=torch.float32, generator=shared_gen
                    )
                ).to(self.text_dtype)
            text_embeddings = self._shared_text
        else:
            text_embeddings = (
                self.feature_scale
                * torch.randn(
                    self.text_seq_len, self.t5_features_dim, dtype=torch.float32, generator=gen
                )
            ).to(self.text_dtype)

        # Pooled CLIP is 768 floats -- cheap enough to always generate per sample.
        pooled = (
            self.feature_scale
            * torch.randn(self.pooled_dim, dtype=torch.float32, generator=gen)
        ).to(self.text_dtype)

        sample = {
            "image_latents": image_latents,
            "text_embeddings": text_embeddings,
            "pooled_prompt_embeds": pooled,
        }
        if self.emit_clip_hidden:
            # The real cache also carries clip_hidden [77, 768] and the stock collate
            # passes it through, but FluxAdapter ignores it. It is ~3% of the per-sample
            # text bytes, so it is off by default and available for the synthetic-vs-real
            # control, where matching the loader's byte volume is the point.
            if self.share_text_features:
                if self._shared_clip_hidden is None:
                    shared_gen = torch.Generator().manual_seed(self.seed + 1)
                    self._shared_clip_hidden = (
                        self.feature_scale
                        * torch.randn(
                            self.clip_hidden_len, self.pooled_dim, dtype=torch.float32, generator=shared_gen
                        )
                    ).to(self.text_dtype)
                sample["clip_hidden"] = self._shared_clip_hidden
            else:
                sample["clip_hidden"] = (
                    self.feature_scale
                    * torch.randn(
                        self.clip_hidden_len, self.pooled_dim, dtype=torch.float32, generator=gen
                    )
                ).to(self.text_dtype)

        if self.cache_in_memory:
            self._cache[idx] = sample
        return sample


def _collate_synthetic_flux(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    """Stack per-sample tensors into the FlowMatchingPipeline batch format.

    Mirrors the key names ``collate_fn_text_to_image`` produces for FLUX so that a
    synthetic arm and a real-cache arm hand the model the same thing.
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


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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

    Every field must be a plain YAML scalar (the recipe validates YAML keys against these
    dataclass fields, and an unknown key is a TypeError). The runtime ``dp_rank`` /
    ``dp_world_size`` / ``batch_size`` are passed to :meth:`build`, not fields.
    """

    num_samples: int = 2048
    in_channels: int = VAE_LATENT_CHANNELS
    # res/8. 32 / 64 / 128 / 256 for 256² / 512² / 1024² / 2048².
    latent_h: int = 128
    latent_w: int = 128
    # FLUX.1-dev's T5 cap. The real cache pads every caption to exactly this, so the
    # synthetic arm must use the same number or the two are not comparable.
    text_seq_len: int = 512
    t5_features_dim: int = T5_FEATURES_DIM
    pooled_dim: int = CLIP_POOLED_DIM
    feature_scale: float = 0.1
    latent_scale: float = 1.0
    seed: int = 1234
    # Match what the upstream FLUX processor writes: fp16 latents, bf16 text. Keeping
    # these equal to the cache's dtypes is what makes the synthetic-vs-real control a
    # comparison of the model rather than of a cast.
    latent_dtype: str = "float16"
    text_dtype: str = "bfloat16"
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = 2
    pin_memory: bool = True
    cache_in_memory: bool = False
    # PERF ONLY -- see SyntheticFluxDataset. Never set for a convergence run.
    share_text_features: bool = False
    emit_clip_hidden: bool = False
    clip_hidden_len: int = 77

    def build(self, *, dp_rank: int, dp_world_size: int, batch_size: int) -> DiffusionDataloaderBuild:
        """Build the synthetic dataset, per-rank sampler, and dataloader."""
        for name in ("latent_dtype", "text_dtype"):
            if getattr(self, name) not in _DTYPES:
                raise ValueError(
                    f"{name}={getattr(self, name)!r} not one of {sorted(_DTYPES)}"
                )

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
            shuffle=(sampler is None and self.shuffle),
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=_collate_synthetic_flux,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            **({"prefetch_factor": 2} if self.num_workers > 0 else {}),
        )
        # This line is the run's evidence of what the loader actually built, and the perf
        # ladder asserts against it. The token grid is derived (latent/2) rather than
        # configured, so a mismatch between the intended resolution and the emitted
        # shapes shows up here instead of silently changing the sequence length.
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
