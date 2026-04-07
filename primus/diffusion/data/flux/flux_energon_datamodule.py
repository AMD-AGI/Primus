# Copyright (c) 2025, Advanced Micro Devices, Inc.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX diffusion dataset config: synthetic latents (path unset) or Megatron-Energon WDS."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Any, Literal, Optional, Tuple

import torch
from torch.utils.data import Dataset

from megatron.bridge.training.config import DatasetBuildContext, DatasetProvider


def _latent_hw(image_h: int, image_w: int, vae_scale_factor: int) -> tuple[int, int]:
    h = max(1, image_h // vae_scale_factor)
    w = max(1, image_w // vae_scale_factor)
    return h, w


class _SyntheticFluxDataset(Dataset):
    """Map-style dataset yielding FLUX Megatron batches (one microbatch row per index)."""

    def __init__(
        self,
        length: int,
        latent_channels: int,
        latent_h: int,
        latent_w: int,
        prompt_seq_len: int,
        context_dim: int,
        pooled_prompt_dim: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._length = max(1, int(length))
        self.latent_channels = int(latent_channels)
        self.latent_h = int(latent_h)
        self.latent_w = int(latent_w)
        self.prompt_seq_len = int(prompt_seq_len)
        self.context_dim = int(context_dim)
        self.pooled_prompt_dim = int(pooled_prompt_dim)
        self._seed = int(seed)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self._seed + index)
        latents = torch.randn(
            (self.latent_channels, self.latent_h, self.latent_w),
            generator=g,
            dtype=torch.float32,
        )
        prompt_embeds = torch.randn(
            (self.prompt_seq_len, self.context_dim),
            generator=g,
            dtype=torch.float32,
        )
        pooled = torch.randn((self.pooled_prompt_dim,), generator=g, dtype=torch.float32)
        text_ids = torch.zeros((self.prompt_seq_len, 3), dtype=torch.long)
        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled,
            "text_ids": text_ids,
        }


def _make_flux_task_encoder() -> Any:
    from megatron.energon import edataclass
    from megatron.energon.flavors.base_dataset import Sample
    from megatron.energon.task_encoder.base import Batch, DefaultTaskEncoder, stateless
    from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys, cooker

    @edataclass
    class FluxPreparedSample(Sample):
        latents: torch.Tensor
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor

    @edataclass
    class FluxPreparedBatch(Batch):
        latents: torch.Tensor
        prompt_embeds: torch.Tensor
        pooled_prompt_embeds: torch.Tensor

    # @cooker does not set __stateless__; TaskEncoder.cook_crude_sample asserts get_stateless(cook).
    @stateless
    @cooker
    def cook_flux_shard(sample: dict) -> FluxPreparedSample:
        # Do not use `a or b`: torch.Tensor and other truthiness breaks (multi-element tensor bool).
        raw_latents = sample.get("pth")
        if raw_latents is None:
            raw_latents = sample.get(".pth")
        if raw_latents is None:
            raise KeyError("Expected 'pth' or '.pth' (VAE latents) in WebDataset sample.")
        latents = raw_latents if isinstance(raw_latents, torch.Tensor) else torch.as_tensor(raw_latents)

        raw_te = sample.get("pickle")
        if raw_te is None:
            raw_te = sample.get(".pickle")
        if raw_te is None:
            raise KeyError("Expected 'pickle' (T5/CLIP tensors) in WebDataset sample.")
        if isinstance(raw_te, (bytes, memoryview, bytearray)):
            te = pickle.loads(raw_te)
        else:
            te = raw_te

        return FluxPreparedSample(
            **basic_sample_keys(sample),
            latents=latents.float(),
            prompt_embeds=te["prompt_embeds"].float(),
            pooled_prompt_embeds=te["pooled_prompt_embeds"].float(),
        )

    class FluxTaskEncoder(DefaultTaskEncoder):
        cookers = (Cooker(cook_flux_shard),)

        def __init__(self) -> None:
            super().__init__(
                encoded_sample_type=FluxPreparedSample,
                raw_batch_type=FluxPreparedBatch,
                batch_type=dict,
            )

        @stateless
        def encode_batch(self, batch: Any) -> dict:
            def _to_batched_tensor(x: Any) -> torch.Tensor:
                if isinstance(x, torch.Tensor):
                    return x.float()
                if isinstance(x, (list, tuple)):
                    if not x:
                        raise ValueError("Empty tensor sequence in Flux batch field")
                    if not all(isinstance(t, torch.Tensor) for t in x):
                        raise TypeError("Flux batch field list must contain only tensors")
                    return torch.stack([t.float() for t in x], dim=0)
                raise TypeError(f"Flux batch field must be Tensor or list[Tensor], got {type(x)}")

            lat = _to_batched_tensor(batch.latents)
            pe = _to_batched_tensor(batch.prompt_embeds)
            pp = _to_batched_tensor(batch.pooled_prompt_embeds)

            # Energon sometimes omits B=1 or yields per-sample tensors without stacking.
            if lat.dim() == 3:
                lat = lat.unsqueeze(0)
            if pe.dim() == 2:
                pe = pe.unsqueeze(0)
            if pp.dim() == 1:
                pp = pp.unsqueeze(0)

            b, seq = pe.shape[0], pe.shape[1]
            text_ids = torch.zeros((b, seq, 3), dtype=torch.long)
            return {
                "latents": lat,
                "prompt_embeds": pe,
                "pooled_prompt_embeds": pp,
                "text_ids": text_ids,
            }

    return FluxTaskEncoder()


def _build_energon_iterators(cfg: "FluxDatasetConfig", context: DatasetBuildContext) -> Tuple[Any, Any, Any]:
    from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule

    module = EnergonMultiModalDataModule(
        path=cfg.path,
        tokenizer=context.tokenizer,
        image_processor=None,
        seq_length=cfg.seq_length,
        micro_batch_size=cfg.micro_batch_size,
        global_batch_size=cfg.global_batch_size,
        num_workers=cfg.num_workers,
        task_encoder=_make_flux_task_encoder(),
        packing_buffer_size=cfg.packing_buffer_size,
        shuffle_buffer_size=100,
    )
    train_dl = module.train_dataloader()
    val_dl = module.val_dataloader()
    return iter(train_dl), iter(val_dl), iter(val_dl)


@dataclass(kw_only=True)
class FluxDatasetConfig(DatasetProvider):
    """DatasetProvider for FLUX pretrain (synthetic or Energon-prepared WDS)."""

    path: Optional[str] = None
    seq_length: int = 1024
    packing_buffer_size: Optional[int] = None
    micro_batch_size: int = 1
    global_batch_size: int = 1
    num_workers: int = 8
    vae_scale_factor: int = 8
    latent_channels: int = 16
    image_H: int = 1024
    image_W: int = 1024
    prompt_seq_len: int = 512
    context_dim: int = 4096
    pooled_prompt_dim: int = 768

    dataloader_type: Optional[Literal["single", "cyclic", "batch", "external"]] = "cyclic"

    def build_datasets(
        self, context: DatasetBuildContext
    ) -> Tuple[Optional[Any], Optional[Any], Optional[Any]]:
        use_path = self.path is not None and (not isinstance(self.path, str) or bool(self.path.strip()))

        if not use_path:
            self.dataloader_type = "cyclic"
            lh, lw = _latent_hw(self.image_H, self.image_W, self.vae_scale_factor)

            def _make_ds(n: int) -> Optional[_SyntheticFluxDataset]:
                if n <= 0:
                    return None
                return _SyntheticFluxDataset(
                    length=max(1, n),
                    latent_channels=self.latent_channels,
                    latent_h=lh,
                    latent_w=lw,
                    prompt_seq_len=self.prompt_seq_len,
                    context_dim=self.context_dim,
                    pooled_prompt_dim=self.pooled_prompt_dim,
                )

            return (
                _make_ds(context.train_samples),
                _make_ds(context.valid_samples),
                _make_ds(context.test_samples),
            )

        try:
            self.dataloader_type = "external"
            return _build_energon_iterators(self, context)
        except ImportError as e:
            raise ImportError(
                "FLUX real-data training requires megatron-energon (and Megatron-Bridge Energon helpers). "
                "Install e.g. `pip install megatron-energon` (see examples/diffusion/mlperf_flux1/"
                "requirements-mlperf-flux1-setup.txt), prepare shards with "
                "examples/diffusion/recipes/flux/prepare_energon_dataset_flux.py, then `energon prepare`. "
                f"Original error: {e}"
            ) from e
