###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from primus.backends.diffusion.amdb_benchmarks.pixels import decode_host_pixels
from primus.backends.diffusion.amdb_benchmarks.sdxl.train_pipeline import (
    SDXLTrainPipeline,
)
from primus.backends.diffusion.models.interface import GenAIModel


class SDXLForTraining(GenAIModel):
    """Primus adapter around the official Stability SDXL UNet."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        autoencoder: nn.Module,
        text_encoder: nn.Module,
        train_pipeline: SDXLTrainPipeline,
        trainable_modules: str | None = None,
    ):
        super().__init__()
        self.dit = dit
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.train_pipeline = train_pipeline
        self.trainable_modules = trainable_modules or "dit"

    def freeze_except(self):
        for param in self.parameters():
            param.requires_grad_(False)
        if self.trainable_modules.lower() in ("dit", "unet", "all"):
            for param in self.dit.parameters():
                param.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        # The official UNet checkpoints ResBlocks and spatial transformers itself
        # when it is constructed with use_checkpoint=True.

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            return self.forward_train(
                args[0], scheduler=args[1] if len(args) > 1 else kwargs.get("scheduler")
            )
        raise TypeError("SDXLForTraining.forward expects (batch_dict, scheduler)")

    def cast_frozen_modules(self, dtype: torch.dtype) -> None:
        self.text_encoder.to(dtype=dtype)
        self.autoencoder.to(dtype=dtype)

    def prepare_host_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        resolution = self.train_pipeline.resolution
        return decode_host_pixels(batch, height=resolution, width=resolution)

    def forward_train(self, batch: dict[str, Any], scheduler: Any = None) -> dict[str, torch.Tensor]:
        del scheduler
        return self.train_pipeline.compute_loss(
            dit=self.dit,
            autoencoder=self.autoencoder,
            text_encoder=self.text_encoder,
            batch=batch,
        )

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        del batch, kwargs
        raise NotImplementedError("SDXL sampling is not part of the training module.")
