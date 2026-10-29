###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from primus.backends.diffusion.models.mochi.train_pipeline import (
    MochiTrainPipeline,
)
from primus.backends.diffusion.utils.pixels import decode_host_pixels
from primus.backends.diffusion.models.interface import GenAIModel


class MochiForTraining(GenAIModel):
    """Primus adapter around the official Mochi AsymmDiTJoint."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        encoder: nn.Module,
        text_encoder: nn.Module,
        train_pipeline: MochiTrainPipeline,
    ):
        super().__init__()
        self.dit = dit
        self.encoder = encoder
        self.text_encoder = text_encoder
        self.train_pipeline = train_pipeline

    def freeze_except(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.dit.parameters():
            param.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if getattr(self.dit, "_torchtitan_checkpoint_wrapped", False):
            return
        ratio = float((gradient_checkpointing_kwargs or {}).get("ratio", 1.0))
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        checkpoint_count = round(len(self.dit.blocks) * ratio)
        for index, block in enumerate(self.dit.blocks):
            if index >= checkpoint_count:
                break
            self.dit.blocks[index] = checkpoint_wrapper(block, preserve_rng_state=False)
        self.dit._torchtitan_checkpoint_wrapped = True

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            return self.forward_train(
                args[0], scheduler=args[1] if len(args) > 1 else kwargs.get("scheduler")
            )
        raise TypeError("MochiForTraining.forward expects (batch_dict, scheduler)")

    def cast_frozen_modules(self, dtype: torch.dtype) -> None:
        self.encoder.to(dtype=dtype)
        self.text_encoder.to(dtype=dtype)

    def prepare_host_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pipeline = self.train_pipeline
        return decode_host_pixels(
            batch, height=pipeline.height, width=pipeline.width, num_frames=pipeline.num_frames
        )

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        param = next(self.dit.parameters())
        return self.train_pipeline.encode_batch(
            encoder=self.encoder,
            text_encoder=self.text_encoder,
            batch=batch,
            device=param.device,
            dtype=param.dtype,
        )

    def offload_frozen_modules(self) -> None:
        self.encoder.to("cpu")
        self.text_encoder.to("cpu")

    def forward_train(self, batch: dict[str, Any], scheduler: Any = None) -> dict[str, torch.Tensor]:
        del scheduler
        return self.train_pipeline.compute_loss(
            dit=self.dit,
            encoder=self.encoder,
            text_encoder=self.text_encoder,
            batch=batch,
        )

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        del batch, kwargs
        raise NotImplementedError("Mochi sampling is not part of the training module.")
