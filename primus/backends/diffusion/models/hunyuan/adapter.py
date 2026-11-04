###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from primus.backends.diffusion.models.hunyuan.train_pipeline import HunyuanTrainPipeline
from primus.backends.diffusion.models.interface import GenAIModel
from primus.backends.diffusion.utils.pixels import decode_host_pixels


class HunyuanVideoForTraining(GenAIModel):
    """Primus adapter around the official HunyuanVideo transformer."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        text_encoder: nn.Module,
        train_pipeline: HunyuanTrainPipeline,
        vae: nn.Module | None = None,
    ):
        super().__init__()
        self.dit = dit
        self.text_encoder = text_encoder
        self.train_pipeline = train_pipeline
        if vae is not None:
            self.vae = vae

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

        block_lists = [self.dit.double_blocks, self.dit.single_blocks]
        checkpoint_count = round(sum(len(blocks) for blocks in block_lists) * ratio)
        block_number = 0
        for blocks in block_lists:
            for index, block in enumerate(blocks):
                if block_number >= checkpoint_count:
                    break
                blocks[index] = checkpoint_wrapper(block, preserve_rng_state=False)
                block_number += 1
        self.dit._torchtitan_checkpoint_wrapped = True

    def forward(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            return self.forward_train(
                args[0], scheduler=args[1] if len(args) > 1 else kwargs.get("scheduler")
            )
        raise TypeError("HunyuanVideoForTraining.forward expects (batch_dict, scheduler)")

    def cast_frozen_modules(self, dtype: torch.dtype) -> None:
        self.text_encoder.to(dtype=dtype)
        vae = getattr(self, "vae", None)
        if vae is not None:
            vae.to(dtype=dtype)

    def prepare_host_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        pipeline = self.train_pipeline
        return decode_host_pixels(
            batch, height=pipeline.height, width=pipeline.width, num_frames=pipeline.num_frames
        )

    def encode_batch(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        param = next(self.dit.parameters())
        return self.train_pipeline.encode_batch(
            text_encoder=self.text_encoder,
            vae=getattr(self, "vae", None),
            batch=batch,
            device=param.device,
            dtype=param.dtype,
        )

    def offload_frozen_modules(self) -> None:
        self.text_encoder.to("cpu")
        vae = getattr(self, "vae", None)
        if vae is not None:
            vae.to("cpu")

    def forward_train(self, batch: dict[str, Any], scheduler: Any = None) -> dict[str, torch.Tensor]:
        del scheduler
        return self.train_pipeline.compute_loss(
            dit=self.dit,
            text_encoder=self.text_encoder,
            vae=getattr(self, "vae", None),
            batch=batch,
        )

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        del batch, kwargs
        raise NotImplementedError("HunyuanVideo sampling is not part of the training module.")
