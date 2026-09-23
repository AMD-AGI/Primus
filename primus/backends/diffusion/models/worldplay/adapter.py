###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from primus.backends.diffusion.models.interface import GenAIModel

from .train_pipeline import WorldPlayARTrainPipeline


class WorldPlayConfigShim:
    def __init__(self, raw: dict[str, Any]):
        self.raw = raw

    def save_pretrained(self, save_directory: str):
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        with (output / "worldplay_config.json").open("w") as handle:
            json.dump(self.raw, handle, indent=2, sort_keys=True)

    def to_dict(self):
        return self.raw


class WorldPlayForTraining(GenAIModel, nn.Module):
    """Primus adapter around the public HY-WorldPlay AR transformer."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        train_pipeline: WorldPlayARTrainPipeline,
        raw_config: dict[str, Any],
    ):
        super().__init__()
        self.dit = dit
        self.train_pipeline = train_pipeline
        self.config = WorldPlayConfigShim(raw_config)
        self.compute_dtype: torch.dtype | None = None

    def freeze_except(self):
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        self.dit.gradient_checkpointing = True

    def configure_distributed(
        self,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        sp_size: int,
    ) -> None:
        """Primus creates and passes the sequence-parallel group in each batch."""
        del rank, world_size, local_rank, sp_size

    def forward(self, batch: dict[str, Any], scheduler=None):
        return self.forward_train(batch, scheduler=scheduler)

    def forward_train(
        self,
        batch: dict[str, Any],
        scheduler=None,
    ) -> dict[str, torch.Tensor]:
        del scheduler
        return self.train_pipeline.compute_loss(dit=self.dit, batch=batch)

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        del batch, kwargs
        raise NotImplementedError("WorldPlay inference is outside this SFT integration")
