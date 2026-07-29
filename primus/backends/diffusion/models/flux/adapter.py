###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from primus.backends.diffusion.models.flux.train_pipeline import (
    FluxFlowMatchTrainPipeline,
)
from primus.backends.diffusion.models.interface import GenAIModel


@dataclass
class FluxConfigShim:
    raw: dict

    def save_pretrained(self, save_directory: str):
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "flux_config.json"), "w") as f:
            json.dump(self.raw, f, indent=2, sort_keys=True)

    def to_dict(self):
        return self.raw


class FluxForTraining(GenAIModel, nn.Module):
    """Thin Primus adapter around the FLUX DiT backbone."""

    def __init__(
        self,
        *,
        dit: nn.Module,
        train_pipeline: FluxFlowMatchTrainPipeline,
        model_config: Any,
        autoencoder: nn.Module | None = None,
        t5_encoder: nn.Module | None = None,
        clip_encoder: nn.Module | None = None,
        raw_config: dict | None = None,
        trainable_modules: str | None = None,
    ):
        super().__init__()
        self.dit = dit
        self.autoencoder = autoencoder
        self.t5_encoder = t5_encoder
        self.clip_encoder = clip_encoder
        self.train_pipeline = train_pipeline
        self.model_config = model_config
        self.trainable_modules = trainable_modules
        self.compute_dtype: torch.dtype | None = None
        self.config = FluxConfigShim(raw=raw_config or {})

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def freeze_except(self):
        mode = (
            self.trainable_modules or getattr(self.model_config, "trainable_modules", None) or "dit"
        ).lower()

        def freeze(module: nn.Module):
            for param in module.parameters():
                param.requires_grad_(False)

        def unfreeze(module: nn.Module):
            for param in module.parameters():
                param.requires_grad_(True)

        freeze(self)
        if mode in ("dit", "diffusion", "backbone"):
            unfreeze(self.dit)
        elif mode == "all":
            unfreeze(self)
        else:
            unfreeze(self.dit)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        del gradient_checkpointing_kwargs
        if getattr(self.dit, "_torchtitan_checkpoint_wrapped", False):
            return

        # Match the MLPerf TorchTitan reference: wrap each transformer block
        # before FSDP and do not preserve RNG state. The older in-forward
        # torch.utils.checkpoint path is not equivalent and is unstable when
        # composed with FSDP2 on ROCm.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
        )

        for blocks_name in ("double_blocks", "single_blocks"):
            blocks = getattr(self.dit, blocks_name)
            for index, block in enumerate(blocks):
                blocks[index] = checkpoint_wrapper(block, preserve_rng_state=False)
        self.dit.gradient_checkpointing = False
        self.dit._torchtitan_checkpoint_wrapped = True

    def forward(self, *args, **kwargs):
        if len(args) >= 1 and isinstance(args[0], dict):
            scheduler = args[1] if len(args) >= 2 else kwargs.get("scheduler")
            return self.forward_train(args[0], scheduler=scheduler)
        raise TypeError("FluxForTraining.forward expects (batch_dict, scheduler)")

    def forward_train(self, batch: dict[str, Any], scheduler: Any = None) -> dict[str, torch.Tensor]:
        del scheduler
        return self.train_pipeline.compute_loss(
            dit=self.dit,
            autoencoder=self.autoencoder,
            t5_encoder=self.t5_encoder,
            clip_encoder=self.clip_encoder,
            batch=batch,
            model_config=self.model_config,
            compute_dtype=self.compute_dtype,
        )

    def forward_inference(self, batch: dict[str, Any], **kwargs):
        raise NotImplementedError("FLUX inference pipeline is not wired for Primus training yet")
