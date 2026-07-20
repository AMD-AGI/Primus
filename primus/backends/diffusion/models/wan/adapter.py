###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from primus.backends.diffusion.models.interface import GenAIModel

from .components import WanComponents
from .train_pipeline import WanFlowMatchTrainPipeline


@dataclass
class WanConfigShim:
    """
    A tiny config shim to keep trainer saving behavior happy.
    We intentionally do not implement a full HF/Diffusers config system.
    """

    raw: dict

    def save_pretrained(self, save_directory: str):
        # Best-effort: keep minimal JSON for reproducibility.
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)
        path = os.path.join(save_directory, "wan_config.json")
        with open(path, "w") as f:
            json.dump(self.raw, f, indent=2, sort_keys=True)

    def to_dict(self):
        return self.raw


class WanForTraining(GenAIModel, nn.Module):
    """
    A thin adapter that exposes the call pattern expected by Wan trainers.

    Trainers call:
      outputs = model(batch, scheduler)  -> {"loss": ...}

    Internally we delegate workflow to the pipeline.
    """

    def __init__(
        self,
        *,
        components: WanComponents,
        train_pipeline: WanFlowMatchTrainPipeline,
        model_config: Any,
        raw_config: Optional[dict] = None,
        trainable_modules: Optional[str] = None,
    ):
        super().__init__()
        self.components = components
        self.train_pipeline = train_pipeline
        self.model_config = model_config
        self.trainable_modules = trainable_modules

        # Expose common attribute names expected by trainers/FSDP ignore regex.
        self.dit = components.dit
        self.vae = components.vae
        self.text_encoder = components.text_encoder
        self.image_encoder = components.image_encoder

        self.config = WanConfigShim(raw=raw_config or {})

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def to(self, *args, **kwargs):
        """
        Override to propagate .to() to non-nn.Module components.

        Wan VAE wrappers (Wan2_1_VAE, Wan2_2_VAE) are plain Python classes
        with a custom .to() method, not nn.Module subclasses.  PyTorch's
        nn.Module.to() only recurses into registered submodules, so the VAE
        would be silently skipped — causing device mismatches on multi-GPU.
        """
        result = super().to(*args, **kwargs)
        for component in (self.vae, self.image_encoder):
            if component is not None and not isinstance(component, nn.Module) and hasattr(component, "to"):
                component.to(*args, **kwargs)
        return result

    def freeze_except(self):
        """
        Keep the Wan training behavior: freeze non-trainable modules.
        Default: train only DiT.
        """
        mode = (
            self.trainable_modules or getattr(self.model_config, "trainable_modules", None) or "dit"
        ).lower()

        def freeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad_(False)

        def unfreeze(m: nn.Module):
            for p in m.parameters():
                p.requires_grad_(True)

        # Freeze everything first
        freeze(self)

        # Unfreeze requested parts
        if mode in ("dit", "diffusion", "backbone"):
            unfreeze(self.dit)
        elif mode in ("all",):
            unfreeze(self)
        else:
            # Conservative default: DiT only
            unfreeze(self.dit)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Activated by HF Trainer when `gradient_checkpointing=True`.
        """
        if self.dit and hasattr(self.dit, "gradient_checkpointing"):
            self.dit.gradient_checkpointing = True

    def forward(self, *args, **kwargs):
        # Match existing trainer call convention: model(batch, scheduler)
        if len(args) >= 1 and isinstance(args[0], dict):
            scheduler = args[1] if len(args) > 1 else kwargs.get("scheduler", None)
            return self.forward_train(args[0], scheduler=scheduler)
        raise TypeError("WanForTraining.forward expects (batch_dict, scheduler)")

    def forward_train(self, batch: Dict[str, Any], scheduler: Any = None) -> Dict[str, torch.Tensor]:
        if scheduler is None:
            raise ValueError("scheduler must be provided by trainer")
        return self.train_pipeline.compute_loss(
            components=self.components,
            batch=batch,
            scheduler=scheduler,
            model_config=self.model_config,
        )

    def forward_inference(self, batch: Dict[str, Any], **kwargs):
        raise NotImplementedError("wan inference pipeline not wired yet")
