###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Tiny multi-view model used to exercise the trainer without upstream weights."""

from __future__ import annotations

import torch

from primus.backends.vision.models.interface import VisionModelForTraining


class SyntheticVisionModel(VisionModelForTraining):
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.proj = torch.nn.Linear(3, embed_dim)
        self.head = torch.nn.Linear(embed_dim, 1)

    def forward(self, batch) -> torch.Tensor:
        images = batch["images"]
        if images.ndim != 5:
            raise ValueError(f"synthetic images must be [B, S, 3, H, W], got {tuple(images.shape)}")
        summary = images.mean(dim=(-1, -2))
        loss = self.head(self.proj(summary)).mean().square()
        self.last_metrics = {
            "loss": loss.detach(),
            "num_images": float(images.shape[0] * images.shape[1]),
        }
        return loss


def build_synthetic_model(model_config: dict) -> SyntheticVisionModel:
    return SyntheticVisionModel(embed_dim=int(model_config.get("embed_dim", 32)))
