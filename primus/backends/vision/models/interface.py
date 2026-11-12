###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Shared training interface for vision models."""

from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn as nn


class VisionModelForTraining(nn.Module):
    """One forward used by the trainer, plus the hooks parallelism needs.

    ``forward`` returns a scalar loss. Component metrics are stored on
    ``last_metrics`` so distributed wrappers only have to sync the loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.last_metrics: dict[str, torch.Tensor | float] = {}
        self.find_unused_parameters = False

    @abstractmethod
    def forward(self, batch) -> torch.Tensor:
        """Run the model and its loss. Return a scalar loss tensor."""

    def block_classes(self) -> tuple[type, ...]:
        """Transformer block classes FSDP2 shards and checkpointing wraps."""
        return ()

    def param_groups(self) -> list[dict]:
        """Optimizer groups. Freeze parameters before returning them."""
        params = [param for param in self.parameters() if param.requires_grad]
        return [{"params": params}]

    def export_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights to write as the final safetensors file."""
        return self.state_dict()
