###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class GenAIModel(nn.Module, ABC):
    """
    Unified interface for Generative Models (DiT, AR, Hybrid).
    Wraps the Backbone (DiT/Transformer), VAE, and TextEncoder.
    """

    @abstractmethod
    def forward_train(self, batch: Dict[str, Any], scheduler: Any = None) -> Dict[str, torch.Tensor]:
        """
        The single entry point for training.

        Responsibilities:
        1. Process raw batch (Tokenization, VAE Encoding if needed).
        2. Apply Training Recipe (e.g., Add Noise for DiT, Shift Tokens for AR).
        3. Forward pass through Backbone.
        4. Calculate Loss.

        Args:
            batch: Raw batch from data loader.
            scheduler: Optional scheduler (e.g., FlowMatchScheduler) passed from Trainer.

        Returns:
            {
                "loss": torch.Tensor,       # Main optimization objective
                "log_metrics": Dict,        # Metrics for WandB (MSE, Accuracy, etc.)
            }
        """

    @abstractmethod
    def forward_inference(self, batch: Dict[str, Any], **kwargs):
        """
        Entry point for validation/inference.
        For DiT: Runs the diffusion sampling loop.
        For AR: Runs autoregressive generation.
        """
