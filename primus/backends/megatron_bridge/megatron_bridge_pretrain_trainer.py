###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgePretrainTrainer: Primus wrapper for Megatron-Bridge pre-training.

This trainer bridges Primus configuration system with Megatron-Bridge's
pre-training framework.

The trainer follows the same pattern as other Megatron-Bridge trainers but is
optimized for large-scale language model pre-training workflows.
"""

from typing import Any

from primus.backends.megatron_bridge.config_utils import load_recipe_config
from primus.backends.megatron_bridge.megatron_bridge_base_trainer import (
    MegatronBridgeBaseTrainer,
)
from primus.modules.module_utils import log_dict_aligned, log_rank_0


class MegatronBridgePretrainTrainer(MegatronBridgeBaseTrainer):
    """
    Trainer class for Megatron-Bridge pre-training.

    This trainer handles:
        - Recipe-based configuration for pre-training
        - Integration with Megatron-Core training infrastructure
        - Support for multiple model architectures (Llama, GPT, Mistral, etc.)
        - Scalable distributed pre-training workflow

    Inherits from MegatronBridgeBaseTrainer which provides:
        - Common Megatron-Bridge initialization and logging
        - Version detection
        - Unified training workflow and patch management
    """

    def __init__(self, backend_args: Any):
        """
        Initialize Megatron-Bridge pretrain trainer.

        Args:
            backend_args: Megatron-Bridge argument namespace (from MegatronBridgeArgBuilder)
        """
        super().__init__(backend_args=backend_args)

    def setup(self):
        """
        Setup phase for Megatron-Bridge pre-training.
        """
        log_rank_0("MegatronBridgePretrainTrainer.setup()")

    def init(self):
        """
        Initialize Megatron-Bridge pre-training components.

        This includes:
            - Model initialization (with or without recipe)
            - Optimizer setup
            - Data pipeline initialization for pre-training data
            - Distributed training setup
        """
        log_rank_0("Initializing Megatron-Bridge pre-training components...")

        self.cfg_container = load_recipe_config(self.backend_args)
        self._apply_nested_overrides()

        log_rank_0("Pre-training initialization completed")

    def train(self):
        """
        Execute Megatron-Bridge pre-training.

        This method is called by BaseTrainer.run() after applying patches.
        It executes the main pre-training loop using Megatron-Bridge's infrastructure.

        Pre-training typically involves:
            - Training on large-scale tokenized corpus
            - Regular evaluation on validation set
            - Saving training checkpoints
        """
        log_rank_0("Executing Megatron-Bridge pre-train...")
        try:
            from megatron.bridge.training.gpt_step import forward_step
            from megatron.bridge.training.pretrain import pretrain

            log_dict_aligned("ConfigContainer", self.cfg_container.to_dict())
            pretrain(self.cfg_container, forward_step_func=forward_step)
        except Exception as e:
            log_rank_0(f"Error during pre-training: {e}")
            raise
        log_rank_0("Megatron-Bridge pre-train execution completed.")
