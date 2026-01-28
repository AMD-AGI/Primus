###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronBridgePosttrainTrainer: Primus wrapper for Megatron-Bridge post-training.

This trainer bridges Primus configuration system with Megatron-Bridge's
post-training framework. Post-training includes supervised fine-tuning (SFT),
instruction tuning, and other post-pretraining tasks.

The trainer follows the same pattern as other Megatron-Bridge trainers but is
optimized for post-training workflows with smaller datasets and specialized
training objectives.
"""

from typing import Any

from primus.backends.megatron_bridge.config_utils import load_recipe_config
from primus.backends.megatron_bridge.megatron_bridge_base_trainer import (
    MegatronBridgeBaseTrainer,
)
from primus.modules.module_utils import log_dict_aligned, log_rank_0


class MegatronBridgePosttrainTrainer(MegatronBridgeBaseTrainer):
    """
    Trainer class for Megatron-Bridge post-training (SFT, instruction tuning, etc.).

    This trainer handles:
        - Supervised fine-tuning (SFT) workflows
        - Instruction tuning with specialized datasets
        - Recipe-based configuration for post-training
        - HuggingFace model conversion (bidirectional)
        - Integration with Megatron-Core training infrastructure
        - Support for multiple model architectures (Llama, GPT, Mistral, etc.)
        - LoRA and other parameter-efficient fine-tuning methods

    Inherits from MegatronBridgeBaseTrainer which provides:
        - Common Megatron-Bridge initialization and logging
        - Version detection
        - Unified training workflow and patch management
    """

    # Task type identifier for logging
    TASK_TYPE = "Post-training (SFT/Instruction Tuning)"

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron-Bridge posttrain trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-Bridge argument namespace (from MegatronBridgeArgBuilder)
        """
        # Initialize MegatronBridgeBaseTrainer (which initializes BaseTrainer)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

    def setup(self):
        """
        Setup phase for Megatron-Bridge post-training.
        """
        log_rank_0("MegatronBridgePosttrainTrainer.setup()")

    def init(self):
        """
        Initialize Megatron-Bridge post-training components.

        This includes:
            - Loading pretrained model checkpoint
            - Model initialization (with or without recipe)
            - LoRA/PEFT adapter initialization if configured
            - Optimizer setup (often with lower learning rate)
            - Data pipeline initialization for instruction/SFT data
            - Distributed training setup
        """
        log_rank_0("Initializing Megatron-Bridge post-training components...")

        self.cfg_container = load_recipe_config(self.backend_args)
        log_dict_aligned("ConfigContainer", self.cfg_container.to_dict())

        log_rank_0("Post-training initialization completed")

    def run_train(self):
        """
        Execute Megatron-Bridge post-training (SFT, instruction tuning).

        This method is called by BaseTrainer.run() after applying patches.
        It executes the main fine-tuning loop using Megatron-Bridge's infrastructure.

        Post-training typically involves:
            - Loading pretrained checkpoint
            - Training on instruction/SFT dataset
            - Regular evaluation on validation set
            - Saving fine-tuned checkpoints
            - Optional conversion to HuggingFace format
        """
        log_rank_0("Executing Megatron-Bridge post-train...")

        try:
            # Execute post-training based on configuration
            from megatron.bridge.training.finetune import finetune
            from megatron.bridge.training.vlm_step import forward_step

            # log_rank_0(f"ConfigContainer: {self.cfg_container}")
            log_dict_aligned("ConfigContainer", self.cfg_container.to_dict())
            finetune(self.cfg_container, forward_step_func=forward_step)

        except Exception as e:
            log_rank_0(f"Error during post-training: {e}")
            raise

        log_rank_0("Megatron-Bridge post-train execution completed.")
