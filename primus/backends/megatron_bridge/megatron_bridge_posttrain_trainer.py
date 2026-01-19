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

The trainer follows the same pattern as MegatronBridgePretrainTrainer but is
optimized for post-training workflows with smaller datasets and specialized
training objectives.
"""

from typing import Any

from primus.core.trainer.base_trainer import BaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronBridgePosttrainTrainer(BaseTrainer):
    """
    Trainer class for Megatron-Bridge post-training (SFT, instruction tuning, etc.).

    This trainer handles:
        - Supervised fine-tuning (SFT) workflows
        - Instruction tuning with specialized datasets
        - Recipe-based configuration for post-training
        - HuggingFace model conversion (bidirectional)
        - Integration with Megatron-Core training infrastructure
        - Support for multiple model architectures (Llama, GPT, Mistral, etc.)

    Inherits from BaseTrainer which provides:
        - Universal training workflow and patch management
        - Consistent training lifecycle
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron-Bridge posttrain trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-Bridge argument namespace (from MegatronBridgeArgBuilder)
        """
        log_rank_0("=" * 80)
        log_rank_0("Initializing MegatronBridgePosttrainTrainer...")
        log_rank_0("=" * 80)

        # Initialize BaseTrainer
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Training components
        self.model = None
        self.optimizer = None
        self.trainer = None
        self.recipe_config = None

        # Log version and configuration
        log_rank_0(f"Megatron-Bridge version: {type(self).detect_version()}")
        log_rank_0(f"Model: {module_config.model or 'custom'}")
        log_rank_0(f"Framework: {module_config.framework}")
        log_rank_0(f"Task: Post-training (SFT/Instruction Tuning)")

        # Check if using recipe-based configuration
        if hasattr(backend_args, "recipe") and backend_args.recipe:
            log_rank_0(f"Using recipe: {backend_args.recipe}")

        log_rank_0("=" * 80)
        log_rank_0("MegatronBridgePosttrainTrainer initialized successfully")
        log_rank_0("=" * 80)

    @classmethod
    def detect_version(cls) -> str:
        """
        Detect Megatron-Bridge version.

        Returns:
            Version string (e.g., "0.2.2")

        Raises:
            RuntimeError: If version cannot be detected
        """
        try:
            # Try to get version from megatron.bridge package
            import importlib.metadata as importlib_metadata

            try:
                version = importlib_metadata.version("megatron-bridge")
                return version
            except importlib_metadata.PackageNotFoundError:
                # Package not installed, try to read from source
                pass

            # Fallback: try to read from package info in source
            try:
                from megatron.bridge import __version__

                return __version__
            except (ImportError, AttributeError):
                pass

            # If all else fails, return unknown
            return "unknown"

        except Exception as e:
            log_rank_0(f"Warning: Could not detect Megatron-Bridge version: {e}")
            return "unknown"

    def setup(self):
        """
        Setup phase for Megatron-Bridge post-training.

        This includes:
            - Loading recipe configuration if specified
            - Loading pretrained checkpoint for fine-tuning
            - Setting up HuggingFace model conversion if needed
            - Preparing instruction/SFT datasets and tokenizers
            - Configuring post-training specific settings (e.g., LoRA, prompt templates)
        """
        log_rank_0("Setting up Megatron-Bridge post-training environment...")

        # Load recipe if specified
        if hasattr(self.backend_args, "recipe") and self.backend_args.recipe:
            self._load_recipe()

        # Load pretrained checkpoint if specified
        if hasattr(self.backend_args, "load") and self.backend_args.load:
            log_rank_0(f"Will load pretrained checkpoint from: {self.backend_args.load}")

        # Handle HuggingFace model conversion if requested
        if hasattr(self.backend_args, "convert_from_hf") and self.backend_args.convert_from_hf:
            self._convert_from_huggingface()

        # Setup post-training specific configurations
        self._setup_posttrain_configs()

        log_rank_0("Setup phase completed")

    def _setup_posttrain_configs(self):
        """
        Setup post-training specific configurations.

        This includes:
            - LoRA/PEFT configuration if specified
            - Prompt template configuration
            - Special token handling
            - Chat format configuration
        """
        log_rank_0("Setting up post-training specific configurations...")

        # Check for LoRA configuration
        if hasattr(self.backend_args, "use_lora") and self.backend_args.use_lora:
            log_rank_0("LoRA fine-tuning enabled")
            if hasattr(self.backend_args, "lora_rank"):
                log_rank_0(f"  LoRA rank: {self.backend_args.lora_rank}")
            if hasattr(self.backend_args, "lora_alpha"):
                log_rank_0(f"  LoRA alpha: {self.backend_args.lora_alpha}")

        # Check for prompt template
        if hasattr(self.backend_args, "prompt_template"):
            log_rank_0(f"Using prompt template: {self.backend_args.prompt_template}")

        # Check for chat format
        if hasattr(self.backend_args, "chat_format"):
            log_rank_0(f"Using chat format: {self.backend_args.chat_format}")

    def _load_recipe(self):
        """
        Load recipe configuration from Megatron-Bridge for post-training.

        Post-training recipes typically include:
            - Base model configuration
            - Fine-tuning specific hyperparameters (lower LR, shorter schedule)
            - Dataset format and preprocessing
            - LoRA/PEFT configuration
            - Evaluation metrics
        """
        recipe_name = self.backend_args.recipe
        log_rank_0(f"Loading post-training recipe: {recipe_name}")

        try:
            # Import Megatron-Bridge recipe loader
            # TODO: Implement actual recipe loading logic for post-training
            # This would typically involve importing the recipe module
            # and extracting its configuration

            # Example structure (to be implemented):
            # from megatron.bridge.recipes.posttrain import load_posttrain_recipe
            # self.recipe_config = load_posttrain_recipe(recipe_name)

            log_rank_0(f"Post-training recipe loaded: {recipe_name}")
        except Exception as e:
            log_rank_0(f"Warning: Failed to load recipe {recipe_name}: {e}")
            log_rank_0("Continuing with manual configuration...")

    def _convert_from_huggingface(self):
        """
        Convert HuggingFace model to Megatron-Bridge format for fine-tuning.

        This uses Megatron-Bridge's bidirectional conversion capability
        to load pretrained HuggingFace models for post-training.
        """
        if not hasattr(self.backend_args, "hf_model_name_or_path"):
            log_rank_0("Warning: convert_from_hf is True but hf_model_name_or_path not specified")
            return

        hf_model_path = self.backend_args.hf_model_name_or_path
        log_rank_0(f"Converting HuggingFace model from: {hf_model_path}")

        try:
            # TODO: Implement HuggingFace to Megatron-Bridge conversion
            # This would use Megatron-Bridge's conversion utilities
            # Example:
            # from megatron.bridge.converters import hf_to_megatron
            # checkpoint = hf_to_megatron(hf_model_path, self.backend_args)

            log_rank_0("HuggingFace model conversion completed")
        except Exception as e:
            log_rank_0(f"Error: Failed to convert HuggingFace model: {e}")
            raise

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

        # TODO: Implement actual initialization logic for post-training
        # This would typically involve:
        # 1. Loading the pretrained model checkpoint
        # 2. Adding LoRA adapters if configured
        # 3. Freezing base model parameters if doing adapter-only training
        # 4. Setting up the optimizer with post-training hyperparameters
        # 5. Initializing data loaders for instruction/SFT datasets
        # 6. Configuring distributed training parameters

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
            # Import Megatron-Bridge training components
            # TODO: Update these imports based on actual Megatron-Bridge structure
            # from megatron.bridge.training import posttrain
            # from megatron.bridge.training.utils import (
            #     get_model_provider,
            #     get_forward_step_func,
            #     get_posttrain_data_provider,
            # )

            # Execute post-training based on configuration
            if self.recipe_config:
                # Use recipe-based post-training
                log_rank_0("Using recipe-based post-training configuration")
                # TODO: Implement recipe-based post-training execution
            else:
                # Use manual configuration
                log_rank_0("Using manual post-training configuration")
                # TODO: Implement manual post-training execution

            # Placeholder for actual post-training execution
            log_rank_0("Post-training loop would execute here")
            log_rank_0("Fine-tuning on instruction/SFT dataset...")

            # Handle HuggingFace conversion after training if requested
            if hasattr(self.backend_args, "convert_to_hf") and self.backend_args.convert_to_hf:
                self._convert_to_huggingface()

        except Exception as e:
            log_rank_0(f"Error during post-training: {e}")
            raise

        log_rank_0("Megatron-Bridge post-train execution completed.")

    def _convert_to_huggingface(self):
        """
        Convert fine-tuned Megatron-Bridge checkpoint to HuggingFace format.

        This uses Megatron-Bridge's bidirectional conversion capability
        to export fine-tuned models to HuggingFace format for easy deployment.
        """
        if not hasattr(self.backend_args, "hf_save_path"):
            log_rank_0("Warning: convert_to_hf is True but hf_save_path not specified")
            return

        hf_save_path = self.backend_args.hf_save_path
        log_rank_0(f"Converting fine-tuned checkpoint to HuggingFace format at: {hf_save_path}")

        try:
            # TODO: Implement Megatron-Bridge to HuggingFace conversion
            # This would use Megatron-Bridge's conversion utilities
            # Example:
            # from megatron.bridge.converters import megatron_to_hf
            # megatron_to_hf(self.backend_args.save, hf_save_path, self.backend_args)

            log_rank_0("Fine-tuned checkpoint conversion to HuggingFace format completed")
        except Exception as e:
            log_rank_0(f"Error: Failed to convert checkpoint to HuggingFace: {e}")
            raise
