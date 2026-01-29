###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronSFTTrainer: Primus wrapper for Megatron-LM supervised fine-tuning.

This trainer bridges Primus configuration system with Megatron-LM's training loop
for supervised fine-tuning (SFT) tasks.

The trainer inherits from MegatronBaseTrainer which handles:
    - Argument injection and Megatron runtime initialization
    - Patch management (before_train, after_train) via template method
    - Common Megatron setup patterns

This class only needs to implement run_train() with the actual SFT training logic.
"""

from typing import Any

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronSFTTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM supervised fine-tuning (SFT).

    Inherits from MegatronBaseTrainer which handles:
        - Argument injection into Megatron runtime
        - Patch management via template method pattern
        - Common Megatron initialization patterns

    This class implements:
        - setup(): Pre-initialization setup (optional)
        - init(): Training-specific initialization
        - run_train(): Execute actual SFT training loop
    """

    # Task type identifier for logging
    TASK_TYPE = "Supervised Fine-Tuning (SFT)"

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron SFT trainer.

        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        # Initialize base class (handles argument injection)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )

        # Training components (will be set during training)
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None

        log_rank_0(f"Initialized SFT trainer for model: {module_config.model or 'custom'}")

    def setup(self):
        """
        Setup phase for SFT training.
        
        Can be used for pre-initialization setup if needed (e.g., loading pretrained checkpoint).
        """
        log_rank_0("MegatronSFTTrainer.setup()")
        # Any SFT-specific pre-initialization setup can go here

    def init(self):
        """
        Initialize Megatron SFT training components.

        Note:
            Argument injection is already done by MegatronBaseTrainer.__init__()
            This method can be used for SFT-specific initialization.
        """
        log_rank_0("Initializing Megatron SFT training components...")
        # SFT-specific initialization can go here (e.g., loading pretrained checkpoint)
        log_rank_0("SFT initialization completed")

    def run_train(self):
        """
        Execute Megatron SFT training.

        This method is called by BaseTrainer.run() after applying patches.
        It uses Megatron's standard pretrain infrastructure but with SFT-specific
        forward_step and loss functions.
        """
        log_rank_0("Executing Megatron SFT training...")

        try:
            import inspect
            from megatron.core.enums import ModelType
            from megatron.training import pretrain  # type: ignore
            from primus.core.utils.import_utils import get_model_provider
            
            # Create SFT-specific forward_step function
            # This will be provided to Megatron's pretrain() call
            from primus.modules.trainer.megatron.sft_trainer import MegatronSFTTrainer as ModuleSFTTrainer
            
            # Create instance to access SFT-specific methods
            # Note: We only use this for the methods, not for running the full trainer
            sft_module = ModuleSFTTrainer(
                module_name="sft_trainer",
                module_config=self.module_config,
                primus_config=self.primus_config,
            )
            
            # Use SFT-specific forward_step
            forward_step = sft_module.forward_step
            
            # Use standard dataset provider (for now - can be customized later)
            # For SFT, the dataset should provide instruction+response pairs with loss_mask
            from pretrain_gpt import train_valid_test_datasets_provider  # type: ignore
            
            # Mark as distributed
            if hasattr(train_valid_test_datasets_provider, "is_distributed"):
                train_valid_test_datasets_provider.is_distributed = True
            
            # Handle different Megatron versions (same as pretrain)
            wrapped_pretrain = pretrain
            store = None
            try:
                from megatron.training import inprocess_restart  # type: ignore
                if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                    wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
            except Exception:
                pass
            
            # Execute SFT training using Megatron's pretrain infrastructure
            sig = inspect.signature(wrapped_pretrain)
            kwargs = {}
            if "args_defaults" in sig.parameters:
                kwargs["args_defaults"] = {"tokenizer_type": "GPT2BPETokenizer"}
            if "store" in sig.parameters and store is not None:
                kwargs["store"] = store
            
            log_rank_0(f"Calling Megatron pretrain() for SFT with model_provider={get_model_provider()}")
            wrapped_pretrain(
                train_valid_test_datasets_provider,
                get_model_provider(),
                ModelType.encoder_or_decoder,
                forward_step,
                **kwargs,
            )

        except Exception as e:
            log_rank_0(f"Error during SFT training: {e}")
            raise

        log_rank_0("Megatron SFT training completed.")
