###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronPretrainTrainer: Primus wrapper for Megatron-LM pre-training.

This trainer bridges Primus configuration system with Megatron-LM's training loop.

The trainer inherits from MegatronBaseTrainer which handles:
    - Argument injection and Megatron runtime initialization
    - Patch management (before_train, after_train) via template method
    - Common Megatron setup patterns

This class only needs to implement run_train() with the actual training logic.
"""

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronPretrainTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM pre-training.

    Inherits from MegatronBaseTrainer which handles:
        - Argument injection into Megatron runtime
        - Patch management via template method pattern
        - Common Megatron initialization patterns

    This class implements:
        - setup(): Pre-initialization setup (optional)
        - init(): Training-specific initialization
        - run_train(): Execute actual training loop (no patch management needed)
    """

    def setup(self):
        """
        Setup phase (optional, for compatibility with BaseModule interface).

        Can be used for pre-initialization setup if needed.
        """
        log_rank_0("Setup phase")
        # Any pre-initialization setup can go here

    def init(self):
        """
        Initialize Megatron training components.

        Note:
            Argument injection is already done by MegatronBaseTrainer.__init__()
            This method can be used for trainer-specific initialization.
        """
        log_rank_0("Initializing Megatron training...")
        # Trainer-specific initialization can go here if needed

    def run_train(self):
        """
        Execute Megatron pre-training using the standard Megatron calling pattern.

        This method is called by MegatronBaseTrainer.run() after applying patches.
        It focuses solely on the training logic without patch management.
        """
        log_rank_0("Executing Megatron pretrain...")

        import inspect

        # Import Megatron components
        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore
        from pretrain_gpt import (  # type: ignore
            forward_step,
            train_valid_test_datasets_provider,
        )

        from primus.core.utils.import_utils import get_model_provider

        # Configure training components
        if hasattr(train_valid_test_datasets_provider, "is_distributed"):
            train_valid_test_datasets_provider.is_distributed = True

        # Megatron versions differ:
        # - v0.12.0: calls `pretrain(...)` directly (no inprocess_restart wrapper).
        # - newer: wraps pretrain via inprocess_restart and may pass `store=...`.
        wrapped_pretrain = pretrain
        store = None
        try:
            from megatron.training import inprocess_restart  # type: ignore

            if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        except Exception:
            pass

        # Execute training
        sig = inspect.signature(wrapped_pretrain)
        kwargs = {}
        if "args_defaults" in sig.parameters:
            # Matches upstream pretrain_gpt entrypoints; harmless if already defaulted.
            kwargs["args_defaults"] = {"tokenizer_type": "GPT2BPETokenizer"}
        if "extra_args_provider" in sig.parameters:
            kwargs["extra_args_provider"] = None
        if "store" in sig.parameters:
            kwargs["store"] = store

        wrapped_pretrain(
            train_valid_test_datasets_provider,
            get_model_provider(),
            ModelType.encoder_or_decoder,
            forward_step,
            **kwargs,
        )

        log_rank_0("Megatron pretrain execution completed.")
