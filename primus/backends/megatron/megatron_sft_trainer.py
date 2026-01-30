###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MegatronSFTTrainer: Primus wrapper for Megatron-LM based supervised fine-tuning.

This trainer bridges Primus configuration system with Megatron-LM's training loop
for supervised fine-tuning (SFT) tasks.

Key features:
    - Direct integration with Megatron-LM (no Megatron-Bridge dependency)
    - Universal dataset interface supporting HuggingFace datasets
    - Multiple conversation format support (Alpaca, ChatML, etc.)
    - Proper loss masking for instruction tuning
    - Compatible with Megatron-LM's distributed training infrastructure
"""

from typing import Any, Optional

from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
from primus.modules.module_utils import log_rank_0


class MegatronSFTTrainer(MegatronBaseTrainer):
    """
    Trainer class for Megatron-LM based supervised fine-tuning.
    
    This trainer handles:
        - SFT workflows with HuggingFace datasets
        - Instruction tuning with proper loss masking
        - Multiple conversation formats (extensible)
        - Direct Megatron-LM integration without Megatron-Bridge
        - Support for various model architectures (Llama, GPT, etc.)
    
    Inherits from MegatronBaseTrainer which provides:
        - Argument injection into Megatron runtime
        - ROCm compatibility patches
        - Common Megatron initialization patterns
    """

    def __init__(self, primus_config: Any, module_config: Any, backend_args: Any):
        """
        Initialize Megatron SFT trainer.
        
        Args:
            primus_config: Full Primus configuration
            module_config: Module-specific configuration
            backend_args: Megatron-LM argument namespace (from MegatronArgBuilder)
        """
        # Initialize MegatronBaseTrainer (which initializes BaseTrainer)
        super().__init__(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args,
        )
        
        # Training components (will be set during training)
        self.model = None
        self.optimizer = None
        self.opt_param_scheduler = None
        
        log_rank_0(f"Initialized MegatronSFTTrainer for model: {module_config.model or 'custom'}")

    def setup(self):
        """
        Setup phase for Megatron SFT training.
        
        Can be used for pre-initialization setup if needed.
        """
        log_rank_0("MegatronSFTTrainer.setup()")
        # Any pre-initialization setup can go here

    def init(self):
        """
        Initialize Megatron SFT training components.
        
        Note:
            Argument injection is already done by MegatronBaseTrainer.__init__()
            This method can be used for trainer-specific initialization.
        """
        log_rank_0("Initializing Megatron SFT training...")
        log_rank_0(f"Model: {self.module_config.model or 'custom'}")
        log_rank_0(f"Framework: {self.module_config.framework}")
        
        # Get SFT-specific configuration from backend_args
        if hasattr(self.backend_args, 'sft_dataset_name'):
            log_rank_0(f"SFT Dataset: {self.backend_args.sft_dataset_name}")
        if hasattr(self.backend_args, 'sft_conversation_format'):
            log_rank_0(f"Conversation Format: {self.backend_args.sft_conversation_format}")

    def run_train(self):
        """
        Execute Megatron SFT training.
        
        This method is called by MegatronBaseTrainer.run() after applying patches.
        It executes the main SFT training loop using Megatron-LM's infrastructure.
        """
        log_rank_0("Executing Megatron SFT training...")
        
        import inspect
        
        # Import Megatron components
        from megatron.core.enums import ModelType
        from megatron.training import pretrain  # type: ignore
        
        from primus.core.utils.import_utils import get_model_provider
        
        # Import SFT-specific components
        from primus.backends.megatron.core.datasets.sft_dataset import (
            build_train_valid_test_datasets,
        )
        
        # Create dataset provider function
        def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
            """Build train, valid, and test datasets for SFT."""
            from megatron.training import get_args, get_tokenizer
            
            args = get_args()
            tokenizer = get_tokenizer()
            
            # Get SFT-specific config (with defaults)
            dataset_name = getattr(args, 'sft_dataset_name', 'tatsu-lab/alpaca')
            conversation_format = getattr(args, 'sft_conversation_format', 'alpaca')
            
            log_rank_0(f"Building SFT datasets from: {dataset_name}")
            log_rank_0(f"Using conversation format: {conversation_format}")
            
            # Build datasets
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_seq_length=args.seq_length,
                train_val_test_num_samples=train_val_test_num_samples,
                formatter=conversation_format,
                seed=args.seed,
            )
            
            return train_ds, valid_ds, test_ds
        
        # Mark as distributed (required by Megatron)
        train_valid_test_datasets_provider.is_distributed = True
        
        # Import forward step function from separate module
        from .sft_forward_step import create_sft_forward_step
        
        # Create forward step function for SFT
        forward_step = create_sft_forward_step()
        
        # Megatron versions differ:
        # - v0.12.0: calls `pretrain(...)` directly (no inprocess_restart wrapper).
        # - newer: wraps pretrain via inprocess_restart and may pass `store=...`.
        wrapped_pretrain = pretrain
        store = None
        try:
            from megatron.training import inprocess_restart  # type: ignore
            
            if hasattr(inprocess_restart, "maybe_wrap_for_inprocess_restart"):
                wrapped_pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)
        except (ImportError, AttributeError) as e:
            # Expected for older Megatron versions that don't have inprocess_restart
            log_rank_0(f"Inprocess restart not available, using standard pretrain: {e}")
            pass
        
        # Execute training
        sig = inspect.signature(wrapped_pretrain)
        kwargs = {}
        if "args_defaults" in sig.parameters:
            # Matches upstream pretrain_gpt entrypoints
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
        
        log_rank_0("Megatron SFT training execution completed.")
