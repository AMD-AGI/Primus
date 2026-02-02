###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MegatronSFTTrainer: Megatron-LM based supervised fine-tuning trainer."""

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
        - Direct Megatron-LM integration
        - Support for various model architectures (Llama, GPT, etc.)
        - LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
    
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
        
        # Initialize LoRA if enabled
        self.peft = None
        self._init_lora()
        
        log_rank_0(f"Initialized MegatronSFTTrainer for model: {module_config.model or 'custom'}")
    
    def _init_lora(self):
        """Initialize LoRA (Low-Rank Adaptation) if enabled in config."""
        lora_enabled = getattr(self.backend_args, 'lora_enabled', False)
        
        if not lora_enabled:
            log_rank_0("LoRA disabled, using full fine-tuning")
            return
        
        from primus.backends.megatron.peft import LoRA
        
        # Get LoRA configuration from backend_args
        lora_rank = getattr(self.backend_args, 'lora_rank', 32)
        lora_alpha = getattr(self.backend_args, 'lora_alpha', 32)
        lora_dropout = getattr(self.backend_args, 'lora_dropout', 0.0)
        lora_dropout_position = getattr(self.backend_args, 'lora_dropout_position', 'pre')
        lora_A_init = getattr(self.backend_args, 'lora_A_init_method', 'xavier')
        lora_B_init = getattr(self.backend_args, 'lora_B_init_method', 'zero')
        lora_target_modules = getattr(
            self.backend_args, 
            'lora_target_modules', 
            ['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']
        )
        
        log_rank_0(f"Initializing LoRA with rank={lora_rank}, alpha={lora_alpha}, "
                   f"dropout={lora_dropout}, targets={lora_target_modules}")
        
        self.peft = LoRA(
            target_modules=lora_target_modules,
            dim=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            dropout_position=lora_dropout_position,
            lora_A_init_method=lora_A_init,
            lora_B_init_method=lora_B_init,
        )

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

    def _create_model_provider_with_lora(self, base_model_provider):
        """
        Wrap the model provider to apply LoRA after model creation.
        
        Args:
            base_model_provider: Original model provider function
            
        Returns:
            Wrapped model provider that applies LoRA to the created model
        """
        peft = self.peft
        
        def model_provider_with_lora(pre_process=True, post_process=True):
            """Model provider that applies LoRA after model creation."""
            # Create the base model
            model = base_model_provider(pre_process=pre_process, post_process=post_process)
            
            # Apply LoRA if enabled
            if peft is not None:
                log_rank_0("=" * 60)
                log_rank_0("Applying LoRA to model...")
                model = peft(model, training=True)
                
                # Log trainable parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                frozen_params = total_params - trainable_params
                
                log_rank_0(f"LoRA Summary:")
                log_rank_0(f"  - Total parameters:     {total_params:,}")
                log_rank_0(f"  - Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
                log_rank_0(f"  - Frozen parameters:    {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
                log_rank_0("=" * 60)
            
            return model
        
        return model_provider_with_lora

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
        
        # Import forward step function from SFT module
        from .sft.forward_step import create_sft_forward_step
        
        # Create forward step function for SFT
        forward_step = create_sft_forward_step()
        
        # Get model provider, wrap with LoRA if enabled
        base_model_provider = get_model_provider()
        if self.peft is not None:
            model_provider = self._create_model_provider_with_lora(base_model_provider)
            log_rank_0("Using LoRA-enabled model provider")
        else:
            model_provider = base_model_provider
        
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
            model_provider,
            ModelType.encoder_or_decoder,
            forward_step,
            **kwargs,
        )
        
        log_rank_0("Megatron SFT training execution completed.")
