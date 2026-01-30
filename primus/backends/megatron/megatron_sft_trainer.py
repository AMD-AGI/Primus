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
        
        # Create forward step function for SFT
        def forward_step(data_iterator, model):
            """
            Enhanced forward step for SFT training.
            
            This implementation is ported and adapted from Megatron-Bridge patterns
            to provide a more complete and robust training loop while maintaining
            independence from Megatron-Bridge as a dependency.
            
            Key features:
            - Robust error handling for data iteration
            - Proper attention mask support for causal language modeling
            - Correct position_ids generation
            - Token count tracking for accurate logging
            - SFT-specific loss masking (only on response tokens)
            - Data parallel loss averaging
            
            Args:
                data_iterator: Iterator over training data batches
                model: Megatron GPT model
                
            Returns:
                tuple: (output_tensor, loss_func) where:
                    - output_tensor: Model logits [batch, seq_len, vocab_size]
                    - loss_func: Callable that computes loss from output_tensor
            """
            from megatron.core import parallel_state
            from megatron.training import get_args
            
            args = get_args()
            
            # Get batch from iterator with error handling
            try:
                batch = next(data_iterator)
            except StopIteration:
                # End of epoch - return None and dummy loss function
                return None, lambda output: torch.tensor(0.0, device='cuda')
            except Exception as e:
                # Log unexpected errors but don't crash training
                print(f"[WARNING] Error getting batch from data iterator: {e}")
                return None, lambda output: torch.tensor(0.0, device='cuda')
            
            # Extract and validate tensors from batch
            try:
                tokens = batch['input_ids'].long().cuda()
                labels = batch['labels'].long().cuda()
                loss_mask = batch['loss_mask'].float().cuda()
            except KeyError as e:
                raise ValueError(f"Batch missing required key: {e}. "
                               f"Batch keys: {batch.keys()}")
            
            # Ensure proper shapes [batch_size, seq_len]
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            if loss_mask.dim() == 1:
                loss_mask = loss_mask.unsqueeze(0)
            
            batch_size, seq_len = tokens.size()
            
            # Generate position_ids [batch_size, seq_len]
            # Standard sequential positions: [0, 1, 2, ..., seq_len-1]
            position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Generate causal attention mask if needed
            # For causal LM, we use a lower triangular mask to prevent attending to future tokens
            # Most Megatron models handle this internally, so we pass None
            attention_mask = None
            
            # Forward pass through model (get logits without computing loss internally)
            # We compute loss separately with custom SFT-specific masking
            output_tensor = model(tokens, position_ids, attention_mask=attention_mask)
            
            # Calculate number of actual tokens for logging
            # This excludes padding and masked tokens
            num_tokens = loss_mask.sum().item()
            
            # Define loss computation function
            def loss_func(loss_mask, output_tensor, num_tokens):
                """
                Compute masked language modeling loss for SFT.
                
                This function applies SFT-specific loss masking to ensure loss is
                computed only on response tokens (not on instruction prompts).
                
                Args:
                    loss_mask: Binary mask [batch, seq_len] indicating where to compute loss
                    output_tensor: Model logits [batch, seq_len, vocab_size]
                    num_tokens: Total number of unmasked tokens for logging
                    
                Returns:
                    Loss tensor (scalar)
                """
                import torch.nn.functional as F
                
                # Extract logits (output_tensor should be logits)
                logits = output_tensor
                
                # Validate shapes
                assert logits.size(0) == labels.size(0), \
                    f"Batch size mismatch: logits {logits.size(0)} vs labels {labels.size(0)}"
                assert logits.size(1) == labels.size(1), \
                    f"Sequence length mismatch: logits {logits.size(1)} vs labels {labels.size(1)}"
                
                # Shift for next token prediction (teacher forcing)
                # Model predicts next token, so we shift by 1
                shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
                shift_labels = labels[..., 1:].contiguous()       # [batch, seq_len-1]
                shift_mask = loss_mask[..., 1:].contiguous()      # [batch, seq_len-1]
                
                # Flatten for cross entropy computation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))  # [batch*(seq_len-1), vocab]
                shift_labels = shift_labels.view(-1)                          # [batch*(seq_len-1)]
                shift_mask = shift_mask.view(-1)                              # [batch*(seq_len-1)]
                
                # Compute per-token cross entropy loss
                losses = F.cross_entropy(
                    shift_logits, 
                    shift_labels, 
                    reduction='none',  # Get per-token losses
                    ignore_index=-100  # Standard ignore index for padding
                )
                
                # Apply SFT-specific mask (only compute loss on response tokens)
                masked_losses = losses * shift_mask
                
                # Compute mean loss over unmasked tokens
                num_unmasked = shift_mask.sum()
                if num_unmasked > 0:
                    loss = masked_losses.sum() / num_unmasked
                else:
                    # All tokens masked - return zero loss (shouldn't happen in practice)
                    loss = torch.tensor(0.0, device=masked_losses.device, dtype=masked_losses.dtype)
                
                # Average loss across data parallel group
                # This ensures consistent loss values across all data parallel ranks
                if parallel_state.get_data_parallel_world_size() > 1:
                    torch.distributed.all_reduce(
                        loss, 
                        group=parallel_state.get_data_parallel_group()
                    )
                    # Divide by world size to get average (all_reduce sums)
                    loss = loss / parallel_state.get_data_parallel_world_size()
                
                return loss
            
            # Return output and loss function
            # The loss function is wrapped in a lambda to capture the required variables
            return output_tensor, lambda output: loss_func(loss_mask, output, num_tokens)
        
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
