###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Forward step function for Megatron SFT training.

This module contains the forward_step function used in supervised fine-tuning,
following the Megatron-Bridge pattern for loss computation.
"""

import torch
from typing import Callable, Iterator, Tuple


def create_sft_forward_step() -> Callable:
    """
    Create and return the forward_step function for SFT training.
    
    This follows the Megatron-Bridge pattern where:
    1. Model is called with labels and returns per-token losses
    2. Loss function applies masking to focus on response tokens
    3. Returns (loss, num_tokens, metrics_dict) for proper DP averaging
    
    Returns:
        forward_step function compatible with Megatron's pretrain loop
    """
    
    def forward_step(data_iterator: Iterator, model) -> Tuple:
        """
        Forward step for SFT training.
        
        Args:
            data_iterator: Iterator over training data batches
            model: Megatron GPT model
            
        Returns:
            Tuple of (output_tensor, loss_function)
            - output_tensor: Per-token losses from model
            - loss_function: Lambda that computes final loss with masking
        """
        from megatron.training import get_args
        
        args = get_args()
        
        # Handle case where data_iterator is None (e.g., during eval without valid dataset)
        if data_iterator is None:
            return None, lambda output: (
                torch.tensor(0.0, device='cuda'),
                torch.tensor(0, device='cuda'),
                {}
            )
        
        # Get batch from iterator
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Return None and a no-op loss function for iteration completion
            return None, lambda output: (
                torch.tensor(0.0, device='cuda'),
                torch.tensor(0, device='cuda'),
                {}
            )
        
        # Extract tensors from batch
        tokens = batch['input_ids'].long().cuda()
        labels = batch['labels'].long().cuda()
        loss_mask = batch['loss_mask'].float().cuda()
        
        # Ensure proper shapes [batch, seq]
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if loss_mask.dim() == 1:
            loss_mask = loss_mask.unsqueeze(0)
        
        # Generate position_ids: [batch, seq]
        # Position IDs are sequential: [0, 1, 2, ..., seq_length-1]
        batch_size, seq_len = tokens.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # attention_mask: None for causal mask (standard GPT autoregressive)
        attention_mask = None
        
        # Forward pass through model with labels
        # When labels are provided, model computes and returns per-token losses
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        
        # Create loss function following Megatron-Bridge pattern
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor) -> Tuple:
            """
            Masked next-token loss function.
            
            This function applies the loss mask to focus training only on
            response tokens (where mask=1), ignoring instruction tokens (mask=0).
            
            Args:
                loss_mask: Binary mask [batch, seq] where 1=compute loss, 0=ignore
                output_tensor: Per-token losses from model [batch, seq] or [batch*seq]
                
            Returns:
                Tuple of (loss, num_tokens, metrics_dict):
                - loss: Summed loss for backpropagation
                - num_tokens: Number of non-masked tokens for proper averaging
                - metrics_dict: Dictionary with reporting metrics for logging
                
            Note:
                This follows Megatron's standard loss function signature.
                The training loop will use num_tokens to properly average loss
                across different micro-batches and data-parallel ranks.
            """
            # Model returns per-token losses, flatten for processing
            losses = output_tensor.view(-1).float()
            loss_mask = loss_mask.view(-1).float()
            
            # Apply mask: only compute loss on response tokens (mask=1)
            # Instruction tokens (mask=0) are ignored
            loss = torch.sum(losses * loss_mask)
            
            # Count number of non-masked tokens
            # This is crucial for proper loss averaging across micro-batches
            num_tokens = loss_mask.sum().clone().detach().to(torch.int)
            
            # Create reporting loss for logging
            # Format: [loss_value, num_tokens] concatenated
            # This allows Megatron to compute proper weighted average across DP ranks
            reporting_loss = torch.cat([
                loss.clone().detach().view(1),
                num_tokens.view(1)
            ])
            
            # Return standard Megatron loss function signature
            # (loss, num_tokens, metrics_dict)
            return (loss, num_tokens, {"lm loss": reporting_loss})
        
        # Return output and loss function
        # The lambda captures loss_mask in its closure
        return output_tensor, lambda output: loss_func(loss_mask, output)
    
    return forward_step
