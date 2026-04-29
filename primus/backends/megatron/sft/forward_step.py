###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Forward step function for Megatron SFT training.

This module contains the forward_step function used in supervised fine-tuning,
following the Megatron-Bridge pattern for loss computation while staying
compatible with newer Megatron-LM forward-step entrypoints.
"""

from functools import partial
from typing import Callable, Iterator, Tuple

import torch


def _move_to_runtime_device(tensor: torch.Tensor) -> torch.Tensor:
    """Move tensors to CUDA when available, keep CPU for unit tests."""
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def _empty_loss_result(device: torch.device | None = None) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Build a no-op loss tuple with the expected Megatron shape."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return (
        torch.tensor(0.0, device=device),
        torch.tensor(0, device=device, dtype=torch.int),
        {},
    )


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
    
    def forward_step(data_iterator: Iterator, model, return_schedule_plan: bool = False) -> Tuple:
        """
        Forward step for SFT training.
        
        Args:
            data_iterator: Iterator over training data batches
            model: Megatron GPT model
            return_schedule_plan: Whether to return a schedule plan for
                newer Megatron pipeline schedulers
            
        Returns:
            Tuple of (output_tensor, loss_function)
            - output_tensor: Per-token losses from model
            - loss_function: Lambda that computes final loss with masking
        """
        from megatron.training import get_args

        args = get_args()

        # Handle case where data_iterator is None (e.g., during eval without valid dataset)
        if data_iterator is None:
            return None, lambda output: _empty_loss_result()
        
        # Get batch from iterator
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Return None and a no-op loss function for iteration completion
            return None, lambda output: _empty_loss_result()
        
        # Extract tensors from batch
        tokens = _move_to_runtime_device(batch["input_ids"]).long()
        labels = _move_to_runtime_device(batch["labels"]).long()
        loss_mask = _move_to_runtime_device(batch["loss_mask"]).float()
        packed_seq_params = batch.get("packed_seq_params")
        
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
        
        # Create loss function following Megatron-Bridge pattern
        def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model=None) -> Tuple:
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

        if getattr(args, "use_legacy_models", False):
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
            return output_tensor, partial(loss_func, loss_mask, model=model)

        if return_schedule_plan:
            if not hasattr(model, "build_schedule_plan"):
                raise AttributeError(
                    "Megatron SFT forward_step received return_schedule_plan=True, "
                    "but the model does not implement build_schedule_plan()."
                )

            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            return schedule_plan, partial(loss_func, loss_mask, model=model)

        model_kwargs = {
            "labels": labels,
            "loss_mask": loss_mask,
        }
        if packed_seq_params is not None:
            model_kwargs["packed_seq_params"] = packed_seq_params

        output_tensor = model(tokens, position_ids, attention_mask, **model_kwargs)

        return output_tensor, partial(loss_func, loss_mask, model=model)
    
    return forward_step
