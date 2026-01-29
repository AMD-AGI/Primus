###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from functools import partial

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model
from megatron.training import get_args, get_timers
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    is_first_or_last_pipeline_stage,
)

from .trainer import MegatronTrainer


class MegatronSFTTrainer(MegatronTrainer):
    """
    Trainer for supervised fine-tuning (SFT) using native Megatron backend.
    
    This trainer implements SFT-specific logic for:
    - Loading instruction-response pairs
    - Computing masked loss (only on response tokens)
    - Supporting standard Megatron parallelism (DP, TP, PP, CP)
    """

    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "sft_trainer"
        super().__init__(*args, **kwargs)

    def get_batch(self, data_iterator, vp_stage=None):
        """
        Generate a batch for SFT training.
        
        For SFT, the batch contains:
        - tokens: concatenated instruction + response tokens
        - labels: same as tokens (shifted by 1 for next-token prediction)
        - loss_mask: 1 for response tokens, 0 for instruction/padding tokens
        - attention_mask: causal attention mask
        - position_ids: position indices
        
        Args:
            data_iterator: Iterator over the dataset
            vp_stage: Virtual pipeline stage (for interleaved pipeline parallelism)
            
        Returns:
            Tuple of (tokens, labels, loss_mask, attention_mask, position_ids)
        """
        # For pipeline stages that don't process data, return None
        if not is_first_or_last_pipeline_stage(vp_stage):
            return None, None, None, None, None

        # Get batches based on the TP rank
        batch = get_batch_on_this_tp_rank(data_iterator)

        # Handle context parallelism
        args = get_args()
        if args.context_parallel_size > 1 and args.enable_primus_turbo and args.use_turbo_attention:
            try:
                from primus.backends.megatron.core.utils import (
                    produce_attention_sharder,
                    shard_batch_on_this_cp_rank,
                )
            except ImportError:
                raise ImportError("Module 'primus_turbo' may not be installed. Please install it")
            sharder = produce_attention_sharder(args.cp_comm_type)
            batch = shard_batch_on_this_cp_rank(sharder, batch)
        else:
            batch = get_batch_on_this_cp_rank(batch)

        return batch.values()

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """
        Loss function for SFT with masked loss computation.
        
        Only computes loss on response tokens (where loss_mask == 1).
        Instruction tokens and padding are masked out (loss_mask == 0).
        
        Args:
            loss_mask: Binary mask indicating which tokens to include in loss (1) or exclude (0)
            output_tensor: The tensor with the losses from the model
            
        Returns:
            Tuple of:
            - loss scalar for this micro-batch
            - number of non-masked tokens in this microbatch
            - dict containing reporting metrics on the loss and number of tokens
        """
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        
        # Apply mask to loss - only sum losses where loss_mask == 1
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        # Reduce across context parallel group if needed
        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check for NaN in loss (if enabled)
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,
                fatal=True,
            )

        # Check for spiky loss (if enabled)
        if args.check_for_spiky_loss:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=2.0,  # Threshold for spiky loss detection
                    context="loss",
                ),
                message="Spiky loss in SFT",
                tolerance=0.0,
                fatal=False,
            )

        # Reduce loss for logging across data parallel ranks
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {"sft loss": (reporting_loss[0], reporting_loss[1])},
        )

    def forward_step(self, data_iterator, model: GPTModel, return_schedule_plan=False):
        """
        Forward training step for SFT.
        
        Args:
            data_iterator: Input data iterator
            model: The GPT Model
            return_schedule_plan: Whether to return schedule plan (for MoE overlap)
            
        Returns:
            Tuple of (output_tensor, loss_func_partial)
        """
        args = get_args()
        timers = get_timers()

        # Get the batch
        timers("batch-generator", log_level=2).start()
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator, vp_stage
        )
        timers("batch-generator").stop()

        # Forward pass through the model
        if return_schedule_plan:
            # For MoE with overlap
            assert (
                args.overlap_moe_expert_parallel_comm
            ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, partial(self.loss_func, loss_mask)
        else:
            # Standard forward pass
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )

        return output_tensor, partial(self.loss_func, loss_mask)
