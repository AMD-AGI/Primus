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
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers

from primus.backends.megatron.data_loader_store import DataLoaderStore, get_batch_func

from .trainer import MegatronTrainer

stimer = StragglerDetector()


class MegatronPretrainTrainer(MegatronTrainer):
    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "pre_trainer"

        # Explicitly reject unknown extra_args
        extra_args = kwargs.pop("extra_args", None)
        if extra_args:
            raise ValueError(
                f"[MegatronPretrainTrainer] Unexpected extra_args detected: {extra_args}. "
                f"Megatron backend does not support unregistered config keys."
            )

        super().__init__(*args, **kwargs)

    def get_batch(self, data_iterator, vp_stage=None):
        """Generate a batch."""
        return get_batch_func(data_iterator, vp_stage)

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            loss_mask (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses

        Returns:
            the loss scalar for this micro-batch
            the number of non-padded tokens in this microbatch
            a dict containing reporting metrics on the loss and number of tokens across
                the data parallel ranks
        """
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message="found NaN in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message="found Inf in local forward loss calculation",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=True,
            )
        # Check for spiky loss
        if args.check_for_spiky_loss:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=partial(
                    rerun_state_machine.is_unexpectedly_large,
                    threshold=SPIKY_LOSS_FACTOR,
                    context="loss",
                ),
                message="Spiky loss",
                tolerance=0.0,  # forward pass calculations are determinisic
                fatal=False,
            )
        # Reduce loss for logging.
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
        # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
        # on loss[0] fixes this
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {"lm loss": (reporting_loss[0], reporting_loss[1])},
        )

    def forward_step(self, data_iterator, model: GPTModel, return_schedule_plan=False):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        if not args.patch_zero_bubble:
            timers("batch-generator", log_level=2).start()
            global stimer
            with stimer(bdata=True):
                vp_stage = get_attr_wrapped_model(model, "vp_stage")
                tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
                    data_iterator, vp_stage
                )
            timers("batch-generator").stop()
        else:
            from collections.abc import Iterable

            vp_stage = get_attr_wrapped_model(model, "vp_stage")
            if (
                not isinstance(data_iterator, Iterable) and not data_iterator is None
            ):  # isinstance(data_iterator, DataLoaderStore):
                tokens, labels, loss_mask, attention_mask, position_ids = data_iterator.pop()
            else:
                DataLoaderStore.push(data_iterator, h2d_stream=False, vp_stage=vp_stage)
                tokens, labels, loss_mask, attention_mask, position_ids = DataLoaderStore.pop()

        with stimer:
            if return_schedule_plan:
                assert (
                    args.overlap_moe_expert_parallel_comm
                ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"

                # Schedule plan building is only supported for GPT models
                # Check if this is a Mamba model
                unwrapped_model = model
                while hasattr(unwrapped_model, "module"):
                    unwrapped_model = unwrapped_model.module
                model_class_name = unwrapped_model.__class__.__name__

                if "Mamba" in model_class_name:
                    raise NotImplementedError(
                        "Schedule plan building is not supported for Mamba models. "
                        "Please disable overlap_moe_expert_parallel_comm for Mamba."
                    )

                if args.patch_moe_overlap:
                    assert (
                        not args.delay_wgrad_compute
                    ), "Primus MoE overlap handles wgrad separately from the original Megatron implementation"
                    from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_utils import (
                        WeightGradStore,
                    )

                    WeightGradStore.enable_split_bw()
                    assert (
                        WeightGradStore.split_bw()
                    ), "WeightGradStore.split_bw is not supported, please make sure overlap_grad_reduce is disabled and gradient_accumulation_fusion is enabled"
                    from primus.backends.megatron.core.models.common.model_chunk_schedule_plan import (
                        TransformerModelChunkSchedulePlan,
                    )

                    schedule_plan = TransformerModelChunkSchedulePlan(
                        model, tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                    )
                else:
                    schedule_plan = model.build_schedule_plan(
                        tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                    )
                return schedule_plan, partial(self.loss_func, loss_mask)
            else:
                # Check if model supports loss_mask parameter
                # MambaModel doesn't accept loss_mask, but GPTModel does
                # Unwrap the model to get the actual model class
                unwrapped_model = model
                while hasattr(unwrapped_model, "module"):
                    unwrapped_model = unwrapped_model.module
                model_class_name = unwrapped_model.__class__.__name__

                if "Mamba" in model_class_name:
                    # MambaModel doesn't accept loss_mask parameter
                    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
                else:
                    # GPTModel and other models accept loss_mask parameter
                    output_tensor = model(
                        tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                    )

        return output_tensor, partial(self.loss_func, loss_mask)
