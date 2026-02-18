###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import collections
from functools import partial

import torch
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector, get_attr_wrapped_model
from megatron.training import get_args, get_timers
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    is_first_or_last_pipeline_stage,
)

stimer = StragglerDetector()

from primus.modules.module_utils import debug_rank_0, log_rank_0

from .trainer import MegatronTrainer

mb_batch = None


def get_batch_func(data_iterator, vp_stage=None):
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage):
        return None, None, None, None, None

    # assert data_iterator is not None, f"data_iterator is None vp_stage: {vp_stage}"
    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    args = get_args()

    if args.patch_zero_bubble:
        from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_vars import (
            get_seq_split_idx,
        )

        global mb_batch
        # "or 0" to support original 1f1b and interleaved-1f1b in schedules.py
        seq_split_idx = get_seq_split_idx() or 0
        if seq_split_idx == 0:
            # get batches based on the TP rank you are on
            mb_batch = get_batch_on_this_tp_rank(data_iterator)
            assert (
                mb_batch["attention_mask"] is None
            ), "attention_mask should be None, please enable --no-create-attention-mask-in-dataloader"
        batch = {}
        for k in mb_batch.keys():
            v = mb_batch[k]
            if v is None:
                batch[k] = v
                continue

            assert v.shape[1] % get_args().num_seq_splits == 0, f"{k} size {v.shape}"
            start_idx = seq_split_idx * v.shape[1] // get_args().num_seq_splits
            end_idx = (seq_split_idx + 1) * v.shape[1] // get_args().num_seq_splits
            if len(v.shape) > 2:
                batch[k] = v[:, start_idx:end_idx, :].contiguous()
            else:
                batch[k] = v[:, start_idx:end_idx].contiguous()

    if args.context_parallel_size > 1 and args.enable_primus_turbo and args.use_turbo_attention:
        try:
            from primus.backends.megatron.core.utils import (
                produce_attention_sharder,
                shard_batch_on_this_cp_rank,
            )
        except:
            raise ImportError("Module 'primus_turbo' may not installed. Please install it")
        sharder = produce_attention_sharder(args.cp_comm_type)
        batch = shard_batch_on_this_cp_rank(sharder, batch)
    else:
        batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


class DataLoaderStore:
    cache = collections.deque()

    @classmethod
    def push(cls, data_iterator, h2d_stream=False, vp_stage=None):
        timers = get_timers()
        # Get the batch.
        timers("batch-generator", log_level=2).start()
        global stimer

        with stimer(bdata=True):
            if h2d_stream:
                from primus.backends.megatron.core.pipeline_parallel.zerobubble.offload import (
                    get_offload_h2d_stream,
                )

                load_event = torch.cuda.Event()
                original_stream = torch.cuda.current_stream()
                with torch.cuda.stream(get_offload_h2d_stream()):
                    data = get_batch_func(data_iterator, vp_stage)
                    for x in data:
                        if x is not None:
                            x.record_stream(original_stream)
                    load_event.record()
                    cls.cache.append((data, load_event))
            else:
                cls.cache.append((get_batch_func(data_iterator, vp_stage), None))
        timers("batch-generator").stop()

    @classmethod
    def pop(cls):
        data, load_event = cls.cache.popleft()
        if load_event:
            load_event.wait()
        return data


class MegatronPretrainTrainer(MegatronTrainer):
    def __init__(self, *args, **kwargs):
        print(f"[PRIMUS-PRETRAINER] MegatronPretrainTrainer.__init__() entered", flush=True)
        kwargs["module_name"] = "pre_trainer"

        # Explicitly reject unknown extra_args
        extra_args = kwargs.pop("extra_args", None)
        if extra_args:
            raise ValueError(
                f"[MegatronPretrainTrainer] Unexpected extra_args detected: {extra_args}. "
                f"Megatron backend does not support unregistered config keys."
            )

        super().__init__(*args, **kwargs)
        print(f"[PRIMUS-PRETRAINER] MegatronPretrainTrainer.__init__() done", flush=True)

    def get_batch(self, data_iterator, vp_stage=None):
        """Generate a batch."""
        debug_rank_0(f"MegatronPretrainTrainer.get_batch() vp_stage={vp_stage}")
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
        debug_rank_0(f"loss_func() entered: loss_mask.shape={loss_mask.shape}, output_tensor.shape={output_tensor.shape}")
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        debug_rank_0(f"loss_func() total_tokens={total_tokens.item()}")
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])
        debug_rank_0(f"loss_func() raw masked loss={loss[0].item():.6f}")

        if args.context_parallel_size > 1:
            debug_rank_0(f"loss_func() all-reducing loss across context parallel group (cp_size={args.context_parallel_size})")
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            debug_rank_0(f"loss_func() loss after CP all-reduce={loss[0].item():.6f}")

        # Check individual rank losses are not NaN prior to DP all-reduce.
        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            debug_rank_0(f"loss_func() checking for NaN/Inf in loss...")
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
            debug_rank_0(f"loss_func() NaN/Inf check passed")
        # Check for spiky loss
        if args.check_for_spiky_loss:
            debug_rank_0(f"loss_func() checking for spiky loss...")
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
            debug_rank_0(f"loss_func() spiky loss check passed")
        # Reduce loss for logging.
        debug_rank_0(f"loss_func() all-reducing loss across DP group for reporting...")
        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
        debug_rank_0(f"loss_func() reporting_loss={reporting_loss[0].item():.6f}, reporting_tokens={reporting_loss[1].item()}")

        # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
        # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
        # on loss[0] fixes this
        local_num_tokens = loss[1].clone().detach().to(torch.int)
        debug_rank_0(f"loss_func() done: loss={loss[0].item():.6f}, local_num_tokens={local_num_tokens.item()}")
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
        debug_rank_0(f"MegatronPretrainTrainer.forward_step() entered: return_schedule_plan={return_schedule_plan}")
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

        debug_rank_0(f"forward_step() batch loaded, running model forward...")
        with stimer:
            if return_schedule_plan:
                assert (
                    args.overlap_moe_expert_parallel_comm
                ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
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
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

        return output_tensor, partial(self.loss_func, loss_mask)
