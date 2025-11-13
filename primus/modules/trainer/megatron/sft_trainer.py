###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################




import collections
from functools import partial

import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_ltor_masks_and_position_ids,
)
from .trainer import MegatronTrainer

stimer = StragglerDetector()

mb_batch = None


def get_eos_id():
    """Return EOS token ID compatible with HF chat templates."""
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643
    if hf_tokenizer.eos_token == "<|return|>":
        return 199999

    return hf_tokenizer.eos_token_id


def get_batch_func(data_iterator):
    """Build supervised fine-tuning batches with optional Primus/CP support."""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    # Only tensor-parallel rank 0 gets data, others broadcast
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    keys = ["input_ids", "loss_mask"]
    datatype = torch.int64
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0 : 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1 : 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1 : 1 + args.seq_length].contiguous()

    # build attention mask and position ids
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        get_eos_id(),
        get_eos_id(),
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        False,
    )

    # combine LM loss mask with answer-only mask
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask.contiguous(),
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

    # Primus zero-bubble support
    if args.patch_zero_bubble:
        from primus.backends.megatron.core.pipeline_parallel.zerobubble.zbpp_vars import (
            get_seq_split_idx,
        )
        global mb_batch
        seq_split_idx = get_seq_split_idx() or 0
        if seq_split_idx == 0:
            mb_batch = batch
        split_batch = {}
        for k, v in mb_batch.items():
            if v is None:
                split_batch[k] = v
                continue
            assert v.shape[1] % get_args().num_seq_splits == 0, f"{k} size {v.shape}"
            start_idx = seq_split_idx * v.shape[1] // get_args().num_seq_splits
            end_idx = (seq_split_idx + 1) * v.shape[1] // get_args().num_seq_splits
            if len(v.shape) > 2:
                split_batch[k] = v[:, start_idx:end_idx, :].contiguous()
            else:
                split_batch[k] = v[:, start_idx:end_idx].contiguous()
        batch = split_batch

    # context-parallel / turbo attention
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
    def push(cls, data_iterator, h2d_stream=False):
        timers = get_timers()
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
                    data = get_batch_func(data_iterator)
                    for x in data:
                        if x is not None:
                            x.record_stream(original_stream)
                    load_event.record()
                    cls.cache.append((data, load_event))
            else:
                cls.cache.append((get_batch_func(data_iterator), None))
        timers("batch-generator").stop()

    @classmethod
    def pop(cls):
        data, load_event = cls.cache.popleft()
        if load_event:
            load_event.wait()
        return data


class MegatronSFTTrainer(MegatronTrainer):
    """Supervised fine-tuning trainer with Primus + context parallel support."""

    def __init__(self, *args, **kwargs):
        kwargs["module_name"] = "sft_trainer"

        extra_args = kwargs.pop("extra_args", None)
        if extra_args:
            raise ValueError(
                f"[MegatronSFTTrainer] Unexpected extra_args detected: {extra_args}. "
                f"Megatron backend does not support unregistered config keys."
            )

        super().__init__(*args, **kwargs)

    def get_batch(self, data_iterator):
        """Generate an SFT batch."""
        return get_batch_func(data_iterator)

    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function for supervised fine-tuning."""
        args = get_args()

        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), total_tokens.view(1)])

        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())

        rerun_state_machine = get_rerun_state_machine()
        if args.check_for_nan_in_loss_and_grad:
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isnan,
                message="found NaN in SFT loss calculation",
                tolerance=0.0,
                fatal=True,
            )
            rerun_state_machine.validate_result(
                result=loss[0],
                rejection_func=torch.isinf,
                message="found Inf in SFT loss calculation",
                tolerance=0.0,
                fatal=True,
            )

        reporting_loss = loss.clone().detach()
        torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())

        local_num_tokens = loss[1].clone().detach().to(torch.int)
        return (
            loss[0].clone(),
            local_num_tokens,
            {"sft loss": (reporting_loss[0], reporting_loss[1])},
        )

    def forward_step(self, data_iterator, model: GPTModel, return_schedule_plan=False):
        """Forward training step for SFT."""
        args = get_args()
        timers = get_timers()

        # Get batch (with zero-bubble support)
        if not args.patch_zero_bubble:
            timers("batch-generator", log_level=2).start()
            global stimer
            with stimer(bdata=True):
                tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(data_iterator)
            timers("batch-generator").stop()
        else:
            from collections.abc import Iterable

            if (
                not isinstance(data_iterator, Iterable) and not data_iterator is None
            ):  # isinstance(data_iterator, DataLoaderStore)
                tokens, labels, loss_mask, attention_mask, position_ids = data_iterator.pop()
            else:
                DataLoaderStore.push(data_iterator, h2d_stream=False)
                tokens, labels, loss_mask, attention_mask, position_ids = DataLoaderStore.pop()

        with stimer:
            if return_schedule_plan:
                assert (
                    args.overlap_moe_expert_parallel_comm
                ), "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(self.loss_func, loss_mask)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

        return output_tensor, partial(self.loss_func, loss_mask)
