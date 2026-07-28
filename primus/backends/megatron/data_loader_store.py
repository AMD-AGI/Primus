###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron batch-loading helpers used by the pipeline schedules.

``get_batch_func`` builds a micro-batch honoring TP/CP sharding (and the
ZeroBubble sequence-split path), and ``DataLoaderStore`` provides the
push/pop cache the ZeroBubble runtime and the ``forward_step`` patch use to
pre-fetch batches (optionally on a dedicated H2D stream).
"""

import collections

import torch
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    is_first_or_last_pipeline_stage,
)

stimer = StragglerDetector()

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

    # Return a stable, explicitly-ordered 5-tuple so both this path and the
    # early not-first/last-stage return path have the same shape/type. The keys
    # match megatron's ``get_batch_on_this_tp_rank`` and the unpack order at all
    # call sites (tokens, labels, loss_mask, attention_mask, position_ids).
    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
    )


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
