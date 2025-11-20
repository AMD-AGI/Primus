###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
from functools import lru_cache

from megatron.core import parallel_state
from primus_turbo.pytorch.ops.attention.attention_utils import (
    All2AllAttentionSharder,
    AttentionSharder,
)


@lru_cache
def produce_attention_sharder(cp_comm_type: str):
    if cp_comm_type == "a2a":
        return All2AllAttentionSharder()
    else:
        raise ValueError(f"Unsupported cp_comm_type: {cp_comm_type}")


def shard_batch_on_this_cp_rank(sharder: AttentionSharder, batch):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_group = parallel_state.get_context_parallel_group()
    if cp_size > 1:
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                batch[key] = sharder.shard_cp_input([val], cp_group, seq_dim)[0]
    return batch
