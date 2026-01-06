###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)


def find_prev_node_with_type(
    scheduler_table: list[SchedulerNode],
    cur_idx: int,
    func_types: list[FuncType],
    mini_batch=None,
    chunk=None,
):

    if mini_batch is None:
        mini_batch = scheduler_table[cur_idx].mini_batch
    if chunk is None:
        chunk = scheduler_table[cur_idx].chunk
    for i in range(cur_idx):
        if (
            scheduler_table[i].func_type in func_types
            and scheduler_table[i].mini_batch == mini_batch
            and scheduler_table[i].chunk == chunk
        ):
            return i
    return None
