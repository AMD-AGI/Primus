###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
import os
from typing import Callable

from primus.core.pipeline_parallel.handler.offload_handler import OFFLOAD_BUFFER
from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType


class ScheduleRunner:
    def __init__(
        self,
        handle_func_dict: dict[FuncType, Callable],
        pre_process_func: Callable = None,
        post_process_func: Callable = None,
    ):
        self.handle_func_dict = handle_func_dict
        self.pre_process_func = pre_process_func
        self.post_process_func = post_process_func

    def run(self, scheduler_table, rank: int):
        for idx, node in enumerate(scheduler_table[rank]):
            if node.args is not None and "combined_group" in node.args:
                func = self.handle_func_dict[FuncType.FB]
                func(node, idx, scheduler_table[rank])
            else:
                if self.pre_process_func is not None:
                    self.pre_process_func(node, idx, scheduler_table[rank])
                func = self.handle_func_dict[node.func_type]
                func(node, idx, scheduler_table[rank])
                if self.post_process_func is not None:
                    self.post_process_func(node, idx, scheduler_table[rank])

        OFFLOAD_BUFFER.check_empty()
        record_offload_memory_info_dir = os.environ.get("RECORD_OFFLOAD_MEMORY_INFO_DIR", "output")
        result_dir = f"{record_offload_memory_info_dir}/offload_memory_info/rank-{rank}"
        if OFFLOAD_BUFFER.record_offload_memory_info:
            os.makedirs(result_dir, exist_ok=True)
            with open(f"{result_dir}/offload_memory_info.txt", "w") as f:
                OFFLOAD_BUFFER.print_offload_memory_info(f)
