###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Callable

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
            if self.pre_process_func is not None:
                self.pre_process_func(node, idx, scheduler_table[rank])
            func = self.handle_func_dict[node.func_type]
            func(node, idx, scheduler_table[rank])
            if self.post_process_func is not None:
                self.post_process_func(node, idx, scheduler_table[rank])
