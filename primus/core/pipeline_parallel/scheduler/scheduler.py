from typing import Callable

import torch

from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType


class ScheduleRunner:
    def __init__(self, handle_func_dict: dict[FuncType, Callable]):
        self.handle_func_dict = handle_func_dict

    def run(self, scheduler_table, rank: int):
        for idx, node in enumerate(scheduler_table[rank]):
            # print(f"node {node} start")
            func = self.handle_func_dict[node.func_type]
            func(node, idx, scheduler_table[rank])
            # print(f"node {node} end")

