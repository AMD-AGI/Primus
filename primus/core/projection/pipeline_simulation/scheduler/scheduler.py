from typing import Callable

import torch

from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import FuncType


class ScheduleRunner:
    def __init__(self, handle_func_dict: dict[FuncType, Callable]):
        self.handle_func_dict = handle_func_dict

    def run(self, scheduler_table, rank: int):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.synchronize()
        for idx, node in enumerate(scheduler_table[rank]):
            func = self.handle_func_dict[node.func_type]
            func(node, idx, scheduler_table[rank])
