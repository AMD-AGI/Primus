from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType, SchedulerNode


def find_prev_node_with_type(scheduler_table: list[SchedulerNode], cur_idx: int, func_types: list[FuncType]):
    for i in range(cur_idx):
        if (
            scheduler_table[i].func_type in func_types
            and scheduler_table[i].mini_batch == scheduler_table[cur_idx].mini_batch
            and scheduler_table[i].chunk == scheduler_table[cur_idx].chunk
        ):
            return i
    return None
