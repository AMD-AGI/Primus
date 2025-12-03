from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import (
    SchedulerNode,
)


def default_bwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]): ...
