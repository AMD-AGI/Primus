""" Handle for BW """

from primus.core.projection.pipeline_simulation.handler.utils import (
    find_prev_node_with_type,
)

from ..scheduler.scheduler_node import FuncType, SchedulerNode


def default_bwd_wgrad_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):

    fwd_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.F])
    assert fwd_node_idx is not None
    output = scheduler_table[fwd_node_idx].args["output"]
    input = scheduler_table[fwd_node_idx].args["input"]
    recv_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RB])

    output_grad = None

    if recv_node_idx is not None:
        if "req" in scheduler_table[recv_node_idx].args:
            scheduler_table[recv_node_idx].args["req"].wait()
            scheduler_table[recv_node_idx].args["req"] = None
            del scheduler_table[recv_node_idx].args["req"]

        output_grad = scheduler_table[recv_node_idx].args["recv_buffer"]

    # run backward
    input_grad = node.args["backward_func"](input, output, output_grad, node.mini_batch, node.chunk)
    node.args["output"] = input_grad

    scheduler_table[fwd_node_idx].args["output"] = None  # release memory
    scheduler_table[fwd_node_idx].args["input"] = None
