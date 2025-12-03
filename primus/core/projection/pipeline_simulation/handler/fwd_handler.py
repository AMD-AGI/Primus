""" Handle for forward """

from ..scheduler.scheduler_node import FuncType, SchedulerNode
from .utils import find_prev_node_with_type
from .wgrad_handler import WGRAD_RUNNING_CACHE


def default_check_fwd_node_valid(node: SchedulerNode):
    assert node.func_type == FuncType.F
    args = node.args
    assert isinstance(args, dict)
    assert "fwd_func" in args


def default_fwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    default_check_fwd_node_valid(node)
    WGRAD_RUNNING_CACHE.set_current_minibatch_and_chunk(node.mini_batch, node.chunk)
    # prepare input, if not found, input is None(fwd_func will handle it)
    recv_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RF])

    input = None
    if recv_node_idx is not None:
        if "req" in scheduler_table[recv_node_idx].args:
            scheduler_table[recv_node_idx].args["req"].wait()
            scheduler_table[recv_node_idx].args["req"] = None
            del scheduler_table[recv_node_idx].args["req"]
        input = scheduler_table[recv_node_idx].args["recv_buffer"]

    output = node.args["fwd_func"](input, node.mini_batch, node.chunk)
    node.args["output"] = output
    node.args["input"] = input

    if recv_node_idx is not None:
        scheduler_table[recv_node_idx].args["recv_buffer"] = None
