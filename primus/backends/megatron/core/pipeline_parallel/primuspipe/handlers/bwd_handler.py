###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.core.pipeline_parallel.schedules import backward_step
from megatron.training.global_vars import get_args

from primus.core.pipeline_parallel.handler.utils import find_prev_node_with_type
from primus.core.pipeline_parallel.handler.wgrad_handler import WGRAD_RUNNING_CACHE
from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)
from primus.modules.trainer.megatron.utils import fwd_bwd_wrapper


def megatron_check_bwd_node_valid(node: SchedulerNode):
    assert node.func_type in [FuncType.B, FuncType.BW], f"node.func_type is {node.func_type}"
    args = node.args
    assert isinstance(args, dict)
    assert "config" in args
    assert "model_type" in args


def megatron_bwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    megatron_check_bwd_node_valid(node)

    # get inputs and grads tensors
    fwd_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.F])

    assert fwd_node_idx is not None
    outputs = scheduler_table[fwd_node_idx].args["outputs"]
    input_tensors = scheduler_table[fwd_node_idx].args["inputs"]

    recv_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RB])

    if recv_node_idx is not None and "req" in scheduler_table[recv_node_idx].args:
        scheduler_table[recv_node_idx].args["req"].wait()
        scheduler_table[recv_node_idx].args["req"] = None
        del scheduler_table[recv_node_idx].args["req"]

    output_grad = (
        scheduler_table[recv_node_idx].args["recv_buffers"]
        if recv_node_idx is not None
        else [None] * len(node.args["send_tensor_shapes"])
    )

    # run backward
    backward_step_ = backward_step
    if get_args().dump_pp_data:
        backward_step_ = fwd_bwd_wrapper(backward_step, "bwd", minibatch=node.mini_batch, chunk=node.chunk)

    WGRAD_RUNNING_CACHE.set_current_minibatch_and_chunk(node.mini_batch, node.chunk)
    input_tensor_grad = backward_step_(
        input_tensors, outputs, output_grad, node.args["model_type"], node.args["config"]
    )

    if fwd_node_idx is not None:  # release memory
        scheduler_table[fwd_node_idx].args["outputs"] = None
        scheduler_table[fwd_node_idx].args["inputs"] = None
    if recv_node_idx is not None:
        scheduler_table[recv_node_idx].args["recv_buffers"] = None

    assert isinstance(input_tensor_grad, list), "input_tensor_grad should be a list"

    node.args["outputs"] = [grad.clone().detach() for grad in input_tensor_grad if grad is not None]
