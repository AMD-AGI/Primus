###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step
from megatron.core.pipeline_parallel.schedules import backward_step
from megatron.training.global_vars import get_args

from primus.core.pipeline_parallel.handler.offload_handler import OFFLOAD_BUFFER
from primus.core.pipeline_parallel.handler.wgrad_handler import WGRAD_RUNNING_CACHE
from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)
from primus.core.pipeline_parallel.utils import find_prev_node_with_type
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
    fwd_node = scheduler_table[fwd_node_idx]

    assert fwd_node_idx is not None
    outputs = scheduler_table[fwd_node_idx].args["outputs"]

    if "should_offload" in fwd_node.args and fwd_node.args["should_offload"]:
        OFFLOAD_BUFFER.wait_reload_done(fwd_node.mini_batch, fwd_node.chunk)

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

    models = fwd_node.args["models"]

    kwargs = None

    if not get_args().overlap_moe_expert_parallel_comm:
        # run backward
        backward_step_ = backward_step
        kwargs = {
            "input_tensor": input_tensors,
            "output_tensor": outputs,
            "output_tensor_grad": output_grad,
            "model_type": node.args["model_type"],
            "config": node.args["config"],
        }
    else:
        backward_step_ = combined_forward_backward_step
        kwargs = {
            "forward_step_func": None,
            "data_iterator": None,
            "f_model": None,
            "num_microbatches": None,
            "input_tensor": None,
            "forward_data_store": None,
            "b_model": models[node.chunk],
            "b_input_tensor": input_tensors,
            "b_output_tensor": outputs,
            "b_output_tensor_grad": output_grad,
            "config": node.args["config"],
            "f_model_chunk_id": None,
            "pre_forward": None,
            "pre_backward": None,
            "post_forward": None,
            "post_backward": None,
            "collect_non_loss_data": fwd_node.args["collect_non_loss_data"],
            "checkpoint_activations_microbatch": None,
            "is_first_microbatch": (node.mini_batch == 0),
            "current_microbatch": node.mini_batch,
            "encoder_decoder_xattn": False,
        }

    if get_args().dump_pp_data:
        backward_step_ = fwd_bwd_wrapper(backward_step_, "bwd", minibatch=node.mini_batch, chunk=node.chunk)

    WGRAD_RUNNING_CACHE.set_current_minibatch_and_chunk(node.mini_batch, node.chunk)

    input_tensor_grad = backward_step_(**kwargs)

    assert input_tensor_grad is not None, "input_tensor_grad should not be None"

    if get_args().overlap_moe_expert_parallel_comm:
        input_tensor_grad = input_tensor_grad[2]

    assert isinstance(
        input_tensor_grad, list
    ), f"input_tensor_grad should be a list, but is {type(input_tensor_grad)}"

    if fwd_node_idx is not None:  # release memory
        scheduler_table[fwd_node_idx].args["outputs"] = None
        scheduler_table[fwd_node_idx].args["inputs"] = None
    if recv_node_idx is not None:
        scheduler_table[recv_node_idx].args["recv_buffers"] = None

    assert isinstance(input_tensor_grad, list), "input_tensor_grad should be a list"

    node.args["outputs"] = [grad.clone().detach() for grad in input_tensor_grad if grad is not None]
