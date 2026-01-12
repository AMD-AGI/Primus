###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor

from primus.backends.megatron.core.pipeline_parallel.primuspipe.handlers.communication_handler import (
    batch_p2p_communication_handler,
)
from primus.core.pipeline_parallel.handler.wgrad_handler import WGRAD_RUNNING_CACHE
from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)
from primus.core.pipeline_parallel.utils import find_prev_node_with_type


def megatron_check_combined_fwd_bkwd_node_valid(node: SchedulerNode):
    args = node.args
    assert "combined_group" in args and args["combined_group"] is not None


def megatron_combined_fwd_bkwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    megatron_check_combined_fwd_bkwd_node_valid(node)
    # only first node need to be handled

    node_key = f"({node.func_type.name}|{node.mini_batch}|{node.chunk})"

    if node_key != node.args["combined_group"][0]:
        return

    combined_group = node.args["combined_group"]
    combined_nodes = scheduler_table[idx : idx + len(combined_group)]

    fwd_node = combined_nodes[0]

    # prepare fwd input data
    fwd_prev_node_idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RF])
    if fwd_prev_node_idx is not None and "req" in scheduler_table[fwd_prev_node_idx].args:
        scheduler_table[fwd_prev_node_idx].args["req"].wait()
        scheduler_table[fwd_prev_node_idx].args["req"] = None
        del scheduler_table[fwd_prev_node_idx].args["req"]

    input_tensors = (
        [None] * len(node.args["recv_tensor_shapes"])
        if fwd_prev_node_idx is None
        else scheduler_table[fwd_prev_node_idx].args["recv_buffers"]
    )

    is_last_stage = (
        fwd_node.meta["last_pp_stage_rank"] == node.meta["pp_rank"]
        and node.chunk == node.meta["vpp_size"] - 1
    )

    # prepare bwd input data
    bwd_node = combined_nodes[-1]
    bwd_idx = idx + len(combined_nodes) - 1
    bwd_fwd_node_idx = find_prev_node_with_type(scheduler_table, bwd_idx, [FuncType.F])

    assert bwd_fwd_node_idx is not None
    bwd_output_tensors = scheduler_table[bwd_fwd_node_idx].args["outputs"]
    bwd_input_tensors = scheduler_table[bwd_fwd_node_idx].args["inputs"]

    bwd_recv_node_idx = find_prev_node_with_type(scheduler_table, bwd_idx, [FuncType.RB])

    if bwd_recv_node_idx is not None and "req" in scheduler_table[bwd_recv_node_idx].args:
        scheduler_table[bwd_recv_node_idx].args["req"].wait()
        scheduler_table[bwd_recv_node_idx].args["req"] = None
        del scheduler_table[bwd_recv_node_idx].args["req"]

    output_grad = (
        scheduler_table[bwd_recv_node_idx].args["recv_buffers"]
        if bwd_recv_node_idx is not None
        else [None] * len(node.args["send_tensor_shapes"])
    )

    models = fwd_node.args["models"]

    WGRAD_RUNNING_CACHE.set_current_minibatch_and_chunk(bwd_node.mini_batch, bwd_node.chunk)

    output_tensor, num_tokens, input_tensor_grad = combined_forward_backward_step(
        forward_step_func=fwd_node.args["forward_step_func"],
        data_iterator=fwd_node.args["data_iterator"][fwd_node.chunk],
        f_model=models[fwd_node.chunk],
        num_microbatches=fwd_node.args["num_microbatches"],
        input_tensor=input_tensors[0],
        forward_data_store=fwd_node.args["forward_data_store"],
        b_model=models[bwd_node.chunk],
        b_input_tensor=bwd_input_tensors,
        b_output_tensor=bwd_output_tensors,
        b_output_tensor_grad=output_grad,
        config=fwd_node.args["config"],
        f_model_chunk_id=fwd_node.chunk,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        collect_non_loss_data=fwd_node.args["collect_non_loss_data"],
        checkpoint_activations_microbatch=None,
        is_first_microbatch=(fwd_node.mini_batch == 0),
        current_microbatch=fwd_node.mini_batch,
        encoder_decoder_xattn=False,
    )

    # fwd post process
    fwd_node.args["total_num_tokens"] += num_tokens

    fwd_node.args["outputs"] = output_tensor if isinstance(output_tensor, list) else [output_tensor]
    fwd_node.args["inputs"] = input_tensors

    if is_last_stage:
        deallocate_output_tensor(output_tensor[0], fwd_node.args["config"].deallocate_pipeline_outputs)
    if fwd_prev_node_idx is not None:
        scheduler_table[fwd_prev_node_idx].args["recv_buffers"] = None

    # bwd post process
    if bwd_fwd_node_idx is not None:  # release memory
        scheduler_table[bwd_fwd_node_idx].args["outputs"] = None
        scheduler_table[bwd_fwd_node_idx].args["inputs"] = None
    if bwd_recv_node_idx is not None:
        scheduler_table[bwd_recv_node_idx].args["recv_buffers"] = None

    assert isinstance(input_tensor_grad, list), "input_tensor_grad should be a list"
    bwd_node.args["outputs"] = [grad.clone().detach() for grad in input_tensor_grad if grad is not None]

    # handle internal communication
    if len(combined_nodes) > 2:
        interal_communication_nodes = combined_nodes[1:-1]
        for i, comm_node in enumerate(interal_communication_nodes):
            batch_p2p_communication_handler(comm_node, idx + i + 1, scheduler_table)
