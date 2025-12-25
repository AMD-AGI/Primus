###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from megatron.core.pipeline_parallel.combined_1f1b import combined_forward_backward_step
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
)
from megatron.training.global_vars import get_args

from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)
from primus.core.pipeline_parallel.utils import find_prev_node_with_type
from primus.modules.trainer.megatron.utils import fwd_bwd_wrapper


def megatron_check_fwd_node_valid(node: SchedulerNode):
    assert node.func_type == FuncType.F
    args = node.args
    assert isinstance(args, dict)
    assert "forward_step_func" in args
    assert "data_iterator" in args
    assert "models" in args
    assert "num_microbatches" in args
    assert "forward_data_store" in args
    assert "config" in args
    assert "collect_non_loss_data" in args
    assert "is_last_stage" in args
    assert "total_num_tokens" in args


def megatron_fwd_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    megatron_check_fwd_node_valid(node)
    # prepare input, if not found, input is None(fwd_func will handle it)
    idx = find_prev_node_with_type(scheduler_table, idx, [FuncType.RF])

    if idx is not None and "req" in scheduler_table[idx].args:
        scheduler_table[idx].args["req"].wait()
        scheduler_table[idx].args["req"] = None
        del scheduler_table[idx].args["req"]

    input_tensors = (
        [None] * len(node.args["recv_tensor_shapes"])
        if idx is None
        else scheduler_table[idx].args["recv_buffers"]
    )

    is_last_stage = (
        node.meta["last_pp_stage_rank"] == node.meta["pp_rank"] and node.chunk == node.meta["vpp_size"] - 1
    )

    if not isinstance(node.args["data_iterator"], list):
        node.args["data_iterator"] = [node.args["data_iterator"]]

    kwargs = {
        "forward_step_func": node.args["forward_step_func"],
        "data_iterator": node.args["data_iterator"][node.chunk],
        "num_microbatches": node.args["num_microbatches"],
        "forward_data_store": node.args["forward_data_store"],
        "config": node.args["config"],
        "collect_non_loss_data": node.args["collect_non_loss_data"],
        "checkpoint_activations_microbatch": None,
        "is_first_microbatch": (node.mini_batch == 0),
        "current_microbatch": node.mini_batch,
    }

    if not get_args().overlap_moe_expert_parallel_comm:
        forward_step_func = forward_step
        kwargs["model"] = node.args["models"][node.chunk]
        kwargs["input_tensor"] = input_tensors
        kwargs["vp_stage"] = node.chunk
        kwargs["cp_group_size"] = node.args["cp_group_size"]
        kwargs["is_last_stage"] = is_last_stage
    else:
        forward_step_func = combined_forward_backward_step
        kwargs["f_model"] = node.args["models"][node.chunk]
        kwargs["f_model_chunk_id"] = node.chunk
        kwargs["input_tensor"] = input_tensors[0]
        kwargs["b_model"] = None
        kwargs["b_input_tensor"] = None
        kwargs["b_output_tensor"] = None
        kwargs["b_output_tensor_grad"] = None

    if get_args().dump_pp_data:
        forward_step_func = fwd_bwd_wrapper(
            forward_step_func, "fwd", minibatch=node.mini_batch, chunk=node.chunk
        )

    fwd_outputs = forward_step_func(**kwargs)

    outputs = fwd_outputs[0]
    num_tokens = fwd_outputs[1]

    node.args["total_num_tokens"] += num_tokens
    node.args["outputs"] = outputs if isinstance(outputs, list) else [outputs]
    node.args["inputs"] = input_tensors

    if get_args().overlap_moe_expert_parallel_comm:
        assert node.args["outputs"][0].schedule_plan is not None

    if is_last_stage:
        deallocate_output_tensor(node.args["outputs"][0], node.args["config"].deallocate_pipeline_outputs)

    if idx is not None:
        scheduler_table[idx].args["recv_buffers"] = None
