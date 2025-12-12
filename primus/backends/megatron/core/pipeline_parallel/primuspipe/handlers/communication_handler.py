###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List, Optional

import torch
from megatron.core.pipeline_parallel.schedules import deallocate_output_tensor
from megatron.training import get_args

from primus.core.pipeline_parallel.handler.utils import find_prev_node_with_type
from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)

COMMUNICATION_NODE_CACHE = []


def _async_send_recv_op(
    *,
    send_prev_nodes: Optional[list[SchedulerNode]],
    recv_prev_nodes: Optional[list[SchedulerNode]],
    send_next_nodes: Optional[list[SchedulerNode]],
    recv_next_nodes: Optional[list[SchedulerNode]],
    group: torch.distributed.ProcessGroup,
):
    even_send_odd_recv_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != "ucc":
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        # The only exception occurs when using the ‘ucc’ backend.
        # Because the global communicator always uses the ‘nccl’ backend,
        # we must ensure the else path is followed for the ‘ucc’ backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    if group.rank() % 2 == 0:
        if send_next_nodes is not None:
            for node in send_next_nodes:
                for send_buffer in node.args["send_buffers"]:
                    send_next_req = torch.distributed.isend(
                        tensor=send_buffer, dst=node.args["to_pp_rank"], group=even_send_odd_recv_group
                    )
                    node.args["req"] = send_next_req

        if recv_prev_nodes is not None:
            for node in recv_prev_nodes:
                for recv_buffer in node.args["recv_buffers"]:
                    recv_prev_req = torch.distributed.irecv(
                        tensor=recv_buffer, src=node.args["from_pp_rank"], group=even_recv_odd_send_group
                    )
                    node.args["req"] = recv_prev_req

        if send_prev_nodes is not None:
            for node in send_prev_nodes:
                for send_buffer in node.args["send_buffers"]:
                    send_prev_req = torch.distributed.isend(
                        tensor=send_buffer, dst=node.args["to_pp_rank"], group=even_send_odd_recv_group
                    )
                    node.args["req"] = send_prev_req

        if recv_next_nodes is not None:
            for node in recv_next_nodes:
                for recv_buffer in node.args["recv_buffers"]:
                    recv_next_req = torch.distributed.irecv(
                        tensor=recv_buffer, src=node.args["from_pp_rank"], group=even_recv_odd_send_group
                    )
                    node.args["req"] = recv_next_req

    else:
        if recv_prev_nodes is not None:
            for node in recv_prev_nodes:
                for recv_buffer in node.args["recv_buffers"]:
                    recv_prev_req = torch.distributed.irecv(
                        tensor=recv_buffer, src=node.args["from_pp_rank"], group=even_send_odd_recv_group
                    )
                    node.args["req"] = recv_prev_req

        if send_next_nodes is not None:
            for node in send_next_nodes:
                for send_buffer in node.args["send_buffers"]:
                    send_next_req = torch.distributed.isend(
                        tensor=send_buffer, dst=node.args["to_pp_rank"], group=even_recv_odd_send_group
                    )
                    node.args["req"] = send_next_req

        if recv_next_nodes is not None:
            for node in recv_next_nodes:
                for recv_buffer in node.args["recv_buffers"]:
                    recv_next_req = torch.distributed.irecv(
                        tensor=recv_buffer, src=node.args["from_pp_rank"], group=even_send_odd_recv_group
                    )
                    node.args["req"] = recv_next_req

        if send_prev_nodes is not None:
            for node in send_prev_nodes:
                for send_buffer in node.args["send_buffers"]:
                    send_prev_req = torch.distributed.isend(
                        tensor=send_buffer, dst=node.args["to_pp_rank"], group=even_recv_odd_send_group
                    )
                    node.args["req"] = send_prev_req


def _init_send_recv_buffers(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):

    if node.func_type in [FuncType.SF, FuncType.SB]:
        prev_nodes_indicate_map = {
            FuncType.SF: [FuncType.F],
            FuncType.SB: [FuncType.B, FuncType.BW],
        }
        prev_node = find_prev_node_with_type(scheduler_table, idx, prev_nodes_indicate_map[node.func_type])
        assert prev_node is not None, f"prev_node not found {node.__str__()}"
        node.args["send_buffers"] = scheduler_table[prev_node].args["outputs"]
        node.args["prev_node_idx"] = prev_node

        if node.func_type == FuncType.SB:
            scheduler_table[prev_node].args["outputs"] = None

        # check send buffer shape and size
        assert len(node.args["send_buffers"]) == len(
            node.args["send_tensor_shapes"]
        ), "send_buffer_shape and send_buffer_size must have the same number of dimensions"
        for i in range(len(node.args["send_tensor_shapes"])):
            assert (
                node.args["send_tensor_shapes"][i] == node.args["send_buffers"][i].shape
            ), f"send_buffer_shape and send_buffer_size must have the same size {node.args['send_tensor_shapes'][i]} {node.args['send_buffers'][i].shape} node.func_type: {node.func_type} node.mini_batch: {node.mini_batch} node.chunk: {node.chunk}"

        # print(f"node {node} send_buffers: {len(node.args['send_buffers'])} | {node.args['send_buffers'][0].shape}")

    elif node.func_type in [FuncType.RF, FuncType.RB]:
        node.args["recv_buffers"] = []
        for recv_buffer_shape in node.args["recv_tensor_shapes"]:
            node.args["recv_buffers"].append(
                torch.empty(
                    *recv_buffer_shape,
                    requires_grad=True,
                    device=torch.cuda.current_device(),
                    dtype=node.args["dtype"],
                )
            )
        # print(f"node {node} recv_buffers: {len(node.args['recv_buffers'])} | {node.args['recv_buffers'][0].shape}")


def _batch_send_recv(p2p_nodes: list[SchedulerNode], mode="batch_p2p"):
    if len(p2p_nodes) == 0:
        return
    ops = []
    send_prev_nodes = []
    resv_prev_nodes = []
    send_next_nodes = []
    resv_next_nodes = []

    for comm_node in p2p_nodes:
        assert comm_node.args["from_pp_rank"] != comm_node.args["to_pp_rank"]

        if comm_node.args["from_pp_rank"] < comm_node.args["to_pp_rank"]:
            if comm_node.func_type in [FuncType.SF, FuncType.SB]:
                send_next_nodes.append(comm_node)
            elif comm_node.func_type in [FuncType.RF, FuncType.RB]:
                resv_prev_nodes.append(comm_node)
        else:
            if comm_node.func_type in [FuncType.SF, FuncType.SB]:
                send_prev_nodes.append(comm_node)
            elif comm_node.func_type in [FuncType.RF, FuncType.RB]:
                resv_next_nodes.append(comm_node)

    if mode == "batch_p2p":
        for comm_nodes in [send_prev_nodes, resv_prev_nodes, send_next_nodes, resv_next_nodes]:
            for node in comm_nodes:
                if node.func_type in [FuncType.SF, FuncType.SB]:
                    for send_buffer in node.args["send_buffers"]:
                        send_op = torch.distributed.P2POp(
                            torch.distributed.isend,
                            send_buffer,
                            group=node.args["pp_group"],
                            group_peer=node.args["to_pp_rank"],
                        )
                        ops.append(send_op)
                elif node.func_type in [FuncType.RF, FuncType.RB]:
                    for recv_buffer in node.args["recv_buffers"]:
                        recv_op = torch.distributed.P2POp(
                            torch.distributed.irecv,
                            recv_buffer,
                            group=node.args["pp_group"],
                            group_peer=node.args["from_pp_rank"],
                        )
                        ops.append(recv_op)

        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()
    else:
        _async_send_recv_op(
            send_prev_nodes=send_prev_nodes,
            recv_prev_nodes=resv_prev_nodes,
            send_next_nodes=send_next_nodes,
            recv_next_nodes=resv_next_nodes,
            group=p2p_nodes[0].args["pp_group"],
        )


def batch_p2p_communication_handler(node: SchedulerNode, idx: int, scheduler_table: List[SchedulerNode]):
    assert node.func_type in [FuncType.SF, FuncType.SB, FuncType.RF, FuncType.RB]

    _init_send_recv_buffers(node, idx, scheduler_table)

    comm_pair = {
        FuncType.RF: FuncType.SF,
        FuncType.RB: FuncType.SB,
    }

    if node.args["from_pp_rank"] == node.args["to_pp_rank"]:  # copy if recv node
        if node.func_type in [FuncType.RF, FuncType.RB]:
            send_idx = find_prev_node_with_type(
                scheduler_table, idx, [comm_pair[node.func_type]], chunk=node.args["recv_from_chunk"]
            )
            assert send_idx is not None, "send_idx not found"
            node.args["recv_buffers"] = [
                x.clone().detach().requires_grad_(True)
                for x in scheduler_table[send_idx].args["send_buffers"]
            ]
            if node.func_type == FuncType.RF:
                deallocate_output_tensor(
                    scheduler_table[send_idx].args["send_buffers"][0],
                    node.args["config"].deallocate_pipeline_outputs,
                )

            scheduler_table[send_idx].args["send_buffers"] = None
    else:
        COMMUNICATION_NODE_CACHE.append(node)

    if idx + 1 < len(scheduler_table) and scheduler_table[idx + 1].func_type in [
        FuncType.SF,
        FuncType.SB,
        FuncType.RF,
        FuncType.RB,
    ]:
        if (
            "time_step" not in scheduler_table[idx + 1].args
            or scheduler_table[idx + 1].args["time_step"] == scheduler_table[idx].args["time_step"]
        ):
            return
    mode = get_args().communication_method
    _batch_send_recv(COMMUNICATION_NODE_CACHE, mode=mode)

    send_fwd_nodes = [node for node in COMMUNICATION_NODE_CACHE if node.func_type == FuncType.SF]
    send_bwd_nodes = [node for node in COMMUNICATION_NODE_CACHE if node.func_type == FuncType.SB]
    # deallocate the send buffers' data
    for node in send_fwd_nodes:
        if "req" in node.args:
            node.args["req"].wait()
            node.args["req"] = None
            del node.args["req"]
        deallocate_output_tensor(
            node.args["send_buffers"][0], node.args["config"].deallocate_pipeline_outputs
        )

    for node in send_bwd_nodes:  # deallocate the send buffers
        if "req" in node.args:
            node.args["req"].wait()
            node.args["req"] = None
            del node.args["req"]
        node.args["send_buffers"] = None
        if node.args["prev_node_idx"] is not None:
            scheduler_table[node.args["prev_node_idx"]].args["outputs"] = None

    COMMUNICATION_NODE_CACHE.clear()

    return
