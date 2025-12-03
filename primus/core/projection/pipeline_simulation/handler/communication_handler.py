""" Handle for communication """

from typing import Optional

import torch

from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)

from .utils import find_prev_node_with_type

COMMUNICATION_NODE_CACHE = []


def _init_send_resv_buffers(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):
    assert node.func_type in [FuncType.SF, FuncType.SB, FuncType.RF, FuncType.RB]

    if node.func_type in [FuncType.SF, FuncType.SB]:
        prev_nodes_indicate_map = {
            FuncType.SF: [FuncType.F],
            FuncType.SB: [FuncType.B, FuncType.BW],
        }
        prev_node = find_prev_node_with_type(scheduler_table, idx, prev_nodes_indicate_map[node.func_type])
        node.args["send_buffer"] = scheduler_table[prev_node].args["output"].clone().detach()
    elif node.func_type in [FuncType.RF, FuncType.RB]:
        node.args["recv_buffer"] = torch.empty(
            *node.args["recv_buffer_size"],
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=node.args["dtype"],
        )


def _async_send_recv(
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
                send_next_req = torch.distributed.isend(
                    tensor=node.args["send_buffer"],
                    dst=node.args["to_pp_rank"],
                    group=even_send_odd_recv_group,
                )
                # node.args["req"] = send_next_req

        if recv_prev_nodes is not None:
            for node in recv_prev_nodes:
                recv_prev_req = torch.distributed.irecv(
                    tensor=node.args["recv_buffer"],
                    src=node.args["from_pp_rank"],
                    group=even_recv_odd_send_group,
                )
                node.args["req"] = recv_prev_req

        if send_prev_nodes is not None:
            for node in send_prev_nodes:
                send_prev_req = torch.distributed.isend(
                    tensor=node.args["send_buffer"],
                    dst=node.args["to_pp_rank"],
                    group=even_send_odd_recv_group,
                )
                # node.args["req"] = send_prev_req

        if recv_next_nodes is not None:
            for node in recv_next_nodes:
                recv_next_req = torch.distributed.irecv(
                    tensor=node.args["recv_buffer"],
                    src=node.args["from_pp_rank"],
                    group=even_recv_odd_send_group,
                )
                node.args["req"] = recv_next_req

    else:
        if recv_prev_nodes is not None:
            for node in recv_prev_nodes:
                recv_prev_req = torch.distributed.irecv(
                    tensor=node.args["recv_buffer"],
                    src=node.args["from_pp_rank"],
                    group=even_send_odd_recv_group,
                )
                node.args["req"] = recv_prev_req

        if send_next_nodes is not None:
            for node in send_next_nodes:
                send_next_req = torch.distributed.isend(
                    tensor=node.args["send_buffer"],
                    dst=node.args["to_pp_rank"],
                    group=even_recv_odd_send_group,
                )
                # node.args["req"] = send_next_req

        if recv_next_nodes is not None:
            for node in recv_next_nodes:
                recv_next_req = torch.distributed.irecv(
                    tensor=node.args["recv_buffer"],
                    src=node.args["from_pp_rank"],
                    group=even_send_odd_recv_group,
                )
                node.args["req"] = recv_next_req

        if send_prev_nodes is not None:
            for node in send_prev_nodes:
                send_prev_req = torch.distributed.isend(
                    tensor=node.args["send_buffer"],
                    dst=node.args["to_pp_rank"],
                    group=even_recv_odd_send_group,
                )
                # node.args["req"] = send_prev_req


def _batch_send_recv(p2p_nodes: list[SchedulerNode], mode: str):
    ops = []
    send_prev_nodes = []
    recv_prev_nodes = []
    send_next_nodes = []
    recv_next_nodes = []

    for comm_node in p2p_nodes:
        if comm_node.args["from_pp_rank"] < comm_node.args["to_pp_rank"]:
            if comm_node.func_type in [FuncType.SF, FuncType.SB]:
                send_next_nodes.append(comm_node)
            elif comm_node.func_type in [FuncType.RF, FuncType.RB]:
                recv_prev_nodes.append(comm_node)
        else:
            if comm_node.func_type in [FuncType.SF, FuncType.SB]:
                send_prev_nodes.append(comm_node)
            elif comm_node.func_type in [FuncType.RF, FuncType.RB]:
                recv_next_nodes.append(comm_node)

    if mode == "batch_p2p":
        send_op = None
        recv_op = None
        for comm_nodes in [send_prev_nodes, recv_prev_nodes, send_next_nodes, recv_next_nodes]:
            for node in comm_nodes:
                if node.func_type in [FuncType.SF, FuncType.SB]:
                    send_op = torch.distributed.P2POp(
                        torch.distributed.isend,
                        node.args["send_buffer"],
                        group=node.args["pp_group"],
                        group_peer=node.args["to_pp_rank"],
                    )
                    ops.append(send_op)
                elif node.func_type in [FuncType.RF, FuncType.RB]:
                    recv_op = torch.distributed.P2POp(
                        torch.distributed.irecv,
                        node.args["recv_buffer"],
                        group=node.args["pp_group"],
                        group_peer=node.args["from_pp_rank"],
                    )
                    ops.append(recv_op)

        # print(f"rank {torch.distributed.get_rank()} send for ops {ops}")

        if len(ops) == 2 and send_op is not None and recv_op is not None:
            group = node.args["pp_group"]
            group_rank = group.rank()

            send_ops = [send_op]
            recv_ops = [recv_op]

            reorder_ops = [recv_ops, send_ops] if group_rank % 2 == 1 else [send_ops, recv_ops]

            for op in reorder_ops:
                reqs = torch.distributed.batch_isend_irecv(op)
                # print(f"rank {torch.distributed.get_rank()} wait for ops {op}")
                for req in reqs:
                    req.wait()
            return

        reqs = torch.distributed.batch_isend_irecv(ops)
        # print(f"rank {torch.distributed.get_rank()} wait for ops {ops}")
        for req in reqs:
            req.wait()

        torch.cuda.synchronize()

    elif mode == "async_p2p":
        _async_send_recv(
            send_prev_nodes=send_prev_nodes,
            recv_prev_nodes=recv_prev_nodes,
            send_next_nodes=send_next_nodes,
            recv_next_nodes=recv_next_nodes,
            group=p2p_nodes[0].args["pp_group"],
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")


def batch_p2p_communication_handler(node: SchedulerNode, idx: int, scheduler_table: list[SchedulerNode]):

    mode = node.meta["communication_mode"]
    _init_send_resv_buffers(node, idx, scheduler_table)

    COMMUNICATION_NODE_CACHE.append(node)

    if idx + 1 < len(scheduler_table) and scheduler_table[idx + 1].func_type in [
        FuncType.SF,
        FuncType.SB,
        FuncType.RF,
        FuncType.RB,
    ]:
        return
    # print(f"rank {torch.distributed.get_rank()} execute communication node {node.__str__()} {[node.__str__() for node in COMMUNICATION_NODE_CACHE]}")
    _batch_send_recv(COMMUNICATION_NODE_CACHE, mode)

    COMMUNICATION_NODE_CACHE.clear()
