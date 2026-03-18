###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from ..scheduler_node import FuncType, SchedulerNode
from .base import PipelineScheduleAlgo

__all__ = [
    "ScheduleZeroBubble",
]


class ScheduleZeroBubble(PipelineScheduleAlgo):
    """Zero Bubble Pipeline Schedule

    This schedule splits backward into B (backward computation) and W (weight gradient)
    to minimize pipeline bubbles by allowing more flexible scheduling.

    The key idea is to separate the backward pass computation from weight updates,
    allowing the pipeline to maintain better utilization.
    """

    def __init__(self, pp_size, vpp_size, micro_batches):
        super().__init__(pp_size, vpp_size, micro_batches)

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, Any]:
        """Map communication directions for zero bubble schedule"""
        assert chunk == 0, "Zero Bubble base version uses chunk=0"

        if func_type == FuncType.F:
            return {
                "prev": (rank - 1 if rank - 1 >= 0 else None, FuncType.RF),
                "next": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.SF),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        elif func_type == FuncType.B:
            # Backward pass: receive gradients from next stage, send to previous stage
            return {
                "prev": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.RB),
                "next": (rank - 1 if rank - 1 >= 0 else None, FuncType.SB),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        else:
            raise ValueError(f"Invalid function type: {func_type}")

    def generate_schedule_table(self):
        """Generate zero bubble schedule table

        The schedule uses B/W split to minimize bubbles:
        - Phase 1: Warmup with forward passes
        - Phase 2: Steady state with F-B-W pattern
        - Phase 3: Cooldown with remaining B and W passes
        """
        schedule_table = [[] for _ in range(self.pp_size)]

        for rank in range(self.pp_size):
            warm_up_phases = self.pp_size - rank - 1

            if self.micro_batches < self.pp_size:
                if rank != self.pp_size - 1:
                    for i in range(self.micro_batches):  # insert all f
                        recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                        if recv_node is not None:
                            schedule_table[rank].append(recv_node)

                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                        )

                        if send_node is not None:
                            schedule_table[rank].append(send_node)

                    for i in range(self.micro_batches):  # insert b-w
                        recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.B)
                        if recv_node is not None:
                            schedule_table[rank].append(recv_node)

                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.B, mini_batch=i, chunk=0, args=None)
                        )

                        if send_node is not None:
                            schedule_table[rank].append(send_node)

                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.W, mini_batch=i, chunk=0, args=None)
                        )

                else:

                    for i in range(self.micro_batches):  # insert f-b
                        fwd_recv_node, fwd_send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                        if fwd_recv_node is not None:
                            schedule_table[rank].append(fwd_recv_node)

                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                        )

                        if fwd_send_node is not None:
                            schedule_table[rank].append(fwd_send_node)

                        bwd_recv_node, bwd_send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.B)
                        if bwd_recv_node is not None:
                            schedule_table[rank].append(bwd_recv_node)

                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.B, mini_batch=i, chunk=0, args=None)
                        )

                        if bwd_send_node is not None:
                            schedule_table[rank].append(bwd_send_node)

                    # insert all w:
                    for i in range(self.micro_batches):
                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.W, mini_batch=i, chunk=0, args=None)
                        )
            else:
                # Warmup: only forward passes
                for i in range(warm_up_phases):

                    recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                    if recv_node is not None:
                        schedule_table[rank].append(recv_node)

                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                    )

                    if send_node is not None:
                        schedule_table[rank].append(send_node)

                # Steady state: F-B-W pattern
                # The key insight: we can do W (weight grad) without blocking the pipeline
                for i in range(warm_up_phases, self.micro_batches):

                    fwd_recv_node, fwd_send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                    if fwd_recv_node is not None:
                        w_node = None

                        if (
                            len(schedule_table[rank]) > 0
                            and schedule_table[rank][len(schedule_table[rank]) - 1].func_type == FuncType.W
                        ):
                            w_node = schedule_table[rank].pop()

                        schedule_table[rank].append(fwd_recv_node)
                        if w_node is not None:
                            schedule_table[rank].append(w_node)
                    # Forward for current microbatch
                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                    )
                    if fwd_send_node is not None:
                        schedule_table[rank].append(fwd_send_node)

                    # Backward for earlier microbatch (B only, without W)
                    bw_idx = i - warm_up_phases
                    bwd_recv_node, bwd_send_node = self.generate_send_recv_nodes(rank, bw_idx, 0, FuncType.B)
                    if bwd_recv_node is not None:
                        schedule_table[rank].append(bwd_recv_node)
                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.B, mini_batch=bw_idx, chunk=0, args=None)
                    )
                    if bwd_send_node is not None:
                        schedule_table[rank].append(bwd_send_node)

                    # Weight gradient can be scheduled flexibly
                    if bw_idx >= rank:
                        schedule_table[rank].append(
                            SchedulerNode(func_type=FuncType.W, mini_batch=bw_idx - rank, chunk=0, args=None)
                        )

                # Cooldown: remaining B and W passes
                for i in range(self.micro_batches - warm_up_phases, self.micro_batches):
                    bwd_recv_node, bwd_send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.B)
                    if bwd_recv_node is not None:
                        w_node = None
                        if (
                            len(schedule_table[rank]) > 0
                            and schedule_table[rank][len(schedule_table[rank]) - 1].func_type == FuncType.W
                        ):
                            w_node = schedule_table[rank].pop()

                        if bwd_recv_node is not None:
                            schedule_table[rank].append(bwd_recv_node)
                        if w_node is not None:
                            schedule_table[rank].append(w_node)

                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.B, mini_batch=i, chunk=0, args=None)
                    )
                    if bwd_send_node is not None:
                        schedule_table[rank].append(bwd_send_node)
                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.W, mini_batch=i - rank, chunk=0, args=None)
                    )

                for i in range(self.micro_batches, self.micro_batches + rank):
                    schedule_table[rank].append(
                        SchedulerNode(func_type=FuncType.W, mini_batch=i - rank, chunk=0, args=None)
                    )

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZeroBubble(pp_size=4, vpp_size=1, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table)
