###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import Any

from ..scheduler_node import FuncType, SchedulerNode
from .base import PipelineScheduleAlgo

__all__ = [
    "Schedule1F1B",
]


class Schedule1F1B(PipelineScheduleAlgo):
    def __init__(self, pp_size, vpp_size, micro_batches):
        super().__init__(pp_size, vpp_size, micro_batches)

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, Any]:
        assert chunk == 0
        if func_type == FuncType.F:
            return {
                "prev": (rank - 1 if rank - 1 >= 0 else None, FuncType.RF),
                "next": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.SF),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        elif func_type == FuncType.BW:
            return {
                "prev": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.RB),
                "next": (rank - 1 if rank - 1 >= 0 else None, FuncType.SB),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        else:
            raise ValueError(f"Invalid function type: {func_type}")

    def generate_schedule_table(self):
        schedule_table = [[] for _ in range(self.pp_size)]

        for rank in range(self.pp_size):
            # warmup
            warm_up_phases = self.pp_size - rank - 1
            for i in range(warm_up_phases):
                recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                if recv_node is not None:
                    schedule_table[rank].append(recv_node)

                schedule_table[rank].append(
                    SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                )

                if send_node is not None:
                    schedule_table[rank].append(send_node)

            # 1f1b steady
            for i in range(warm_up_phases, self.micro_batches):
                recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.F)
                if recv_node is not None:
                    schedule_table[rank].append(recv_node)

                schedule_table[rank].append(
                    SchedulerNode(func_type=FuncType.F, mini_batch=i, chunk=0, args=None)
                )

                if send_node is not None:
                    schedule_table[rank].append(send_node)

                recv_node, send_node = self.generate_send_recv_nodes(rank, i - warm_up_phases, 0, FuncType.BW)
                if recv_node is not None:
                    schedule_table[rank].append(recv_node)

                schedule_table[rank].append(
                    SchedulerNode(func_type=FuncType.BW, mini_batch=i - warm_up_phases, chunk=0, args=None)
                )
                if send_node is not None:
                    schedule_table[rank].append(send_node)

            # cool down
            for i in range(self.micro_batches - warm_up_phases, self.micro_batches):
                recv_node, send_node = self.generate_send_recv_nodes(rank, i, 0, FuncType.BW)
                if recv_node is not None:
                    schedule_table[rank].append(recv_node)

                schedule_table[rank].append(
                    SchedulerNode(func_type=FuncType.BW, mini_batch=i, chunk=0, args=None)
                )
                if send_node is not None:
                    schedule_table[rank].append(send_node)

        return schedule_table


if __name__ == "__main__":
    schedule = Schedule1F1B(pp_size=4, vpp_size=1, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table)
