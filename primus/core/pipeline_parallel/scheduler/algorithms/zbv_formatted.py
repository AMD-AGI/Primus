###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from ..scheduler_node import FuncType
from .base import VFoldScheduleAlgo

__all__ = [
    "ScheduleZBVFormatted",
]


class ScheduleZBVFormatted(VFoldScheduleAlgo):
    """ZBV Formatted Pipeline Schedule"""

    def __init__(self, pp_size, vpp_size, micro_batches, combined_forward_backward=False, offload=False):
        assert vpp_size == 2, "VFold1F1B requires vpp_size == 2"
        super().__init__(pp_size, vpp_size, micro_batches)
        self.combined_forward_backward = combined_forward_backward
        self.offload = offload

    def generate_schedule_table(self):
        # max(2 * self.pp_size - 1, ...) ensure the number of microbatches is at least
        # as large of the number of microbatches needed to fully utilize the pipeline
        n_micro = max(2 * self.pp_size - 1, self.micro_batches)
        # rank_ops: list[Optional[_Action]] = [None for _ in range(rank)]
        time_step = [i for i in range(self.pp_size)]

        self.time_step_nodes = [dict() for _ in range(self.pp_size)]

        # Forward and backward action counts for stage chunk 0 and chunk 1

        for rank in range(self.pp_size):
            # warm-up phase
            warmup_n1 = 2 * (self.pp_size - rank) - 1
            f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0

            for _ in range(warmup_n1):

                self._insert_compute_node(rank, f0_cnt, 0, FuncType.F, time_step[rank])
                f0_cnt += 1
                time_step[rank] += 1

            warmup_n2 = rank
            for _ in range(warmup_n2):
                self._insert_compute_node(rank, f1_cnt, 1, FuncType.F, time_step[rank])
                f1_cnt += 1
                time_step[rank] += 1

                self._insert_compute_node(rank, f0_cnt, 0, FuncType.F, time_step[rank])
                f0_cnt += 1
                time_step[rank] += 1

            warmup_n3 = self.pp_size - rank
            for _ in range(warmup_n3):
                self._insert_compute_node(rank, f1_cnt, 1, FuncType.F, time_step[rank])
                f1_cnt += 1
                time_step[rank] += 1

                self._insert_compute_node(rank, b1_cnt, 1, FuncType.B, time_step[rank])
                time_step[rank] += 1
                self._insert_compute_node(rank, b1_cnt, 1, FuncType.W, time_step[rank])
                time_step[rank] += 1
                b1_cnt += 1

            # stable phase
            while f1_cnt < f0_cnt or f0_cnt < n_micro:
                if f0_cnt < n_micro:
                    self._insert_compute_node(rank, f0_cnt, 0, FuncType.F, time_step[rank])
                    time_step[rank] += 1
                    f0_cnt += 1

                self._insert_compute_node(rank, b0_cnt, 0, FuncType.B, time_step[rank])
                time_step[rank] += 1
                self._insert_compute_node(rank, b0_cnt, 0, FuncType.W, time_step[rank])
                time_step[rank] += 1
                b0_cnt += 1

                self._insert_compute_node(rank, f1_cnt, 1, FuncType.F, time_step[rank])
                time_step[rank] += 1
                f1_cnt += 1
                self._insert_compute_node(rank, b1_cnt, 1, FuncType.B, time_step[rank])
                time_step[rank] += 1
                self._insert_compute_node(rank, b1_cnt, 1, FuncType.W, time_step[rank])
                time_step[rank] += 1
                b1_cnt += 1
            # cool-down phase
            w0_cnt, w1_cnt = b0_cnt, b1_cnt
            cooldown_n1 = rank
            for _ in range(cooldown_n1):
                self._insert_compute_node(rank, b0_cnt, 0, FuncType.B, time_step[rank])
                time_step[rank] += 1
                b0_cnt += 1
                self._insert_compute_node(rank, b1_cnt, 1, FuncType.B, time_step[rank])
                time_step[rank] += 1
                b1_cnt += 1
            cooldown_n2 = self.pp_size - rank
            for _ in range(cooldown_n2):
                self._insert_compute_node(rank, b0_cnt, 0, FuncType.B, time_step[rank])
                time_step[rank] += 1
                b0_cnt += 1
                self._insert_compute_node(rank, w0_cnt, 0, FuncType.W, time_step[rank])
                time_step[rank] += 1
                w0_cnt += 1
            while w1_cnt < b1_cnt:
                self._insert_compute_node(rank, w1_cnt, 1, FuncType.W, time_step[rank])
                time_step[rank] += 1
                w1_cnt += 1
            while w0_cnt < b0_cnt:
                self._insert_compute_node(rank, w0_cnt, 0, FuncType.W, time_step[rank])
                time_step[rank] += 1
                w0_cnt += 1

            assert w0_cnt == b0_cnt and b0_cnt == f0_cnt
            assert w1_cnt == b1_cnt and b1_cnt == f1_cnt

        schedule_table = self._calculate_schedule_table_by_time_step_nodes()

        if self.combined_forward_backward:
            schedule_table = self.add_combine_1f1b_info_for_schedule_table(schedule_table)

        if self.offload:
            assert (
                not self.combined_forward_backward
            ), "combined_forward_backward and offload cannot be True at the same time"
            schedule_table = self.add_offload_nodes_to_schedule_table(schedule_table)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZBVFormatted(
        pp_size=4, vpp_size=2, micro_batches=16, combined_forward_backward=False, offload=True
    )
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table)

    for rank in range(schedule.pp_size):
        print(f"rank {rank}: {len(schedule_table[rank])}")
