###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from typing import List

from ..scheduler_node import FuncType, SchedulerNode
from .base import VFoldScheduleAlgo

__all__ = [
    "ScheduleZBVGreedy",
]


class ScheduleZBVGreedy(VFoldScheduleAlgo):
    """ZBV Greedy Pipeline Schedule"""

    def __init__(self, pp_size, vpp_size, micro_batches, memory_config, offload=False):
        assert vpp_size == 2, "ZBV Greedy requires vpp_size == 2"
        assert memory_config in ["min", "half"], "Memory config must be 'min' or 'half'"
        super().__init__(pp_size, vpp_size, micro_batches)
        self.pp_group_size = pp_size
        self.mem_config = memory_config
        self.memory_delta = {
            FuncType.F: 1,
            FuncType.B: 0,
            FuncType.W: -1,
        }

        self.offload = offload

    def stable_pattern_v_half(self) -> List[List[int]]:
        interval = 3 if self.pp_size % 2 == 0 else 0
        schedule_patten = []

        for stage_idx in range(self.pp_size):
            # (func_type, chunk, start_time)
            schedule_patten.append(
                [
                    (FuncType.F, 0, stage_idx * 2),
                    (FuncType.F, 1, self.pp_size * 3 - stage_idx - 2),
                    (FuncType.B, 1, self.pp_size * 3 + interval + stage_idx * 2 - 1),
                    (FuncType.B, 0, self.pp_size * 6 + interval - stage_idx - 2),
                ]
            )
        return schedule_patten

    def stable_pattern_v_min(self) -> List[List[int]]:
        interval = 2 if self.pp_size % 3 == 0 else 0
        schedule_patten = []

        for pp_rank in range(self.pp_size):
            # (func_type, chunk, start_time)
            schedule_patten.append(
                [
                    (FuncType.F, 0, pp_rank),
                    (FuncType.F, 1, self.pp_size * 2 - pp_rank - 1),
                    (FuncType.B, 1, self.pp_size * 2 + interval + pp_rank),
                    (FuncType.B, 0, self.pp_size * 4 + interval - pp_rank - 1),
                ]
            )
        return schedule_patten

    def add_w_nodes(self, node_execute_time_map, time_step_compute_node):
        memory_footprint = [None for _ in range(self.pp_size)]

        for pp_rank in range(self.pp_size):
            node_execute_time_map[pp_rank][FuncType.W] = [
                [0 for _ in range(self.micro_batches)] for _ in range(2)
            ]
            w_nodes_queue = []
            max_time_step = max(time_step_compute_node[pp_rank].keys()) + 2
            memory_footprint[pp_rank] = [0 for _ in range(max_time_step + 1)]

            phase = "WARMUP"

            for time_step in range(max_time_step + 1):
                if time_step > 0:
                    memory_footprint[pp_rank][time_step] = memory_footprint[pp_rank][time_step - 1]
                if time_step in time_step_compute_node[pp_rank]:
                    node = time_step_compute_node[pp_rank][time_step]
                    if node.args is None:
                        node.args = {}
                    node.args["phase"] = phase
                    if node.func_type == FuncType.B:
                        w_nodes_queue.append(node)

                    assert node.func_type in [FuncType.F, FuncType.B]

                    memory_footprint[pp_rank][time_step] += self.memory_delta[node.func_type]

                    if node.mini_batch == 0 and node.func_type == FuncType.B and node.chunk == 0:
                        phase = "STEADY"
                    elif (
                        node.mini_batch == self.micro_batches - 1
                        and node.func_type == FuncType.F
                        and node.chunk == 1
                    ):
                        phase = "COOLDOWN"

                elif len(w_nodes_queue) > 0:
                    node = w_nodes_queue.pop(0)
                    w_node = SchedulerNode(
                        func_type=FuncType.W,
                        mini_batch=node.mini_batch,
                        chunk=node.chunk,
                        args={"phase": phase},
                    )

                    node_execute_time_map[pp_rank][FuncType.W][w_node.chunk][w_node.mini_batch] = time_step
                    time_step_compute_node[pp_rank][time_step] = w_node
                    memory_footprint[pp_rank][time_step] += self.memory_delta[FuncType.W]

        return node_execute_time_map, time_step_compute_node, memory_footprint

    def calculate_earlist_time(self, rank, node_execute_time_map, node):

        earliest_time = 0

        if node.mini_batch > 0:
            earliest_time = max(
                earliest_time, node_execute_time_map[rank][node.func_type][node.chunk][node.mini_batch - 1]
            )

        if node.func_type == FuncType.W:  # after backward_nodes
            earliest_time = max(
                earliest_time, node_execute_time_map[rank][FuncType.B][node.chunk][node.mini_batch]
            )
            return earliest_time

        direction_info = self.direction_map(rank, node.chunk, node.func_type)
        prev_rank = direction_info["prev"][0]
        prev_chunk = direction_info["recv_from_chunk"]

        if prev_rank is not None:  # need prev node done
            earliest_time = max(
                earliest_time, node_execute_time_map[prev_rank][node.func_type][prev_chunk][node.mini_batch]
            )
        elif node.func_type == FuncType.B:  # first backward node for a mini-batch need forward end
            earliest_time = max(
                earliest_time, node_execute_time_map[rank][FuncType.F][node.chunk][node.mini_batch]
            )

        return earliest_time + 1

    def squeeze(
        self, node_execute_time_map, time_step_compute_node, memory_footprint, reorder_cooldown=False
    ):

        max_memory = max([max(memory_footprint[pp_rank]) for pp_rank in range(self.pp_size)])

        max_time_step = max([max(time_step_compute_node[pp_rank].keys()) for pp_rank in range(self.pp_size)])
        # memory_footprint = [[0 for _ in range(max_time_step + 1)] for _ in range(self.pp_size)]

        pending_w_nodes = [[] for _ in range(self.pp_size)]

        def greedy_pre_insert(pp_rank, node, time_step, earliest_time):
            for i in range(earliest_time, time_step + 1):
                if i not in time_step_compute_node[pp_rank]:  # free time slot
                    if not reorder_cooldown:
                        insert_succ = True

                        for j in range(i, time_step):
                            if memory_footprint[pp_rank][j] + self.memory_delta[node.func_type] > max_memory:
                                insert_succ = False
                                break

                        if not insert_succ:
                            continue
                        else:
                            for j in range(i, time_step):
                                memory_footprint[pp_rank][j] += self.memory_delta[node.func_type]
                            return i
                    else:
                        return i

            return time_step

        for time_step in range(max_time_step + 1):
            for pp_rank in range(self.pp_size):
                if time_step in time_step_compute_node[pp_rank]:
                    node = time_step_compute_node[pp_rank][time_step]
                    if (
                        reorder_cooldown and node.args["phase"] != "COOLDOWN"
                    ):  # reorder cooldown will freeze other phases nodes
                        continue
                    if node.func_type == FuncType.W and reorder_cooldown:
                        pending_w_nodes[pp_rank].append(node)
                        del time_step_compute_node[pp_rank][time_step]
                        continue
                    earlist_time = self.calculate_earlist_time(pp_rank, node_execute_time_map, node)

                    earlist_time = greedy_pre_insert(pp_rank, node, time_step, earlist_time)

                    time_step_compute_node[pp_rank][earlist_time] = node
                    node_execute_time_map[pp_rank][node.func_type][node.chunk][node.mini_batch] = earlist_time
                    if time_step != earlist_time:
                        del time_step_compute_node[pp_rank][time_step]

        for pp_rank in range(self.pp_size):
            for node in pending_w_nodes[pp_rank]:
                earlist_time = self.calculate_earlist_time(pp_rank, node_execute_time_map, node)
                for i in range(earlist_time, max_time_step + 1):
                    if i not in time_step_compute_node[pp_rank]:
                        time_step_compute_node[pp_rank][i] = node
                        node_execute_time_map[pp_rank][node.func_type][node.chunk][node.mini_batch] = i
                        break

        return node_execute_time_map, time_step_compute_node

    def schedule_from_pattern(
        self,
        schedule_patten,
        num_microbatches: int,
    ):
        # [rank][func_type][chunk][microbatch] -> time_step
        node_execute_time_map = [dict() for _ in range(self.pp_size)]
        # [rank][time_step] -> node
        time_step_compute_node = [dict() for _ in range(self.pp_size)]

        for pp_rank in range(self.pp_size):
            for microbatch in range(num_microbatches):
                for func_type, chunk, start_time in schedule_patten[pp_rank]:
                    if func_type not in node_execute_time_map[pp_rank]:
                        node_execute_time_map[pp_rank][func_type] = [
                            [0 for _ in range(num_microbatches)] for _ in range(2)
                        ]

                    execute_time = start_time + 6 * microbatch
                    node_execute_time_map[pp_rank][func_type][chunk][microbatch] = execute_time

                    assert (
                        execute_time not in time_step_compute_node[pp_rank]
                    ), f"start_time {execute_time} already in time_step_compute_node[{pp_rank}] time_step_compute_node: {time_step_compute_node[pp_rank][execute_time].__str__()}"

                    time_step_compute_node[pp_rank][execute_time] = SchedulerNode(
                        func_type=func_type, mini_batch=microbatch, chunk=chunk, args=None
                    )

        node_execute_time_map, time_step_compute_node, memory_footprint = self.add_w_nodes(
            node_execute_time_map, time_step_compute_node
        )

        # squeeze warmup
        node_execute_time_map, time_step_compute_node = self.squeeze(
            node_execute_time_map, time_step_compute_node, memory_footprint
        )

        # squeeze cooldown
        node_execute_time_map, time_step_compute_node = self.squeeze(
            node_execute_time_map, time_step_compute_node, memory_footprint, True
        )

        return time_step_compute_node

    def generate_schedule_table(self):

        schedule_pattern_funcs = {
            "min": self.stable_pattern_v_min,
            "half": self.stable_pattern_v_half,
        }
        pattern_func = schedule_pattern_funcs[self.mem_config]

        time_step_compute_node = self.schedule_from_pattern(
            pattern_func(),
            self.micro_batches,
        )

        self.time_step_nodes = [dict() for _ in range(self.pp_size)]

        for pp_rank in range(self.pp_size):
            for time_step in sorted(time_step_compute_node[pp_rank].keys()):
                compute_node = time_step_compute_node[pp_rank][time_step]
                self._insert_compute_node(
                    pp_rank, compute_node.mini_batch, compute_node.chunk, compute_node.func_type, time_step
                )

        schedule_table = self._calculate_schedule_table_by_time_step_nodes()

        if self.offload:
            schedule_table = self.add_offload_nodes_to_schedule_table(schedule_table)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZBVGreedy(pp_size=4, vpp_size=2, micro_batches=16, memory_config="half")
    schedule_table = schedule.generate_schedule_table()

    schedule.print_schedule_table(schedule_table)
