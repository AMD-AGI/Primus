from abc import ABC, abstractmethod
from typing import Any

from primus.core.pipeline_parallel.scheduler.scheduler_node import FuncType, SchedulerNode


class PipelineScheduleAlgo(ABC):
    """Base class of Pipeline schedule algorithm"""

    def __init__(self, pp_size, vpp_size, micro_batches):
        self.pp_size = pp_size
        self.vpp_size = vpp_size
        self.micro_batches = micro_batches

    @abstractmethod
    def generate_schedule_table(self) -> list[list[SchedulerNode]]:
        raise NotImplementedError

    @abstractmethod
    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, int]:
        raise NotImplementedError

    def print_schedule_table(
        self, schedule_table: list[list[SchedulerNode]], filter: list[FuncType] = None, file=None
    ):
        for rank in range(len(schedule_table)):
            print(
                f"Rank {rank}: {','.join([node.__str__() for node in schedule_table[rank] if filter is None or node.func_type in filter])}",
                file=file,
            )

    def add_communication_nodes(self, schedule_table: list[list[SchedulerNode]], mode="batch_p2p"):
        new_schedule_table = [[] for _ in range(self.pp_size)]
        if mode == "batch_p2p":
            for rank in range(self.pp_size):
                for i in range(len(schedule_table[rank])):
                    if schedule_table[rank][i].func_type == FuncType.W:
                        new_schedule_table[rank].append(schedule_table[rank][i])
                        continue
                    direction_info = self.direction_map(
                        rank, schedule_table[rank][i].chunk, schedule_table[rank][i].func_type
                    )

                    prev_node, prev_node_type = direction_info["prev"]
                    next_node, next_node_type = direction_info["next"]
                    recv_from_chunk = direction_info["recv_from_chunk"]
                    send_to_chunk = direction_info["send_to_chunk"]
                    if prev_node is not None:
                        new_schedule_table[rank].append(
                            SchedulerNode(
                                func_type=prev_node_type,
                                mini_batch=schedule_table[rank][i].mini_batch,
                                chunk=schedule_table[rank][i].chunk,
                                args={
                                    "from_pp_rank": prev_node,
                                    "to_pp_rank": rank,
                                    "recv_from_chunk": recv_from_chunk,
                                },
                            )
                        )
                    new_schedule_table[rank].append(schedule_table[rank][i])
                    if next_node is not None:
                        new_schedule_table[rank].append(
                            SchedulerNode(
                                func_type=next_node_type,
                                mini_batch=schedule_table[rank][i].mini_batch,
                                chunk=schedule_table[rank][i].chunk,
                                args={
                                    "from_pp_rank": rank,
                                    "to_pp_rank": next_node,
                                    "send_to_chunk": send_to_chunk,
                                },
                            )
                        )
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return new_schedule_table


    def first_pp_stage_rank(self) -> int:
        return 0

    def last_pp_stage_rank(self) -> int:
        return self.pp_size - 1
    
class VFoldScheduleAlgo(PipelineScheduleAlgo):
    def __init__(self, pp_size, vpp_size, micro_batches):
        super().__init__(pp_size, vpp_size, micro_batches)
        assert vpp_size == 2, "VFold requires vpp_size == 2"

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, Any]:
        left_v = (chunk == 0) if func_type == FuncType.F else (chunk == 1)

        if left_v:
            prev_rank = rank - 1 if rank - 1 >= 0 else None
            next_rank = rank + 1 if rank + 1 < self.pp_size else rank
            send_to_chunk = chunk if rank < self.pp_size - 1 else 1 - chunk
            recv_from_chunk = chunk
        else:
            prev_rank = rank + 1 if rank + 1 < self.pp_size else rank
            next_rank = rank - 1 if rank - 1 >= 0 else None
            send_to_chunk = chunk
            recv_from_chunk = chunk if rank + 1 < self.pp_size else 1 - chunk

        return {
            "prev": (prev_rank, FuncType.RF if func_type == FuncType.F else FuncType.RB),
            "next": (next_rank, FuncType.SF if func_type == FuncType.F else FuncType.SB),
            "recv_from_chunk": recv_from_chunk,
            "send_to_chunk": send_to_chunk,
        }

    def generate_schedule_table(self) -> list[list[SchedulerNode]]:
        raise NotImplementedError

    def first_pp_stage_rank(self) -> int:
        return 0

    def last_pp_stage_rank(self) -> int:
        return 0
