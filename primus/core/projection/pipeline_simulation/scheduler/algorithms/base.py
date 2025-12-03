from abc import ABC, abstractmethod

from primus.core.projection.pipeline_simulation.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)


class PipelineScheduleAlgo(ABC):
    """Base class of Pipeline schedule algorithm"""

    def __init__(self, pp_size, vpp_size, micro_batches):
        self.pp_size = pp_size
        self.vpp_size = vpp_size
        self.micro_batches = micro_batches

        self.time_ref_dict = {
            FuncType.F: 1000.0,
            FuncType.B: 1000.0,
            FuncType.W: 1000.0,
            FuncType.BW: 2000.0,
        }

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

    def generate_send_recv_nodes(self, rank: int, mini_batch: int, chunk: int, func_type: FuncType):
        direction_info = self.direction_map(rank, chunk, func_type)
        prev_node, prev_node_type = direction_info["prev"]
        next_node, next_node_type = direction_info["next"]
        recv_from_chunk = direction_info["recv_from_chunk"]
        send_to_chunk = direction_info["send_to_chunk"]
        send_node, recv_node = None, None

        if prev_node is not None:
            recv_node = SchedulerNode(
                func_type=prev_node_type,
                mini_batch=mini_batch,
                chunk=chunk,
                args={
                    "from_pp_rank": prev_node,
                    "to_pp_rank": rank,
                    "recv_from_chunk": recv_from_chunk,
                },
            )
        if next_node is not None:
            send_node = SchedulerNode(
                func_type=next_node_type,
                mini_batch=mini_batch,
                chunk=chunk,
                args={
                    "from_pp_rank": rank,
                    "to_pp_rank": next_node,
                    "send_to_chunk": send_to_chunk,
                },
            )

        return send_node, recv_node

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

    def _predict_node_time_info(self, schedule_table: list[list[SchedulerNode]]):

        node_time_info = [dict() for _ in range(self.pp_size)]
        communication_map = dict()

        comm_time = 1.0
        latency_time = 0.01

        current_rank = 0

        rank_timer = [0.0 for _ in range(self.pp_size)]
        rank_idx = [0 for _ in range(self.pp_size)]

        max_retry = len(schedule_table) * len(schedule_table[0])

        def add_time_info(node: SchedulerNode, rank: int, time: float, info_type: str):
            if node not in node_time_info[rank]:
                node_time_info[rank][node] = dict()
            assert (
                info_type not in node_time_info[rank][node]
            ), f"info_type {info_type} already exists for node {node.__str__()}"
            node_time_info[rank][node][info_type] = time

        while True:
            max_retry -= 1
            if max_retry <= 0:
                raise ValueError("Max retry reached, May have bugs in the schedule table")

            all_finished = True
            for rank in range(self.pp_size):
                if len(schedule_table[rank]) == rank_idx[rank]:
                    continue
                all_finished = False
            if all_finished:
                break

            while rank_idx[current_rank] < len(schedule_table[current_rank]):
                node = schedule_table[current_rank][rank_idx[current_rank]]
                if node.func_type in [FuncType.SF, FuncType.SB]:
                    send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{node.func_type}_{node.mini_batch}_{node.args['send_to_chunk']}"
                    if send_key not in communication_map:
                        add_time_info(node, current_rank, rank_timer[current_rank], "start_time")
                        rank_timer[current_rank] += comm_time
                        add_time_info(node, current_rank, rank_timer[current_rank], "end_time")
                        communication_map[send_key] = (rank_timer[current_rank], node)

                elif node.func_type in [FuncType.RF, FuncType.RB]:
                    send_func_type = FuncType.SF if node.func_type == FuncType.RF else FuncType.SB
                    send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{send_func_type}_{node.mini_batch}_{node.chunk}"

                    if send_key not in communication_map:
                        merge_comm_index = rank_idx[current_rank] + 1
                        for i in range(merge_comm_index, len(schedule_table[current_rank])):
                            if schedule_table[current_rank][i].func_type in [FuncType.SF, FuncType.SB]:
                                send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{node.func_type}_{node.mini_batch}_{node.args['send_to_chunk']}"
                                add_time_info(
                                    schedule_table[current_rank][i],
                                    current_rank,
                                    rank_timer[current_rank],
                                    "start_time",
                                )
                                rank_timer[current_rank] += comm_time
                                add_time_info(
                                    schedule_table[current_rank][i],
                                    current_rank,
                                    rank_timer[current_rank],
                                    "end_time",
                                )
                            else:
                                break
                        break

                    send_time, send_node = communication_map.pop(send_key)
                    add_time_info(
                        node, current_rank, send_time + latency_time, "start_time"
                    )  # here we supporse resv time is the same as send time
                    add_time_info(node, current_rank, send_time + comm_time + latency_time, "end_time")

                    rank_timer[current_rank] = max(
                        rank_timer[current_rank], send_time + comm_time + latency_time
                    )

                if node.func_type in [FuncType.F, FuncType.B, FuncType.W, FuncType.BW, FuncType.FB]:
                    add_time_info(node, current_rank, rank_timer[current_rank], "start_time")

                    rank_timer[current_rank] += self.time_ref_dict[node.func_type] / self.vpp_size

                    add_time_info(node, current_rank, rank_timer[current_rank], "end_time")

                rank_idx[current_rank] += 1

            current_rank = (current_rank + 1) % len(schedule_table)

        return node_time_info


class VFoldScheduleAlgo(PipelineScheduleAlgo):
    def __init__(self, pp_size, vpp_size, micro_batches):
        super().__init__(pp_size, vpp_size, micro_batches)
        assert vpp_size == 2, "VFold requires vpp_size == 2"

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, int]:
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
