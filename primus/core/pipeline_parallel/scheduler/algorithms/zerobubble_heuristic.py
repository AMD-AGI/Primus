###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Zero Bubble Pipeline Schedule with Graph-based Heuristic.

Ported from primus/backends/megatron/core/pipeline_parallel/zerobubble/scheduler/zb.py.
The manual_order heuristic tries 8 combinations of 3 boolean parameters
(allow_bubble_before_first_b, prioritize_b, no_bubble_greedy) and picks the
schedule with the lowest bubble time.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..scheduler_node import FuncType, SchedulerNode
from .base import PipelineScheduleAlgo

__all__ = [
    "ScheduleZeroBubbleHeuristic",
]


# ---------------------------------------------------------------------------
# Internal graph data structures
# ---------------------------------------------------------------------------


@dataclass
class _GraphConfig:
    cost_f: List[float] = field(default_factory=list)
    cost_b: List[float] = field(default_factory=list)
    cost_w: List[float] = field(default_factory=list)
    cost_comm: float = 0.0
    mem_f: List[float] = field(default_factory=list)
    mem_b: List[float] = field(default_factory=list)
    mem_w: List[float] = field(default_factory=list)
    max_mem: Optional[List[float]] = None


class _Graph:
    """DAG for zero-bubble pipeline scheduling.

    Node ID layout (type * S * M + stage * M + mb):
      F: [0,           S*M)
      B: [S*M,       2*S*M)
      W: [2*S*M,     3*S*M)
    """

    def __init__(self, nstages: int, nmb: int, config: _GraphConfig):
        self.nstages = nstages
        self.nmb = nmb
        self.nnodes = nstages * nmb * 3
        self.config = config

    def get_id(self, type_idx: int, stage: int, mb: int) -> int:
        return type_idx * (self.nstages * self.nmb) + stage * self.nmb + mb

    def get_stage(self, node_id: int) -> int:
        return (node_id // self.nmb) % self.nstages

    def get_cost(self, node_id: int) -> float:
        type_idx = node_id // (self.nstages * self.nmb)
        stage = self.get_stage(node_id)
        return [self.config.cost_f[stage], self.config.cost_b[stage], self.config.cost_w[stage]][type_idx]

    def get_mem(self, node_id: int) -> float:
        type_idx = node_id // (self.nstages * self.nmb)
        stage = self.get_stage(node_id)
        return [self.config.mem_f[stage], self.config.mem_b[stage], self.config.mem_w[stage]][type_idx]

    # ------------------------------------------------------------------
    # Core heuristic scheduler
    # ------------------------------------------------------------------

    def manual_order(
        self,
        allow_bubble_before_first_b: bool = False,
        prioritize_b: bool = False,
        no_bubble_greedy: bool = True,
    ):
        """Heuristic ordering that produces schedules like:
            fffffffbfbfbfbfbfbwbwbwbwbwbwbwwwwww
             fffffbfbfbfbfbfbfbfbwbwbwbwbwwwwwwww
              fffbfbfbfbfbfbfbfbfbfbwbwbwwwwwwwwww
               fbfbfbfbfbfbfbfbfbfbfbfbwwwwwwwwwwww

        Returns (order, completion_times, best_time).
        """
        order = [0] * self.nnodes
        f = [0] * self.nstages
        b = [0] * self.nstages
        w = [0] * self.nstages
        o = [0] * self.nstages
        m = [0] * self.nstages
        e = [0] * self.nstages
        t = [0] * self.nnodes
        max_mem = self.config.max_mem or [
            self.get_mem(self.get_id(0, stage, 0)) * self.nmb * 3 for stage in range(self.nstages)
        ]
        comm = self.config.cost_comm
        stage_bubble = [0] * self.nstages

        def get_max_bubble():
            return max(stage_bubble)

        def put(stage_j, type_k):
            if type_k == 0:
                _i = f[stage_j]
            elif type_k == 1:
                _i = b[stage_j]
            else:
                _i = w[stage_j]
            _j = stage_j
            _id = self.get_id(type_k, _j, _i)
            _cost = self.get_cost(_id)

            tmp = e[_j] + _cost
            no_bubble = tmp
            if _j > 0 and type_k == 0:
                tmp = max(tmp, t[self.get_id(0, _j - 1, _i)] + comm + _cost)
            if _j < self.nstages - 1 and type_k == 1:
                tmp = max(tmp, t[self.get_id(1, _j + 1, _i)] + comm + _cost)
            if f[_j] > 0:
                stage_bubble[_j] += tmp - no_bubble
            e[_j] = tmp
            t[_id] = tmp
            m[_j] += self.get_mem(_id)
            order[_id] = o[_j]
            if type_k == 0:
                f[_j] += 1
            elif type_k == 1:
                b[_j] += 1
            else:
                w[_j] += 1
            o[_j] += 1

        for i in range(self.nmb):
            if i == 0:
                # -- First microbatch: warmup all stages with F, then schedule B --
                for j in range(self.nstages):
                    put(j, 0)
                f_required = [0] * self.nstages
                last_t = 0
                for j in range(self.nstages - 1, -1, -1):
                    if j == self.nstages - 1:
                        last_t = t[self.get_id(0, j, i)] + self.get_cost(self.get_id(1, j, i))
                        continue
                    mem = m[j]
                    cost = e[j]
                    while True:
                        f_id = self.get_id(0, j, f[j] + f_required[j])
                        if f[j] + f_required[j] < self.nmb and mem + self.get_mem(f_id) <= max_mem[j]:
                            if allow_bubble_before_first_b:
                                if cost + self.get_cost(f_id) > last_t + comm:
                                    break
                            else:
                                if cost >= last_t + comm:
                                    break
                            mem += self.get_mem(f_id)
                            cost += self.get_cost(f_id)
                            f_required[j] += 1
                        else:
                            break
                    last_t = max(cost, last_t + comm) + self.get_cost(self.get_id(1, j, i))
                for j in range(self.nstages):
                    while (
                        j > 0
                        and f_required[j] > 0
                        and f_required[j] >= f_required[j - 1]
                        and f[j] + f_required[j] < self.nmb
                    ):
                        f_required[j] -= 1
                for j in range(self.nstages):
                    for _ in range(f_required[j]):
                        put(j, 0)
                for j in range(self.nstages - 1, -1, -1):
                    put(j, 1)
                continue

            # -- Subsequent microbatches --
            f_required = [0] * self.nstages
            for j in range(self.nstages):
                if f[j] >= self.nmb:
                    continue
                if j + 1 < self.nstages and f[j] >= f[j + 1] + 2 and prioritize_b:
                    next_plus_fw = (
                        e[j + 1]
                        + self.get_cost(self.get_id(0, j + 1, f[j + 1]))
                        + self.get_cost(self.get_id(1, j + 1, b[j + 1]))
                        + comm
                    )
                    if e[j] >= next_plus_fw:
                        continue
                    f_id = self.get_id(0, j, f[j])
                    f_mem = self.get_mem(f_id)
                    w_cost, w_cnt = 0, 0
                    mem_with_w = m[j] + f_mem
                    while mem_with_w > max_mem[j] and w[j] + w_cnt < b[j]:
                        w_id = self.get_id(2, j, w[j] + w_cnt)
                        w_cost += self.get_cost(w_id)
                        mem_with_w += self.get_mem(w_id)
                        w_cnt += 1
                    if e[j] + self.get_cost(f_id) + w_cost <= next_plus_fw:
                        f_required[j] = 1
                        continue

                    w_cost, w_cnt = 0, 0
                    if next_plus_fw - (e[j] + w_cost) <= get_max_bubble() - stage_bubble[j]:
                        continue
                f_required[j] = 1
            for j in range(self.nstages - 2, -1, -1):
                f_required[j] = min(f_required[j], f_required[j + 1])

            for j in range(self.nstages):
                if f_required[j] == 0:
                    continue
                f_id = self.get_id(0, j, f[j])
                mem = self.get_mem(f_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j > 0:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(0, j - 1, f[j])] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(0, j - 1, f[j])] + comm:
                        if t[self.get_id(0, j - 1, f[j])] + comm - e[j] <= get_max_bubble() - stage_bubble[j]:
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                put(j, 0)

            for j in range(self.nstages - 1, -1, -1):
                assert b[j] == i
                b_id = self.get_id(1, j, b[j])
                mem = self.get_mem(b_id)
                while m[j] + mem > max_mem[j]:
                    if w[j] >= b[j]:
                        raise ValueError("Cannot fit memory")
                    put(j, 2)
                if j + 1 < self.nstages:
                    while (
                        w[j] < b[j]
                        and e[j] + self.get_cost(self.get_id(2, j, w[j]))
                        <= t[self.get_id(1, j + 1, i)] + comm
                    ):
                        put(j, 2)
                    if w[j] < b[j] and e[j] < t[self.get_id(1, j + 1, i)] + comm:
                        if t[self.get_id(1, j + 1, i)] + comm - e[j] <= get_max_bubble() - stage_bubble[j]:
                            if no_bubble_greedy:
                                put(j, 2)
                        else:
                            put(j, 2)
                if j == 0 and f[j] == self.nmb:
                    while w[j] < b[j]:
                        put(j, 2)
                put(j, 1)

        for i in range(self.nstages):
            while w[i] < self.nmb:
                put(i, 2)

        best_time = 0
        for i in range(self.nstages):
            time_i = (
                t[self.get_id(2, i, self.nmb - 1)]
                - t[self.get_id(0, i, 0)]
                + self.get_cost(self.get_id(0, i, 0))
            )
            best_time = max(best_time, time_i)

        return order, t, best_time

    @classmethod
    def build(cls, nstages: int, nmb: int, config: _GraphConfig) -> "_Graph":
        return cls(nstages, nmb, config)


def _initial_solution(graph: _Graph):
    """Exhaustively try all 8 heuristic parameter combinations."""
    best_time, order, complete_time = None, None, None
    for allow_bubble_before_first_b in [True, False]:
        for prioritize_b in [True, False]:
            for no_bubble_greedy in [True, False]:
                order_t, complete_time_t, best_time_t = graph.manual_order(
                    allow_bubble_before_first_b=allow_bubble_before_first_b,
                    prioritize_b=prioritize_b,
                    no_bubble_greedy=no_bubble_greedy,
                )
                if best_time is None or best_time_t < best_time:
                    best_time = best_time_t
                    order = order_t
                    complete_time = complete_time_t
    return best_time, order, complete_time


# ---------------------------------------------------------------------------
# Public scheduler class
# ---------------------------------------------------------------------------


class ScheduleZeroBubbleHeuristic(PipelineScheduleAlgo):
    """Zero Bubble Pipeline Schedule using graph-based heuristic search.

    Tries all 8 combinations of (allow_bubble_before_first_b, prioritize_b,
    no_bubble_greedy) and selects the F/B/W ordering that minimises the
    longest-stage execution time.

    Constructor kwargs beyond the base class:
        cost_f   – Forward cost  (scalar or per-stage list, default 1000)
        cost_b   – Backward cost (scalar or per-stage list, default 1000)
        cost_w   – W-grad cost   (scalar or per-stage list, default 1000)
        cost_comm – Communication latency between adjacent stages (default 0)
        max_mem  – Peak activation memory budget per stage (None = unlimited)
    """

    def __init__(
        self,
        pp_size: int,
        vpp_size: int,
        micro_batches: int,
        cost_f=1000,
        cost_b=1000,
        cost_w=1000,
        cost_comm=1,
        max_mem=None,
    ):
        super().__init__(pp_size, vpp_size, micro_batches)
        assert vpp_size == 1, "ScheduleZeroBubbleHeuristic requires vpp_size == 1"

        self._cost_f = [cost_f] * pp_size if isinstance(cost_f, (int, float)) else list(cost_f)
        self._cost_b = [cost_b] * pp_size if isinstance(cost_b, (int, float)) else list(cost_b)
        self._cost_w = [cost_w] * pp_size if isinstance(cost_w, (int, float)) else list(cost_w)
        self._cost_comm = float(cost_comm)
        self._max_mem = max_mem

    # -- direction_map (identical to ScheduleZeroBubble) -------------------

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, Any]:
        assert chunk == 0
        if func_type == FuncType.F:
            return {
                "prev": (rank - 1 if rank >= 1 else None, FuncType.RF),
                "next": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.SF),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        elif func_type == FuncType.B:
            return {
                "prev": (rank + 1 if rank + 1 < self.pp_size else None, FuncType.RB),
                "next": (rank - 1 if rank >= 1 else None, FuncType.SB),
                "recv_from_chunk": chunk,
                "send_to_chunk": chunk,
            }
        raise ValueError(f"Invalid func_type for direction_map: {func_type}")

    # -- schedule generation -----------------------------------------------

    def _build_graph_config(self) -> _GraphConfig:
        n = self.pp_size
        # Memory model must match the runtime/simulation convention:
        # F allocates activation (+1), W releases activation (-1), B is neutral (0).
        return _GraphConfig(
            cost_f=self._cost_f,
            cost_b=self._cost_b,
            cost_w=self._cost_w,
            cost_comm=self._cost_comm,
            mem_f=[1.0] * n,
            mem_b=[0.0] * n,
            mem_w=[-1.0] * n,
            max_mem=(
                (
                    [float(self._max_mem)] * n
                    if isinstance(self._max_mem, (int, float))
                    else [float(x) for x in self._max_mem]
                )
                if self._max_mem is not None
                else None
            ),
        )

    def generate_schedule_table(self) -> list[list[SchedulerNode]]:
        config = self._build_graph_config()
        graph = _Graph.build(self.pp_size, self.micro_batches, config)
        _, _, complete_time = _initial_solution(graph)

        _TYPE_F, _TYPE_B, _TYPE_W = 0, 1, 2
        _func_type_map = {_TYPE_F: FuncType.F, _TYPE_B: FuncType.B, _TYPE_W: FuncType.W}

        # Per-rank dict: time_step -> list[SchedulerNode]
        # Following the same timestamp-based approach as zbv-formatted / zbv-greedy
        # to ensure send/recv pairs land at the same logical time_step on both
        # ranks, preventing NCCL communication deadlocks.
        time_step_nodes: list[dict[int, list[SchedulerNode]]] = [{} for _ in range(self.pp_size)]

        def _insert(rank: int, ts: int, node: SchedulerNode):
            if node.args is None:
                node.args = {}
            node.args["time_step"] = ts
            time_step_nodes[rank].setdefault(ts, []).append(node)

        for stage in range(self.pp_size):
            nodes: list[tuple[float, int, int]] = []
            for type_idx in range(3):
                for mb in range(self.micro_batches):
                    node_id = graph.get_id(type_idx, stage, mb)
                    nodes.append((complete_time[node_id], type_idx, mb))
            nodes.sort()

            for ct, type_idx, mb in nodes:
                ft = _func_type_map[type_idx]
                _insert(stage, ct, SchedulerNode(func_type=ft, mini_batch=mb, chunk=0, args=None))

                if ft in (FuncType.F, FuncType.B):
                    send_info, recv_info = self.generate_send_recv_nodes_comm_pair(stage, mb, 0, ft)
                    if send_info is not None:
                        send_rank, send_node = send_info
                        recv_rank, recv_node = recv_info
                        _insert(send_rank, ct + 1, send_node)
                        _insert(recv_rank, ct + 1, recv_node)

        schedule_table: list[list[SchedulerNode]] = [[] for _ in range(self.pp_size)]
        for rank in range(self.pp_size):
            for ts in sorted(time_step_nodes[rank].keys()):
                step_nodes = time_step_nodes[rank][ts]
                comm = [
                    n
                    for n in step_nodes
                    if n.func_type in (FuncType.SF, FuncType.SB, FuncType.RF, FuncType.RB)
                ]
                comp = [n for n in step_nodes if n.func_type in (FuncType.F, FuncType.B, FuncType.W)]
                schedule_table[rank].extend(comm)
                schedule_table[rank].extend(comp)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZeroBubbleHeuristic(
        pp_size=4, vpp_size=1, micro_batches=16, cost_f=1000, cost_b=1000, cost_w=1000, cost_comm=10
    )
    table = schedule.generate_schedule_table()
    schedule.print_schedule_table(table)
