from ..scheduler_node import FuncType, SchedulerNode
from .base import VFoldScheduleAlgo


class ScheduleZBVGreedy(VFoldScheduleAlgo):
    """ZBVGreedy Pipeline Schedule"""

    def __init__(self, pp_size, vpp_size, micro_batches, cached_fwd_chunks_limit=None):

        if cached_fwd_chunks_limit is None:
            cached_fwd_chunks_limit = pp_size * vpp_size
        assert vpp_size == 2, "VFold1F1B requires vpp_size == 2"
        super().__init__(pp_size, vpp_size, micro_batches)
        self.cached_fwd_chunks_limit = cached_fwd_chunks_limit

    def generate_schedule_table(self):
        schedule_table = [[] for _ in range(self.pp_size)]

        time_stamp_map = [[None] * (2 * 3 * self.micro_batches * self.pp_size) for _ in range(self.pp_size)]
        memory_time_map = [[0] * (2 * 3 * self.micro_batches * self.pp_size) for _ in range(self.pp_size)]

        wgrad_node_queue = [[] for _ in range(self.pp_size)]

        def greedy_insert_node(rank, cur_time_stamp, node):
            for i in range(cur_time_stamp, len(time_stamp_map[rank])):
                if node.func_type == FuncType.F:
                    if (
                        memory_time_map[rank][i] < self.cached_fwd_chunks_limit
                        and time_stamp_map[rank][i] is None
                    ):
                        exceed_mem = False
                        for j in range(i, len(memory_time_map[rank])):
                            if memory_time_map[rank][j] >= self.cached_fwd_chunks_limit:
                                exceed_mem = True
                                break
                        if not exceed_mem:
                            time_stamp_map[rank][i] = node
                            return i
                elif time_stamp_map[rank][i] is None:
                    time_stamp_map[rank][i] = node
                    return i
            return -1

        def find_backward_node_time_stamp(rank, micro_batch, chunk):
            for i in range(len(time_stamp_map[rank])):
                if (
                    time_stamp_map[rank][i] is not None
                    and time_stamp_map[rank][i].func_type == FuncType.B
                    and time_stamp_map[rank][i].mini_batch == micro_batch
                    and time_stamp_map[rank][i].chunk == chunk
                ):
                    return i
            return -1

        for micro_batch in range(self.micro_batches):
            # insert forward / backward nodes
            cur_time_stamp = micro_batch
            assert cur_time_stamp != -1, "No valid time stamp found"
            # insert forward / backward nodes
            for insert_nodes_type in [FuncType.F, FuncType.B]:
                cur_rank = 0
                cur_chunk = 0 if insert_nodes_type == FuncType.F else 1
                for i in range(2 * self.pp_size):
                    schedule_node = SchedulerNode(
                        func_type=insert_nodes_type, mini_batch=micro_batch, chunk=cur_chunk, args=None
                    )
                    insert_idx = greedy_insert_node(cur_rank, cur_time_stamp, schedule_node)

                    while (
                        insert_idx == -1
                    ):  # failed to insert FWD node, execeed the limit of cached fwd chunks
                        assert len(wgrad_node_queue[cur_rank]) > 0, "No wgrad node to insert"
                        wgrad_node = wgrad_node_queue[cur_rank].pop(0)
                        backward_node_time = find_backward_node_time_stamp(
                            cur_rank, wgrad_node.mini_batch, wgrad_node.chunk
                        )
                        assert backward_node_time != -1
                        wgrad_insert_idx = greedy_insert_node(cur_rank, backward_node_time, wgrad_node)

                        for j in range(wgrad_insert_idx, len(time_stamp_map[cur_rank])):
                            memory_time_map[cur_rank][j] -= 1

                        insert_idx = greedy_insert_node(cur_rank, cur_time_stamp, schedule_node)

                    cur_time_stamp = insert_idx

                    if insert_nodes_type == FuncType.F:  # forward may increase activation memory
                        for j in range(cur_time_stamp, len(memory_time_map[cur_rank])):
                            memory_time_map[cur_rank][j] += 1
                    else:
                        wgrad_node_queue[cur_rank].append(
                            SchedulerNode(
                                func_type=FuncType.W, mini_batch=micro_batch, chunk=cur_chunk, args=None
                            )
                        )

                    dir_map = self.direction_map(cur_rank, cur_chunk, insert_nodes_type)
                    next_rank, next_chunk = dir_map["next"][0], dir_map["send_to_chunk"]

                    if next_rank is not None:
                        cur_rank = next_rank
                        cur_chunk = next_chunk
                    cur_time_stamp += 1

        # insert remain wgrad nodes
        for rank in range(self.pp_size):
            for i in range(len(wgrad_node_queue[rank])):
                wgrad_node = wgrad_node_queue[rank][i]
                backward_node_time = find_backward_node_time_stamp(
                    rank, wgrad_node.mini_batch, wgrad_node.chunk
                )
                assert backward_node_time != -1
                wgrad_insert_idx = greedy_insert_node(rank, backward_node_time, wgrad_node)

        # convert time_stamp_map to schedule_table
        for rank in range(self.pp_size):
            for node in time_stamp_map[rank]:
                if node is not None and not isinstance(node, bool):
                    schedule_table[rank].append(node)
        schedule_table = self.add_communication_nodes(schedule_table)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZBVGreedy(pp_size=4, vpp_size=2, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table, [FuncType.F, FuncType.B, FuncType.W])
