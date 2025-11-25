from ..scheduler_node import FuncType, SchedulerNode
from .base import VFoldScheduleAlgo


class ScheduleVFold1F1B(VFoldScheduleAlgo):
    """VFold1F1B Pipeline Schedule"""

    def __init__(self, pp_size, vpp_size, micro_batches):
        assert vpp_size == 2, "VFold1F1B requires vpp_size == 2"
        super().__init__(pp_size, vpp_size, micro_batches)

    def generate_schedule_table(self):
        schedule_table = [[] for _ in range(self.pp_size)]

        time_stamp_map = [[None] * (2 * 3 * self.micro_batches * 2) for _ in range(self.pp_size)]

        def greedy_insert_node(rank, cur_time_stamp, node, width=1):
            for i in range(cur_time_stamp, 2 * 3 * self.micro_batches * 2):
                if width == 1:
                    if time_stamp_map[rank][i] is None:
                        time_stamp_map[rank][i] = node
                        return i
                elif width == 2:
                    if time_stamp_map[rank][i] is None and time_stamp_map[rank][i + 1] is None:
                        time_stamp_map[rank][i] = node
                        time_stamp_map[rank][i + 1] = True  # fake node to indicate the width is 2
                        return i + 1
            return -1

        for micro_batch in range(self.micro_batches):
            cur_time_stamp = micro_batch
            # insert forward / backward nodes
            for insert_nodes_type in [FuncType.F, FuncType.BW]:
                cur_rank = 0
                cur_chunk = 0 if insert_nodes_type == FuncType.F else 1
                width = 1 if insert_nodes_type == FuncType.F else 2
                for i in range(2 * self.pp_size):

                    cur_time_stamp = greedy_insert_node(
                        cur_rank,
                        cur_time_stamp,
                        SchedulerNode(
                            func_type=insert_nodes_type, mini_batch=micro_batch, chunk=cur_chunk, args=None
                        ),
                        width,
                    )
                    if cur_time_stamp == -1:
                        raise ValueError("Failed to insert node")

                    dir_map = self.direction_map(cur_rank, cur_chunk, insert_nodes_type)
                    next_rank, next_chunk = dir_map["next"][0], dir_map["send_to_chunk"]

                    if next_rank is not None:
                        cur_rank = next_rank
                        cur_chunk = next_chunk

                    cur_time_stamp += 1

        # convert time_stamp_map to schedule_table
        for rank in range(self.pp_size):
            for node in time_stamp_map[rank]:
                if node is not None and not isinstance(node, bool):
                    schedule_table[rank].append(node)

        schedule_table = self.add_communication_nodes(schedule_table)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleVFold1F1B(pp_size=4, vpp_size=2, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table, [FuncType.F, FuncType.BW])
