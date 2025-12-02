from ..scheduler_node import FuncType, SchedulerNode
from .base import VFoldScheduleAlgo


class ScheduleZBVFormatted(VFoldScheduleAlgo):
    """ZBV Formatted Pipeline Schedule"""

    def __init__(self, pp_size, vpp_size, micro_batches, wgrad_delayed_batches=0):
        assert vpp_size == 2, "VFold1F1B requires vpp_size == 2"
        super().__init__(pp_size, vpp_size, micro_batches)
        self.wgrad_delayed_batches = wgrad_delayed_batches
        self.pp_group_size = pp_size

    def generate_schedule_table(self):
        # max(2 * self.pp_group_size - 1, ...) ensure the number of microbatches is at least
        # as large of the number of microbatches needed to fully utilize the pipeline
        n_micro = max(2 * self.pp_group_size - 1, self.micro_batches)
        # rank_ops: list[Optional[_Action]] = [None for _ in range(rank)]
        schedule_table = [[] for _ in range(self.pp_size)]
        time_step = [i for i in range(self.pp_size)]

        time_step_nodes = [dict() for _ in range(self.pp_size)]

        # Forward and backward action counts for stage chunk 0 and chunk 1

        def insert_time_step_nodes(rank, time_step, node):
            if node is None: return
            if node.args is None:
                node.args = {}
            node.args["time_step"] = time_step
            if time_step in time_step_nodes[rank]:
                time_step_nodes[rank][time_step].append(node)
            else:
                time_step_nodes[rank][time_step] = [node]

        
        def insert_computation_node(rank, mini_batch, chunk, func_type):
            if func_type == FuncType.W:
                insert_time_step_nodes(rank, time_step[rank], SchedulerNode(func_type=func_type, mini_batch=mini_batch, chunk=chunk, args=None))
                time_step[rank] += 1
            else:
                compute_node = SchedulerNode(func_type=func_type, mini_batch=mini_batch, chunk=chunk, args=None)
                insert_time_step_nodes(rank, time_step[rank], compute_node)
                time_step[rank] += 1
                send_node_info, recv_node_info = self.generate_send_recv_nodes_comm_pair(
                    rank, mini_batch, chunk, func_type
                )
                if send_node_info is not None:
                    send_rank, send_node = send_node_info
                    recv_rank, recv_node = recv_node_info
                    insert_time_step_nodes(send_rank, time_step[send_rank], send_node)
                    insert_time_step_nodes(recv_rank, time_step[rank], recv_node)
        
        for rank in range(self.pp_size):
            # warm-up phase
            warmup_n1 = 2 * (self.pp_size - rank) - 1
            f0_cnt, f1_cnt, b0_cnt, b1_cnt = 0, 0, 0, 0

            for _ in range(warmup_n1):

                insert_computation_node(rank, f0_cnt, 0, FuncType.F)
                f0_cnt += 1

            warmup_n2 = rank
            for _ in range(warmup_n2):
                insert_computation_node(rank, f1_cnt, 1, FuncType.F)
                f1_cnt += 1

                insert_computation_node(rank, f0_cnt, 0, FuncType.F)
                f0_cnt += 1

            warmup_n3 = self.pp_group_size - rank
            for _ in range(warmup_n3):
                insert_computation_node(rank, f1_cnt, 1, FuncType.F)
                f1_cnt += 1
                

                insert_computation_node(rank, b1_cnt, 1, FuncType.B)
                insert_computation_node(rank, b1_cnt, 1, FuncType.W)
                b1_cnt += 1

            # stable phase
            while f1_cnt < f0_cnt or f0_cnt < n_micro:
                if f0_cnt < n_micro:
                    insert_computation_node(rank, f0_cnt, 0, FuncType.F)
                    f0_cnt += 1

                insert_computation_node(rank, b0_cnt, 0, FuncType.B)
                insert_computation_node(rank, b0_cnt, 0, FuncType.W)
                b0_cnt += 1

                insert_computation_node(rank, f1_cnt, 1, FuncType.F)
                f1_cnt += 1
                insert_computation_node(rank, b1_cnt, 1, FuncType.B)
                insert_computation_node(rank, b1_cnt, 1, FuncType.W)
                b1_cnt += 1
            # cool-down phase
            w0_cnt, w1_cnt = b0_cnt, b1_cnt
            cooldown_n1 = rank
            for _ in range(cooldown_n1):
                insert_computation_node(rank, b0_cnt, 0, FuncType.B)
                b0_cnt += 1
                insert_computation_node(rank, b1_cnt, 1, FuncType.B)
                b1_cnt += 1
            cooldown_n2 = self.pp_group_size - rank
            for _ in range(cooldown_n2):
                insert_computation_node(rank, b0_cnt, 0, FuncType.B)
                b0_cnt += 1
                insert_computation_node(rank, w0_cnt, 0, FuncType.W)
                w0_cnt += 1
            while w1_cnt < b1_cnt:
                insert_computation_node(rank, w1_cnt, 1, FuncType.W)
                w1_cnt += 1
            while w0_cnt < b0_cnt:
                insert_computation_node(rank, w0_cnt, 0, FuncType.W)
                w0_cnt += 1

            assert w0_cnt == b0_cnt and b0_cnt == f0_cnt
            assert w1_cnt == b1_cnt and b1_cnt == f1_cnt

        for rank in range(self.pp_size):

            for time_step in sorted(time_step_nodes[rank].keys()):
                nodes = time_step_nodes[rank][time_step]

                compute_nodes = [node for node in nodes if node.func_type in [FuncType.F, FuncType.B, FuncType.W]]
                comm_nodes = [node for node in nodes if node.func_type in [FuncType.SF, FuncType.SB, FuncType.RF, FuncType.RB]]

                schedule_table[rank].extend(comm_nodes)
                schedule_table[rank].extend(compute_nodes)

        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleZBVFormatted(pp_size=4, vpp_size=2, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table)
