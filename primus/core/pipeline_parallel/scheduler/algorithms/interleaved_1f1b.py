from ..scheduler_node import FuncType, SchedulerNode
from .base import PipelineScheduleAlgo


class ScheduleInterleaved1F1B(PipelineScheduleAlgo):
    """Interleaved 1F1B Pipeline Schedule with Virtual Pipeline Parallelism

    This schedule uses virtual pipeline parallelism (VPP) where each device handles
    multiple model chunks to reduce pipeline bubbles.

    Key concepts:
    - Each rank processes multiple chunks (vpp_size chunks)
    - Each chunk processes all microbatches
    - Chunks are interleaved based on their stage IDs to minimize bubbles
    """

    def __init__(self, pp_size, vpp_size, micro_batches):
        super().__init__(pp_size, vpp_size, micro_batches)
        assert vpp_size > 1, "Interleaved 1F1B requires vpp_size > 1"

    def direction_map(self, rank: int, chunk: int, func_type: FuncType) -> dict[str, int]:
        """Map communication directions for interleaved schedule

        In interleaved mode, even chunks go forward, odd chunks go backward.
        """
        if func_type == FuncType.F:
            # Forward: even chunks go forward, odd chunks go backward
            if rank > 0 and rank < self.pp_size - 1:
                prev_rank = rank - 1
                next_rank = rank + 1
                recv_from_chunk = chunk
                send_to_chunk = chunk

            elif rank == 0:
                prev_rank = None if chunk == 0 else self.pp_size - 1
                next_rank = rank + 1
                recv_from_chunk = chunk - 1 if chunk > 0 else None
                send_to_chunk = chunk
            elif rank == self.pp_size - 1:
                prev_rank = rank - 1
                next_rank = None if chunk == self.vpp_size - 1 else 0

                recv_from_chunk = chunk
                send_to_chunk = chunk + 1 if chunk < self.vpp_size - 1 else None

            return {
                "prev": (prev_rank, FuncType.RF),
                "next": (next_rank, FuncType.SF),
                "recv_from_chunk": recv_from_chunk,
                "send_to_chunk": send_to_chunk,
            }

        elif func_type == FuncType.BW:
            # Backward: reverse of forward
            if rank > 0 and rank < self.pp_size - 1:
                prev_rank = rank + 1
                next_rank = rank - 1

                recv_from_chunk = chunk
                send_to_chunk = chunk
            elif rank == 0:
                prev_rank = rank + 1
                next_rank = None if chunk == 0 else self.pp_size - 1

                recv_from_chunk = chunk
                send_to_chunk = chunk - 1 if chunk > 0 else None
            elif rank == self.pp_size - 1:
                prev_rank = None if chunk == self.vpp_size - 1 else 0
                next_rank = rank - 1

                recv_from_chunk = chunk + 1 if chunk < self.vpp_size - 1 else None
                send_to_chunk = chunk

            return {
                "prev": (prev_rank, FuncType.RB),
                "next": (next_rank, FuncType.SB),
                "recv_from_chunk": recv_from_chunk,
                "send_to_chunk": send_to_chunk,
            }
        else:
            raise ValueError(f"Invalid function type: {func_type}")

    def generate_schedule_table(self):
        """Generate interleaved 1F1B schedule table

        Strategy:
        - Build complete schedule for each chunk independently
        - Merge schedules by prioritizing operations from chunks with earlier stage IDs
        - This allows chunk 0 to execute multiple forward passes before chunk 1 starts
        """
        schedule_table = [[] for _ in range(self.pp_size)]
        vpp_range_len = self.pp_size * self.vpp_size

        for rank in range(self.pp_size):
            # Build schedule for each chunk with timing information
            rank_schedule = schedule_table[rank]

            num_of_warmup_phases = self.pp_size + (self.pp_size - 1 - rank) * 2

            fwd_chunk = 0
            bwd_chunk = self.vpp_size - 1

            # warmup phase
            for i in range(num_of_warmup_phases):
                if i % vpp_range_len == 0:
                    fwd_chunk = 0
                elif i % self.pp_size == 0:
                    fwd_chunk += 1

                fwd_mini_batch = i % self.pp_size + (i // vpp_range_len * self.pp_size)
                rank_schedule.append(
                    SchedulerNode(func_type=FuncType.F, mini_batch=fwd_mini_batch, chunk=fwd_chunk, args=None)
                )

            # 1f1b steady phase
            for i in range(num_of_warmup_phases, self.micro_batches * self.vpp_size):
                if i % vpp_range_len == 0:
                    fwd_chunk = 0
                elif i % self.pp_size == 0:
                    fwd_chunk += 1

                fwd_mini_batch = i % self.pp_size + (i // vpp_range_len * self.pp_size)

                rank_schedule.append(
                    SchedulerNode(func_type=FuncType.F, mini_batch=fwd_mini_batch, chunk=fwd_chunk, args=None)
                )

                backward_i = i - num_of_warmup_phases
                if backward_i % vpp_range_len == 0:
                    bwd_chunk = self.vpp_size - 1
                elif backward_i % self.pp_size == 0:
                    bwd_chunk -= 1

                bwd_mini_batch = backward_i % self.pp_size + (backward_i // vpp_range_len * self.pp_size)
                rank_schedule.append(
                    SchedulerNode(
                        func_type=FuncType.BW, mini_batch=bwd_mini_batch, chunk=bwd_chunk, args=None
                    )
                )

            # cooldown phase
            for backward_i in range(
                self.micro_batches * self.vpp_size - num_of_warmup_phases, self.micro_batches * self.vpp_size
            ):
                if backward_i % vpp_range_len == 0:
                    bwd_chunk = self.vpp_size - 1
                elif backward_i % self.pp_size == 0:
                    bwd_chunk -= 1

                bwd_mini_batch = backward_i % self.pp_size + (backward_i // vpp_range_len * self.pp_size)

                rank_schedule.append(
                    SchedulerNode(
                        func_type=FuncType.BW, mini_batch=bwd_mini_batch, chunk=bwd_chunk, args=None
                    )
                )

        schedule_table = self.add_communication_nodes(schedule_table)
        return schedule_table


if __name__ == "__main__":
    schedule = ScheduleInterleaved1F1B(pp_size=4, vpp_size=2, micro_batches=16)
    schedule_table = schedule.generate_schedule_table()
    schedule.print_schedule_table(schedule_table, filter=[FuncType.BW])
