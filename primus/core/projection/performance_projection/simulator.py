import importlib
import json
import os

from primus.core.pipeline_parallel.scheduler.scheduler_node import (
    FuncType,
    SchedulerNode,
)


class SchedulerSimulationRunner:
    def __init__(self, config: dict):
        self.config = config
        self.chunk_time_ms = self.config.get("chunk_time_ms")
        self._result_key_dict = {
            FuncType.F: "fwd",
            FuncType.B: "bwd",
            FuncType.W: "wgrad",
            FuncType.BW: "bwd",
        }

        self.debug_simulator = int(os.getenv("DEBUG_SIMULATOR", "0") == "1")

    def _summarize_simulation_result(self, simulation_result: list[dict], scheduler_config: dict) -> dict:
        rank_totals = [rank.get("total", 0.0) for rank in simulation_result]
        step_time_ms = max(rank_totals) if rank_totals else 0.0
        critical_rank = rank_totals.index(step_time_ms) if rank_totals else None
        max_memory = max((rank.get("memory", 0.0) for rank in simulation_result), default=0.0)
        return {
            "step_time_ms": step_time_ms,
            "rank_totals": rank_totals,
            "critical_rank": critical_rank,
            "max_memory": max_memory,
            "micro_batches": scheduler_config.get("micro_batches"),
            "pp_size": scheduler_config.get("pp_size"),
            "vpp_size": scheduler_config.get("vpp_size"),
        }

    def _chunk_duration(
        self, rank: int, chunk: int | None, func_type: FuncType, scheduler_config: dict
    ) -> float:
        chunk_idx = chunk or 0
        assert self.chunk_time_ms is not None
        chunk_entry = self.chunk_time_ms[rank][chunk_idx]
        assert chunk_entry is not None
        if func_type == FuncType.F:
            duration = chunk_entry.get("fwd")
        elif func_type == FuncType.B:
            duration = chunk_entry.get("bwd")
        elif func_type == FuncType.W:
            duration = chunk_entry.get("wgrad")
        elif func_type == FuncType.BW:
            bwd = chunk_entry.get("bwd", 0.0)
            wgrad = chunk_entry.get("wgrad", bwd)
            duration = bwd + wgrad
        else:
            duration = None
        if duration is not None and duration > 0:
            return float(duration)
        else:
            raise ValueError(f"Duration is not found.")

    def _chunk_activation(self, rank: int, chunk: int | None) -> float:
        if self.chunk_time_ms is None:
            return 0.0
        chunk_idx = chunk or 0
        chunk_entry = self.chunk_time_ms[rank][chunk_idx]
        return float(chunk_entry.get("activation", 0.0) or 0.0)

    def run(self):
        run_summaries = []
        for scheduler_config in self.config["schedulers"]:
            class_path = scheduler_config["class"]
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            scheduler_class = getattr(module, class_name)

            scheduler_params = {k: v for k, v in scheduler_config.items() if k not in ["name", "class"]}

            scheduler_instance = scheduler_class(**scheduler_params)
            schedule_table = scheduler_instance.generate_schedule_table()
            print(f"{'='*80}")
            print(f"Scheduler: {scheduler_config['name']}")
            print(f"{'='*80}")
            if self.debug_simulator:
                scheduler_instance.print_schedule_table(schedule_table)
            simulation_result = self.simulate_scheduler_table(schedule_table, scheduler_config)
            self.dump_simulation_result(simulation_result, scheduler_config)
            summary = self._summarize_simulation_result(simulation_result, scheduler_config)
            run_summaries.append(
                {
                    "name": scheduler_config["name"],
                    "config": scheduler_config,
                    "summary": summary,
                    "per_rank": simulation_result,
                }
            )
        return run_summaries

    def simulate_scheduler_table(self, schedule_table: list[list[SchedulerNode]], scheduler_config: dict):

        current_rank = 0
        rank_clock = [0.0 for _ in range(len(schedule_table))]
        rank_idx = [0 for _ in range(len(schedule_table))]

        rank_memory = [0.0 for _ in range(len(schedule_table))]

        communication_map = dict()

        simulation_result = [
            {
                "total": 0.0,
                "memory": 0.0,
                "fwd_start": [],
                "fwd_end": [],
                "bwd_start": [],
                "bwd_end": [],
                "wgrad_start": [],
                "wgrad_end": [],
                "fwd_minibatch": [],
                "fwd_chunk": [],
                "bwd_minibatch": [],
                "bwd_chunk": [],
                "wgrad_minibatch": [],
                "wgrad_chunk": [],
                "activation_memory_usage": [],
            }
            for _ in range(len(schedule_table))
        ]

        max_retry = len(schedule_table) * len(schedule_table[0]) * 2
        while True:
            max_retry -= 1
            if max_retry <= 0:
                print(f"Max retry reached, May have bugs in the schedule table")
                print(communication_map)
                break

            all_finished = True
            for rank in range(len(schedule_table)):
                if len(schedule_table[rank]) == rank_idx[rank]:
                    simulation_result[rank]["total"] = rank_clock[rank]
                    continue
                all_finished = False

            if all_finished:
                break

            while rank_idx[current_rank] < len(schedule_table[current_rank]):
                node = schedule_table[current_rank][rank_idx[current_rank]]

                if node.func_type in [FuncType.SF, FuncType.SB]:
                    send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{node.func_type}_{node.mini_batch}_{node.args['send_to_chunk']}"
                    if self.debug_simulator:
                        print(f"rank {current_rank} send_key: {send_key}")
                    communication_map[send_key] = rank_clock[current_rank]
                if node.func_type in [FuncType.RF, FuncType.RB]:
                    send_func_type = FuncType.SF if node.func_type == FuncType.RF else FuncType.SB
                    send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{send_func_type}_{node.mini_batch}_{node.chunk}"
                    if send_key not in communication_map:
                        merge_comm_index = rank_idx[current_rank] + 1
                        if self.debug_simulator:
                            print(f"rank {current_rank} wait send_key {send_key}")

                        # merge the send op behind
                        for i in range(merge_comm_index, len(schedule_table[current_rank])):
                            if schedule_table[current_rank][i].func_type in [FuncType.SF, FuncType.SB]:
                                send_key = f"{node.args['from_pp_rank']}_{node.args['to_pp_rank']}_{node.func_type}_{node.mini_batch}_{node.args['send_to_chunk']}"
                                communication_map[send_key] = rank_clock[current_rank]
                            else:
                                break
                        break
                    else:
                        if self.debug_simulator:
                            print(f"rank {current_rank} send_key {send_key} recved")

                    send_time = communication_map.pop(send_key)

                    rank_clock[current_rank] = max(rank_clock[current_rank], send_time)

                if node.func_type in [FuncType.F, FuncType.B, FuncType.W, FuncType.BW, FuncType.FB]:
                    simulation_result[current_rank][f"{self._result_key_dict[node.func_type]}_start"].append(
                        rank_clock[current_rank]
                    )
                    duration = self._chunk_duration(
                        current_rank, getattr(node, "chunk", 0), node.func_type, scheduler_config
                    )
                    rank_clock[current_rank] += duration
                    simulation_result[current_rank][f"{self._result_key_dict[node.func_type]}_end"].append(
                        rank_clock[current_rank]
                    )
                    simulation_result[current_rank][
                        f"{self._result_key_dict[node.func_type]}_minibatch"
                    ].append(node.mini_batch)
                    simulation_result[current_rank][f"{self._result_key_dict[node.func_type]}_chunk"].append(
                        node.chunk
                    )
                    if node.func_type == FuncType.F:
                        act_gb = self._chunk_activation(current_rank, getattr(node, "chunk", 0))
                        rank_memory[current_rank] += act_gb
                        simulation_result[current_rank]["memory"] = max(
                            simulation_result[current_rank]["memory"], rank_memory[current_rank]
                        )
                        simulation_result[current_rank]["activation_memory_usage"].append(
                            rank_memory[current_rank]
                        )
                    elif node.func_type in [FuncType.BW, FuncType.W]:
                        act_gb = self._chunk_activation(current_rank, getattr(node, "chunk", 0))
                        rank_memory[current_rank] = rank_memory[current_rank] - act_gb
                        simulation_result[current_rank]["activation_memory_usage"].append(
                            rank_memory[current_rank]
                        )
                rank_idx[current_rank] += 1

            current_rank = (current_rank + 1) % len(schedule_table)

        return simulation_result

    def dump_simulation_result(self, simulation_result: list[dict], scheduler_config: dict):
        result_dir = f"{self.config['output_dir']}/{scheduler_config['name']}"
        os.makedirs(result_dir, exist_ok=True)
        with open(f"{result_dir}/config.json", "w") as f:
            dump_dict = {
                "pp_size": scheduler_config["pp_size"],
                "vpp_size": scheduler_config["vpp_size"],
                "micro_batches": scheduler_config["micro_batches"],
            }
            json.dump(dump_dict, f, indent=2)

        for i in range(len(simulation_result)):
            with open(f"{result_dir}/pp_rank_{i}.json", "w") as f:
                dump_dict = {"0": simulation_result[i]}
                json.dump(dump_dict, f, indent=2)
