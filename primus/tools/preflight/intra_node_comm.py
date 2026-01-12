###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist

from primus.tools.preflight.global_vars import (
    ITERATION,
    LOCAL_RANK,
    LOCAL_WORLD_SIZE,
    RANK,
    WARMUP,
    WORLD_SIZE,
    get_hostnames,
)
from primus.tools.preflight.utility import (
    create_dir,
    extract_first_middle_last,
    extract_number,
    log,
)


def run_intra_node_comm(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    cases = {
        "allreduce": [2, 4, 8],
        "alltoall": [2, 4, 8],
    }

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# IntraNode Comm Perf\n")

    for comm, parallel in cases.items():
        if RANK == 0:
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"## IntraNode - {comm}\n")
        for num_procs in parallel:
            bandwidth_results = {}
            latency_results = {}
            case_name = f"{comm}-{num_procs}gpu"

            assert LOCAL_WORLD_SIZE % num_procs == 0
            assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
            num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE
            num_groups_per_node = LOCAL_WORLD_SIZE // num_procs
            group = None
            for i_node in range(num_nodes):
                for i_group in range(num_groups_per_node):
                    group_ranks = [
                        i_node * LOCAL_WORLD_SIZE + i_group * num_procs + r for r in range(num_procs)
                    ]
                    tmp_group = dist.new_group(ranks=group_ranks)
                    if RANK in group_ranks:
                        assert group is None
                        group = tmp_group
            assert group is not None

            for size in sizes:
                tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
                for _ in range(WARMUP):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(ITERATION):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / ITERATION
                scale = 2 if comm == "allreduce" else 1
                comm_size = scale * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])

            # destroy this parallel group
            dist.destroy_process_group(group)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number
                )
                max_len = max(len(s) for s in get_hostnames()) + 2

                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======IntraNodeComm - {case_name} (us)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (us)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_latency_results):
                        hostname = get_hostnames()[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    f.write(f"=======IntraNodeComm - {case_name} (GB/s)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (GB/s)=======")

                    f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    formatted_keys = [f"{key:<6}" for key in keys]
                    log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
                    for rank, r in enumerate(all_bandwidth_results):
                        hostname = get_hostnames()[rank]
                        if rank % num_procs != 0:
                            continue
                        node_id = rank // LOCAL_WORLD_SIZE

                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                        f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    if not args.plot:
                        continue

                log(f"=======Plot IntraNode {case_name} Bandwidth=======")
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======Plot IntraNode {case_name} Bandwidth=======\n")
                plot_case = f"intra_node_comm/{comm}"
                dump_path = f"{args.dump_path}/{plot_case}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                first_rank_bandwidth_results = [
                    all_bandwidth_results[i] for i in range(len(all_bandwidth_results)) if i % num_procs == 0
                ]
                num_print_ranks = len(first_rank_bandwidth_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_bandwidth_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("Bandwidth")
                    plt.title(f"Intra Node {case_name} bandwidth for {size_key}")
                    xtick_labels = [f"{i*num_procs}" for i in range(num_print_ranks)]
                    plt.xticks(range(num_print_ranks), xtick_labels)
                    plt.grid(True, axis="y")

                    # plt value
                    for bar in bars:
                        height = bar.get_height()
                        plt.text(
                            bar.get_x() + bar.get_width() / 2,
                            height,
                            f"{height:.2f}",
                            ha="center",
                            va="bottom",
                        )

                    png_file = f"intra_node_{case_name}_bandwidth_{size_key.replace('x', '_')}.png"
                    plt.tight_layout()
                    plt.savefig(f"{dump_path}/{png_file}")
                    plt.close()
                    with open(args.markdown_file, "a", encoding="utf-8") as f:
                        f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")

                # Bar chart visualization for rank 0
                rank_0_values = [all_bandwidth_results[0][size_key] for size_key in keys]
                plt.figure(figsize=(10, 4))
                bars = plt.bar(keys, rank_0_values)
                plt.xlabel("Size")
                plt.ylabel("Bandwidth")
                plt.title(f"Intra Node {case_name} bandwidth for Rank 0")
                plt.grid(True, axis="y")

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                png_file = f"intra_node_{case_name}_bandwidth_rank_0.png"
                plt.tight_layout()
                plt.savefig(f"{dump_path}/{png_file}")
                plt.close()
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
                    f.write(f"\n")
                log(f"")
