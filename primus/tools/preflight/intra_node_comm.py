###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time
from typing import Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist

from primus.tools.preflight.global_vars import (
    LOCAL_RANK,
    LOCAL_WORLD_SIZE,
    RANK,
    WORLD_SIZE,
    get_hostnames,
    get_iteration,
    get_warmup,
)
from primus.tools.preflight.utility import (
    barrier_after_comm_destroy,
    create_dir,
    extract_first_middle_last,
    extract_number,
    format_int_range,
    log,
)


def run_intra_node_comm(
    args,
    enabled_comms: Optional[Iterable[str]] = None,
    sizes_mb: Optional[Sequence[int]] = None,
    group_sizes: Optional[Sequence[int]] = None,
):
    """Intra-node allreduce / alltoall benchmark.

    Args:
        args: parsed namespace (must have markdown_file, dump_path, plot, ib_bw).
        enabled_comms: subset of {"allreduce", "alltoall"} to run. Defaults to both.
        sizes_mb: message sizes in MB. Defaults to powers of two from 2..1024.
        group_sizes: GPU group sizes; each must divide LOCAL_WORLD_SIZE.
            Defaults to [2, 4, 8].
    """
    device = torch.device(f"cuda:{LOCAL_RANK}")

    if sizes_mb is None or len(sizes_mb) == 0:
        sizes_mb = [2**i for i in range(1, 11)]
    sizes = [int(mb) * 1024 * 1024 for mb in sizes_mb]

    enabled_set = set(enabled_comms) if enabled_comms else {"allreduce", "alltoall"}
    enabled_set &= {"allreduce", "alltoall"}
    if not enabled_set:
        log("Skip intra-node comm benchmark (no enabled comms)")
        return

    if group_sizes is None or len(group_sizes) == 0:
        group_sizes = [2, 4, 8]
    # Filter out invalid group sizes (must divide LOCAL_WORLD_SIZE) and de-dupe.
    group_sizes = sorted({int(g) for g in group_sizes if int(g) > 0 and LOCAL_WORLD_SIZE % int(g) == 0})
    if not group_sizes:
        log(f"Skip intra-node comm benchmark, no valid group sizes for LOCAL_WORLD_SIZE={LOCAL_WORLD_SIZE}")
        return

    cases = {comm: list(group_sizes) for comm in ("allreduce", "alltoall") if comm in enabled_set}

    warmup = get_warmup()
    iteration = get_iteration()

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
            # Track which ranks belong to each group so we can render compact rows.
            all_group_ranks: List[List[int]] = []
            my_group_index = -1
            for i_node in range(num_nodes):
                for i_group in range(num_groups_per_node):
                    group_ranks = [
                        i_node * LOCAL_WORLD_SIZE + i_group * num_procs + r for r in range(num_procs)
                    ]
                    tmp_group = dist.new_group(ranks=group_ranks)
                    if RANK in group_ranks:
                        assert group is None
                        group = tmp_group
                        my_group_index = len(all_group_ranks)
                    all_group_ranks.append(group_ranks)
            assert group is not None
            assert my_group_index >= 0

            for size in sizes:
                tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
                dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
                for _ in range(warmup):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iteration):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=group)
                    else:
                        assert False
                torch.cuda.synchronize()
                elapsed = (time.time() - start) / iteration
                scale = 2 if comm == "allreduce" else 1
                comm_size = scale * size * (num_procs - 1) / num_procs
                gb_per_sec = comm_size / elapsed / 1e9
                latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
                bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

            dist.barrier(device_ids=[torch.cuda.current_device()])

            # destroy this parallel group
            dist.destroy_process_group(group)
            barrier_after_comm_destroy(args.comm_cleanup_delay_sec)

            all_latency_results = [None for _ in range(WORLD_SIZE)]
            all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                keys = sorted(
                    list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number
                )
                hostnames = get_hostnames()

                # Each row corresponds to one group (group_ranks). Use the first
                # rank's results since all members observe the same collective.
                # The Hostname column shows only the leader's host (compact);
                # use the Node range plus the legend at the top of the report
                # to look up the rest.
                def _row_for(group_ranks: List[int], results):
                    leader = group_ranks[0]
                    host_str = hostnames[leader]
                    node_str = format_int_range([r // LOCAL_WORLD_SIZE for r in group_ranks])
                    rank_str = format_int_range(group_ranks)
                    return host_str, node_str, rank_str, results[leader]

                formatted_keys = [f"{key:<6}" for key in keys]
                host_col_label = "Leader hostname"
                host_col_w = max(20, len(host_col_label) + 2)
                header_line = (
                    f"{host_col_label:<{host_col_w}} {'Node':<10} {'Rank':<10} " f"{' '.join(formatted_keys)}"
                )

                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======IntraNodeComm - {case_name} (us)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (us)=======")
                    log(header_line)

                    f.write(f"| {host_col_label} | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    for group_ranks in all_group_ranks:
                        host_str, node_str, rank_str, r = _row_for(group_ranks, all_latency_results)
                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(
                            f"{host_str:<{host_col_w}} {node_str:<10} {rank_str:<10} "
                            f"{' '.join(formatted_values)}"
                        )
                        f.write(f"| {host_str} | {node_str} | {rank_str} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    f.write(f"=======IntraNodeComm - {case_name} (GB/s)=======\n")
                    log(f"=======IntraNodeComm - {case_name} (GB/s)=======")
                    log(header_line)

                    f.write(f"| {host_col_label} | Node | Rank | {' | '.join(keys)}|\n")
                    f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
                    for group_ranks in all_group_ranks:
                        host_str, node_str, rank_str, r = _row_for(group_ranks, all_bandwidth_results)
                        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
                        log(
                            f"{host_str:<{host_col_w}} {node_str:<10} {rank_str:<10} "
                            f"{' '.join(formatted_values)}"
                        )
                        f.write(f"| {host_str} | {node_str} | {rank_str} | {' | '.join(formatted_values)}|\n")
                    f.write(f"\n")

                    if not args.plot:
                        continue

                import matplotlib.pyplot as plt

                log(f"=======Plot IntraNode {case_name} Bandwidth=======")
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======Plot IntraNode {case_name} Bandwidth=======\n")
                plot_case = f"intra_node_comm/{comm}"
                dump_path = f"{args.dump_path}/{plot_case}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                # Use leader rank of each group for plotting.
                leader_ranks = [g[0] for g in all_group_ranks]
                first_rank_bandwidth_results = [all_bandwidth_results[i] for i in leader_ranks]
                num_print_ranks = len(first_rank_bandwidth_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_bandwidth_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"RankPair ({num_procs} ranks)")
                    plt.ylabel("Bandwidth")
                    plt.title(f"Intra Node {case_name} bandwidth for {size_key}")
                    xtick_labels = [f"{leader_ranks[i]}" for i in range(num_print_ranks)]
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
