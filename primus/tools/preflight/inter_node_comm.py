###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time
from typing import Iterable, List, Optional, Sequence, Union

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


def _resolve_inter_group_sizes(
    group_sizes: Optional[Sequence[Union[int, str]]],
    num_nodes: int,
) -> List[int]:
    """Translate user-supplied inter-node group sizes (with 'all') to ints.

    - 'all' is mapped to num_nodes.
    - values > num_nodes are dropped.
    - duplicates are removed and the result is sorted ascending.
    """
    if group_sizes is None or len(group_sizes) == 0:
        candidates = [2, 4, num_nodes]
    else:
        candidates = []
        for g in group_sizes:
            if isinstance(g, str) and g.strip().lower() == "all":
                candidates.append(num_nodes)
            else:
                candidates.append(int(g))
    candidates = [c for c in candidates if c >= 2 and c <= num_nodes]
    return sorted(set(candidates))


def run_inter_node_comm(
    args,
    enabled_comms: Optional[Iterable[str]] = None,
    sizes_mb: Optional[Sequence[int]] = None,
    group_sizes: Optional[Sequence[Union[int, str]]] = None,
):
    """Inter-node allreduce / alltoall benchmark.

    Args:
        args: parsed namespace.
        enabled_comms: subset of {"allreduce", "alltoall"} to run. Defaults to both.
        sizes_mb: message sizes in MB.
        group_sizes: list of node group sizes; values > num_nodes are dropped.
            'all' is accepted as a synonym for num_nodes. Defaults to [2, 4, num_nodes].
    """
    device = torch.device(f"cuda:{LOCAL_RANK}")

    if sizes_mb is None or len(sizes_mb) == 0:
        sizes_mb = [2**i for i in range(1, 11)]
    sizes = [int(mb) * 1024 * 1024 for mb in sizes_mb]

    enabled_set = set(enabled_comms) if enabled_comms else {"allreduce", "alltoall"}
    enabled_set &= {"allreduce", "alltoall"}
    if not enabled_set:
        log("Skip inter-node comm benchmark (no enabled comms)")
        return

    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    if num_nodes <= 1:
        log(f"Skip inter node comm benchmark, {num_nodes=}")
        return

    node_counts = _resolve_inter_group_sizes(group_sizes, num_nodes)
    if not node_counts:
        log("Skip inter-node comm benchmark, no valid group sizes")
        return

    cases = {comm: list(node_counts) for comm in ("allreduce", "alltoall") if comm in enabled_set}

    warmup = get_warmup()
    iteration = get_iteration()

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"# InterNode Comm\n")

    for comm, adjacent_node_list in cases.items():
        if RANK == 0:
            with open(args.markdown_file, "a", encoding="utf-8") as f:
                f.write(f"## InterNode - {comm}\n")
        for adjacent_nodes in adjacent_node_list:
            if adjacent_nodes > num_nodes:
                continue

            case_name = f"{comm}-{adjacent_nodes}nodes"
            latency_results = {}
            bandwidth_results = {}

            num_full_groups = num_nodes // adjacent_nodes
            remainder_nodes = num_nodes % adjacent_nodes
            adjacent_group = None
            # Track per-group member ranks for compact reporting.
            all_group_ranks: List[List[int]] = []
            group_node_counts: List[int] = []

            for i_group in range(num_full_groups):
                group_start = i_group * adjacent_nodes * LOCAL_WORLD_SIZE
                group_ranks = [group_start + r for r in range(adjacent_nodes * LOCAL_WORLD_SIZE)]
                tmp_group = dist.new_group(ranks=group_ranks)
                if RANK in group_ranks:
                    assert adjacent_group is None
                    adjacent_group = tmp_group
                all_group_ranks.append(group_ranks)
                group_node_counts.append(adjacent_nodes)

            if remainder_nodes >= 2:
                group_start = num_full_groups * adjacent_nodes * LOCAL_WORLD_SIZE
                group_ranks = [group_start + r for r in range(remainder_nodes * LOCAL_WORLD_SIZE)]
                tmp_group = dist.new_group(ranks=group_ranks)
                if RANK in group_ranks:
                    assert adjacent_group is None
                    adjacent_group = tmp_group
                all_group_ranks.append(group_ranks)
                group_node_counts.append(remainder_nodes)

            num_procs = dist.get_world_size(adjacent_group) if adjacent_group is not None else 0

            total_grouped_ranks = num_full_groups * adjacent_nodes * LOCAL_WORLD_SIZE
            if remainder_nodes >= 2:
                total_grouped_ranks += remainder_nodes * LOCAL_WORLD_SIZE
            if RANK < total_grouped_ranks:
                assert adjacent_group is not None

            for size in sizes:
                if adjacent_group is None:
                    break

                tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
                dist.barrier(group=adjacent_group, device_ids=[torch.cuda.current_device()])
                for _ in range(warmup):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=adjacent_group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=adjacent_group)
                    else:
                        assert False
                torch.cuda.synchronize()
                start = time.time()
                for _ in range(iteration):
                    if "allreduce" == comm:
                        dist.all_reduce(tensor, group=adjacent_group)
                    elif "alltoall" == comm:
                        dist.all_to_all_single(tensor, tensor, group=adjacent_group)
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
            if adjacent_group is not None:
                dist.destroy_process_group(adjacent_group)
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

                # Show only the leader node's hostname; the Node range plus the
                # legend at the top of the report cover the rest.
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
                    f.write(f"=======InterNodeComm - {case_name} (us)=======\n")
                    log(f"=======InterNodeComm - {case_name} (us)=======")
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

                    f.write(f"=======InterNodeComm - {case_name} (GB/s)=======\n")
                    log(f"=======InterNodeComm - {case_name} (GB/s)=======")
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

                log(f"=======Plot InterNode {case_name} Bandwidth=======")
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"=======Plot InterNode {case_name} Bandwidth=======\n")
                plot_case = f"inter_node_comm/{comm}"
                dump_path = f"{args.dump_path}/{plot_case}"
                create_dir(dump_path)
                print_keys = extract_first_middle_last(keys)
                leader_ranks = [g[0] for g in all_group_ranks]
                first_rank_bandwidth_results = [all_bandwidth_results[i] for i in leader_ranks]
                num_print_ranks = len(first_rank_bandwidth_results)
                for size_key in print_keys:
                    values = [r[size_key] for r in first_rank_bandwidth_results]
                    plt.figure(figsize=(10, 4))
                    bars = plt.bar(range(num_print_ranks), values)
                    plt.xlabel(f"Group (starting rank)")
                    plt.ylabel("Bandwidth")
                    plt.title(f"Inter Node {case_name} Bandwidth for {size_key}")
                    xtick_labels = [
                        f"{leader_ranks[i]} ({group_node_counts[i]}N)" for i in range(num_print_ranks)
                    ]
                    plt.xticks(range(num_print_ranks), xtick_labels)
                    plt.grid(True, axis="y")

                    # Add roofline
                    roofline_bandwidth = args.ib_bw
                    plt.axhline(
                        y=roofline_bandwidth,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"IB Unidirectional BW Roofline: {roofline_bandwidth} GB/s",
                    )
                    plt.legend()

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

                    png_file = f"inter_node_{case_name}_bandwidth_{size_key.replace('x', '_')}.png"
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
                plt.title(f"Inter Node {case_name} Bandwidth for Rank 0")
                plt.grid(True, axis="y")
                # Add roofline
                roofline_bandwidth = args.ib_bw
                plt.axhline(
                    y=roofline_bandwidth,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    label=f"IB Unidirectional BW Roofline: {roofline_bandwidth} GB/s",
                )
                plt.legend()

                # plt value
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
                    )

                png_file = f"inter_node_{case_name}_bandwidth_rank_0.png"
                plt.tight_layout()
                plt.savefig(f"{dump_path}/{png_file}")
                plt.close()
                with open(args.markdown_file, "a", encoding="utf-8") as f:
                    f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
                    f.write(f"\n")
                log(f"")
