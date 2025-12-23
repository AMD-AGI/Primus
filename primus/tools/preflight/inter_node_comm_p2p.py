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


def run_inter_node_comm_p2p(args):
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sizes = [2**i * 1024 * 1024 for i in range(1, 11)]
    # sizes = [2**i * 1024 * 1024 for i in range(1, 5)]
    assert WORLD_SIZE % LOCAL_WORLD_SIZE == 0
    num_nodes = WORLD_SIZE // LOCAL_WORLD_SIZE

    if num_nodes <= 1:
        log(f"Skip inter node comm benchmark, {num_nodes=}")
        return
    # 2-node p2p
    #   pair nodes: [0, 1]
    #        ranks: [0, 8], [1, 9], [2, 10], ...
    #   pair nodes: [2, 3]
    #        ranks: [16, 24], [17, 25], [18, 26], ...
    comm = "p2p"
    adjacent_nodes = 2
    case_name = f"{comm}-{adjacent_nodes}nodes"
    latency_results = {}
    bandwidth_results = {}

    num_adjacent_groups = num_nodes // adjacent_nodes
    p2p_group = None
    is_src_rank = ((RANK // LOCAL_WORLD_SIZE) % 2) == 0
    peer_rank = RANK + LOCAL_WORLD_SIZE if is_src_rank else RANK - LOCAL_WORLD_SIZE
    assert peer_rank >= 0 and peer_rank < WORLD_SIZE
    for i_group in range(num_adjacent_groups):
        for i_r in range(LOCAL_WORLD_SIZE):
            group_ranks = [
                i_group * adjacent_nodes * LOCAL_WORLD_SIZE + i_r,
                i_group * adjacent_nodes * LOCAL_WORLD_SIZE + i_r + LOCAL_WORLD_SIZE,
            ]
            tmp_group = dist.new_group(ranks=group_ranks)
            if RANK in group_ranks:
                assert p2p_group is None
                p2p_group = tmp_group
    if RANK < num_adjacent_groups * adjacent_nodes * LOCAL_WORLD_SIZE:
        assert p2p_group is not None

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"## InterNode - P2P\n")

    for size in sizes:
        if p2p_group is None:
            break

        tensor = torch.rand(size // 2, dtype=torch.bfloat16, device=device)
        dist.barrier(group=p2p_group, device_ids=[torch.cuda.current_device()])
        for _ in range(WARMUP):
            if is_src_rank:
                dist.send(tensor, dst=peer_rank, group=p2p_group)
            else:
                dist.recv(tensor, src=peer_rank, group=p2p_group)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERATION):
            if is_src_rank:
                dist.send(tensor, dst=peer_rank, group=p2p_group)
            else:
                dist.recv(tensor, src=peer_rank, group=p2p_group)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / ITERATION
        comm_size = size
        gb_per_sec = comm_size / elapsed / 1e9
        latency_results[f"{size//1024//1024}MB"] = elapsed * 1e6
        bandwidth_results[f"{size//1024//1024}MB"] = gb_per_sec

    dist.barrier(device_ids=[torch.cuda.current_device()])
    if p2p_group is not None:
        dist.destroy_process_group(p2p_group)

    all_latency_results = [None for _ in range(WORLD_SIZE)]
    all_bandwidth_results = [None for _ in range(WORLD_SIZE)]
    dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
    dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

    if RANK == 0:
        keys = sorted(list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number)
        max_len = max(len(s) for s in get_hostnames()) + 2

        # result of src ranks will be print
        src_ranks = []
        peer_ranks = []
        src_rank_latency_results = []
        src_rank_bandwidth_results = []
        for rank, r in enumerate(all_bandwidth_results):
            is_src_rank = ((rank // LOCAL_WORLD_SIZE) % 2) == 0
            peer_rank = rank + LOCAL_WORLD_SIZE if is_src_rank else rank - LOCAL_WORLD_SIZE
            assert peer_rank >= 0 and peer_rank < WORLD_SIZE
            if not is_src_rank:
                continue
            src_ranks.append(rank)
            peer_ranks.append(peer_rank)
            src_rank_latency_results.append(all_latency_results[rank])
            src_rank_bandwidth_results.append(r)

        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======InterNodeComm - {case_name} (us)=======\n")
            log(f"=======InterNodeComm - {case_name} (us)=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")

            formatted_keys = [f"{key:<6}" for key in keys]
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            for i_r in range(len(src_ranks)):
                rank = src_ranks[i_r]
                hostname = get_hostnames()[rank]
                node_id = rank // LOCAL_WORLD_SIZE

                formatted_values = [f"{src_rank_latency_results[i_r].get(key, 0):<6.2f}" for key in keys]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

            f.write(f"=======InterNodeComm - {case_name} (GB/s)=======\n")
            log(f"=======InterNodeComm - {case_name} (GB/s)=======")

            f.write(f"| Hostname | Node | Rank | {' | '.join(keys)}|\n")
            f.write(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
            formatted_keys = [f"{key:<6}" for key in keys]
            log(f"{'Hostname':<{max_len}} {'Node':<5} {'Rank':<5} {' '.join(formatted_keys)}")
            for i_r in range(len(src_ranks)):
                rank = src_ranks[i_r]
                hostname = get_hostnames()[rank]
                node_id = rank // LOCAL_WORLD_SIZE

                formatted_values = [f"{src_rank_bandwidth_results[i_r].get(key, 0):<6.2f}" for key in keys]
                log(f"{hostname:<{max_len}} {node_id:<5} {rank:<5} {' '.join(formatted_values)}")
                f.write(f"| {hostname} | {node_id} | {rank} | {' | '.join(formatted_values)}|\n")
            f.write(f"\n")

        if not args.plot:
            return

        log(f"=======Plot InterNode {case_name} Bandwidth=======")
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"=======Plot InterNode {case_name} Bandwidth=======\n")
        plot_case = f"inter_node_comm/{comm}"
        dump_path = f"{args.dump_path}/{plot_case}"
        create_dir(dump_path)
        print_keys = extract_first_middle_last(keys)

        for size_key in print_keys:
            values = [r[size_key] for r in src_rank_bandwidth_results]
            plt.figure(figsize=(10, 4))
            bars = plt.bar(range(len(src_ranks)), values)
            plt.xlabel(f"RankPair (rank-i <-> rank-i+{LOCAL_WORLD_SIZE})")
            plt.ylabel("Bandwidth")
            plt.title(f"Inter Node {case_name} Bandwidth for {size_key}")
            xtick_labels = [f"r-{src_ranks[i]}/{peer_ranks[i]}" for i in range(len(src_ranks))]
            plt.xticks(range(len(src_ranks)), xtick_labels)
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
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom")

        png_file = f"inter_node_{case_name}_bandwidth_rank_0.png"
        plt.tight_layout()
        plt.savefig(f"{dump_path}/{png_file}")
        plt.close()
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write(f"![{plot_case}](./{plot_case}/{png_file})\n")
            f.write(f"\n")
        log(f"")
