###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

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


def _format_inter_comm_chunk(
    args,
    comm: str,
    case_name: str,
    all_latency_results: List,
    all_bandwidth_results: List,
    all_group_ranks: List[List[int]],
    group_node_counts: List[int],
) -> str:
    """Build the markdown chunk for one (comm, adjacent_nodes) case (rank 0 only).

    Streaming console ``log(...)`` calls happen here in execution order. Plot
    PNGs are written eagerly via ``plt.savefig``. The returned string is the
    buffered markdown text (headers + tables + plot refs); it is emitted later
    in comm-major order by ``run_inter_node_comm``'s post-loop pass so the
    report layout matches the pre-refactor structure.
    """
    keys = sorted(
        list({k for r in all_bandwidth_results for k in (r or {}).keys()}), key=extract_number
    )
    hostnames = get_hostnames()

    # Show only the leader node's hostname; the Node range plus the legend at
    # the top of the report cover the rest.
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

    parts: List[str] = []

    parts.append(f"=======InterNodeComm - {case_name} (us)=======\n")
    log(f"=======InterNodeComm - {case_name} (us)=======")
    log(header_line)
    parts.append(f"| {host_col_label} | Node | Rank | {' | '.join(keys)}|\n")
    parts.append(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
    for group_ranks in all_group_ranks:
        host_str, node_str, rank_str, r = _row_for(group_ranks, all_latency_results)
        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
        log(
            f"{host_str:<{host_col_w}} {node_str:<10} {rank_str:<10} "
            f"{' '.join(formatted_values)}"
        )
        parts.append(f"| {host_str} | {node_str} | {rank_str} | {' | '.join(formatted_values)}|\n")
    parts.append("\n")

    parts.append(f"=======InterNodeComm - {case_name} (GB/s)=======\n")
    log(f"=======InterNodeComm - {case_name} (GB/s)=======")
    log(header_line)
    parts.append(f"| {host_col_label} | Node | Rank | {' | '.join(keys)}|\n")
    parts.append(f"|----------|----------|----------{'|----------' * len(keys)}|\n")
    for group_ranks in all_group_ranks:
        host_str, node_str, rank_str, r = _row_for(group_ranks, all_bandwidth_results)
        formatted_values = [f"{r.get(key, 0):<6.2f}" for key in keys]
        log(
            f"{host_str:<{host_col_w}} {node_str:<10} {rank_str:<10} "
            f"{' '.join(formatted_values)}"
        )
        parts.append(f"| {host_str} | {node_str} | {rank_str} | {' | '.join(formatted_values)}|\n")
    parts.append("\n")

    if not args.plot:
        return "".join(parts)

    import matplotlib.pyplot as plt

    log(f"=======Plot InterNode {case_name} Bandwidth=======")
    parts.append(f"=======Plot InterNode {case_name} Bandwidth=======\n")
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
        plt.xlabel("Group (starting rank)")
        plt.ylabel("Bandwidth")
        plt.title(f"Inter Node {case_name} Bandwidth for {size_key}")
        xtick_labels = [
            f"{leader_ranks[i]} ({group_node_counts[i]}N)" for i in range(num_print_ranks)
        ]
        plt.xticks(range(num_print_ranks), xtick_labels)
        plt.grid(True, axis="y")
        roofline_bandwidth = args.ib_bw
        plt.axhline(
            y=roofline_bandwidth,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"IB Unidirectional BW Roofline: {roofline_bandwidth} GB/s",
        )
        plt.legend()
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
        parts.append(f"![{plot_case}](./{plot_case}/{png_file})\n")

    # Bar chart visualization for rank 0
    rank_0_values = [all_bandwidth_results[0][size_key] for size_key in keys]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(keys, rank_0_values)
    plt.xlabel("Size")
    plt.ylabel("Bandwidth")
    plt.title(f"Inter Node {case_name} Bandwidth for Rank 0")
    plt.grid(True, axis="y")
    roofline_bandwidth = args.ib_bw
    plt.axhline(
        y=roofline_bandwidth,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"IB Unidirectional BW Roofline: {roofline_bandwidth} GB/s",
    )
    plt.legend()
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, height, f"{height:.2f}", ha="center", va="bottom"
        )

    png_file = f"inter_node_{case_name}_bandwidth_rank_0.png"
    plt.tight_layout()
    plt.savefig(f"{dump_path}/{png_file}")
    plt.close()
    parts.append(f"![{plot_case}](./{plot_case}/{png_file})\n")
    parts.append("\n")
    log("")

    return "".join(parts)


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

    Loop structure (build once, benchmark in parallel, destroy once):
        The outer loop iterates over `adjacent_nodes`. For each value, we build
        all subgroups once (every rank ends up owning at most one), then run
        each enabled comm's benchmark on the rank's owned subgroup, then
        destroy. This is the same parallelism as the pre-refactor code; the
        only structural change is that allreduce and alltoall now share the
        same subgroup instead of each creating a fresh one (Win 1), and the
        all-nodes case reuses ``dist.group.WORLD`` directly instead of
        constructing a duplicate 1024-rank communicator (Win 2). The latter
        is the change that eliminates the EADDRINUSE failure at 128N: it
        removes the consecutive 1024-rank ``ncclCommSplit`` operations whose
        destroy-side TIME_WAIT churn exhausts the ephemeral-port pool.

    Destroy phase is rank-uniform:
        Every rank calls the same sequence of WORLD-level collectives
        (``dist.barrier`` + ``barrier_after_comm_destroy``). Only the local
        ``destroy_process_group`` is gated on per-rank state. The cleanup
        sleep is suppressed (delay=0) on the WORLD path where nothing was
        destroyed.

    Report layout:
        ``## InterNode - allreduce`` / ``## InterNode - alltoall`` and their
        tables are preserved exactly. Per-(comm, adjacent_nodes) markdown
        chunks are buffered on rank 0 during the benchmark phase and emitted
        in comm-major order after the outer loop completes. Plot PNG files
        are written eagerly during the benchmark phase; only the
        ``![...](...)`` lines that reference them are buffered.
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

    # `cases` is a dict {comm: list(node_counts)} preserved for backward
    # compatibility with the post-loop emit pass (which iterates `cases` in
    # the historical order: allreduce first, then alltoall).
    cases = {comm: list(node_counts) for comm in ("allreduce", "alltoall") if comm in enabled_set}

    warmup = get_warmup()
    iteration = get_iteration()

    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            f.write("# InterNode Comm\n")

    # Per-(comm, adjacent_nodes) markdown chunks, rank 0 only. Buffered here
    # and emitted in comm-major order after the outer loop so the report
    # structure matches the pre-refactor layout.
    buffered_chunks: Dict[Tuple[str, int], str] = {}

    for adjacent_nodes in node_counts:
        num_full_groups = num_nodes // adjacent_nodes
        remainder_nodes = num_nodes % adjacent_nodes

        # ------------------------------------------------------------------
        # Build phase (executed once per adjacent_nodes).
        # ------------------------------------------------------------------
        adjacent_group = None
        all_group_ranks: List[List[int]] = []
        group_node_counts: List[int] = []

        # Win 2: when the requested subgroup is exactly WORLD (one full group
        # spanning every rank, no remainder), reuse `dist.group.WORLD`
        # instead of constructing a fresh duplicate. This avoids the
        # ncclCommSplit + destroy cycle whose IB OOB TIME_WAIT churn drives
        # the 128N EADDRINUSE failure.
        if adjacent_nodes == num_nodes and num_full_groups == 1 and remainder_nodes < 2:
            adjacent_group = dist.group.WORLD
            all_group_ranks.append(list(range(WORLD_SIZE)))
            group_node_counts.append(adjacent_nodes)
        else:
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

        # Today's invariant: any rank whose global index falls inside the
        # covered range must own a subgroup. For the WORLD path the covered
        # range is the entire world.
        if adjacent_group is dist.group.WORLD:
            total_grouped_ranks = WORLD_SIZE
        else:
            total_grouped_ranks = num_full_groups * adjacent_nodes * LOCAL_WORLD_SIZE
            if remainder_nodes >= 2:
                total_grouped_ranks += remainder_nodes * LOCAL_WORLD_SIZE
        if RANK < total_grouped_ranks:
            assert adjacent_group is not None

        # ------------------------------------------------------------------
        # Benchmark phase: run each enabled comm on the same owned group
        # (Win 1). The outer loop's `comm` order is the dict's insertion
        # order; it doesn't affect the final report ordering because the
        # post-loop emit pass re-iterates `cases` directly.
        # ------------------------------------------------------------------
        for comm in cases:
            case_name = f"{comm}-{adjacent_nodes}nodes"
            latency_results: Dict[str, float] = {}
            bandwidth_results: Dict[str, float] = {}

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

            # Gather is a WORLD collective: every rank participates, even
            # ranks with adjacent_group is None (e.g. uneven node counts).
            all_latency_results: List = [None for _ in range(WORLD_SIZE)]
            all_bandwidth_results: List = [None for _ in range(WORLD_SIZE)]
            dist.gather_object(latency_results, all_latency_results if RANK == 0 else None, dst=0)
            dist.gather_object(bandwidth_results, all_bandwidth_results if RANK == 0 else None, dst=0)

            if RANK == 0:
                buffered_chunks[(comm, adjacent_nodes)] = _format_inter_comm_chunk(
                    args=args,
                    comm=comm,
                    case_name=case_name,
                    all_latency_results=all_latency_results,
                    all_bandwidth_results=all_bandwidth_results,
                    all_group_ranks=all_group_ranks,
                    group_node_counts=group_node_counts,
                )

        # ------------------------------------------------------------------
        # Destroy phase: rank-uniform sequence. The branch condition on the
        # cleanup-sleep duration is rank-uniform (`adjacent_nodes ==
        # num_nodes` is computed identically on every rank); the
        # `destroy_process_group` itself is rank-local and may safely be
        # gated on `adjacent_group is None` / WORLD.
        # ------------------------------------------------------------------
        dist.barrier(device_ids=[torch.cuda.current_device()])
        if adjacent_group is not None and adjacent_group is not dist.group.WORLD:
            dist.destroy_process_group(adjacent_group)
        # WORLD path: nothing was destroyed by anyone, no ephemeral-port drain
        # needed. Non-WORLD path: at least some ranks destroyed; wait the
        # configured cleanup window so the next outer iteration starts clean.
        delay = args.comm_cleanup_delay_sec if adjacent_nodes != num_nodes else 0.0
        barrier_after_comm_destroy(delay)

    # ----------------------------------------------------------------------
    # Post-loop emit: comm-major ordering, identical to the pre-refactor
    # report. Subsection headers (## InterNode - {comm}) are written exactly
    # once per comm here, not per-chunk.
    # ----------------------------------------------------------------------
    if RANK == 0:
        with open(args.markdown_file, "a", encoding="utf-8") as f:
            for comm in cases:
                f.write(f"## InterNode - {comm}\n")
                for adjacent_nodes in node_counts:
                    chunk = buffered_chunks.get((comm, adjacent_nodes))
                    if chunk:
                        f.write(chunk)
