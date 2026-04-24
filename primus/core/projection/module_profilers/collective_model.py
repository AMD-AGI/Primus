###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from math import ceil

import numpy as np

# ---------------------------
# Utility Functions
# ---------------------------


def get_effective_node_bw(args, group_size=None, a2a=False):
    """
    Get effective intra-node bandwidth, applying mesh topology derate.

    Note: bw_eff is already applied to args.node_bw at initialization.

    Mesh topology derate: When a communicating group uses fewer GPUs than node_size
    on mesh topology, the effective bandwidth is reduced because only (g-1) links
    are used out of (node_size-1) available.

    Formula: effective_node_bw = node_bw × (g - 1) / (node_size - 1)
    Only applies when: g > 1 AND g < node_size AND node_topology == "mesh"

    A2A contention derate: Full-mesh AllToAll saturates all xGMI links
    simultaneously. Applied only when a2a=True.

    Args:
        args: CollectiveArgs instance (node_bw already has bw_eff applied)
        group_size: Optional override for the communicating group size.
                    Defaults to args.hp (TP size) for AllReduce-style collectives.
                    AllToAll should pass the actual EP/A2A group size.
        a2a: If True, apply A2A mesh contention derate for full-mesh patterns.

    Returns:
        Effective node bandwidth in GB/s (after mesh derate and optional contention)
    """
    g = group_size if group_size is not None else args.hp
    if args.node_topology == "mesh" and g > 1 and g < args.node_size:
        derate_factor = (g - 1) / (args.node_size - 1)
        bw = args.node_bw * derate_factor
    else:
        bw = args.node_bw

    if a2a and args.node_topology == "mesh" and g > 1 and args.node_size > 2:
        contention = getattr(args, "a2a_mesh_contention", 0.0)
        if contention > 0:
            link_saturation = (g - 1) / (args.node_size - 1)
            bw *= 1 - contention * link_saturation**2

    return bw


def get_bandwidth_and_latency(args, domain_size):
    """
    Determine bandwidth and latency for a given communication domain size.
    Selects node, pod, or cluster bandwidth/latency depending on the size.
    """
    if domain_size <= args.node_size:
        # Communication fits within a node
        bw = get_effective_node_bw(args)
        lat = args.node_lat
    elif domain_size <= args.pod_size:
        # Communication fits within a pod (multiple nodes)
        bw = args.pod_bw
        lat = args.pod_lat
    else:
        # Communication spans the cluster (multiple pods)
        bw = args.cluster_bw
        lat = args.cluster_lat
    return bw, lat


def node_latency_and_volume_protocol(args, msg_size, protocol):
    """
    Calculate node latency and message size for a given protocol.
    Protocols affect packetization and latency.
    """
    if protocol == "simple":
        # Simple protocol: one packet, add header
        node_lat = args.write_latency + args.write_resp + args.write_latency
        msg_size = msg_size + 8
    elif protocol == "ll":
        # Low-latency protocol: 4-byte packets
        node_lat = args.write_latency
        msg_size = ceil(msg_size / 4) * 8
    elif protocol == "ll64":
        # 64-byte packets
        node_lat = args.write_latency
        msg_size = ceil(msg_size / 56) * 64
    elif protocol == "ll128":
        # 128-byte packets
        node_lat = args.write_latency
        msg_size = ceil(msg_size / 120) * 128
    else:
        raise ValueError(f"Unknown Protocol {protocol}")
    node_lat += args.hbm_latency  # Add HBM latency
    return node_lat, msg_size


def pod_latency_and_volume_protocol(args, msg_size, protocol):
    """
    Calculate pod latency and message size for a given protocol.
    Similar to node_latency_and_volume_protocol but for pod-level.
    """
    if protocol == "simple":
        pod_lat = args.pod_lat * 3
        msg_size = max(8, msg_size + 8)
    elif protocol == "ll":
        pod_lat = args.pod_lat
        msg_size = ceil(msg_size / 4) * 8
    elif protocol == "ll64":
        pod_lat = args.pod_lat
        msg_size = ceil(msg_size / 56) * 64
    elif protocol == "ll128":
        pod_lat = args.pod_lat
        msg_size = ceil(msg_size / 120) * 128
    else:
        raise ValueError(f"Unknown Protocol {protocol}")
    pod_lat += args.hbm_latency
    return pod_lat, msg_size


def get_max_fanout(args):
    """
    Return intra-node and inter-node fanout.
    Used for single-shot collectives to determine parallelism.
    """
    intra_node_fan_out = args.node_size - 1
    inter_node_fan_out = args.pod_size - 1
    return intra_node_fan_out, inter_node_fan_out


# ---------------------------
# Collective Algorithms
# ---------------------------


def sendrecv(args, msg_size):
    """
    Point-to-point send/recv latency calculation.
    Used for basic communication between two GPUs (e.g. pipeline parallelism).

    Determines intra-node vs inter-node based on PP stage placement:
    - If each PP stage fills >= 1 full node, adjacent stages are on different
      nodes → inter-node P2P using NIC bandwidth.
    - Otherwise, P2P is intra-node using a single xGMI link with p2p_bw_eff.
    """
    pp = getattr(args, "pp", 1)
    num_nodes = getattr(args, "num_nodes", 1)
    total_gpus = num_nodes * args.node_size
    gpus_per_pp_stage = total_gpus // max(pp, 1)

    pp_is_inter_node = (pp > 1) and (num_nodes > 1) and (gpus_per_pp_stage >= args.node_size)

    if pp_is_inter_node:
        raw_pod = getattr(args, "_raw_pod_bw", args.pod_bw / args.bw_eff)
        p2p_eff = getattr(args, "p2p_bw_eff", 0.80)
        bw = raw_pod * p2p_eff
        lat = args.pod_lat
    else:
        domain = args.hp * args.cp * args.ep
        if domain <= args.node_size:
            raw_bw = getattr(args, "_raw_node_bw", args.node_bw / args.bw_eff)
            p2p_eff = getattr(args, "p2p_bw_eff", 0.80)
            p2p_node_bw = raw_bw * p2p_eff
            if args.node_topology == "mesh" and args.node_size > 2:
                p2p_node_bw *= 1 / (args.node_size - 1)
            bw = p2p_node_bw
            lat = args.node_lat
        elif domain <= args.pod_size:
            bw = args.pod_bw
            lat = args.pod_lat
        else:
            bw = args.cluster_bw
            lat = args.cluster_lat
    t = (msg_size / bw) * 1.0e-3 + lat + args.kernel_launch_latency
    return t


def direct_alltoall(args, msg_size, gpus, groups=["ep"], protocol=None, original_msg_size=None):
    """
    Direct alltoall for HP=1, hierarchical with parallel NIC utilization.

    In all-to-all:
    - Total data = msg_size (scaled by (gpus-1)/gpus before this function)
    - This volume is what each GPU sends to all its peers

    For inter-node with switch topology:
    - All NICs are used in parallel for inter-node traffic
    - Per-peer latency overhead accounts for QP setup, work request posting, etc.
    """
    assert args.hp == 1
    assert (args.hp * gpus > args.node_size) and (args.hp * gpus) <= args.pod_size

    gpus_per_node = args.node_size
    num_nodes = int(np.ceil(gpus / gpus_per_node))
    nics_per_node = args.nics_per_node if args.nics_per_node else gpus_per_node

    # Calculate number of inter-node peers (GPUs on remote nodes)
    inter_node_peers = gpus - gpus_per_node

    # Split volume between intra-node and inter-node
    intra_fraction = (gpus_per_node - 1) / (gpus - 1)
    inter_fraction = 1 - intra_fraction

    intra_node_volume = msg_size * intra_fraction
    inter_node_volume_per_gpu = msg_size * inter_fraction

    node_lat, intra_vol_adj = node_latency_and_volume_protocol(args, intra_node_volume, protocol)
    pod_lat = args.pod_lat

    # Intra-node time (with A2A contention derate)
    t_intra = (
        intra_vol_adj / get_effective_node_bw(args, group_size=gpus_per_node, a2a=True) * 1.0e-3 + node_lat
    )

    # Inter-node time with all NICs — A2A uses P2P-like NIC streams
    # plus a multi-destination contention derate (NIC QP multiplexing cost grows with node count).
    raw_pod = getattr(args, "_raw_pod_bw", args.pod_bw / args.bw_eff)
    p2p_eff = getattr(args, "p2p_bw_eff", 0.80)
    remote_contention = getattr(args, "a2a_remote_contention", 0.0)
    contention_factor = max(0.0, 1.0 - remote_contention * (num_nodes - 1))
    effective_pod_bw = raw_pod * p2p_eff * contention_factor
    if args.switch_topology:
        total_inter_volume = inter_node_volume_per_gpu * gpus_per_node
        aggregate_inter_bw = effective_pod_bw * nics_per_node
        t_inter = total_inter_volume / aggregate_inter_bw * 1.0e-3 + pod_lat
    else:
        remote_nodes = num_nodes - 1
        t_inter = inter_node_volume_per_gpu / effective_pod_bw * 1.0e-3 + pod_lat * remote_nodes

    # Overlap intra and inter
    t_a2a = max(t_intra, t_inter)

    # Add synchronization overhead for multi-node all-to-all
    # This accounts for barrier synchronization and RCCL setup
    sync_overhead = (num_nodes - 1) * args.pod_lat * 0.5
    t_a2a += sync_overhead

    # Add per-peer latency overhead for inter-node communication
    # This accounts for RDMA QP setup, work request posting, completion polling, etc.
    if hasattr(args, "a2a_peer_lat") and args.a2a_peer_lat > 0:
        peer_overhead = args.a2a_peer_lat * inter_node_peers
        t_a2a += peer_overhead

    t_a2a += args.kernel_launch_latency

    return t_a2a


def run_alltoall(args, msg_size, gpus, groups=["ep"], protocol=None):
    """
    Run alltoall collective.
    Chooses between node, pod, or cluster domain based on GPU count.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    # Save original msg_size for NIC estimation
    original_msg_size = msg_size
    # Scale message size for alltoall
    msg_size = int(msg_size * (gpus - 1) / gpus)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    # tensor parallelism groups will require alltoall across hp dimension
    if (args.hp * gpus) <= args.node_size:
        # Alltoall fits within node — derate based on actual A2A group size
        bw = get_effective_node_bw(args, group_size=gpus, a2a=True)
        lat = node_lat
    elif (args.hp * gpus > args.node_size) and (args.hp * gpus) <= args.pod_size:
        # Alltoall fits within pod
        if args.hp == 1:
            return direct_alltoall(args, msg_size, gpus, groups, protocol, original_msg_size)
        bw = args.pod_bw
        lat = args.pod_lat
    else:
        # Alltoall spans cluster
        bw = args.cluster_bw
        lat = args.cluster_lat
    t = msg_size / bw * 1.0e-3 + lat + args.kernel_launch_latency
    return t


def cp_allgather(args, msg_size, gpus, protocol=None):
    """
    Allgather for CP domain.
    Used when communication is across CP (cross-pod) group.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    cpxhp = args.cp * args.hp
    if cpxhp > args.node_size and cpxhp <= args.pod_size:
        # CP domain fits within pod
        bw = args.pod_bw
        lat = pod_lat
    elif cpxhp <= args.node_size:
        # CP domain fits within node
        bw = get_effective_node_bw(args)
        lat = node_lat
    else:
        # CP domain spans cluster
        bw = args.cluster_bw
        lat = args.cluster_lat
    # Logarithmic steps for tree allgather
    t = msg_size / bw * 1.0e-3 + lat * np.ceil(np.log2(gpus)) + args.kernel_launch_latency
    return t


def run_allgather(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Run allgather collective.
    Handles node and pod domains, and CP group special case.
    When spanning multiple nodes, uses overlap model instead of additive timing.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    if "cp" in groups:
        return cp_allgather(args, msg_size, gpus, protocol)
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    lat = 0
    if gpus > args.node_size:
        base_overlap = getattr(args, "ar_overlap_factor", 0.9)
        warmup_bytes = getattr(args, "ar_warmup_chunk_bytes", 32 * 1024 * 1024)
        chunk_per_gpu = (msg_size * gpus / (gpus - 1)) / args.node_size
        warmup_ratio = min(1.0, chunk_per_gpu / warmup_bytes) if warmup_bytes > 0 else 1.0
        effective_overlap = base_overlap * warmup_ratio
        bw = get_effective_node_bw(args)
        node_msg_volume = msg_size * (args.node_size - 1) / args.node_size
        t_intra = node_msg_volume / bw * 1.0e-3 + node_lat * np.ceil(np.log2(args.node_size))
        bw = args.pod_bw
        pod_msg_volume = msg_size - node_msg_volume
        t_inter = pod_msg_volume / bw * 1.0e-3 + pod_lat * np.ceil(np.log2(gpus / args.node_size))
        t = max(t_intra, t_inter) + (1 - effective_overlap) * min(t_intra, t_inter)
    else:
        bw = get_effective_node_bw(args)
        t = msg_size / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(gpus))
        t += lat
    t += args.kernel_launch_latency
    return t


def run_reduce_scatter(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Run reduce_scatter collective.
    Handles node and pod domains, includes compute time for reduction.
    When spanning multiple nodes, uses overlap model instead of additive timing.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    t = 0
    lat = 0
    if gpus > args.node_size:
        base_overlap = getattr(args, "ar_overlap_factor", 0.9)
        warmup_bytes = getattr(args, "ar_warmup_chunk_bytes", 32 * 1024 * 1024)
        chunk_per_gpu = (msg_size * gpus / (gpus - 1)) / args.node_size
        warmup_ratio = min(1.0, chunk_per_gpu / warmup_bytes) if warmup_bytes > 0 else 1.0
        effective_overlap = base_overlap * warmup_ratio
        bw = get_effective_node_bw(args)
        node_msg_volume = msg_size * (args.node_size - 1) / args.node_size
        t_intra = node_msg_volume / bw * 1.0e-3 + node_lat * np.ceil(np.log2(args.node_size))
        bw = args.pod_bw
        pod_msg_volume = msg_size - node_msg_volume
        pod_lat = args.pod_lat
        t_inter = pod_msg_volume / bw * 1.0e-3 + pod_lat * np.ceil(np.log2(gpus / args.node_size))
        t = max(t_intra, t_inter) + (1 - effective_overlap) * min(t_intra, t_inter)
    else:
        bw = get_effective_node_bw(args)
        t = msg_size / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(gpus))
        t += lat
    tensor_elems = msg_size / 2
    t += tensor_elems / (args.vector_flops) * 1.0e6
    t += args.kernel_launch_latency
    return t


def RingAllreduce(args, msg_size, gpus, groups=["dp"], protocol=None):
    """
    Ring Allreduce algorithm.
    Communication is performed in a ring, with two passes.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    if gpus <= args.node_size:
        # Ring fits within node
        t += node_lat * (gpus - 1)
        t += msg_size / get_effective_node_bw(args) * 1.0e-3
    elif gpus <= args.pod_size:
        # Ring fits within pod
        t += pod_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.pod_bw)
        t += msg_size / bw * 1.0e-3
    else:
        # Ring spans cluster
        t += args.cluster_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.cluster_bw)
        t += msg_size / bw * 1.0e-3
    t = 2 * t  # Two passes in ring
    tensor_elems = msg_size * gpus / 2
    t += tensor_elems / (args.vector_flops) * 1.0e6
    t += args.kernel_launch_latency
    return t


def RingAllgather(args, msg_size, gpus, groups=["dp"], protocol=None):
    """
    Ring Allgather algorithm.
    Communication is performed in a ring, single pass.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    if gpus <= args.node_size:
        t += node_lat * (gpus - 1)
        t += msg_size / get_effective_node_bw(args) * 1.0e-3
    elif gpus <= args.pod_size:
        t += pod_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.pod_bw)
        t += msg_size / bw * 1.0e-3
    else:
        t += args.cluster_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.cluster_bw)
        t += msg_size / bw * 1.0e-3
    t += args.kernel_launch_latency
    return t


def RingRS(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Ring ReduceScatter algorithm.
    Communication is performed in a ring, single pass, includes compute.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    if gpus <= args.node_size:
        t += node_lat * (gpus - 1)
        t += msg_size / get_effective_node_bw(args) * 1.0e-3
    elif gpus <= args.pod_size:
        t += pod_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.pod_bw)
        t += msg_size / bw * 1.0e-3
    else:
        t += args.cluster_lat * (gpus - 1)
        bw = min(args.node_bw, args.node_size * args.cluster_bw)
        t += msg_size / bw * 1.0e-3
    tensor_elems = msg_size * gpus / 2
    t += tensor_elems / (args.vector_flops) * 1.0e6
    t += args.kernel_launch_latency
    return t


def oneshotHCallreduce(args, msg_size, gpus, groups=["dp"], protocol=None):
    """
    One-shot Hypercube Allreduce algorithm.
    Uses log2 steps for communication, includes compute.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    lat = 0
    if gpus > args.node_size:
        # Communication spans node and pod
        bw = get_effective_node_bw(args)
        node_msg_volume = msg_size * (args.node_size - 1)
        t = node_msg_volume / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(args.node_size))
        bw = args.pod_bw
        pod_msg_volume = msg_size * np.ceil(np.log2((gpus - args.node_size)))
        t += pod_msg_volume / bw * 1.0e-3
        lat += pod_lat * np.ceil(np.log2(gpus / args.node_size))
    else:
        # Allreduce fits within node
        bw = get_effective_node_bw(args)
        node_msg_volume = msg_size * (gpus - 1)
        t = node_msg_volume / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(gpus))
    t += lat
    tensor_elems = msg_size * gpus / 2
    t += tensor_elems / (args.vector_flops) * 1.0e6
    t += args.kernel_launch_latency
    return t


# ---------------------------
# Single-Shot Collectives
# ---------------------------


def single_shot_alltoall(args, msg_size, gpus, groups=None, protocol=None):
    """
    Single shot alltoall with max fanout and overlap.
    Uses parallel communication rounds.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    intra_node_fanout, inter_node_fanout = get_max_fanout(args)
    msg_size_per_peer = ceil(msg_size / gpus)
    # Account for TP striding: with TP (hp) > 1, each EP rank occupies
    # hp GPUs, so only node_size/hp EP ranks fit on a single node.
    hp = getattr(args, "hp", 1)
    gpus_per_node = min(gpus, args.node_size // max(hp, 1))
    nics_per_node = args.nics_per_node if args.nics_per_node else gpus_per_node
    intra_node_gpus = gpus_per_node - 1
    inter_node_gpus = max(0, gpus - gpus_per_node)

    t_intra_node = 0
    t_inter_node = 0
    if intra_node_gpus > 0:
        node_lat, msg_size_per_peer_adj = node_latency_and_volume_protocol(args, msg_size_per_peer, protocol)
        node_bw = get_effective_node_bw(args, group_size=gpus_per_node, a2a=True)
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_adj * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        inter_node_msg_size_per_gpu = msg_size_per_peer * inter_node_gpus
        raw_pod = getattr(args, "_raw_pod_bw", args.pod_bw / args.bw_eff)
        p2p_eff = getattr(args, "p2p_bw_eff", 0.80)
        num_nodes_a2a = int(np.ceil(gpus / args.node_size))
        remote_contention = getattr(args, "a2a_remote_contention", 0.0)
        contention_factor = max(0.0, 1.0 - remote_contention * (num_nodes_a2a - 1))
        eff_pod_bw = raw_pod * p2p_eff * contention_factor
        if args.switch_topology:
            total_inter_volume = inter_node_msg_size_per_gpu * gpus_per_node
            aggregate_bw = eff_pod_bw * nics_per_node
            inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
            t_inter_node = inter_node_rounds * pod_lat
            t_inter_node += total_inter_volume / aggregate_bw * 1.0e-3
        else:
            inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
            t_inter_node = inter_node_rounds * pod_lat
            t_inter_node += inter_node_msg_size_per_gpu / eff_pod_bw * 1.0e-3
    t_a2a = max(t_intra_node, t_inter_node)
    t_a2a += args.kernel_launch_latency
    return t_a2a


def hierarchical_alltoall(args, msg_size, gpus, groups=None, protocol=None):
    """
    Hierarchical alltoall with parallel NIC utilization.

    For inter-node traffic with switch topology:
    - All NICs are used in parallel
    """
    if gpus == 1 or msg_size == 0:
        return 0

    # Account for TP striding: with TP (hp) > 1, each EP rank occupies
    # hp GPUs, so only node_size/hp EP ranks fit on a single node.
    hp = getattr(args, "hp", 1)
    gpus_per_node = min(gpus, args.node_size // max(hp, 1))
    num_nodes = ceil(gpus / max(gpus_per_node, 1))
    nics_per_node = args.nics_per_node if args.nics_per_node else gpus_per_node

    if num_nodes == 1:
        return single_shot_alltoall(args, msg_size, gpus, groups, protocol)

    # Volume breakdown per GPU
    intra_node_volume = msg_size * (gpus_per_node - 1) / gpus
    inter_node_volume_per_gpu = msg_size * (gpus - gpus_per_node) / gpus

    # Intra-node time — derate based on intra-node A2A group size + contention
    node_lat, intra_vol_adj = node_latency_and_volume_protocol(args, intra_node_volume, protocol)
    node_bw = get_effective_node_bw(args, group_size=gpus_per_node, a2a=True)
    t_intra = node_lat + intra_vol_adj / node_bw * 1.0e-3

    # Inter-node time — A2A uses P2P-like NIC streams with multi-dest contention
    raw_pod = getattr(args, "_raw_pod_bw", args.pod_bw / args.bw_eff)
    p2p_eff = getattr(args, "p2p_bw_eff", 0.80)
    remote_contention = getattr(args, "a2a_remote_contention", 0.0)
    contention_factor = max(0.0, 1.0 - remote_contention * (num_nodes - 1))
    eff_pod_bw = raw_pod * p2p_eff * contention_factor
    if args.switch_topology:
        total_inter_volume = inter_node_volume_per_gpu * gpus_per_node
        aggregate_inter_bw = eff_pod_bw * nics_per_node
        t_inter = args.pod_lat + total_inter_volume / aggregate_inter_bw * 1.0e-3
    else:
        t_inter = args.pod_lat * num_nodes + inter_node_volume_per_gpu / eff_pod_bw * 1.0e-3

    t_total = max(t_intra, t_inter)
    t_total += args.kernel_launch_latency

    return t_total


def single_shot_allgather(args, msg_size, gpus, groups=None, protocol=None):
    """
    Single shot allgather with max fanout and overlap.
    Uses parallel communication rounds.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    intra_node_fanout, inter_node_fanout = get_max_fanout(args)
    msg_size_per_peer = ceil(msg_size / gpus)
    gpus_per_node = min(gpus, args.node_size)
    intra_node_gpus = gpus_per_node - 1
    inter_node_gpus = max(0, gpus - gpus_per_node)
    t_intra_node = 0
    t_inter_node = 0
    if intra_node_gpus > 0:
        node_lat, msg_size_per_peer_node = node_latency_and_volume_protocol(args, msg_size_per_peer, protocol)
        node_bw = get_effective_node_bw(args)
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_node * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
        t_inter_node = inter_node_rounds * pod_lat
        inter_node_msg_size = msg_size_per_peer * inter_node_gpus
        t_inter_node += inter_node_msg_size / args.pod_bw * 1.0e-3
    t_ag = max(t_intra_node, t_inter_node)
    t_ag += args.kernel_launch_latency
    return t_ag


def single_shot_reduce_scatter(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Single shot reduce scatter with max fanout and overlap.
    Includes compute time for reduction.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    intra_node_fanout, inter_node_fanout = get_max_fanout(args)
    msg_size_per_peer = ceil(msg_size / gpus)
    gpus_per_node = min(gpus, args.node_size)
    intra_node_gpus = gpus_per_node - 1
    inter_node_gpus = max(0, gpus - gpus_per_node)
    t_intra_node = 0
    t_inter_node = 0
    if intra_node_gpus > 0:
        node_lat, msg_size_per_peer_node = node_latency_and_volume_protocol(args, msg_size_per_peer, protocol)
        node_bw = get_effective_node_bw(args)
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_node * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
        t_inter_node = inter_node_rounds * pod_lat
        inter_node_msg_size = msg_size_per_peer * inter_node_gpus
        t_inter_node += inter_node_msg_size / args.pod_bw * 1.0e-3
    t_rs = max(t_intra_node, t_inter_node)
    # Add compute time for reduction
    tensor_elems = np.ceil((msg_size_per_peer * (gpus - 1)) / 2)
    t_rs += tensor_elems / (args.vector_flops) * 1.0e6
    t_rs += args.kernel_launch_latency
    return t_rs


def single_shot_allreduce(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Single shot allreduce = reduce scatter + allgather.
    Combines single shot reduce scatter and allgather.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    t_rs = single_shot_reduce_scatter(args, msg_size, gpus, groups, protocol)
    t_ag = single_shot_allgather(args, msg_size, gpus, groups, protocol)
    t_ar = t_rs + t_ag - args.kernel_launch_latency  # Remove duplicate kernel launch latency
    return t_ar


# ---------------------------
# Hierarchical AllReduce
# ---------------------------


def hierarchical_allreduce(args, msg_size, gpus, groups=["dp"], protocol=None):
    """
    Hierarchical AllReduce: intra-RS → inter-AR → intra-AG with pipelining.

    Models the 3-phase approach used by RCCL for multi-node AllReduce:
      Phase 1: Intra-node reduce-scatter (each node reduces locally)
      Phase 2: Inter-node allreduce (rank-0s allreduce across nodes via NICs)
      Phase 3: Intra-node allgather (broadcast result within each node)

    Phases 1+3 use intra-node links; Phase 2 uses inter-node links.
    With pipelining, intra and inter phases overlap:
      t = max(t_intra, t_inter) + (1 - overlap_factor) * min(t_intra, t_inter)
    """
    if gpus == 1 or msg_size == 0:
        return 0
    if gpus <= args.node_size:
        return (
            run_reduce_scatter(args, msg_size, gpus, groups, protocol)
            + run_allgather(args, msg_size, gpus, groups, protocol)
            - args.kernel_launch_latency
        )

    node_size = args.node_size
    num_nodes = int(np.ceil(gpus / node_size))
    nics_per_node = args.nics_per_node if args.nics_per_node else node_size
    base_overlap = getattr(args, "ar_overlap_factor", 0.9)
    warmup_bytes = getattr(args, "ar_warmup_chunk_bytes", 32 * 1024 * 1024)

    # --- Phase 1: Intra-node reduce-scatter ---
    rs_volume = msg_size * (node_size - 1) / node_size
    node_lat_rs, rs_vol_adj = node_latency_and_volume_protocol(args, rs_volume, protocol)
    node_bw = get_effective_node_bw(args)
    t_intra_rs = rs_vol_adj / node_bw * 1.0e-3 + node_lat_rs * np.ceil(np.log2(node_size))
    rs_compute = (rs_vol_adj / 2) / args.vector_flops * 1.0e6
    t_intra_rs += rs_compute

    # --- Phase 2: Inter-node allreduce of the reduced chunk ---
    chunk_size = msg_size / node_size
    inter_ar_volume = chunk_size * (num_nodes - 1) / num_nodes
    if args.switch_topology:
        inter_bw = args.pod_bw * nics_per_node
    else:
        inter_bw = args.pod_bw
    pod_lat = args.pod_lat
    t_inter_ar = 2.0 * (inter_ar_volume / inter_bw * 1.0e-3 + pod_lat * np.ceil(np.log2(num_nodes)))
    inter_compute = (inter_ar_volume / 2) / args.vector_flops * 1.0e6
    t_inter_ar += inter_compute

    # --- Phase 3: Intra-node allgather ---
    ag_volume = msg_size * (node_size - 1) / node_size
    node_lat_ag, ag_vol_adj = node_latency_and_volume_protocol(args, ag_volume, protocol)
    t_intra_ag = ag_vol_adj / node_bw * 1.0e-3 + node_lat_ag * np.ceil(np.log2(node_size))

    # --- Size-dependent pipelined overlap ---
    # RCCL's pipeline efficiency depends on per-GPU chunk size;
    # small chunks can't fill the pipeline, reducing effective overlap.
    chunk_per_gpu = msg_size / node_size
    warmup_ratio = min(1.0, chunk_per_gpu / warmup_bytes) if warmup_bytes > 0 else 1.0
    effective_overlap = base_overlap * warmup_ratio

    t_intra = t_intra_rs + t_intra_ag
    t_inter = t_inter_ar
    t = max(t_intra, t_inter) + (1 - effective_overlap) * min(t_intra, t_inter)
    t += args.kernel_launch_latency

    return t


# ---------------------------
# Algorithm Selection Wrappers
# ---------------------------


def allreduce(args, msg_size, gpus, groups=["dp"]):
    """
    Select best allreduce algorithm among several options.
    Tries multiple protocols and algorithms, returns fastest.
    Includes hierarchical allreduce with pipelining for multi-node.
    Adds fixed RCCL overhead for protocol negotiation and sync.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    min_ar_time = float("inf")
    for p in ["simple", "ll", "ll64", "ll128"]:
        rs_time = run_reduce_scatter(args, msg_size, gpus, protocol=p)
        ag_time = run_allgather(args, msg_size, gpus, protocol=p)
        bruck_time = rs_time + ag_time
        hypercubeallreduce = oneshotHCallreduce(args, msg_size, gpus, protocol=p)
        ss_allreduce = single_shot_allreduce(args, msg_size, gpus, protocol=p)
        ringallreduce = RingAllreduce(args, msg_size, gpus, protocol=p)
        hier_allreduce = hierarchical_allreduce(args, msg_size, gpus, groups, protocol=p)
        min_ar_alg_time = min(ringallreduce, bruck_time, hypercubeallreduce, ss_allreduce, hier_allreduce)
        if min_ar_alg_time < min_ar_time:
            min_ar_time = min_ar_alg_time
    rccl_overhead = getattr(args, "rccl_overhead_us", 0.0)
    min_ar_time += rccl_overhead
    if gpus > args.node_size:
        nics = getattr(args, "nics_per_node", None) or args.node_size
        num_nodes = gpus // args.node_size
        node_steps = max(1, int(np.ceil(np.log2(num_nodes))))
        per_nic_bytes = msg_size / (nics * node_steps)
        warmup = getattr(args, "nic_warmup_bytes", 32 * 1024 * 1024)
        ratio = min(1.0, per_nic_bytes / warmup) if warmup > 0 else 1.0
        if ratio < 1.0:
            setup_us = getattr(args, "nic_rdma_setup_us", 0.0) * node_steps
            min_ar_time += setup_us * (1.0 - ratio**2.5)
    return min_ar_time


def alltoall(args, msg_size, gpus, groups=["ep"]):
    """
    Select best alltoall algorithm among several options.
    Tries multiple protocols and algorithms, returns fastest.
    Applies per-peer latency overhead and minimum latency floor.
    """
    min_a2a_time = float("inf")
    for p in ["simple", "ll", "ll64", "ll128"]:
        direct_a2a_time = run_alltoall(args, msg_size, gpus, protocol=p)
        single_shot_a2a_time = single_shot_alltoall(args, msg_size, gpus, protocol=p)
        hierarchical_a2a_time = hierarchical_alltoall(args, msg_size, gpus, protocol=p)
        a2a_time = min(direct_a2a_time, single_shot_a2a_time, hierarchical_a2a_time)

        if a2a_time < min_a2a_time:
            min_a2a_time = a2a_time

    # Add overhead for A2A communication (intra-node and inter-node components)
    gpus_per_node = args.node_size
    intra_node_peers = min(gpus - 1, gpus_per_node - 1)
    inter_node_peers = max(0, gpus - gpus_per_node)

    # Intra-node: fixed sync overhead + small per-peer scheduling cost
    # RCCL parallelizes intra-node P2P transfers so overhead is sub-linear in peers.
    # Calibrated against MI325X preflight (2/4/8 GPU A2A measurements).
    intra_sync = getattr(args, "a2a_intra_sync_overhead", 50.0)
    intra_per_peer = getattr(args, "a2a_intra_node_peer_lat", 2.5)
    intra_overhead = (intra_sync + intra_per_peer * intra_node_peers) if intra_node_peers > 0 else 0

    # Inter-node: per-peer RDMA setup cost (sequential QP establishment)
    inter_per_peer = getattr(args, "a2a_peer_lat", 0.45)
    inter_overhead = inter_per_peer * inter_node_peers

    min_a2a_time += intra_overhead + inter_overhead

    # Fixed RCCL A2A kernel/protocol overhead (size-independent)
    min_a2a_time += getattr(args, "a2a_rccl_overhead_us", 0.0)

    return min_a2a_time


def allgather(args, msg_size, gpus, groups=["hp"]):
    """
    Select best allgather algorithm among several options.
    Tries multiple protocols and algorithms, returns fastest.
    """
    min_ag_time = float("inf")
    for p in ["simple", "ll", "ll64", "ll128"]:
        bruck_ag_time = run_allgather(args, msg_size, gpus, protocol=p)
        single_shot_ag_time = single_shot_allgather(args, msg_size, gpus, protocol=p)
        ring_ag_time = RingAllgather(args, msg_size, gpus, protocol=p)
        best_ag_time = min(bruck_ag_time, ring_ag_time, single_shot_ag_time)
        if best_ag_time < min_ag_time:
            min_ag_time = best_ag_time
    return min_ag_time


def reduce_scatter(args, msg_size, gpus, groups=["hp"]):
    """
    Select best reduce_scatter algorithm among several options.
    Tries multiple protocols and algorithms, returns fastest.
    """
    min_rs_time = float("inf")
    for p in ["simple", "ll", "ll64", "ll128"]:
        rs_bruck = run_reduce_scatter(args, msg_size, gpus, protocol=p)
        rs_ring = RingRS(args, msg_size, gpus, protocol=p)
        rs_single_shot = single_shot_reduce_scatter(args, msg_size, gpus, protocol=p)
        best_rs_time = min(rs_bruck, rs_ring, rs_single_shot)
        if best_rs_time < min_rs_time:
            min_rs_time = best_rs_time
    return min_rs_time
