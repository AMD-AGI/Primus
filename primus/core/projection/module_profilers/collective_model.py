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


def get_bandwidth_and_latency(args, domain_size):
    """
    Determine bandwidth and latency for a given communication domain size.
    Selects node, pod, or cluster bandwidth/latency depending on the size.
    """
    if domain_size <= args.node_size:
        # Communication fits within a node
        bw = args.bw_eff * args.node_bw
        lat = args.node_lat
    elif domain_size <= args.pod_size:
        # Communication fits within a pod (multiple nodes)
        bw = args.bw_eff * args.pod_bw
        lat = args.pod_lat
    else:
        # Communication spans the cluster (multiple pods)
        bw = args.bw_eff * args.cluster_bw
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
    Used for basic communication between two GPUs.
    """
    domain = args.hp * args.cp * args.ep
    bw, lat = get_bandwidth_and_latency(args, domain)
    # Time = transmission + latency + kernel launch overhead
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

    # Intra-node time
    t_intra = intra_vol_adj / (args.bw_eff * args.node_bw) * 1.0e-3 + node_lat

    # Inter-node time with all NICs
    if args.switch_topology:
        # Total inter-node volume from the node
        total_inter_volume = inter_node_volume_per_gpu * gpus_per_node
        # Aggregate bandwidth using all NICs
        aggregate_inter_bw = args.bw_eff * args.pod_bw * nics_per_node
        t_inter = total_inter_volume / aggregate_inter_bw * 1.0e-3 + pod_lat
    else:
        remote_nodes = num_nodes - 1
        t_inter = inter_node_volume_per_gpu / (args.bw_eff * args.pod_bw) * 1.0e-3 + pod_lat * remote_nodes

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
        # Alltoall fits within node
        bw = args.bw_eff * args.node_bw
        lat = node_lat
    elif (args.hp * gpus > args.node_size) and (args.hp * gpus) <= args.pod_size:
        # Alltoall fits within pod
        if args.hp == 1:
            return direct_alltoall(args, msg_size, gpus, groups, protocol, original_msg_size)
        bw = args.bw_eff * args.pod_bw
        lat = args.pod_lat
    else:
        # Alltoall spans cluster
        bw = args.bw_eff * args.cluster_bw
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
        bw = args.pod_bw * args.bw_eff
        lat = pod_lat
    elif cpxhp <= args.node_size:
        # CP domain fits within node
        bw = args.node_bw * args.bw_eff
        lat = node_lat
    else:
        # CP domain spans cluster
        bw = args.cluster_bw * args.bw_eff
        lat = args.cluster_lat
    # Logarithmic steps for tree allgather
    t = msg_size / bw * 1.0e-3 + lat * np.ceil(np.log2(gpus)) + args.kernel_launch_latency
    return t


def run_allgather(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Run allgather collective.
    Handles node and pod domains, and CP group special case.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    if "cp" in groups:
        # Use CP allgather if requested
        return cp_allgather(args, msg_size, gpus, protocol)
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    pod_lat = args.pod_lat
    t = 0
    lat = 0
    if gpus > args.node_size:
        # Communication spans node and pod
        bw = args.bw_eff * args.node_bw
        node_msg_volume = msg_size * (args.node_size - 1) / args.node_size
        t = node_msg_volume / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(args.node_size))
        bw = args.bw_eff * args.pod_bw
        pod_msg_volume = msg_size - node_msg_volume
        t += pod_msg_volume / bw * 1.0e-3
        lat += pod_lat * np.ceil(np.log2(gpus / args.node_size))
    else:
        # Allgather fits within node
        bw = args.bw_eff * args.node_bw
        t = msg_size / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(gpus))
    t += lat + args.kernel_launch_latency
    return t


def run_reduce_scatter(args, msg_size, gpus, groups=["hp"], protocol=None):
    """
    Run reduce_scatter collective.
    Handles node and pod domains, includes compute time for reduction.
    """
    if gpus == 1 or msg_size == 0:
        return 0
    msg_scale = (gpus - 1) / gpus
    msg_size = int(msg_size * msg_scale)
    node_lat, msg_size = node_latency_and_volume_protocol(args, msg_size, protocol)
    t = 0
    lat = 0
    if gpus > args.node_size:
        # Communication spans node and pod
        bw = args.bw_eff * args.node_bw
        node_msg_volume = msg_size * (args.node_size - 1) / args.node_size
        t = node_msg_volume / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(args.node_size))
        bw = args.bw_eff * args.pod_bw
        pod_msg_volume = msg_size - node_msg_volume
        pod_lat = args.pod_lat
        t += pod_msg_volume / bw * 1.0e-3
        lat += pod_lat * np.ceil(np.log2(gpus / args.node_size))
    else:
        # Reduce scatter fits within node
        bw = args.bw_eff * args.node_bw
        t = msg_size / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(gpus))
    t += lat
    # Add compute time for reduction (vector flops)
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
        t += msg_size / args.node_bw * 1.0e-3
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
        t += msg_size / args.node_bw * 1.0e-3
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
        t += msg_size / args.node_bw * 1.0e-3
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
        bw = args.bw_eff * args.node_bw
        node_msg_volume = msg_size * (args.node_size - 1)
        t = node_msg_volume / bw * 1.0e-3
        lat += node_lat * np.ceil(np.log2(args.node_size))
        bw = args.bw_eff * args.pod_bw
        pod_msg_volume = msg_size * np.ceil(np.log2((gpus - args.node_size)))
        t += pod_msg_volume / bw * 1.0e-3
        lat += pod_lat * np.ceil(np.log2(gpus / args.node_size))
    else:
        # Allreduce fits within node
        bw = args.bw_eff * args.node_bw
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
    gpus_per_node = min(gpus, args.node_size)
    nics_per_node = args.nics_per_node if args.nics_per_node else gpus_per_node
    intra_node_gpus = gpus_per_node - 1
    inter_node_gpus = max(0, gpus - gpus_per_node)

    t_intra_node = 0
    t_inter_node = 0
    if intra_node_gpus > 0:
        node_lat, msg_size_per_peer_adj = node_latency_and_volume_protocol(args, msg_size_per_peer, protocol)
        node_bw = args.bw_eff * args.node_bw
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_adj * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        inter_node_msg_size_per_gpu = msg_size_per_peer * inter_node_gpus
        if args.switch_topology:
            # With switch topology, use all NICs
            total_inter_volume = inter_node_msg_size_per_gpu * gpus_per_node
            aggregate_bw = args.bw_eff * args.pod_bw * nics_per_node
            inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
            t_inter_node = inter_node_rounds * pod_lat
            t_inter_node += total_inter_volume / aggregate_bw * 1.0e-3
        else:
            pod_bw = args.bw_eff * args.pod_bw
            inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
            t_inter_node = inter_node_rounds * pod_lat
            t_inter_node += inter_node_msg_size_per_gpu / pod_bw * 1.0e-3
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

    gpus_per_node = min(gpus, args.node_size)
    num_nodes = ceil(gpus / args.node_size)
    nics_per_node = args.nics_per_node if args.nics_per_node else gpus_per_node

    if num_nodes == 1:
        return single_shot_alltoall(args, msg_size, gpus, groups, protocol)

    # Volume breakdown per GPU
    intra_node_volume = msg_size * (gpus_per_node - 1) / gpus
    inter_node_volume_per_gpu = msg_size * (gpus - gpus_per_node) / gpus

    # Intra-node time
    node_lat, intra_vol_adj = node_latency_and_volume_protocol(args, intra_node_volume, protocol)
    node_bw = args.bw_eff * args.node_bw
    t_intra = node_lat + intra_vol_adj / node_bw * 1.0e-3

    # Inter-node time with all NICs
    if args.switch_topology:
        total_inter_volume = inter_node_volume_per_gpu * gpus_per_node
        aggregate_inter_bw = args.bw_eff * args.pod_bw * nics_per_node
        t_inter = args.pod_lat + total_inter_volume / aggregate_inter_bw * 1.0e-3
    else:
        effective_pod_bw = args.bw_eff * args.pod_bw
        t_inter = args.pod_lat * num_nodes + inter_node_volume_per_gpu / effective_pod_bw * 1.0e-3

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
        node_bw = args.bw_eff * args.node_bw
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_node * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        pod_bw = args.bw_eff * args.pod_bw
        inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
        t_inter_node = inter_node_rounds * pod_lat
        inter_node_msg_size = msg_size_per_peer * inter_node_gpus
        t_inter_node += inter_node_msg_size / pod_bw * 1.0e-3
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
        node_bw = args.bw_eff * args.node_bw
        intra_node_rounds = ceil(intra_node_gpus / intra_node_fanout)
        t_intra_node = intra_node_rounds * node_lat
        intra_node_msg_size = msg_size_per_peer_node * intra_node_gpus
        t_intra_node += intra_node_msg_size / node_bw * 1.0e-3
    if inter_node_gpus > 0:
        pod_lat = args.pod_lat
        pod_bw = args.bw_eff * args.pod_bw
        inter_node_rounds = ceil(inter_node_gpus / inter_node_fanout)
        t_inter_node = inter_node_rounds * pod_lat
        inter_node_msg_size = msg_size_per_peer * inter_node_gpus
        t_inter_node += inter_node_msg_size / pod_bw * 1.0e-3
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
# Algorithm Selection Wrappers
# ---------------------------


def allreduce(args, msg_size, gpus, groups=["dp"]):
    """
    Select best allreduce algorithm among several options.
    Tries multiple protocols and algorithms, returns fastest.
    """
    min_ar_time = float("inf")
    for p in ["simple", "ll", "ll64", "ll128"]:
        rs_time = run_reduce_scatter(args, msg_size, gpus, protocol=p)
        ag_time = run_allgather(args, msg_size, gpus, protocol=p)
        bruck_time = rs_time + ag_time
        hypercubeallreduce = oneshotHCallreduce(args, msg_size, gpus, protocol=p)
        ss_allreduce = single_shot_allreduce(args, msg_size, gpus, protocol=p)
        ringallreduce = RingAllreduce(args, msg_size, gpus, protocol=p)
        min_ar_alg_time = min(ringallreduce, bruck_time, hypercubeallreduce, ss_allreduce)
        if min_ar_alg_time < min_ar_time:
            min_ar_time = min_ar_alg_time
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

    # Add per-peer latency overhead for inter-node communication
    # This accounts for RDMA QP setup, work request posting, completion polling, etc.
    if hasattr(args, "a2a_peer_lat") and args.a2a_peer_lat > 0:
        gpus_per_node = args.node_size
        inter_node_peers = max(0, gpus - gpus_per_node)
        peer_overhead = args.a2a_peer_lat * inter_node_peers
        min_a2a_time += peer_overhead

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
