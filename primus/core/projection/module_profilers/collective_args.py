###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Default configuration for collective communication modeling.
Hardware parameters can be customized via config file.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CollectiveArgs:
    """
    Hardware and topology configuration for collective communication modeling.

    All parameters can be overridden via configuration file.
    """

    # Topology
    node_size: int = 8  # GPUs per node
    pod_size: int = 64  # GPUs per pod (cluster)
    num_nodes: int = 1  # Number of nodes
    hp: int = 1  # Horizontal parallelism groups
    pp: int = 1  # Pipeline parallelism
    cp: int = 1  # Context parallelism
    ep: int = 1  # Expert parallelism

    # Bandwidth in GB/s (bidirectional)
    node_bw: float = 1024.0  # Intra-node bandwidth per GPU
    pod_bw: float = 50.0  # Inter-node bandwidth per NIC
    cluster_bw: float = 25.0  # Cluster-level bandwidth
    bw_eff: float = 0.70  # Bandwidth efficiency for collectives (AllReduce, AllToAll)
    p2p_bw_eff: float = 0.80  # Bandwidth efficiency for point-to-point (SendRecv)
    # Single-link P2P transfers have less contention and achieve higher
    # efficiency than full-mesh collectives where all GPUs compete for
    # aggregate mesh bandwidth simultaneously.

    # Latency in microseconds
    node_lat: float = 0.45  # Intra-node latency
    pod_lat: float = 2.0  # Inter-node latency
    cluster_lat: float = 10.0  # Cluster-level latency
    hbm_latency: float = 0.09  # HBM access latency
    write_latency: float = 0.28  # Write operation latency
    write_resp: float = 0.09  # Write response latency

    # Compute
    kernel_launch_latency: float = 2.8  # Kernel launch overhead (us)
    vector_flops: float = 3.2e12  # Vector FLOPS (for reduction compute)

    # RCCL fixed overhead per collective call (us)
    rccl_overhead_us: float = 200.0
    # Per-collective setup cost covering protocol negotiation, synchronization
    # barriers, memory registration, and GPU scheduler overhead. Independent
    # of message size. Calibrated against MI325X preflight measurements.

    # Network topology
    switch_topology: bool = True  # Whether using switch-based topology
    node_topology: str = "switch"  # Node topology: "switch" or "mesh" (for mesh derate)
    nics_per_node: Optional[int] = 8  # NICs per node (None = gpus_per_node)

    # Hierarchical AllReduce pipelining
    ar_overlap_factor: float = 0.9  # Peak overlap between intra-node and inter-node phases (0-1)
    ar_warmup_chunk_bytes: int = 32 * 1024 * 1024  # 32 MB per-GPU chunk for full pipelining
    # RCCL pipelines intra/inter phases using fine-grained chunking. The effective
    # overlap scales with per-GPU chunk size: small chunks can't fill the pipeline,
    # so overlap is reduced proportionally below ar_warmup_chunk_bytes.
    # effective_overlap = ar_overlap_factor * min(1, chunk_per_gpu / ar_warmup_chunk_bytes)
    # Calibrated against MI325X 2-node/4-node preflight measurements.
    nic_rdma_setup_us: float = 260.0
    nic_warmup_bytes: int = 32 * 1024 * 1024  # 32 MB per-NIC for peak RDMA throughput
    # Small RDMA transfers don't achieve peak NIC throughput due to DMA engine
    # warmup, QP setup, and PCIe round-trip overhead. The penalty decays as
    # per-NIC data reaches nic_warmup_bytes via a power-law:
    #   overhead = nic_rdma_setup_us * max(0, 1 - (per_nic_data / nic_warmup_bytes)^2.5)
    # At 8MB/NIC: ~252us overhead (NIC underutilized)
    # At 32MB+/NIC: 0us overhead (peak RDMA throughput achieved)
    # Calibrated against MI325X 2-node preflight AllReduce measurements.

    # All-to-all specific
    a2a_peer_lat: float = 0.45  # Per-peer latency overhead for inter-node a2a (RDMA setup)
    a2a_intra_sync_overhead: float = 50.0  # Fixed intra-node A2A sync overhead (us)
    a2a_intra_node_peer_lat: float = 2.5  # Per-peer scheduling overhead for intra-node a2a (us)
    # Intra-node A2A overhead model: a2a_intra_sync_overhead + a2a_intra_node_peer_lat * peers
    # RCCL parallelizes intra-node P2P transfers, so overhead does NOT scale
    # linearly with peer count. The fixed sync component (barrier, kernel setup)
    # dominates, with a small per-peer scheduling cost.
    # Inter-node overhead is per-peer (~0.45 us) due to sequential RDMA QP setup.
    a2a_mesh_contention: float = 0.12  # Peak BW derate for full-mesh A2A contention
    # Full-mesh AllToAll saturates all xGMI links simultaneously, causing HBM
    # controller and xGMI bridge contention that reduces per-link efficiency.
    # Applied as: effective_bw *= 1 - contention * link_saturation^2
    #   link_saturation = (gpus - 1) / (node_size - 1)
    # The quadratic scaling captures that contention grows super-linearly as
    # the mesh approaches full saturation. Only affects intra-node A2A bandwidth.
    # Calibrated against MI325X 8-GPU AllToAll preflight measurements.
    a2a_rccl_overhead_us: float = 150.0  # Fixed A2A kernel/protocol overhead (us)
    # RCCL AllToAll has a size-independent setup cost (algorithm selection,
    # communicator state, kernel launch chain) observed as a ~150us floor in
    # measured A2A times across all configurations (intra and inter-node).
    # Distinct from rccl_overhead_us (AllReduce) because A2A uses a different
    # RCCL protocol path with additional per-peer coordination.
    a2a_remote_contention: float = 0.04  # Per-remote-node inter-node A2A BW derate
    # In multi-node AllToAll, each GPU sends to (num_nodes-1) * gpus_per_node
    # remote peers through a single NIC. As the number of remote destinations
    # grows, the NIC must multiplex between more QPs, causing PFC/switching
    # overhead and reduced per-flow efficiency. Applied as:
    #   effective_pod_bw *= 1 - a2a_remote_contention * (num_nodes - 1)
    # Calibrated: 2N has no derate; 4N has 8% derate matching measured BW drop.


def get_default_args(
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    tp: int = 1,
    pp: int = 1,
    _dp: int = -1,  # Auto-calculated if -1 (currently unused, retained for API compatibility)
    ep: int = 1,
    cp: int = 1,
    hardware_config: Optional[Dict[str, Any]] = None,
) -> CollectiveArgs:
    """
    Get CollectiveArgs with customizable hardware configuration.

    This function creates a CollectiveArgs instance with default values that can be
    overridden via the hardware_config dictionary. This allows customers to specify
    their own hardware characteristics through config files.

    Args:
        num_nodes: Number of nodes in the cluster
        gpus_per_node: GPUs per node
        tp: Tensor parallelism size
        pp: Pipeline parallelism size
        _dp: Data parallelism size (currently unused, retained for API compatibility)
        ep: Expert parallelism size
        cp: Context parallelism size
        hardware_config: Optional dictionary to override default hardware parameters.
                        Supported keys:
                        - node_bw: Intra-node bandwidth (GB/s)
                        - pod_bw: Inter-node bandwidth (GB/s)
                        - cluster_bw: Cluster-level bandwidth (GB/s)
                        - bw_eff: Bandwidth efficiency factor (0-1)
                        - node_lat: Intra-node latency (us)
                        - pod_lat: Inter-node latency (us)
                        - cluster_lat: Cluster-level latency (us)
                        - hbm_latency: HBM access latency (us)
                        - write_latency: Write operation latency (us)
                        - write_resp: Write response latency (us)
                        - kernel_launch_latency: Kernel launch overhead (us)
                        - vector_flops: Vector FLOPS for compute
                        - switch_topology: Whether using switch-based topology (bool)
                        - node_topology: Node topology type ("switch" or "mesh") for mesh derate
                        - nics_per_node: Number of NICs per node (int)
                        - p2p_bw_eff: Bandwidth efficiency for P2P SendRecv (0-1)
                        - rccl_overhead_us: Fixed per-collective RCCL setup overhead (us)
                        - ar_overlap_factor: Overlap factor for hierarchical AllReduce pipelining (0-1)
                        - ar_warmup_chunk_bytes: Min per-GPU chunk for full pipeline overlap (bytes)
                        - nic_rdma_setup_us: Peak NIC RDMA setup overhead for small transfers (us)
                        - nic_warmup_bytes: Per-NIC data threshold for peak RDMA throughput (bytes)
                        - a2a_peer_lat: Per-peer latency for inter-node all-to-all (us)
                        - a2a_intra_sync_overhead: Fixed intra-node A2A sync overhead (us)
                        - a2a_intra_node_peer_lat: Per-peer scheduling overhead for intra-node A2A (us)

    Returns:
        CollectiveArgs configured with specified parameters

    Example:
        >>> # Use default configuration
        >>> args = get_default_args(num_nodes=4, gpus_per_node=8)

        >>> # Override hardware parameters
        >>> hw_config = {
        >>>     'node_bw': 1024.0,  # Higher intra-node bandwidth
        >>>     'bw_eff': 0.92,     # Better efficiency
        >>>     'node_lat': 0.45,   # Lower latency
        >>> }
        >>> args = get_default_args(num_nodes=4, gpus_per_node=8,
        >>>                         hardware_config=hw_config)
    """
    # Calculate total GPUs
    total_gpus = num_nodes * gpus_per_node

    # Start with default CollectiveArgs
    args = CollectiveArgs(
        node_size=gpus_per_node,
        pod_size=total_gpus,
        num_nodes=num_nodes,
        hp=tp,
        pp=pp,
        cp=cp,
        ep=ep,
    )

    # Override with hardware_config if provided
    if hardware_config:
        for key, value in hardware_config.items():
            if hasattr(args, key):
                # Get the expected type from the dataclass field
                field_type = type(getattr(args, key))

                # Convert value to the expected type if it's a string representation of a number
                if isinstance(value, str) and field_type in (int, float):
                    try:
                        if field_type == int:
                            # Handle scientific notation for int (e.g., "1e3" -> 1000)
                            if "e" in value.lower() or "E" in value.lower():
                                value = int(float(value))
                            else:
                                value = int(value)
                        elif field_type == float:
                            # Handle scientific notation for float (e.g., "3.2e12" -> 3.2e12)
                            value = float(value)
                    except (ValueError, TypeError):
                        # If conversion fails, keep original value and let it fail later
                        pass
                elif field_type == bool and isinstance(value, str):
                    # Convert string booleans to actual booleans
                    if value.lower() in ("true", "1", "yes", "on"):
                        value = True
                    elif value.lower() in ("false", "0", "no", "off"):
                        value = False

                setattr(args, key, value)
            else:
                print(f"[Primus:WARNING] Unknown hardware parameter '{key}' in config. Skipping.")

    # Set nics_per_node to gpus_per_node if not explicitly set
    if args.nics_per_node is None:
        args.nics_per_node = gpus_per_node

    # Store raw bandwidths before applying bw_eff (needed for P2P which uses p2p_bw_eff)
    args._raw_node_bw = args.node_bw
    args._raw_pod_bw = args.pod_bw

    # Apply bw_eff to bandwidth values once at initialization
    args.node_bw = args.node_bw * args.bw_eff
    args.pod_bw = args.pod_bw * args.bw_eff
    args.cluster_bw = args.cluster_bw * args.bw_eff

    return args
