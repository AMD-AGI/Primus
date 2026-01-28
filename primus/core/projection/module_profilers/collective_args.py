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
from typing import Optional, Dict, Any


@dataclass
class CollectiveArgs:
    """
    Hardware and topology configuration for collective communication modeling.

    All parameters can be overridden via configuration file.
    """

    # Topology
    node_size: int = 8  # GPUs per node
    pod_size: int = 64  # GPUs per pod (cluster)
    hp: int = 1  # Horizontal parallelism groups
    cp: int = 1  # Context parallelism
    ep: int = 1  # Expert parallelism

    # Bandwidth in GB/s (bidirectional)
    node_bw: float = 1024.0  # Intra-node bandwidth per GPU
    pod_bw: float = 50.0  # Inter-node bandwidth per NIC
    cluster_bw: float = 25.0  # Cluster-level bandwidth
    bw_eff: float = 0.91  # Bandwidth efficiency factor

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

    # Network topology
    switch_topology: bool = True  # Whether using switch-based topology
    nics_per_node: Optional[int] = 8  # NICs per node (None = gpus_per_node)

    # All-to-all specific
    a2a_peer_lat: float = 0.45  # Per-peer latency overhead for a2a


def get_default_args(
    num_nodes: int = 1,
    gpus_per_node: int = 8,
    tp: int = 1,
    pp: int = 1,
    dp: int = -1,  # Auto-calculated if -1
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
        dp: Data parallelism size (auto-calculated if -1)
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
                        - nics_per_node: Number of NICs per node (int)
                        - a2a_peer_lat: Per-peer latency for all-to-all (us)

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
    total_gpus = num_nodes * gpus_per_node

    if dp == -1:
        dp = total_gpus // (tp * pp * ep * cp)

    # Start with default CollectiveArgs
    args = CollectiveArgs(
        node_size=gpus_per_node,
        pod_size=total_gpus,
        hp=tp,
        cp=cp,
        ep=ep,
    )

    # Override with hardware_config if provided
    if hardware_config:
        for key, value in hardware_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise ValueError(f"Unknown hardware parameter: {key}")

    # Set nics_per_node to gpus_per_node if not explicitly set
    if args.nics_per_node is None:
        args.nics_per_node = gpus_per_node

    return args
