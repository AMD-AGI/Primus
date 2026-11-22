###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Node / rack / facility power and power-constrained capacity planning.

GPU power comes from :mod:`primus.core.projection.power.model`; this module
adds what sits around the GPUs:

* per node: host CPUs, memory/storage/fans/BMC, scale-out and front-end NICs,
  optional scale-up switch share, divided by PSU efficiency;
* per cluster: scale-out switches for a two-tier non-blocking fabric sized
  from the NIC count and switch radix;
* facility: IT power times PUE.

Every default here is a generic placeholder in the range of public datasheets
for current server CPUs, NICs and switches; replace them with the real bill of
materials in the ``infrastructure:`` block of ``--power-config``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class InfraConfig:
    gpus_per_node: int = 8
    cpu_sockets: int = 2
    cpu_w_per_socket: float = 360.0
    host_other_w: float = 800.0  # DRAM, storage, fans, BMC, board
    nics_per_node: Optional[int] = None  # scale-out NICs; default one per GPU
    nic_w: float = 25.0
    frontend_nics: int = 1
    scaleup_switch_w_per_node: float = 0.0
    psu_efficiency: float = 0.94
    switch_w: float = 1500.0
    switch_radix: int = 64
    pue: float = 1.3
    rack_power_kw: Optional[float] = None
    # How GPUs are provisioned against a budget: at their cap ("cap") or at the
    # projected average draw ("average").  Facilities provision for the cap.
    provision: str = "cap"

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], **defaults: Any) -> "InfraConfig":
        merged = dict(defaults)
        merged.update({k: v for k, v in (data or {}).items() if k in {f.name for f in fields(cls)}})
        return cls(**merged)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def scaleout_nics(self) -> int:
        return self.nics_per_node if self.nics_per_node is not None else self.gpus_per_node


def node_power_w(gpu_power_w: float, cfg: InfraConfig) -> Dict[str, float]:
    gpus = cfg.gpus_per_node * gpu_power_w
    cpus = cfg.cpu_sockets * cfg.cpu_w_per_socket
    nics = (cfg.scaleout_nics + cfg.frontend_nics) * cfg.nic_w
    dc = gpus + cpus + cfg.host_other_w + nics + cfg.scaleup_switch_w_per_node
    wall = dc / max(cfg.psu_efficiency, 1e-3)
    return {
        "gpus_w": gpus,
        "cpus_w": cpus,
        "host_other_w": cfg.host_other_w,
        "nics_w": nics,
        "scaleup_switch_w": cfg.scaleup_switch_w_per_node,
        "psu_loss_w": wall - dc,
        "node_w": wall,
    }


def scaleout_switch_count(num_nodes: int, cfg: InfraConfig) -> int:
    """Two-tier non-blocking leaf/spine sized from the scale-out NIC ports."""
    ports = num_nodes * cfg.scaleout_nics
    if ports <= 0:
        return 0
    radix = max(2, cfg.switch_radix)
    if ports <= radix:
        return 1
    down = radix // 2
    leaves = math.ceil(ports / down)
    spines = math.ceil(leaves * down / radix)
    return leaves + spines


@dataclass
class ClusterPower:
    num_nodes: int
    gpu_power_w: float
    node: Dict[str, float]
    switches: int
    network_w: float
    it_w: float
    facility_w: float
    racks: Optional[int]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def cluster_power(num_nodes: int, gpu_power_w: float, cfg: InfraConfig) -> ClusterPower:
    node = node_power_w(gpu_power_w, cfg)
    switches = scaleout_switch_count(num_nodes, cfg)
    network_w = switches * cfg.switch_w
    it_w = num_nodes * node["node_w"] + network_w
    racks = None
    if cfg.rack_power_kw:
        per_rack = max(1, int(cfg.rack_power_kw * 1000.0 // node["node_w"]))
        racks = math.ceil(num_nodes / per_rack)
    return ClusterPower(
        num_nodes=num_nodes,
        gpu_power_w=gpu_power_w,
        node=node,
        switches=switches,
        network_w=network_w,
        it_w=it_w,
        facility_w=it_w * cfg.pue,
        racks=racks,
    )


@dataclass
class CapacityOption:
    gpu_cap_w: float
    tokens_per_s_per_gpu: float
    avg_gpu_power_w: float
    num_nodes: int
    num_gpus: int
    cluster_tokens_per_s: float
    facility_w: float
    tokens_per_kwh: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def max_nodes_within(
    budget_w: float, gpu_power_w: float, cfg: InfraConfig, budget_is_facility: bool = True
) -> int:
    """Largest node count whose (facility or IT) power fits *budget_w*."""

    def power(n: int) -> float:
        cp = cluster_power(n, gpu_power_w, cfg)
        return cp.facility_w if budget_is_facility else cp.it_w

    if power(1) > budget_w:
        return 0
    lo, hi = 1, 2
    while power(hi) <= budget_w:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if power(mid) <= budget_w:
            lo = mid
        else:
            hi = mid
    return lo


def capacity_plan(
    budget_w: float,
    caps_w: Sequence[float],
    throughput_at_cap: Callable[[float], Dict[str, float]],
    cfg: InfraConfig,
    budget_is_facility: bool = True,
    max_nodes: Optional[int] = None,
    node_multiple: int = 1,
) -> List[CapacityOption]:
    """Cluster throughput for each per-GPU cap under a fixed power budget.

    *throughput_at_cap(cap_w)* returns ``{"tokens_per_s_per_gpu", "avg_power_w"}``
    for one GPU running the workload under that cap.  Per-GPU throughput is
    assumed independent of cluster size (weak scaling at fixed parallelism);
    re-run the projection at the chosen node count to confirm.
    """
    options: List[CapacityOption] = []
    for cap in caps_w:
        perf = throughput_at_cap(cap)
        tps = float(perf.get("tokens_per_s_per_gpu", 0.0))
        avg = float(perf.get("avg_power_w", cap))
        provisioned = cap if cfg.provision == "cap" else avg
        n = max_nodes_within(budget_w, provisioned, cfg, budget_is_facility)
        if max_nodes is not None:
            n = min(n, max_nodes)
        if node_multiple > 1:
            n -= n % node_multiple
        running = cluster_power(n, avg, cfg) if n > 0 else None
        cluster_tps = n * cfg.gpus_per_node * tps
        facility = running.facility_w if running else 0.0
        tokens_per_kwh = cluster_tps * 3600.0 / (facility / 1000.0) if facility > 0 else 0.0
        options.append(
            CapacityOption(
                gpu_cap_w=cap,
                tokens_per_s_per_gpu=tps,
                avg_gpu_power_w=avg,
                num_nodes=n,
                num_gpus=n * cfg.gpus_per_node,
                cluster_tokens_per_s=cluster_tps,
                facility_w=facility,
                tokens_per_kwh=tokens_per_kwh,
            )
        )
    return options
