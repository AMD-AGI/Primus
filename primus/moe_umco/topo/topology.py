from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyInfo:
    nodes: int
    gpus_per_node: int
    rank_to_node: list[int]
    intra_node_groups: list[list[int]]

    @classmethod
    def from_env_or_default(cls, world_size: int, rank: int) -> "TopologyInfo":
        _ = rank
        ws = max(1, world_size)
        local_size = _first_int_env(
            [
                "LOCAL_WORLD_SIZE",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "SLURM_GPUS_ON_NODE",
            ]
        )
        local_rank = _first_int_env(["LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK"], default=0)
        node_rank = _first_int_env(["NODE_RANK", "SLURM_NODEID", "OMPI_COMM_WORLD_NODE_RANK"], default=0)

        if local_size is None or local_size <= 0:
            local_size = ws
            node_rank = 0
            local_rank = 0

        nodes = max(1, (ws + local_size - 1) // local_size)
        rank_to_node = [min(r // local_size, nodes - 1) for r in range(ws)]
        if ws > 1 and node_rank >= 0 and local_rank >= 0:
            inferred_rank = node_rank * local_size + local_rank
            if inferred_rank < ws:
                rank_to_node[inferred_rank] = node_rank

        groups: list[list[int]] = []
        for n in range(nodes):
            groups.append([r for r in range(ws) if rank_to_node[r] == n])
        return cls(nodes=nodes, gpus_per_node=local_size, rank_to_node=rank_to_node, intra_node_groups=groups)


def _first_int_env(names: list[str], default: int | None = None) -> int | None:
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default
