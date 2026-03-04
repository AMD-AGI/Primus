from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from primus.moe_umco.topo.topology import TopologyInfo

if TYPE_CHECKING:
    from primus.moe_umco.types import MoEWorldInfo


@dataclass(frozen=True)
class TopologyPlan:
    ep_groups: list[list[int]]
    ep_rank_to_group: dict[int, int]
    prefer_intra_node: bool


class TopologyMapper:
    def map_ep_groups(self, world_info: "MoEWorldInfo", topo: TopologyInfo) -> TopologyPlan:
        world_size = max(1, world_info.world_size)
        ep_size = max(1, world_info.ep_size)
        all_ranks = list(range(world_size))
        sorted_ranks = sorted(all_ranks, key=lambda r: (topo.rank_to_node[r], r))

        prefer_intra = (
            topo.gpus_per_node > 0 and world_size % topo.gpus_per_node == 0 and ep_size <= topo.gpus_per_node
        )
        if prefer_intra:
            groups = self._intra_first_groups(sorted_ranks, topo, ep_size, world_size)
        else:
            groups = self._greedy_groups(sorted_ranks, ep_size)

        rank_to_group = {}
        for group_idx, group in enumerate(groups):
            for rank in group:
                rank_to_group[rank] = group_idx
        return TopologyPlan(ep_groups=groups, ep_rank_to_group=rank_to_group, prefer_intra_node=prefer_intra)

    def _intra_first_groups(
        self, sorted_ranks: list[int], topo: TopologyInfo, ep_size: int, world_size: int
    ) -> list[list[int]]:
        groups: list[list[int]] = []
        for node in range(topo.nodes):
            node_ranks = [r for r in sorted_ranks if topo.rank_to_node[r] == node]
            start = 0
            while start < len(node_ranks):
                group = node_ranks[start : start + ep_size]
                if len(group) == ep_size:
                    groups.append(group)
                start += ep_size

        used = {r for g in groups for r in g}
        remaining = [r for r in range(world_size) if r not in used]
        if remaining:
            groups.extend(self._greedy_groups(remaining, ep_size))
        return groups

    def _greedy_groups(self, ranks: list[int], ep_size: int) -> list[list[int]]:
        groups: list[list[int]] = []
        current: list[int] = []
        for rank in ranks:
            current.append(rank)
            if len(current) == ep_size:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return groups
