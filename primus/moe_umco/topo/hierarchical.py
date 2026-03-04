from __future__ import annotations

from dataclasses import dataclass

from primus.moe_umco.topo.mapper import TopologyPlan
from primus.moe_umco.topo.topology import TopologyInfo


@dataclass(frozen=True)
class HierarchicalStage:
    name: str
    groups: list[list[int]]


@dataclass(frozen=True)
class HierarchicalTopologyPlan:
    stage1_intra_node: HierarchicalStage
    stage2_inter_node: HierarchicalStage


def build_hierarchical_plan(topo: TopologyInfo, topo_plan: TopologyPlan) -> HierarchicalTopologyPlan:
    stage1 = HierarchicalStage(name="intra_node", groups=topo.intra_node_groups)
    inter_groups = [
        group for group in topo_plan.ep_groups if len(set(topo.rank_to_node[r] for r in group)) > 1
    ]
    stage2 = HierarchicalStage(name="inter_node", groups=inter_groups)
    return HierarchicalTopologyPlan(stage1_intra_node=stage1, stage2_inter_node=stage2)
