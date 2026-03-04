from primus.moe_umco.topo.mapper import TopologyMapper
from primus.moe_umco.topo.topology import TopologyInfo
from primus.moe_umco.types import MoEWorldInfo


def _world(ep_size: int = 8) -> MoEWorldInfo:
    return MoEWorldInfo(
        world_size=16,
        rank=0,
        local_rank=0,
        ep_size=ep_size,
        tp_size=1,
        pp_size=1,
    )


def test_topology_mapper_prefers_intra_node_groups():
    topo = TopologyInfo(
        nodes=2,
        gpus_per_node=8,
        rank_to_node=[0] * 8 + [1] * 8,
        intra_node_groups=[list(range(8)), list(range(8, 16))],
    )
    plan = TopologyMapper().map_ep_groups(_world(ep_size=8), topo)
    assert plan.ep_groups[0] == list(range(8))
    assert plan.ep_groups[1] == list(range(8, 16))


def test_topology_mapper_limits_cross_node_spread():
    topo = TopologyInfo(
        nodes=4,
        gpus_per_node=4,
        rank_to_node=[0] * 4 + [1] * 4 + [2] * 4 + [3] * 4,
        intra_node_groups=[list(range(0, 4)), list(range(4, 8)), list(range(8, 12)), list(range(12, 16))],
    )
    plan = TopologyMapper().map_ep_groups(_world(ep_size=8), topo)
    assert len(plan.ep_groups) == 2
    assert all(len(set(topo.rank_to_node[r] for r in g)) <= 2 for g in plan.ep_groups)
