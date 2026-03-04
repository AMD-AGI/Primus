from primus.moe_umco.config import UmcoConfig
from primus.moe_umco.planner import MoEStepPlanner
from primus.moe_umco.types import MoEWorldInfo


def test_umco_step_plan_basic_and_deterministic():
    planner = MoEStepPlanner()
    world_info = MoEWorldInfo(
        world_size=16,
        rank=0,
        local_rank=0,
        ep_size=8,
        tp_size=1,
        pp_size=1,
    )
    cfg = UmcoConfig(enable=True, chunk_tokens=2048, max_inflight=2, topo_enable=False, log_level="INFO")
    p1 = planner.plan(world_info=world_info, max_tokens=16384, dtype_bytes=2, cfg=cfg, topo=None)
    p2 = planner.plan(world_info=world_info, max_tokens=16384, dtype_bytes=2, cfg=cfg, topo=None)

    assert p1.num_chunks == 8
    assert p1.chunks[0].token_begin == 0
    assert p1.chunks[-1].token_end == 16384
    assert p1.buffer_layout.dispatch_in_bytes > 0
    assert p1.buffer_layout.gather_out_bytes > 0
    assert p1.chunks == p2.chunks
