from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from primus.moe_umco.config import UmcoConfig
from primus.moe_umco.dispatcher import (
    BaselineMegatronDispatcher,
    MoEDispatcher,
    UmcoDispatcher,
)
from primus.moe_umco.planner import MoEStepPlanner
from primus.moe_umco.topo.mapper import TopologyMapper, TopologyPlan
from primus.moe_umco.topo.topology import TopologyInfo
from primus.moe_umco.types import MoEStepPlan, MoEWorldInfo

logger = logging.getLogger("primus.moe_umco")


@dataclass
class UnifiedMoECommOrchestrator:
    cfg: UmcoConfig
    world_info: MoEWorldInfo
    topo_plan: TopologyPlan | None = None
    planner: MoEStepPlanner = field(default_factory=MoEStepPlanner)
    _plan_cache: dict[tuple[int, int, int, int], MoEStepPlan] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cfg.topo_enable:
            topo = TopologyInfo.from_env_or_default(self.world_info.world_size, self.world_info.rank)
            self.topo_plan = TopologyMapper().map_ep_groups(self.world_info, topo)
            logger.info("UMCO topology EP groups: %s", self.topo_plan.ep_groups)
        logger.info(
            "UMCO world info rank=%s world=%s ep=%s tp=%s pp=%s",
            self.world_info.rank,
            self.world_info.world_size,
            self.world_info.ep_size,
            self.world_info.tp_size,
            self.world_info.pp_size,
        )

    def build_plan(self, max_tokens: int, dtype_bytes: int) -> MoEStepPlan:
        cache_key = (max_tokens, dtype_bytes, self.world_info.ep_size, self.world_info.world_size)
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]
        plan = self.planner.plan(
            world_info=self.world_info,
            max_tokens=max_tokens,
            dtype_bytes=dtype_bytes,
            cfg=self.cfg,
            topo=self.topo_plan,
        )
        self._plan_cache[cache_key] = plan
        logger.info("UMCO plan chunk_tokens=%s num_chunks=%s", plan.chunk_tokens, plan.num_chunks)
        return plan

    def get_dispatcher(
        self,
        baseline_dispatch_impl: Any,
        baseline_gather_impl: Any,
        baseline_dispatch_fn_ref: Any,
        baseline_gather_fn_ref: Any,
        max_tokens: int,
        dtype_bytes: int,
    ) -> MoEDispatcher:
        baseline = BaselineMegatronDispatcher(
            dispatch_impl=baseline_dispatch_impl,
            gather_impl=baseline_gather_impl,
            dispatch_fn_ref=baseline_dispatch_fn_ref,
            gather_fn_ref=baseline_gather_fn_ref,
        )
        if not self.cfg.enable:
            return baseline
        plan = self.build_plan(max_tokens=max_tokens, dtype_bytes=dtype_bytes)
        return UmcoDispatcher(baseline=baseline, plan=plan)

    def get_ep_groups(self) -> list[list[int]]:
        if self.topo_plan is None:
            return []
        return self.topo_plan.ep_groups
