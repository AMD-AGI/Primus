from __future__ import annotations

from primus.moe_umco.config import UmcoConfig
from primus.moe_umco.overlap import OverlapController
from primus.moe_umco.step_plan import compute_buffer_layout, compute_chunking
from primus.moe_umco.topo.mapper import TopologyPlan
from primus.moe_umco.types import MoEStepPlan, MoEWorldInfo


class MoEStepPlanner:
    def plan(
        self,
        world_info: MoEWorldInfo,
        max_tokens: int,
        dtype_bytes: int,
        cfg: UmcoConfig,
        topo: TopologyPlan | None,
    ) -> MoEStepPlan:
        chunk_tokens = cfg.chunk_tokens
        if chunk_tokens <= 0:
            if max_tokens >= 4096:
                chunk_tokens = max(512, min(2048, max_tokens))
            else:
                chunk_tokens = max(1, min(2048, max_tokens))

        chunks = compute_chunking(max_tokens=max_tokens, chunk_tokens=chunk_tokens)
        layout = compute_buffer_layout(
            max_tokens=max_tokens,
            dtype_bytes=dtype_bytes,
            ep_size=world_info.ep_size,
            alignment=128,
            chunk_tokens=chunk_tokens,
        )
        streams = OverlapController.default().stream_plan
        return MoEStepPlan(
            chunk_tokens=chunk_tokens,
            num_chunks=len(chunks),
            chunks=chunks,
            max_inflight=max(1, cfg.max_inflight),
            stream_plan=streams,
            buffer_layout=layout,
            topo_plan=topo,
        )
