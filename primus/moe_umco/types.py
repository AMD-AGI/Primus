from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from primus.moe_umco.topo.mapper import TopologyPlan


@dataclass(frozen=True)
class MoEWorldInfo:
    world_size: int
    rank: int
    local_rank: int
    ep_size: int
    tp_size: int
    pp_size: int
    node_id: Optional[int] = None
    gpus_per_node: Optional[int] = None


@dataclass(frozen=True)
class MoEBufferLayout:
    dispatch_in_bytes: int
    dispatch_out_bytes: int
    gather_in_bytes: int
    gather_out_bytes: int
    alignment_bytes: int
    chunk_tokens: int


@dataclass(frozen=True)
class MoEChunkSpec:
    chunk_id: int
    token_begin: int
    token_end: int
    tokens: int


class StreamTag(str, Enum):
    DISPATCH = "dispatch"
    COMPUTE = "compute"
    GATHER = "gather"


@dataclass(frozen=True)
class StreamPlan:
    dispatch: StreamTag = StreamTag.DISPATCH
    compute: StreamTag = StreamTag.COMPUTE
    gather: StreamTag = StreamTag.GATHER


@dataclass(frozen=True)
class MoEStepPlan:
    chunk_tokens: int
    num_chunks: int
    chunks: list[MoEChunkSpec]
    max_inflight: int
    stream_plan: StreamPlan
    buffer_layout: MoEBufferLayout
    topo_plan: Optional[TopologyPlan] = None
