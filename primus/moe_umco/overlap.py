from __future__ import annotations

from dataclasses import dataclass

from primus.moe_umco.types import StreamPlan, StreamTag


@dataclass(frozen=True)
class OverlapController:
    stream_plan: StreamPlan = StreamPlan()

    @staticmethod
    def default() -> "OverlapController":
        return OverlapController(
            stream_plan=StreamPlan(
                dispatch=StreamTag.DISPATCH,
                compute=StreamTag.COMPUTE,
                gather=StreamTag.GATHER,
            )
        )
