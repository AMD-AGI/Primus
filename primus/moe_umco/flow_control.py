from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlowController:
    max_inflight: int
    _inflight: int = 0

    def acquire(self) -> bool:
        if self._inflight >= self.max_inflight:
            return False
        self._inflight += 1
        return True

    def release(self) -> None:
        self._inflight = max(0, self._inflight - 1)

    @property
    def inflight(self) -> int:
        return self._inflight
