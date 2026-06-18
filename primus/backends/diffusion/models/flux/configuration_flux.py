from __future__ import annotations

from types import SimpleNamespace


class FluxTrainingConfig(SimpleNamespace):
    def to_dict(self) -> dict:
        return dict(self.__dict__)
