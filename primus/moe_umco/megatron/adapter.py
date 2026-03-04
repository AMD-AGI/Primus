from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RoutingInfo:
    routing_map: Any
    probs: Any


def build_dispatch_impl(dispatch_fn: Callable[..., Any]) -> Callable[..., Any]:
    return dispatch_fn


def build_gather_impl(gather_fn: Callable[..., Any]) -> Callable[..., Any]:
    return gather_fn
