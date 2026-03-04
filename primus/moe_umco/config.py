from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


def _env_bool(name: str) -> Optional[bool]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _nested_get(data: dict[str, Any], path: list[str], default: Any) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@dataclass(frozen=True)
class UmcoConfig:
    enable: bool = False
    chunk_tokens: int = 0
    max_inflight: int = 2
    topo_enable: bool = False
    log_level: str = "INFO"


def load_umco_config(exp_config: Optional[dict[str, Any]] = None) -> UmcoConfig:
    exp_config = exp_config or {}
    base_cfg = _nested_get(exp_config, ["exp", "moe", "comm_orchestrator"], {}) or {}
    topo_cfg = base_cfg.get("topology", {}) if isinstance(base_cfg, dict) else {}

    enable = bool(base_cfg.get("enable", False))
    chunk_tokens = int(base_cfg.get("chunk_tokens", 0) or 0)
    max_inflight = int(base_cfg.get("max_inflight", 2) or 2)
    topo_enable = bool(topo_cfg.get("enable", False)) if isinstance(topo_cfg, dict) else False
    log_level = str(base_cfg.get("log_level", "INFO"))

    env_enable = _env_bool("PRIMUS_UMCO_ENABLE")
    env_chunk_tokens = _env_int("PRIMUS_UMCO_CHUNK_TOKENS")
    env_max_inflight = _env_int("PRIMUS_UMCO_MAX_INFLIGHT")
    env_topo_enable = _env_bool("PRIMUS_UMCO_TOPO_ENABLE")
    env_log_level = os.environ.get("PRIMUS_UMCO_LOG_LEVEL")

    if env_enable is not None:
        enable = env_enable
    if env_chunk_tokens is not None:
        chunk_tokens = env_chunk_tokens
    if env_max_inflight is not None:
        max_inflight = env_max_inflight
    if env_topo_enable is not None:
        topo_enable = env_topo_enable
    if env_log_level:
        log_level = env_log_level

    return UmcoConfig(
        enable=enable,
        chunk_tokens=max(chunk_tokens, 0),
        max_inflight=max(1, max_inflight),
        topo_enable=topo_enable,
        log_level=log_level.upper(),
    )
