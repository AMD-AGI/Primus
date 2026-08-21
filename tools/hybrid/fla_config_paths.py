"""Resolve FLA legacy training config JSON paths."""

from __future__ import annotations

import os
from pathlib import Path


def fla_training_configs_dir() -> Path:
    fla_root = os.environ.get("FLA_ROOT", os.path.expanduser("~/flash-linear-attention"))
    primary = Path(fla_root) / "legacy" / "training" / "configs"
    if primary.exists():
        return primary
    alt = (
        Path(__file__).resolve().parent.parent.parent
        / "third_party"
        / "flash-linear-attention"
        / "legacy"
        / "training"
        / "configs"
    )
    return primary if primary.exists() else alt


def kda_fla_config(configs_dir: Path, *, size: str) -> Path:
    name = "kda_300M.json" if size == "300M" else "kda_1B.json"
    return configs_dir / name


def gdn_fla_config(configs_dir: Path, *, size: str, hundred_b: bool = False) -> Path:
    if hundred_b:
        return configs_dir / "gated_deltanet_1B_100B.json"
    if size == "300M":
        return configs_dir / "gated_deltanet_300M.json"
    return configs_dir / "gated_deltanet_1B.json"
