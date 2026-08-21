"""Resolve FLA legacy training config JSON paths with Primus canonical names."""

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


def resolve_fla_training_config(configs_dir: Path, *names: str) -> Path:
    """Return the first existing config under configs_dir; else the canonical (first) name."""
    for name in names:
        path = configs_dir / name
        if path.exists():
            return path
    return configs_dir / names[0]


def kda_fla_config(configs_dir: Path, *, size: str) -> Path:
    if size == "300M":
        return resolve_fla_training_config(configs_dir, "kda_300M.json", "kda_300M_pure.json")
    return resolve_fla_training_config(configs_dir, "kda_1B.json", "kda_1B_pure.json")


def gdn_fla_config(configs_dir: Path, *, size: str, hundred_b: bool = False) -> Path:
    if hundred_b:
        return resolve_fla_training_config(
            configs_dir,
            "gated_deltanet_1B_100B.json",
            "gated_deltanet_1B_pure_100B.json",
        )
    if size == "300M":
        return resolve_fla_training_config(
            configs_dir,
            "gated_deltanet_300M.json",
            "gated_deltanet_300M_pure.json",
        )
    return resolve_fla_training_config(
        configs_dir,
        "gated_deltanet_1B.json",
        "gated_deltanet_1B_pure.json",
    )
