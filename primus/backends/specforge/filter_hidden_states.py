###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Drop captures with too few anchorable loss-mask tokens."""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import torch


def _load_record(path: Path) -> dict:
    if path.suffix == ".gz" or path.name.endswith(".ckpt.gz"):
        with gzip.open(path, "rb") as handle:
            return torch.load(handle, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu", weights_only=False)


def _keep(record: dict, block_size: int) -> bool:
    mask = record["loss_mask"]
    if mask.ndim > 1:
        mask = mask.reshape(-1)
    if mask.numel() <= block_size:
        return False
    return int(mask[:-block_size].sum().item()) >= 2


def filter_dflash_dir(source: str | Path, output: str | Path, block_size: int = 16) -> tuple[int, int]:
    """Copy keepable ``*.ckpt`` shards. Returns (kept, dropped)."""

    src = Path(source)
    dest_root = Path(output)
    files = sorted(set(list(src.rglob("*.ckpt")) + list(src.rglob("*.ckpt.gz"))))
    if not files:
        raise RuntimeError(f"[Primus:specforge] no .ckpt files under {src}")

    dest_root.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped = 0
    for item in files:
        if not _keep(_load_record(item), block_size):
            dropped += 1
            continue
        dest = dest_root / item.relative_to(src)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest)
        kept += 1
    return kept, dropped
