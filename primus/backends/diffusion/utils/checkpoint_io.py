###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""Load official DiT / UNet state dicts from a file or a checkpoint directory."""

from __future__ import annotations

import glob
import os

import torch
from safetensors.torch import load_file as safe_load_file

_CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pth", ".pt", ".ckpt")


def _directory_checkpoints(path: str) -> list[str]:
    """Files to load from a checkpoint directory.

    Shards of one format are merged. Safetensors win when a directory also
    holds a pickle copy of the same weights. Two non-safetensors formats are
    refused so ``dict.update`` cannot silently overlay keys.
    """
    grouped: dict[str, list[str]] = {}
    for suffix in _CHECKPOINT_SUFFIXES:
        matches = sorted(glob.glob(os.path.join(path, f"*{suffix}")))
        if matches:
            grouped[suffix] = matches
    if not grouped:
        raise FileNotFoundError(f"No checkpoint files found under {path}")
    if ".safetensors" in grouped:
        return grouped[".safetensors"]
    if len(grouped) > 1:
        kinds = ", ".join(sorted(grouped))
        raise ValueError(
            f"Mixed checkpoint formats under {path} ({kinds}). "
            "Keep a single format, or provide safetensors shards."
        )
    return next(iter(grouped.values()))


def _merge_state_dicts(parts: list[tuple[str, dict[str, torch.Tensor]]]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for candidate, part in parts:
        overlap = sorted(set(merged).intersection(part))
        if overlap:
            sample = ", ".join(overlap[:3])
            raise ValueError(f"Checkpoint {candidate} repeats keys already loaded ({sample})")
        merged.update(part)
    return merged


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if os.path.isdir(path):
        candidates = _directory_checkpoints(path)
        return _merge_state_dicts([(candidate, load_state_dict(candidate)) for candidate in candidates])
    if path.endswith(".safetensors"):
        return dict(safe_load_file(path))
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "module"):
            value = obj.get(key)
            if isinstance(value, dict) and value and all(torch.is_tensor(item) for item in value.values()):
                obj = value
                break
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")
    return obj


def strip_prefixes(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        stripped = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix) :]
                    changed = True
        out[stripped] = value
    return out


def load_matching(module: torch.nn.Module, path: str, *, prefixes: tuple[str, ...] = ()) -> tuple[int, int]:
    if not path:
        return 0, 0
    state = strip_prefixes(load_state_dict(path), prefixes)
    result = module.load_state_dict(state, strict=False)
    return len(result.missing_keys), len(result.unexpected_keys)
