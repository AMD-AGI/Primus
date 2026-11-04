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


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    if os.path.isdir(path):
        candidates = []
        for pattern in ("*.safetensors", "*.pth", "*.pt", "*.bin", "*.ckpt"):
            candidates.extend(sorted(glob.glob(os.path.join(path, pattern))))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint files found under {path}")
        merged: dict[str, torch.Tensor] = {}
        for candidate in candidates:
            merged.update(load_state_dict(candidate))
        return merged
    if path.endswith(".safetensors"):
        return dict(safe_load_file(path))
    obj = torch.load(path, map_location="cpu")
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
