###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
Distributed Tensor Checkpointing (DTCP) save / load utilities.

Handles FSDP2 sharded checkpoints without gathering to rank 0.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, load, save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from primus.backends.diffusion.utils.log import logger

from .mesh import _ensure_process_group

_META_FILENAME = "meta.json"


def save_checkpoint_dtcp(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    epoch: int,
    step: int,
    additional_data: Dict[str, Any] = None,
    *,
    model_state_options: Optional[StateDictOptions] = None,
    optim_state_options: Optional[StateDictOptions] = None,
):
    """Save checkpoint using DTCP (sharded, no gather to rank 0)."""
    os.makedirs(path, exist_ok=True)
    _ensure_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    model_state_options = model_state_options or StateDictOptions(full_state_dict=False)
    optim_state_options = optim_state_options or StateDictOptions(full_state_dict=False)

    model_state = get_model_state_dict(model, options=model_state_options)
    optim_state = None
    if optimizer is not None:
        optim_state = get_optimizer_state_dict(model, optimizer, options=optim_state_options)

    meta = {"epoch": epoch, "step": step, **(additional_data or {})}
    state_dict = {
        "model": model_state,
        **({"optimizer": optim_state} if optim_state is not None else {}),
        "meta": meta,
    }

    save(state_dict, FileSystemWriter(path))
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        with open(os.path.join(path, _META_FILENAME), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        logger.info(f"Saved DTCP checkpoint to {path}")


def load_checkpoint_dtcp(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    *,
    model_state_options: Optional[StateDictOptions] = None,
    optim_state_options: Optional[StateDictOptions] = None,
) -> Dict[str, Any]:
    """Load checkpoint using DTCP. Updates model/optimizer in-place. Returns metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    _ensure_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    model_state_options = model_state_options or StateDictOptions(full_state_dict=False)
    optim_state_options = optim_state_options or StateDictOptions(full_state_dict=False)

    model_state = get_model_state_dict(model, options=model_state_options)
    optim_state = None
    if optimizer is not None:
        optim_state = get_optimizer_state_dict(model, optimizer, options=optim_state_options)

    state_dict = {
        "model": model_state,
        **({"optimizer": optim_state} if optim_state is not None else {}),
        "meta": {},
    }

    load(state_dict, FileSystemReader(path))

    set_model_state_dict(model, model_state, options=model_state_options)
    if optimizer is not None and optim_state is not None:
        set_optimizer_state_dict(model, optimizer, optim_state, options=optim_state_options)

    meta = state_dict.get("meta", {})
    if not meta:
        meta_path = os.path.join(path, _META_FILENAME)
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
    return meta
