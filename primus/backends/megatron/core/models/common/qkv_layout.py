###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Megatron ``linear_qkv`` row layout for checkpoint mappings (no Megatron or converter imports):
shared by the native HF converters (``native_convert_common``) and model weight mappings
(``inkling_hf_mapping``)."""

import torch

__all__ = ["fuse_qkv"]


def fuse_qkv(q, k, v, *, heads: int, groups: int, head_dim: int) -> torch.Tensor:
    """HF q / k / v projection weights -> mcore ``linear_qkv``: per query group, its
    ``heads / groups`` query heads, then its key head, then its value head (no RoPE
    permutation; keyword-only). Shapes are checked:
    ``q`` is ``[heads * head_dim, hidden]``, ``k`` / ``v`` are ``[groups * head_dim, hidden]``."""
    if groups <= 0 or heads <= 0 or heads % groups:
        raise ValueError(f"heads ({heads}) must be a positive multiple of groups ({groups})")
    hidden = q.shape[-1]
    for name, w, rows in (
        ("q", q, heads * head_dim),
        ("k", k, groups * head_dim),
        ("v", v, groups * head_dim),
    ):
        if tuple(w.shape) != (rows, hidden):
            raise ValueError(
                f"{name} projection weight has shape {tuple(w.shape)}, expected ({rows}, {hidden}) for "
                f"heads={heads}, groups={groups}, head_dim={head_dim}"
            )
    per = heads // groups
    parts = (
        q.reshape(groups, per * head_dim, hidden),
        k.reshape(groups, head_dim, hidden),
        v.reshape(groups, head_dim, hidden),
    )
    return torch.cat(parts, dim=1).reshape(-1, hidden)
