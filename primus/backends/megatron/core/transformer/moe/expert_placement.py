###############################################################################
# Copyright (c) 2026, Zyphra Technologies Inc. All rights reserved.
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Custom global-expert-to-EP-rank placement for the all-to-all MoE dispatcher.

Ported from ROCm/Megatron-LM PR #166 ("co-activation expert placement"). The
placement is a permutation of ``range(num_moe_experts)`` in rank-major slot
order: reshaped to ``[ep_size, num_local_experts]``, row ``r`` lists the global
expert ids owned by EP rank ``r`` (in local-slot order). The default
(contiguous) placement is ``range(num_experts)``.

A placement file is JSON of the form::

    {"num_experts": N, "default": [perm of 0..N-1], "layers": {"<layer_number>": [perm]}}

``default`` applies to every MoE layer; per-layer entries in ``layers`` (keyed by
the global layer number) override it. Files are produced offline by
``tools/moe_expert_placement/build_expert_placement.py`` from a route capture.
"""

from __future__ import annotations

import functools
import json
from typing import Dict, List, Optional


def validate_expert_placement_perm(perm, num_experts: int, ep_size: int, ctx: str) -> List[int]:
    """Validate one placement array: a permutation of ``range(num_experts)`` that
    reshapes cleanly to ``[ep_size, num_experts // ep_size]``."""
    if not isinstance(perm, (list, tuple)):
        raise ValueError(f"moe_expert_placement {ctx}: expected a list, got {type(perm)}.")
    perm = [int(x) for x in perm]
    if len(perm) != num_experts:
        raise ValueError(f"moe_expert_placement {ctx}: expected {num_experts} entries, got {len(perm)}.")
    if sorted(perm) != list(range(num_experts)):
        raise ValueError(
            f"moe_expert_placement {ctx}: entries must be a permutation of 0..{num_experts - 1}."
        )
    if num_experts % ep_size != 0:
        raise ValueError(
            f"moe_expert_placement {ctx}: num_experts ({num_experts}) must be divisible by "
            f"expert_model_parallel_size ({ep_size})."
        )
    return perm


def parse_moe_expert_placement(path: str, num_experts: Optional[int], ep_size: int) -> Dict:
    """Load and validate a placement JSON file.

    Returns ``{'default': list|None, 'layers': {int: list}}`` with validated permutations.
    """
    if num_experts is None:
        raise ValueError("moe_expert_placement requires num_experts to be set.")
    with open(path, "r") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("moe_expert_placement JSON must be an object.")
    if "num_experts" in raw and int(raw["num_experts"]) != num_experts:
        raise ValueError(
            f"moe_expert_placement num_experts ({raw['num_experts']}) does not match "
            f"num_experts ({num_experts})."
        )
    parsed: Dict = {"default": None, "layers": {}}
    if raw.get("default") is not None:
        parsed["default"] = validate_expert_placement_perm(raw["default"], num_experts, ep_size, "'default'")
    for layer_key, perm in (raw.get("layers") or {}).items():
        layer_number = int(layer_key)
        parsed["layers"][layer_number] = validate_expert_placement_perm(
            perm, num_experts, ep_size, f"layer {layer_number}"
        )
    if parsed["default"] is None and not parsed["layers"]:
        raise ValueError("moe_expert_placement JSON must define at least a 'default' or 'layers' entry.")
    return parsed


@functools.lru_cache(maxsize=None)
def load_moe_expert_placement(path: str, num_experts: Optional[int], ep_size: int) -> Dict:
    """Cached :func:`parse_moe_expert_placement`.

    The parsed table has integer layer keys, so it must not be stored on the Primus
    module params (``merge_namespace`` turns dicts into ``SimpleNamespace`` keyword
    arguments). Callers re-load it from the path instead; the result is shared, so
    treat it as read-only.
    """
    return parse_moe_expert_placement(path, num_experts, ep_size)


def get_expert_placement_order(config, layer_number, ep_size: int) -> List[int]:
    """Resolve the global-expert-to-EP-rank placement for a single MoE layer.

    Returns a list of length ``config.num_moe_experts`` giving global expert ids in
    rank-major slot order. ``config.moe_expert_placement`` (the parsed placement
    table, or None) may override the contiguous default with a per-layer
    permutation (keyed by ``layer_number``) or a global ``'default'`` permutation.
    """
    num_experts = config.num_moe_experts
    placement = getattr(config, "moe_expert_placement", None)
    if placement is None:
        return list(range(num_experts))

    order = None
    layers = placement.get("layers") or {}
    if layer_number is not None and layer_number in layers:
        order = layers[layer_number]
    elif placement.get("default") is not None:
        order = placement["default"]

    if order is None:
        # No entry for this layer -> fall back to the default contiguous placement.
        return list(range(num_experts))

    assert len(order) == num_experts, (
        f"moe_expert_placement for layer {layer_number} has {len(order)} entries, " f"expected {num_experts}."
    )
    assert (
        num_experts % ep_size == 0
    ), f"num_moe_experts ({num_experts}) must be divisible by expert parallel size ({ep_size})."
    assert sorted(order) == list(
        range(num_experts)
    ), f"moe_expert_placement for layer {layer_number} is not a permutation of 0..{num_experts - 1}."
    return list(order)


def resolve_global_expert_ids(local_expert_indices, num_local_experts: int, ep_rank: int) -> List[int]:
    """Global expert ids owned by this rank in local-slot order.

    With a custom placement ``local_expert_indices`` is given (and may be
    non-contiguous); ``None`` means the contiguous block
    ``[ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts)``.
    """
    if local_expert_indices is not None:
        assert len(local_expert_indices) == num_local_experts, (
            len(local_expert_indices),
            num_local_experts,
        )
        return list(local_expert_indices)
    offset = ep_rank * num_local_experts
    return [offset + i for i in range(num_local_experts)]


def compute_layer_expert_placement(config, layer_number, ep_size: int, ep_rank: int):
    """``(expert_order, local_expert_indices)`` for one MoE layer on one EP rank."""
    expert_order = get_expert_placement_order(config, layer_number, ep_size)
    num_local_experts = config.num_moe_experts // ep_size
    offset = ep_rank * num_local_experts
    local_expert_indices = [expert_order[offset + i] for i in range(num_local_experts)]
    return expert_order, local_expert_indices
