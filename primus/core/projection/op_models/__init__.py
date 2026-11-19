###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Pluggable analytical op models; see :mod:`.base` for the contract."""

from .base import (
    SLOT_ATTENTION,
    SLOT_ATTENTION_EXTRA,
    SLOT_LAYER_EXTRA,
    SLOT_MLP,
    SLOTS,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
    full_num_layers,
    interleave_fraction,
    is_active,
)
from .registry import (
    active_op_models,
    discover_op_models,
    discovery_errors,
    load_op_model_file,
    register_op_model,
    registered_op_models,
    unregister_op_model,
)

__all__ = [
    "SLOT_ATTENTION",
    "SLOT_ATTENTION_EXTRA",
    "SLOT_LAYER_EXTRA",
    "SLOT_MLP",
    "SLOTS",
    "OpContext",
    "OpCost",
    "OpModel",
    "cfg_value",
    "full_num_layers",
    "interleave_fraction",
    "is_active",
    "active_op_models",
    "discover_op_models",
    "discovery_errors",
    "load_op_model_file",
    "register_op_model",
    "registered_op_models",
    "unregister_op_model",
]
