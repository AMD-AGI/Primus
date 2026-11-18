###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib
import importlib.util

from primus.core.patches import PatchContext, get_args
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

# Megatron-core >= 0.19 runs the per-layer quantization contexts from these modules
# (full recompute goes through megatron.core.recompute.checkpointed_forward), and each
# binds get_fp8_context / get_fp4_context by name at import.
_LAYER_LOOP_MODULES = (
    "megatron.core.recompute",
    "megatron.core.models.hybrid.hybrid_block",
)


def optional_layer_loop_modules() -> list:
    """The ``_LAYER_LOOP_MODULES`` present in this Megatron."""
    modules = []
    for name in _LAYER_LOOP_MODULES:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue
    return modules


def is_primus_turbo_can_patch(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo is enabled and can be used.

    Requires:
      - primus_turbo package is installed
      - enable_primus_turbo == True

    Tensor parallelism is intentionally not part of this gate. Only the Turbo
    linear / grouped-linear layers need ``tensor_model_parallel_size == 1``, and
    each enforces that in its own ``__init__``. Checking it here also disabled
    the TP-agnostic features (attention, norm, DeepEP, FP8 context, ...).
    """
    args = get_args(ctx)
    if not bool(getattr(args, "enable_primus_turbo", False)):
        return False

    if importlib.util.find_spec("primus_turbo") is None:
        warning_rank_0("[Patch:megatron.turbo] primus_turbo not found, use TE backend...")
        return False

    log_rank_0("[Patch:megatron.turbo] Primus Turbo enabled; using Primus Turbo backend...")
    return True
