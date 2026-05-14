###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib.util

from primus.core.patches import PatchContext, get_args
from primus.modules.module_utils import log_rank_0, warning_rank_0


def _is_primus_turbo_enabled(ctx: PatchContext) -> bool:
    """
    Check if PrimusTurbo is enabled and can be used.

    Requires:
      - primus_turbo package is installed
      - tensor_model_parallel_size == 1
      - enable_primus_turbo == True
    """
    # Check if primus_turbo package is available
    args = get_args(ctx)
    enable_primus_turbo = bool(getattr(args, "enable_primus_turbo", False))

    if enable_primus_turbo:
        if importlib.util.find_spec("primus_turbo") is None:
            warning_rank_0(
                "[Patch:megatron.turbo.te_spec_provider] primus_turbo not found, use TE backend..."
            )
            return False

        log_rank_0(
            "[Patch:megatron.turbo.te_spec_provider] Primus Turbo enabled; using Primus Turbo backend..."
        )
        return True
    else:
        return False


def _is_tp_parallel_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    tensor_model_parallel_size = getattr(args, "tensor_model_parallel_size", 1)

    if tensor_model_parallel_size != 1:
        return True
    else:
        return False


def is_primus_turbo_can_patch(ctx: PatchContext) -> bool:

    enable_primus_turbo = _is_primus_turbo_enabled(ctx)

    if enable_primus_turbo:
        if _is_tp_parallel_enabled(ctx):
            warning_rank_0(
                "[Patch:megatron.turbo.te_spec_provider] Primus Turbo backend does not support TP > 1; Please use TE backend instead..."
            )
            return False

        return True
    else:
        return False
