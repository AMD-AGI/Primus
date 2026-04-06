###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve diffusion forward-step callables for Megatron-Bridge pretrain."""

from __future__ import annotations

import inspect
from typing import Any


def load_diffusion_forward_step(backend_args: Any) -> Any:
    """
    Instantiate the forward step used by megatron.bridge.training.pretrain for diffusion.

    Args:
        backend_args: Namespace with optional ``step_func`` (default ``flux_step``) and
            ``step_func_mode`` (for ``wan_step``, e.g. ``pretrain``).

    Returns:
        Callable or functor compatible with Megatron-Bridge pretrain.
    """
    step_key = (getattr(backend_args, "step_func", None) or "flux_step").lower()
    mode = getattr(backend_args, "step_func_mode", None) or "pretrain"

    if step_key == "flux_step":
        from primus.diffusion.models.flux.flux_step import FluxForwardStep

        return FluxForwardStep()

    if step_key == "wan_step":
        from primus.diffusion.models.wan.wan_step import WanForwardStep

        wan_cls = WanForwardStep
        if "mode" in inspect.signature(wan_cls.__init__).parameters:
            return wan_cls(mode=mode)
        return wan_cls()

    raise ValueError(
        f"Unsupported diffusion step_func '{step_key}'. "
        "Use 'flux_step' or 'wan_step' (optionally set step_func_mode for WAN)."
    )
