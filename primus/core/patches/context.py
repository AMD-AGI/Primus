###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Patch Context and Phase Management.

This module defines the execution context for patches and a list of valid phases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# -----------------------------------------------------------------------------
# Patch Execution Phases
# -----------------------------------------------------------------------------

# Core lifecycle phases used by Primus
PHASES = [
    "setup",  # Before building args or trainer construction
    "build_args",  # During backend argument construction
    "before_train",  # Right before training starts
    # Additional phases could be added in the future:
    # "after_train",
    # "before_eval",
    # "after_eval",
]


# -----------------------------------------------------------------------------
# Patch Context
# -----------------------------------------------------------------------------


@dataclass
class PatchContext:
    """
    Patch execution context object passed to all patches.

    Attributes:
        backend:
            Backend name (e.g., "megatron", "torchtitan").
        phase:
            Training lifecycle phase (e.g., "setup", "build_args", "before_train").
        backend_version:
            Version or commit hash of the backend (e.g., "0.8.0", "commit:abc123").
        primus_version:
            Primus version string (used for Primus-side compatibility patches).
        model_name:
            Model preset name (e.g., "llama3_70B", "deepseek_v3").
        module_name:
            Primus module name (e.g., "pre_trainer", "sft_trainer").
        platform:
            Hardware platform identifier (e.g., "MI300X", "MI355X").
        extra:
            Arbitrary additional information passed by the caller
            (e.g., {"args": megatron_args, "config": module_config}).
    """

    backend: str
    phase: str
    backend_version: Optional[str] = None
    primus_version: Optional[str] = None
    model_name: Optional[str] = None
    module_name: Optional[str] = None
    platform: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_args(ctx: PatchContext) -> Any:
    """
    Get module configuration parameters from patch context.

    This is a convenience helper that extracts module_config.params from the
    context's extra dict. Raises AssertionError if module_config or params
    is not available.

    Args:
        ctx: The patch context

    Returns:
        module_config.params (SimpleNamespace)

    Raises:
        AssertionError: If module_config or params is not available in context

    Example:
        @register_patch(...)
        def my_patch(ctx: PatchContext):
            args = get_args(ctx)

            if getattr(args, "my_flag", False):
                # Apply patch...
    """
    module_config = ctx.extra.get("module_config")
    assert module_config is not None, "module_config is required in patch context"
    params = getattr(module_config, "params", None)
    assert params is not None, "module_config.params is required in patch context"
    return params
