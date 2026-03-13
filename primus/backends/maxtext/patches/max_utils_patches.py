###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText max_utils Patches

- Wraps ``jax.distributed.initialize`` to inject ``heartbeat_timeout_seconds``
  from the Primus config (Dec version only — older JAX may not support it).
- Replaces ``print_system_information`` and ``save_device_information`` with
  Primus implementations that provide richer device logging and
  local-filesystem support for device-info JSON dumps.
"""

import functools

import jax

from primus.core.patches import PatchContext, register_patch
from primus.core.patches.context import get_param
from primus.modules.module_utils import log_rank_0, warning_rank_0


def _patch_print_and_save(orig_max_utils) -> None:
    """Shared helper: replace print_system_information and save_device_information."""
    from primus.backends.maxtext.max_utils import (
        print_system_information,
        save_device_information,
    )

    orig_max_utils.print_system_information = print_system_information
    orig_max_utils.save_device_information = save_device_information


@register_patch(
    patch_id="maxtext.max_utils",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText max_utils helpers with Primus implementations (Dec version)",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
)
def patch_max_utils(ctx: PatchContext) -> None:
    """
    1. Wrap ``jax.distributed.initialize`` so every call automatically includes
       ``heartbeat_timeout_seconds`` (Primus-specific config field).
    2. Monkey-patch ``print_system_information`` and ``save_device_information``.
    """
    log_rank_0("[Patch:maxtext.max_utils] Patching MaxText max_utils...")

    # --- Wrap jax.distributed.initialize with heartbeat_timeout_seconds ---
    heartbeat_timeout = get_param(ctx, "jax_distributed_heartbeat_timeout_seconds", 100)

    _orig_jax_init = jax.distributed.initialize

    @functools.wraps(_orig_jax_init)
    def _jax_init_with_heartbeat(*args, **kwargs):
        kwargs.setdefault("heartbeat_timeout_seconds", heartbeat_timeout)
        return _orig_jax_init(*args, **kwargs)

    jax.distributed.initialize = _jax_init_with_heartbeat
    log_rank_0(
        f"[Patch:maxtext.max_utils] jax.distributed.initialize wrapped "
        f"(heartbeat_timeout_seconds={heartbeat_timeout})"
    )

    # --- Replace print_system_information / save_device_information ---
    import MaxText.max_utils as orig_max_utils

    _patch_print_and_save(orig_max_utils)

    warning_rank_0("[Patch:maxtext.max_utils] max_utils patched successfully.")


@register_patch(
    patch_id="maxtext.max_utils.legacy",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText max_utils helpers with Primus implementations (Aug version)",
    condition=lambda ctx: True,
    backend_versions=["2025.*"],
)
def patch_max_utils_legacy(ctx: PatchContext) -> None:
    """
    Monkey-patch ``print_system_information`` and ``save_device_information``.

    The ``jax.distributed.initialize`` wrapping is NOT applied here because
    the older JAX version bundled with Aug-dated MaxText may not support
    the ``heartbeat_timeout_seconds`` parameter.
    """
    log_rank_0("[Patch:maxtext.max_utils.legacy] Patching MaxText max_utils (legacy)...")

    import MaxText.max_utils as orig_max_utils

    _patch_print_and_save(orig_max_utils)

    warning_rank_0("[Patch:maxtext.max_utils.legacy] max_utils patched successfully.")
