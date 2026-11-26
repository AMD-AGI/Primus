###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText JAX distributed heartbeat timeout patch (ROCm/maxtext 6d3eab29e).

Fixes "stop sending heartbeats" failures (e.g. during slow NFS checkpoint
restores) by forwarding ``jax_distributed_heartbeat_timeout_seconds`` to every
``jax.distributed.initialize`` call MaxText makes. All of those calls are made
from ``max_utils.maybe_initialize_jax_distributed_system`` (directly or via
``initialize_jax_for_{gpu,cpu,tpu_with_emergency_checkpointing}``), so the
wrapper scopes the extra kwarg to that call.
"""

import functools
import inspect

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_PREFIX = "[Patch:maxtext.distributed.heartbeat_timeout]"
_KEY = "jax_distributed_heartbeat_timeout_seconds"
_DEFAULT_TIMEOUT = 100


@register_patch(
    patch_id="maxtext.distributed.heartbeat_timeout",
    backend="maxtext",
    phase="setup",
    description="Forward jax_distributed_heartbeat_timeout_seconds to jax.distributed.initialize",
    condition=lambda ctx: True,
)
def patch_heartbeat_timeout(ctx: PatchContext) -> None:
    del ctx
    try:
        import jax
        from maxtext.utils import max_utils
    except ImportError as e:
        warning_rank_0(f"{_PREFIX} maxtext.utils.max_utils not importable; skipped: {e}")
        return

    if "heartbeat_timeout_seconds" in inspect.getsource(max_utils):
        log_rank_0(f"{_PREFIX} MaxText already passes heartbeat_timeout_seconds; not patched.")
        return

    orig = max_utils.maybe_initialize_jax_distributed_system

    @functools.wraps(orig)
    def maybe_initialize_jax_distributed_system(raw_keys, *args, **kwargs):
        timeout = raw_keys.get(_KEY, _DEFAULT_TIMEOUT)
        orig_initialize = jax.distributed.initialize
        jax.distributed.initialize = functools.partial(orig_initialize, heartbeat_timeout_seconds=timeout)
        try:
            return orig(raw_keys, *args, **kwargs)
        finally:
            jax.distributed.initialize = orig_initialize

    max_utils.maybe_initialize_jax_distributed_system = maybe_initialize_jax_distributed_system
    log_rank_0(f"{_PREFIX} jax.distributed.initialize now receives heartbeat_timeout_seconds.")
