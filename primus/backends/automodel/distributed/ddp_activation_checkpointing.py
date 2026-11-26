###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Forward ``ddp.activation_checkpointing`` to AutoModel's DDP path.

WHY:
  AutoModel's distributed-config parser pops ``activation_checkpointing`` into
  the ``MeshContext`` and builds ``DDPConfig`` without it, while
  ``parallelize_ddp`` reads the flag from ``DDPConfig`` only. So under DDP the
  setting is parsed, echoed in the config dump, and never applied: the run
  trains with no checkpointing at all. FSDP2 is unaffected because its path
  reads the flag from the ``MeshContext``.

WHAT (NO AutoModel fork):
  Wraps the DDP dispatch in ``model_parallelizer`` and copies the
  ``MeshContext`` value into ``DDPConfig`` when it is missing there. Once
  upstream carries the flag itself the wrapper finds nothing to copy and does
  nothing, so it can be deleted without a behaviour change.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusDDPActivationCheckpointing]"
_PATCHED_ATTR = "_primus_ddp_ac_forwarded"


def forward_flag(mesh_context) -> bool:
    """Copy the requested AC mode into ``DDPConfig``. True if a copy was needed."""
    config = mesh_context.strategy_config
    requested = mesh_context.activation_checkpointing
    if not requested or getattr(config, "activation_checkpointing", None):
        return False
    config.activation_checkpointing = requested
    return True


def install() -> bool:
    """Wrap AutoModel's DDP dispatch. Idempotent; False if the hook is gone."""
    from nemo_automodel.components.distributed import model_parallelizer as mp

    original = getattr(mp, "_parallelize_ddp", None)
    if original is None:
        logger.warning(
            "%s model_parallelizer._parallelize_ddp is missing; the AutoModel layout "
            "changed, so ddp.activation_checkpointing is not forwarded. Check whether "
            "upstream now applies it natively.",
            _LOG_PREFIX,
        )
        return False
    if getattr(original, _PATCHED_ATTR, False):
        return True

    @functools.wraps(original)
    def _parallelize_ddp(model, mesh_context):
        if forward_flag(mesh_context):
            logger.info(
                "%s forwarded activation_checkpointing=%s into DDPConfig, which the "
                "upstream parser left unset.",
                _LOG_PREFIX,
                mesh_context.activation_checkpointing,
            )
        return original(model, mesh_context)

    setattr(_parallelize_ddp, _PATCHED_ATTR, True)
    mp._parallelize_ddp = _parallelize_ddp
    return True
