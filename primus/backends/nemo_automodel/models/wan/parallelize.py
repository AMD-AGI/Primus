###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Honour selective AC and ``reshard_after_forward`` on the Wan path.

WHY:
  AutoModel's ``WanModelParallelizer`` drops two config values, silently:

  1. ``activation_checkpointing`` is a bare truthiness test, so ``"selective"``
     (a non-empty string) runs the full-AC branch.
  2. ``reshard_after_forward`` lands in ``**kwargs`` and is never read: the
     sharding helper is called positionally, one argument short of it, so the
     per-layer heuristic always applies.

  The config echo shows the requested values either way.

WHAT (NO diffusers / AutoModel fork):
  Subclasses upstream's Wan sidecar (``distributed/sidecar.py``) so its tensor-
  parallel plan and root sharding stay authoritative, and fills in only what it
  drops:

  * selective AC is applied here through AutoModel's own selective helper, and
    the parent then sees AC off; ``full`` and off pass through unchanged.
  * ``reshard_after_forward`` is supplied to the parent's sharding call when the
    parent does not pass it. Once upstream passes it, this finds nothing to
    fill, logs that, and can be deleted.
"""
from __future__ import annotations

import contextlib
import functools
import logging
import sys

from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac
from primus.backends.nemo_automodel.distributed import sidecar

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusWanParallelize]"
_WAN_MODEL_NAME = "WanTransformer3DModel"
_WAN_BLOCK_ATTRS = ("blocks",)
_SHARDING_FN = "apply_fsdp2_sharding_recursively"
# Position of reshard_after_forward in the sharding helper's signature.
_RESHARD_POSITION = 7


@contextlib.contextmanager
def _forwarding_reshard(base, value):
    """Supply ``reshard_after_forward`` to the parent's sharding call if it omits it."""
    module = sys.modules.get(base._apply.__module__)
    original = getattr(module, _SHARDING_FN, None)
    if value is None or original is None:
        yield
        return

    seen = {"called": False, "filled": False}

    @functools.wraps(original)
    def _with_reshard(*args, **kwargs):
        seen["called"] = True
        if len(args) <= _RESHARD_POSITION and "reshard_after_forward" not in kwargs:
            kwargs["reshard_after_forward"] = value
            seen["filled"] = True
        return original(*args, **kwargs)

    setattr(module, _SHARDING_FN, _with_reshard)
    try:
        yield
    finally:
        setattr(module, _SHARDING_FN, original)

    if seen["filled"]:
        logger.info(
            "%s forwarded reshard_after_forward=%s, which %s drops", _LOG_PREFIX, value, base.__name__
        )
    elif seen["called"]:
        logger.info(
            "%s %s passes reshard_after_forward itself; nothing to forward", _LOG_PREFIX, base.__name__
        )
    else:
        logger.warning(
            "%s %s no longer calls %s, so reshard_after_forward=%s was not forwarded; "
            "re-check this repair against the AutoModel pin.",
            _LOG_PREFIX,
            base.__name__,
            _SHARDING_FN,
            value,
        )


def _build(base):
    class PrimusWanParallelizer(base):
        def _apply(self, model, *args, **kwargs):
            value = ac.normalize(kwargs.get("activation_checkpointing", False))
            helpers = ac.upstream_helpers()
            if helpers.is_selective_activation_checkpointing(value):
                ac.apply(
                    helpers,
                    model,
                    _WAN_BLOCK_ATTRS,
                    value,
                    enable_compile=bool(kwargs.get("enable_compile", False)),
                    log_prefix=_LOG_PREFIX,
                )
                value = False
            kwargs["activation_checkpointing"] = value
            with _forwarding_reshard(base, kwargs.get("reshard_after_forward")):
                return super()._apply(model, *args, **kwargs)

    return PrimusWanParallelizer


def install() -> bool:
    """Install the Wan sidecar. Idempotent; edits no AutoModel source."""
    return sidecar.install(_WAN_MODEL_NAME, _build, log_prefix=_LOG_PREFIX)
