###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Make ``fsdp.reshard_after_forward`` reach PyTorch on the diffusion path.

WHY:
  ``reshard_after_forward`` is a real ``FSDP2Config`` field and the diffusion recipe
  parses it correctly from YAML into ``manager_args``. But
  ``_create_parallel_manager`` rebuilds ``FSDP2Config`` from an **explicit keyword
  whitelist** that does not include it, so the value is present in ``manager_args``
  and simply never read back. ``FSDP2Manager.reshard_after_forward`` is therefore
  always ``None``.

  With ``None``, the per-layer heuristic in ``apply_fsdp2_sharding_recursively``
  takes over and reshards every transformer block except the last, re-all-gathering
  each one during backward. That is ZeRO-3 communication volume where ZeRO-2 was
  asked for. The ``reshard_after_forward=False`` on the root ``fully_shard()`` in
  the parallelization strategies only covers the root unit -- embeddings, norms,
  adaLN heads -- not the blocks, which hold nearly all the parameters.

  This affects **every** model on the shared diffusion path, and is worth reporting
  upstream.

WHAT:
  Wraps ``_create_parallel_manager`` and re-applies the key the whitelist drops. It
  **repairs rather than invents**: the value comes from ``manager_args``, which came
  from YAML. Applied unconditionally with no env gate, because it is
  behaviour-neutral when the YAML omits the key (``None`` stays ``None`` and today's
  heuristic applies), so there is nothing to gate.

  Note this is only half the fix. A parallelization strategy must also forward the
  value into ``apply_fsdp2_sharding_recursively``; see the per-model
  ``parallelize`` modules. **Fixing either half alone changes nothing**, which is
  why the strategy side reads the provenance globals below to check its partner is
  present.

FORWARD COMPATIBILITY:
  If AutoModel later adds ``reshard_after_forward`` to the whitelist, this wrapper
  writes the value the manager already holds -- both read the same ``manager_args``
  -- so it degrades to a no-op rather than conflicting. Do not remove it on a
  submodule bump on the assumption that it now conflicts.
"""
from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_primus_reshard_patch"

# Provenance for the strategy-side guard. ``patch_installed`` says the wrapper is in
# place; ``applied_reshard_after_forward`` is the value it last wrote. Read these
# through a module reference (``import fsdp2_reshard``), never a ``from`` import:
# a from-import binds the value at import time, when it is still False/None, and
# never sees install() update it -- so the guard would fire on every healthy run.
patch_installed = False
applied_reshard_after_forward = None


def install() -> bool:
    """Patch ``_create_parallel_manager`` to re-apply ``reshard_after_forward``."""
    global patch_installed

    # Imported inside install() so that importing this module does not drag in
    # AutoModel; the unit tests rely on that to run without the submodule.
    import nemo_automodel._diffusers.auto_diffusion_pipeline as adp

    target = getattr(adp, "_create_parallel_manager", None)
    if target is None or not callable(target):
        # An explicit raise, not an assert: asserts are stripped under `python -O`.
        raise RuntimeError(
            "auto_diffusion_pipeline._create_parallel_manager is missing; the AutoModel "
            "layout changed and the reshard_after_forward repair cannot apply."
        )
    if getattr(target, _SENTINEL, False):
        patch_installed = True
        return True  # already installed

    @functools.wraps(target)
    def _patched(*args, **kwargs):
        global applied_reshard_after_forward

        # Locate the dict defensively: a rename upstream should degrade to today's
        # behaviour, not raise a KeyError from inside the wrapper at parallelize
        # time, long after install.
        manager_args = args[0] if args else kwargs.get("manager_args")
        manager = target(*args, **kwargs)

        if isinstance(manager_args, dict) and manager_args.get("_manager_type", "fsdp2") == "fsdp2":
            # The callee copies manager_args and pops _manager_type from the copy, so
            # the caller's dict we inspect here still carries both keys.
            value = manager_args.get("reshard_after_forward")
            manager.reshard_after_forward = value
            applied_reshard_after_forward = value
            logger.info("[PrimusFSDP2Reshard] applied reshard_after_forward=%s", value)

        return manager

    # The sentinel is what makes this idempotent. Without it a second install() would
    # capture the already-patched function as `target` and nest wrappers.
    setattr(_patched, _SENTINEL, True)
    adp._create_parallel_manager = _patched
    patch_installed = True
    return True
