###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText train.py Patches

Replaces ``MaxText.train.train_loop``, ``initialize``, and ``run`` with Primus
implementations that add:

- ``jax.block_until_ready`` after each training step for deterministic timing.
- A host-synchronisation barrier before the training loop to prevent collective
  operation timeouts.
- Factored initialisation (``setup_maxtext_runtime`` / ``post_initialize``) so
  the config can be built in-memory without a YAML round-trip.
- Error logging with traceback in ``run``.
"""

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0


@register_patch(
    patch_id="maxtext.train",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText train functions with Primus implementations (Dec version)",
    condition=lambda ctx: True,
    backend_versions=["0.1.1"],
)
def patch_train(ctx: PatchContext) -> None:
    """
    Monkey-patch ``MaxText.train`` with Primus versions of:
    - ``train_loop``  — adds ``jax.block_until_ready`` and pre-loop barrier
    - ``initialize``  — factored into ``setup_maxtext_runtime`` + ``post_initialize``
    - ``run``         — adds error logging with traceback
    """
    log_rank_0("[Patch:maxtext.train] Patching MaxText train module...")

    import MaxText.train as orig_train

    from primus.backends.maxtext.train import initialize, run, train_loop

    orig_train.train_loop = train_loop
    orig_train.initialize = initialize
    orig_train.run = run

    warning_rank_0("[Patch:maxtext.train] MaxText train module patched successfully.")


@register_patch(
    patch_id="maxtext.train.legacy",
    backend="maxtext",
    phase="setup",
    description="Replace MaxText train functions with Primus implementations (Aug version)",
    condition=lambda ctx: True,
    backend_versions=["2025.*"],
)
def patch_train_legacy(ctx: PatchContext) -> None:
    """
      Monkey-patch ``MaxText.train`` with Primus versions of:

    - ``train_loop``  — adds ``jax.block_until_ready`` and pre-loop barrier
    """
    log_rank_0("[Patch:maxtext.train] Patching MaxText train module...")

    import MaxText.train as orig_train

    from primus.backends.maxtext.legacy.train import (
        initialize,
        run,
        train_loop,
        validate_train_config,
    )

    orig_train.train_loop = train_loop
    orig_train.initialize = initialize
    orig_train.validate_train_config = validate_train_config
    orig_train.run = run

    warning_rank_0("[Patch:maxtext.train] MaxText train module patched successfully.")
