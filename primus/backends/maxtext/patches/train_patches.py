###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText train.py Patches

Replaces ``MaxText.train.initialize`` with a thin wrapper that forwards
``**kwargs`` to ``pyconfig.initialize``, allowing Primus to inject CLI
overrides without copying the entire upstream function.
"""

import functools
from typing import Any, Sequence

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0


def _resolve_train_and_pyconfig():
    """Resolve MaxText's train module and pyconfig across MaxText versions.

    MaxText v26.4+ exposes the training loop at
    ``maxtext.trainers.pre_train.train`` and config at ``maxtext.configs.pyconfig``.
    MaxText v26.3 and earlier expose ``MaxText.train`` and ``MaxText.pyconfig``.
    """
    try:
        from maxtext.configs import pyconfig
        from maxtext.trainers.pre_train import train as orig_train

        return orig_train, pyconfig
    except ImportError:
        import MaxText.train as orig_train
        from MaxText import pyconfig

        return orig_train, pyconfig


@register_patch(
    patch_id="maxtext.train",
    backend="maxtext",
    phase="setup",
    description="Wrap MaxText initialize to forward **kwargs to pyconfig.initialize",
    condition=lambda ctx: True,
)
def patch_train(ctx: PatchContext) -> None:
    """
    Monkey-patch MaxText's ``train.initialize`` so that callers can pass
    ``**kwargs`` which are transparently forwarded to ``pyconfig.initialize``.
    """
    log_rank_0("[Patch:maxtext.train] Patching MaxText train module...")

    try:
        orig_train, pyconfig = _resolve_train_and_pyconfig()
    except ImportError as e:
        warning_rank_0(
            f"[Patch:maxtext.train] Could not locate MaxText train/pyconfig module; "
            f"skipping override-forwarding patch: {e}"
        )
        return

    _upstream_initialize = orig_train.initialize

    def initialize(argv: Sequence[str], **kwargs) -> tuple[Any, Any, Any]:
        if not kwargs:
            return _upstream_initialize(argv)
        _orig = pyconfig.initialize
        pyconfig.initialize = functools.partial(_orig, **kwargs)
        try:
            return _upstream_initialize(argv)
        finally:
            pyconfig.initialize = _orig

    orig_train.initialize = initialize

    warning_rank_0("[Patch:maxtext.train] MaxText train module patched successfully.")
