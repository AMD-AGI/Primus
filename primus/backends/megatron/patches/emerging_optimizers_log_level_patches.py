###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Raise the ``emerging_optimizers`` (absl) logger to INFO.

The Muon path pulls in ``emerging_optimizers``, which logs via ``absl`` (``from
absl import logging``). Its Newton-Schulz helper emits a per-call ``DEBUG`` line
for the coefficient schedule (``muon_utils.get_coefficient_iterator``)::

    Iterating through 10 steps with cycle mode.
    Coefficient sets: [(3.4445, -4.775, 2.0315), ..., (2.0, -1.5, 0.5)]

Because Muon orthogonalizes every matrix parameter on every optimizer step, this
fires repeatedly and floods the logs.

All ``absl`` logging records flow through the single stdlib logger named
``"absl"`` (Primus routes stdlib logging into loguru via an InterceptHandler on
the root logger, with ``root`` at NOTSET, so these DEBUG records are emitted).
Raising the ``"absl"`` logger to INFO drops the DEBUG spam at the source while
keeping INFO and above. We do it in the ``before_train`` phase, which runs after
the optimizer is built but before the first training step.

Set ``PRIMUS_VERBOSE_EMERGING_OPTIMIZERS=1`` to keep the full DEBUG trace.
"""

import logging
import os

from primus.core.patches import PatchContext, register_patch
from primus.modules.module_utils import log_rank_0


@register_patch(
    "megatron.emerging_optimizers.log_level",
    backend="megatron",
    phase="before_train",
    description=(
        "Raise the 'absl' logger (used by emerging_optimizers) to INFO so the "
        "Muon Newton-Schulz coefficient DEBUG lines do not flood the logs."
    ),
    condition=lambda ctx: os.environ.get("PRIMUS_VERBOSE_EMERGING_OPTIMIZERS", "0") != "1",
)
def patch_emerging_optimizers_log_level(ctx: PatchContext):
    """Set the absl logger threshold to INFO. Idempotent / best-effort."""
    del ctx
    try:
        logging.getLogger("absl").setLevel(logging.INFO)
    except Exception as exc:  # never block training over a logging tweak.
        log_rank_0(
            f"[Patch:megatron.emerging_optimizers.log_level] could not set " f"'absl' logger level: {exc!r}"
        )
        return

    log_rank_0("[Patch:megatron.emerging_optimizers.log_level] 'absl' logger raised to INFO.")
