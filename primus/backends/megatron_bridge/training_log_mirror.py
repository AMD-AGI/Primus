###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Mirror Megatron-Bridge ``print_rank_last`` training summaries to rank 0.

``megatron.bridge.training.utils.train_utils.training_log`` prints a short
``Step Time`` line with ``print_rank_0`` (visible on rank 0) but emits the full
iteration line (loss, LR, batch size, …) only via ``print_rank_last``, which
prints on the **last** global rank only. When you tail a single rank-0 log
stream (common in direct / container setups), that line never appears.

We wrap ``megatron.bridge.utils.common_utils.print_rank_last`` so that, on
global rank 0 only, the same summary is also sent through Primus ``log_rank_0``.

We also rebind ``print_rank_last`` on ``eval`` and ``train_utils`` modules,
because they import it by value from ``common_utils``.

Disable with: ``export PRIMUS_MIRROR_MEGATRON_BRIDGE_TRAINING_LOG=0``
"""

from __future__ import annotations

import os


def _is_iteration_summary_line(message: str) -> bool:
    return (
        "iteration" in message
        and "consumed samples:" in message
        and "elapsed time per iteration" in message
    )


def _is_validation_log_line(message: str) -> bool:
    """Megatron-Bridge prints validation loss only via ``print_rank_last``; mirror to rank 0."""
    return "validation loss at" in message


def install_megatron_bridge_training_log_mirror_to_rank0() -> None:
    """Wrap ``print_rank_last`` once so rank 0 also logs the full training line."""
    raw = os.environ.get("PRIMUS_MIRROR_MEGATRON_BRIDGE_TRAINING_LOG", "1").strip().lower()
    if raw in ("0", "false", "no", "off"):
        return

    try:
        import megatron.bridge.utils.common_utils as cu
    except ImportError:
        return

    if getattr(cu.print_rank_last, "_primus_training_log_mirror", False):
        return

    _orig = cu.print_rank_last

    def print_rank_last_wrapped(message: str) -> None:
        if _is_iteration_summary_line(message) or _is_validation_log_line(message):
            try:
                from megatron.bridge.utils.common_utils import get_rank_safe
                from primus.modules.module_utils import log_rank_0

                if get_rank_safe() == 0:
                    log_rank_0(message)
            except Exception:
                pass
        _orig(message)

    setattr(print_rank_last_wrapped, "_primus_training_log_mirror", True)
    cu.print_rank_last = print_rank_last_wrapped

    # Megatron-Bridge does ``from ...common_utils import print_rank_last`` in eval.py and
    # train_utils.py — rebinding only ``common_utils.print_rank_last`` is not enough.
    try:
        import megatron.bridge.training.eval as mb_eval
        import megatron.bridge.training.utils.train_utils as tu

        mb_eval.print_rank_last = print_rank_last_wrapped
        tu.print_rank_last = print_rank_last_wrapped
    except ImportError:
        pass

    try:
        from primus.modules.module_utils import log_rank_0

        log_rank_0(
            "[Primus:Megatron-Bridge] print_rank_last is wired for rank-0 Primus logs (training + validation "
            "summaries) and last-rank Bridge stdout; eval module + train_utils aliases updated. "
            "Set PRIMUS_MIRROR_MEGATRON_BRIDGE_TRAINING_LOG=0 to use raw Megatron-Bridge only."
        )
    except Exception:
        pass
