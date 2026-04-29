###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Megatron-Bridge training_log patch.

Megatron-Bridge has its own ``training_log`` function in
``megatron.bridge.training.utils.train_utils`` that calls
``print_rank_last`` (plain ``print()`` to stdout on the last rank).

Because Primus routes stdout to DEBUG level, the per-step metrics become
invisible on the INFO-filtered stderr sink that captures log output.

This patch wraps Megatron-Bridge's ``training_log`` so that the
``print_rank_last`` call inside it is redirected through a custom handler
that:

    1. Parses the log string to extract elapsed time and batch size.
    2. Computes and appends **tokens/s/GPU** (not provided by Megatron-Bridge).
    3. Emits the enriched line via ``log_rank_0`` (Python ``logging`` at INFO).

For SFT / finetuning workloads the batch dataloader pads to the **actual**
max sequence length in each batch, which can be much shorter than the
configured ``seq_length``.  The patch intercepts
``megatron.bridge.data.finetuning.prepare_finetuning_batch`` to capture
the real per-step sequence length so that the tokens/s/GPU metric reflects
the positions the model actually processes rather than the configured
maximum.
"""

import re
import threading
from typing import Optional

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0

# Per-step actual sequence length captured from prepare_finetuning_batch.
# Accessed from the same training-loop thread; the lock guards against
# unlikely concurrent reads during logging.
_actual_seq_length_lock = threading.Lock()
_actual_seq_length: Optional[int] = None


def _set_actual_seq_length(value: int) -> None:
    global _actual_seq_length
    with _actual_seq_length_lock:
        _actual_seq_length = value


def _get_actual_seq_length() -> Optional[int]:
    with _actual_seq_length_lock:
        return _actual_seq_length


def _parse_and_enrich(
    log_string: str,
    configured_seq_length: Optional[int],
    world_size: Optional[int],
) -> str:
    """Append tokens/s/GPU to *log_string* if the required fields are present."""
    seq_length = _get_actual_seq_length() or configured_seq_length
    if seq_length is None or world_size is None or world_size == 0:
        return log_string

    try:
        elapsed_match = re.search(
            r"elapsed time per iteration \(ms\):\s*([\d.]+)", log_string
        )
        batch_match = re.search(r"global batch size:\s*(\d+)", log_string)
        if not elapsed_match or not batch_match:
            return log_string

        elapsed_ms = float(elapsed_match.group(1))
        batch_size = int(batch_match.group(1))

        if elapsed_ms <= 0:
            return log_string

        elapsed_s = elapsed_ms / 1000.0
        tokens_per_iter = seq_length * batch_size
        tokens_per_gpu = tokens_per_iter / elapsed_s / world_size

        throughput_seg = (
            f" tokens per GPU (tokens/s/GPU): {tokens_per_gpu:.1f} |"
            f" seq_length: {seq_length} |"
        )

        tflop_match = re.search(
            r"throughput per GPU \(TFLOP/s/GPU\):\s*[\d.]+\s*\|", log_string
        )
        if tflop_match:
            insert_pos = tflop_match.end()
            return log_string[:insert_pos] + throughput_seg + log_string[insert_pos:]

        elapsed_end = re.search(
            r"elapsed time per iteration \(ms\):\s*[\d.]+\s*\|", log_string
        )
        if elapsed_end:
            insert_pos = elapsed_end.end()
            return log_string[:insert_pos] + throughput_seg + log_string[insert_pos:]

        return log_string + throughput_seg

    except Exception:
        return log_string


def _install_seq_length_hook() -> None:
    """Wrap ``prepare_finetuning_batch`` to capture per-step seq_length."""
    try:
        import megatron.bridge.data.finetuning as bridge_ft
    except ImportError:
        return

    original_fn = getattr(bridge_ft, "prepare_finetuning_batch", None)
    if original_fn is None or getattr(original_fn, "_primus_seq_hook", False):
        return

    def _hooked_prepare(*args, **kwargs):
        iterator, seq_length = original_fn(*args, **kwargs)
        _set_actual_seq_length(seq_length)
        return iterator, seq_length

    _hooked_prepare._primus_seq_hook = True
    bridge_ft.prepare_finetuning_batch = _hooked_prepare


@register_patch(
    "megatron_bridge.training_log.print_rank_last_patch",
    backend="megatron_bridge",
    phase="before_train",
    description=(
        "Redirect Megatron-Bridge training_log output through Primus logger "
        "and append tokens/s/GPU metric."
    ),
)
def patch_bridge_training_log(ctx: PatchContext):
    """
    Wrap Megatron-Bridge's ``training_log`` so that ``print_rank_last``
    calls made inside it are routed through ``log_rank_0`` with an
    additional tokens/s/GPU metric appended to the log string.

    Also hooks ``prepare_finetuning_batch`` so that the actual per-batch
    sequence length (which can be shorter than the configured max for SFT
    workloads) is used in the throughput calculation.
    """
    try:
        import megatron.bridge.training.utils.train_utils as bridge_train_utils
    except ImportError:
        return

    original_training_log = getattr(bridge_train_utils, "training_log", None)
    if original_training_log is None:
        return

    if getattr(original_training_log, "_primus_bridge_training_log_wrapper", False):
        return

    args = get_args(ctx)
    configured_seq_length = getattr(args, "seq_length", None)

    _install_seq_length_hook()

    def _enriched_log_rank_0(log_string: str) -> None:
        import torch

        ws = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        enriched = _parse_and_enrich(log_string, configured_seq_length, ws)
        log_rank_0(enriched)

    def primus_bridge_training_log(*args, **kwargs):
        original_print = bridge_train_utils.print_rank_last
        try:
            bridge_train_utils.print_rank_last = _enriched_log_rank_0
            return original_training_log(*args, **kwargs)
        finally:
            bridge_train_utils.print_rank_last = original_print

    primus_bridge_training_log._primus_bridge_training_log_wrapper = True
    bridge_train_utils.training_log = primus_bridge_training_log

    log_rank_0(
        f"[Patch:megatron_bridge.training_log] "
        f"Wrapped Megatron-Bridge training_log with Primus log_rank_0 hook "
        f"(configured_seq_length={configured_seq_length})"
    )
