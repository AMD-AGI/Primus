###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Whole-block activation checkpointing for Ideogram-4.

WHY:
  AutoModel has no sidecar for ``Ideogram4Transformer2DModel``, so the default
  ``ModelParallelizer`` applies, and its checkpointing wraps only the attention
  and feed-forward submodules of each block. That leaves the norms, modulation
  and residual activations of every block resident, and there is no way to
  checkpoint only part of the stack.

WHAT (NO diffusers / AutoModel fork):
  Installs a block-checkpointing sidecar (``distributed/sidecar.py``) that wraps
  whole blocks, optionally every nth one, then hands the model to upstream's
  parallelizer with checkpointing off. It acts in ``parallelize``, so the FSDP2
  and DDP paths get the same checkpointing; sharding, mixed precision and
  ``reshard_after_forward`` are upstream's. If upstream adds an Ideogram sidecar
  it becomes the base.

  Non-reentrant checkpointing re-runs each block's forward during backward, so
  anything a block reads from module state -- for the var-len attention path,
  the packing buffer -- has to stay valid until its backward has run.

  ``primus_ideogram4.ac_every: n`` checkpoints only every nth block under full AC.
"""
from __future__ import annotations

from primus.backends.nemo_automodel import options
from primus.backends.nemo_automodel.distributed import sidecar

_LOG_PREFIX = "[PrimusIdeogramAC]"
_MODEL_NAME = "Ideogram4Transformer2DModel"
# The single-stream transformer keeps every block in one list.
_BLOCK_ATTRS = ("layers",)


def ac_stride() -> int:
    """Blocks between checkpoints under full AC, or 0 to checkpoint every block.

    A stride of 1 means the same as not setting it, so it normalizes to 0.

    Set by ``primus_ideogram4.ac_every``, and parsed strictly: this decides how
    much memory a run uses, and falling back to the default would quietly give a
    run that asked for a partial stride the full checkpointing it meant to avoid.
    """
    value = options.integer("primus_ideogram4.ac_every", None)
    if value is None:
        return 0
    if value < 1:
        raise ValueError(f"primus_ideogram4.ac_every must be at least 1, got {value}")
    return 0 if value == 1 else value


def install() -> bool:
    """Install the Ideogram-4 sidecar. Idempotent; edits no AutoModel source."""
    # Read once, so a bad value fails here rather than inside parallelize, where
    # the trainer would report it as a parallelization failure.
    stride = ac_stride()

    def _build(base):
        return sidecar.block_checkpointing(
            base, block_attrs=_BLOCK_ATTRS, log_prefix=_LOG_PREFIX, stride=lambda: stride
        )

    return sidecar.install(_MODEL_NAME, _build, log_prefix=_LOG_PREFIX)
