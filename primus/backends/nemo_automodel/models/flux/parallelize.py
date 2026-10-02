###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Whole-block activation checkpointing for FLUX.

WHY:
  AutoModel has no sidecar for ``FluxTransformer2DModel``, so the default
  ``ModelParallelizer`` applies. Its layer discovery keeps only the longest
  ``ModuleList`` -- the 38 single-stream blocks -- and its checkpointing wraps
  submodules by conventional name, which in a FLUX block matches only ``attn``.
  The 19 dual-stream blocks are not checkpointed at all, so
  ``fsdp.activation_checkpointing: true`` saves far less memory than it should.

WHAT (NO diffusers / AutoModel fork):
  Installs a block-checkpointing sidecar (``distributed/sidecar.py``) that wraps
  every block in both lists, then hands the model to upstream's parallelizer
  with checkpointing off. Sharding, mixed precision and ``reshard_after_forward``
  are upstream's. If upstream adds a FLUX sidecar it becomes the base, and this
  can be deleted once it checkpoints both lists itself.
"""
from __future__ import annotations

from primus.backends.nemo_automodel.distributed import sidecar

_LOG_PREFIX = "[PrimusFluxAC]"
_FLUX_MODEL_NAME = "FluxTransformer2DModel"
# Dual-stream then single-stream, so the logged count is reproducible.
_FLUX_BLOCK_ATTRS = ("transformer_blocks", "single_transformer_blocks")


def _build(base):
    return sidecar.block_checkpointing(base, block_attrs=_FLUX_BLOCK_ATTRS, log_prefix=_LOG_PREFIX)


def install() -> bool:
    """Install the FLUX sidecar. Idempotent; edits no AutoModel source."""
    return sidecar.install(_FLUX_MODEL_NAME, _build, log_prefix=_LOG_PREFIX)
