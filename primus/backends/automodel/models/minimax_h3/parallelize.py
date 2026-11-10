###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""MiniMax-H3's parallelization sidecar: whole-block checkpointing, FP32 compute for its
keep-in-FP32 modules under FSDP2, and releasing the load's cached memory after sharding
(``release_cached_memory``).

WHOLE-BLOCK CHECKPOINTING:
  AutoModel's default checkpointing reports every transformer block as selected, but
  measured memory grows by about 9-11 GB per block at a 5.2 s, 768x1344 clip (38K rows),
  which is what an uncheckpointed block holds: the q/k/v and feed-forward activations,
  the norms and the per-row AdaLN modulation. Checkpointing a whole block keeps only its
  input, about 0.4 GB there. Without it the full 50-block model ran out of memory at step
  0 on 8x MI355X (253 GB allocated). This uses the same Primus sidecar as Ideogram-4
  (``distributed/sidecar.block_checkpointing``), over ``transformer_blocks``; the two
  token-refiner blocks only see the text rows and are left alone.

WHY:
  MiniMax-H3 is a mixed-precision checkpoint: diffusers keeps the time embedder, the
  four patch projections and the rotary buffer in FP32 (``_keep_in_fp32_modules``)
  whatever the load dtype, and its forward aligns each input with its projection's
  parameter dtype. AutoModel has no sidecar for the model, so its default parallelizer
  applies, and that shards the root with one mixed-precision policy. Under full
  fine-tuning every parameter is an FP32 master weight, so FSDP2 casts those modules
  to BF16 along with the rest, and the model computes them in BF16. It trains, but not
  as the model was built: the time embedding feeds the AdaLN modulation of every block.

WHAT (NO diffusers / AutoModel fork):
  Overrides ``_fully_shard_module``, the hook AutoModel's ``ModelParallelizer`` documents
  for a model that needs a different FSDP primitive, with AutoModel's own
  ``fully_shard_by_dtype``, pinned to the model's ``_keep_in_fp32_modules``. That puts
  the pinned parameters in FP32 FSDP units of their own, and leaves the rest computing
  in the policy dtype, FP32 master weights included. Every unit goes through it: a
  transformer block has nothing pinned and is sharded exactly as before, in one unit.
  Layer discovery, activation checkpointing, ``reshard_after_forward`` and prefetch
  stay upstream's. If upstream adds a MiniMax-H3 sidecar it becomes the base.

  Only the sharded FSDP2 path calls the hook. On one GPU AutoModel skips FSDP2, and
  with it any cast, so the weights' own dtype is the compute dtype there.

ONLY WHERE THE WEIGHTS ARE STORED IN FP32:
  A module is pinned only if its parameters and buffers are FP32 in storage. AutoModel's
  LoRA path loads the model in BF16 and then casts the whole module with ``.to()``,
  which overrides diffusers' keep-in-FP32 (diffusers warns about exactly that), so
  there every weight is BF16. Pinning such a module would make FSDP2 cast it up to
  FP32 only inside its forward, while the model picks the dtype of that module's input
  from the still-sharded BF16 weight, and the matmul fails on mixed dtypes. Those
  modules are frozen under LoRA anyway, so they are left to upstream's primitive and
  compute in BF16, as without the sidecar.
"""
from __future__ import annotations

import logging
from typing import Tuple

from primus.backends.automodel.distributed import sidecar

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusMiniMaxH3]"
_MODEL_NAME = "MiniMaxH3Transformer3DModel"
# The keyword arguments AutoModel passes to ``_fully_shard_module``, all of which
# ``fully_shard_by_dtype`` takes. Anything else means the call changed upstream.
_FORWARDED = ("mesh", "mp_policy", "offload_policy", "reshard_after_forward", "ignored_params")
_BLOCK_ATTRS = ("transformer_blocks",)


def fp32_compute_names() -> Tuple[str, ...]:
    """The model's keep-in-FP32 names, as diffusers declares them."""
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    return tuple(MiniMaxH3Transformer3DModel._keep_in_fp32_modules or ())


def pinned_in_fp32_storage(module, names) -> Tuple[str, ...]:
    """The names in ``names`` whose matching parameters and buffers in ``module`` are all FP32."""
    import torch

    tensors = (*module.named_parameters(), *module.named_buffers())
    keep = []
    for token in names:
        matched = [t for name, t in tensors if token in name]
        if matched and all(t.dtype == torch.float32 for t in matched):
            keep.append(token)
    return tuple(keep)


def build(base: type) -> type:
    """The sidecar class: whole-block checkpointing over a ``fully_shard_by_dtype`` primitive."""

    class MiniMaxH3Parallelizer(base):
        def _fully_shard_module(self, module, **kwargs):
            unexpected = sorted(set(kwargs) - set(_FORWARDED))
            if unexpected:
                # A new argument would be dropped silently; the stock primitive keeps the
                # run going, in BF16 for the pinned modules, and the warning says why.
                logger.warning(
                    "%s _fully_shard_module got arguments it does not forward (%s); sharding "
                    "with upstream's primitive, so the keep-in-FP32 modules compute in the "
                    "policy dtype.",
                    _LOG_PREFIX,
                    ", ".join(unexpected),
                )
                return super()._fully_shard_module(module, **kwargs)
            names = pinned_in_fp32_storage(module, fp32_compute_names())
            if not names:
                # A transformer block, or a model stored in BF16 (LoRA): nothing to keep in FP32.
                return super()._fully_shard_module(module, **kwargs)
            from nemo_automodel.components.distributed import parallelizer_utils

            parallelizer_utils.fully_shard_by_dtype(
                module,
                kwargs["mesh"],
                kwargs.get("mp_policy"),
                kwargs.get("offload_policy"),
                fp32_compute_module_names=names,
                reshard_after_forward=kwargs.get("reshard_after_forward"),
                ignored_params=kwargs.get("ignored_params"),
            )
            return module

    checkpointed = sidecar.block_checkpointing(
        MiniMaxH3Parallelizer, block_attrs=_BLOCK_ATTRS, log_prefix=_LOG_PREFIX
    )

    class MiniMaxH3ReleasingParallelizer(checkpointed):
        def parallelize(self, model, mesh_context, /):
            result = super().parallelize(model, mesh_context)
            release_cached_memory()
            return result

    MiniMaxH3ReleasingParallelizer.__name__ = MiniMaxH3ReleasingParallelizer.__qualname__ = "MiniMaxH3Parallelizer"
    return MiniMaxH3ReleasingParallelizer


def release_cached_memory() -> float:
    """Return the allocator's cached blocks to the device, once, after sharding. Returns GiB released.

    AutoModel moves the whole model onto every GPU before sharding it (124 GiB for the
    33B model in FP32), and after sharding the caching allocator keeps that memory
    reserved. Training then allocates on top of the pool, fragmenting it: measured on
    8x MI355X at full size, 5.2 s clip, the device peak was 159 GB against 86 GiB
    allocated. Releasing the pool here brought it to 118 GiB, with step time
    unchanged (20.39 s both). One call, at setup; nothing in the training loop.
    """
    import torch

    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return 0.0
    before = torch.cuda.memory_reserved()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    released = (before - torch.cuda.memory_reserved()) / 2**30
    logger.info(
        "%s released %.1f GiB of cached memory after sharding (%.1f GiB allocated, %.1f GiB reserved).",
        _LOG_PREFIX,
        released,
        torch.cuda.memory_allocated() / 2**30,
        torch.cuda.memory_reserved() / 2**30,
    )
    return released


def install() -> bool:
    """Install the MiniMax-H3 sidecar. Idempotent; edits no AutoModel source."""
    return sidecar.install(_MODEL_NAME, build, log_prefix=_LOG_PREFIX)
