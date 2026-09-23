###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real activation checkpointing for Ideogram-4.

WHY THIS EXISTS:
  AutoModel picks a parallelization strategy by the model's class name.
  Ideogram-4 is not registered, so it falls back to the default strategy, whose
  activation-checkpointing path reaches for per-layer submodule attributes that a
  single-stream diffusion block does not have. The result is that
  ``fsdp.activation_checkpointing: true`` is a SILENT NO-OP -- the config says
  checkpointing is on, the log says nothing, and the run uses the memory of a run
  with none. It is the same failure the FLUX strategy in this backend was added to
  fix, and it is found the same way: by noticing that turning the setting on
  changed nothing.

WHAT IT DOES:
  Registers an Ideogram-specific strategy through AutoModel's own entry point, so
  nothing upstream is edited. The blocks are wrapped BEFORE sharding, then sharded
  the way the in-tree diffusion strategies do.

  The checkpointing itself is delegated to the shared helper, which is where the
  off/selective/full decision and the block stride live. That decision has two
  known ways of going wrong -- ``"selective"`` and ``"false"`` are both truthy
  strings -- and both are invisible at runtime, so it is deliberately not
  reimplemented here.

TWO INTERACTIONS WORTH KNOWING ABOUT, because neither is visible from this file:

  Non-reentrant checkpointing re-runs each block's forward during the backward
  pass. Anything the block reads from module state has to still be valid then. For
  the var-len attention path that is the packing buffer, and it is why nothing may
  republish between a forward and its backward.

  A stride leaves some blocks unwrapped, so under per-layer compilation the
  wrapped and unwrapped blocks compile differently. That is fine -- they are
  separate graphs either way -- but it does mean a stride changes how many
  distinct graphs a run builds.

Activation, by environment, with no config schema change:
  PRIMUS_IDEOGRAM_REAL_AC=1     register the strategy, so
                                fsdp.activation_checkpointing takes effect
  PRIMUS_IDEOGRAM_AC_EVERY=n    under full AC, checkpoint only every nth block
"""
from __future__ import annotations

import logging
import os

from primus.backends.nemo_automodel._env import env_flag
from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusIdeogramAC]"
_MODEL_NAME = "Ideogram4Transformer2DModel"
# The single-stream transformer keeps every block in one list.
_BLOCK_ATTRS = ("layers",)


def is_enabled() -> bool:
    """Whether the real-AC Ideogram strategy should be registered."""
    return env_flag("PRIMUS_IDEOGRAM_REAL_AC")


def ac_stride() -> int:
    """Blocks between checkpoints under full AC, or 0 to checkpoint every block.

    See the shared helper for why this axis exists. A stride of 1 means the same
    thing as not setting it, so it normalizes to 0 rather than being carried
    through as a special case.

    Parsed strictly, unlike the diagnostic environment helpers, which fall back to
    their default on anything unparseable. This one decides how much memory a run
    uses. Falling back would give a run that asked for a partial stride the full
    checkpointing it was trying to avoid, and the only evidence would be a step
    time nobody had a baseline for.
    """
    raw = os.getenv("PRIMUS_IDEOGRAM_AC_EVERY")
    if raw is None or not raw.strip():
        return 0
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"PRIMUS_IDEOGRAM_AC_EVERY must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"PRIMUS_IDEOGRAM_AC_EVERY must be at least 1, got {value}")
    return 0 if value == 1 else value


def install() -> bool:
    """Register the real-AC Ideogram parallelization strategy.

    A no-op returning False unless ``PRIMUS_IDEOGRAM_REAL_AC`` is set. Idempotent:
    returns True if the strategy is already registered. Edits no AutoModel source.
    """
    if not is_enabled():
        return False

    import nemo_automodel.components.distributed.parallelizer as P
    import torch

    if _MODEL_NAME in P.PARALLELIZATION_STRATEGIES:
        return True

    # Read once, here, so a bad value fails at registration rather than inside the
    # strategy -- where the trainer would report it as a parallelization failure
    # with no mention of the environment variable that caused it.
    stride = ac_stride()

    class Ideogram4ParallelizationStrategy(P.ParallelizationStrategy):
        """FSDP2 plus real activation checkpointing for the single-stream DiT."""

        def parallelize(
            self,
            model,
            device_mesh,
            mp_policy=None,
            offload_policy=None,
            sequence_parallel: bool = False,
            activation_checkpointing: bool = False,
            tp_shard_plan=None,
            dp_replicate_mesh_name: str = "dp_replicate",
            dp_shard_cp_mesh_name: str = "dp_shard_cp",
            tp_mesh_name: str = "tp",
            reshard_after_forward: bool | None = None,
            **kwargs,
        ):
            dp_mesh = P.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

            # Read the sibling repair's state through the module, never a
            # from-import: a from-import binds False at import time and the guard
            # below would then fire on every healthy run.
            from primus.backends.nemo_automodel.distributed import fsdp2_reshard

            logger.info(
                "%s reshard_after_forward: received=%s applied_by_repair=%s",
                _LOG_PREFIX,
                reshard_after_forward,
                fsdp2_reshard.applied_reshard_after_forward,
            )
            if not fsdp2_reshard.patch_installed:
                # The patch runner isolates failures, so a failed install shows up
                # as one log line and then a whole run at ZeRO-3 traffic. This is
                # the one unambiguous in-process signal that it happened.
                logger.error(
                    "%s the FSDP2 reshard repair is NOT installed, so "
                    "fsdp.reshard_after_forward from YAML was dropped upstream and "
                    "this run will use ZeRO-3-style per-block resharding regardless "
                    "of config.",
                    _LOG_PREFIX,
                )

            if (
                tp_mesh_name in getattr(device_mesh, "mesh_dim_names", ())
                and device_mesh[tp_mesh_name].size() > 1
            ):
                logger.warning(
                    "%s tensor parallelism requested but this strategy has no "
                    "Ideogram-specific TP plan; proceeding with FSDP only.",
                    _LOG_PREFIX,
                )

            # Before sharding: fully_shard indexes parameters, so the module
            # structure has to be final first.
            ac.apply(
                P,
                model,
                _BLOCK_ATTRS,
                activation_checkpointing,
                enable_compile=bool(kwargs.get("enable_compile", False)),
                stride=stride,
                log_prefix=_LOG_PREFIX,
            )

            if not mp_policy:
                mp_policy = P.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                )

            # Keyword form throughout. The in-tree strategies pass seven positional
            # arguments and stop one short of reshard_after_forward, silently
            # discarding it; keywords make that class of bug impossible to repeat.
            P.apply_fsdp2_sharding_recursively(
                model,
                dp_mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch=kwargs.get("enable_fsdp2_prefetch", True),
                fsdp2_backward_prefetch_depth=kwargs.get("fsdp2_backward_prefetch_depth", 2),
                fsdp2_forward_prefetch_depth=kwargs.get("fsdp2_forward_prefetch_depth", 1),
                # Forwarded faithfully, including None: YAML decides, not this code.
                reshard_after_forward=reshard_after_forward,
            )

            # The root unit holds embeddings, norms and heads -- a small fraction of
            # the parameters -- and is deliberately never resharded.
            return P.fully_shard(
                model,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=False,
            )

    P.register_parallel_strategy(name=_MODEL_NAME)(Ideogram4ParallelizationStrategy)
    logger.info(
        "%s registered a real activation-checkpointing parallelization strategy for "
        "%s; fsdp.activation_checkpointing now takes effect%s.",
        _LOG_PREFIX,
        _MODEL_NAME,
        f" (full AC will wrap every {stride} blocks)" if stride else "",
    )
    return True
