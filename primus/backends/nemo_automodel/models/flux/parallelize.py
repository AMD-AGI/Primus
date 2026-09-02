###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real activation checkpointing for FLUX in the AutoModel diffusion recipe.

WHY:
  AutoModel selects an FSDP/AC parallelization strategy by the model's class name
  (``PARALLELIZATION_STRATEGIES`` in
  ``nemo_automodel.components.distributed.parallelizer``). ``FluxTransformer2DModel``
  is NOT registered, so it falls back to the default strategy -- which applies
  activation checkpointing by wrapping each layer's ``self_attn`` and ``mlp``
  submodules, attributes a FLUX block does not have.

  So ``fsdp.activation_checkpointing: true`` is a SILENT NO-OP for FLUX. The flag
  is accepted, nothing is checkpointed, and the activation-memory ceiling -- and
  therefore the batch size that fits -- is unchanged. Nothing raises, and nothing
  in the config echo suggests the setting was ignored.

WHAT (NO diffusers / AutoModel fork):
  Registers a FLUX strategy for ``FluxTransformer2DModel`` through the submodule's
  own ``register_parallel_strategy`` entry point. Unlike the Wan strategy, no
  re-registration wrapper is needed here: FLUX is not in the hard-coded list that
  ``_init_parallelizer`` resets, so the registration survives.

  Activation checkpointing is applied through the shared helper in
  ``distributed/activation_checkpointing.py``, so ``full`` / ``selective`` / ``off``
  mean the same thing here as for every other model on this path. It happens
  BEFORE FSDP2 sharding, so the module structure is stable when ``fully_shard``
  indexes parameters.

  Env-gated by ``PRIMUS_FLUX_REAL_AC=1`` (default off). Off is current behaviour,
  which keeps this an explicit, reversible lever rather than a change that lands
  on everyone at once.

  TP note: this strategy targets the DP/FSDP path. If a TP mesh is requested it
  warns and proceeds without a FLUX-specific TP plan rather than silently
  producing a wrong one.

Activation (env, no config schema change):
    PRIMUS_FLUX_REAL_AC=1    register the real-AC FLUX parallelization strategy
"""
from __future__ import annotations

import logging

from primus.backends.nemo_automodel._env import env_flag
from primus.backends.nemo_automodel.distributed import activation_checkpointing as ac

logger = logging.getLogger(__name__)

_LOG_PREFIX = "[PrimusFluxAC]"
_FLUX_MODEL_NAME = "FluxTransformer2DModel"
# FLUX keeps dual-stream and single-stream blocks in two separate ModuleLists;
# both are checkpointed, in this order, so the logged count is reproducible.
_FLUX_BLOCK_ATTRS = ("transformer_blocks", "single_transformer_blocks")


def is_enabled() -> bool:
    """Whether the real-AC FLUX strategy should be registered."""
    return env_flag("PRIMUS_FLUX_REAL_AC")


def install() -> bool:
    """Register the real-AC FLUX parallelization strategy.

    Idempotent: returns True if it is already present. Modifies NO AutoModel
    source.
    """
    import nemo_automodel.components.distributed.parallelizer as P
    import torch

    if _FLUX_MODEL_NAME in P.PARALLELIZATION_STRATEGIES:
        return True  # already registered

    class FluxParallelizationStrategy(P.ParallelizationStrategy):
        """FSDP2 plus real activation checkpointing for FLUX MMDiT transformers."""

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
            # would then fire on every healthy run.
            from primus.backends.nemo_automodel.distributed import fsdp2_reshard

            logger.info(
                "%s reshard_after_forward: received=%s applied_by_repair=%s",
                _LOG_PREFIX,
                reshard_after_forward,
                fsdp2_reshard.applied_reshard_after_forward,
            )
            if not fsdp2_reshard.patch_installed:
                # The patch runner isolates failures, so a failed install shows up
                # as one log line and then a full run at ZeRO-3 traffic. This is
                # the one unambiguous in-process signal that it happened.
                logger.error(
                    "%s the FSDP2 reshard repair is NOT installed, so "
                    "fsdp.reshard_after_forward from YAML was dropped upstream and this run "
                    "will use ZeRO-3-style per-block resharding regardless of config.",
                    _LOG_PREFIX,
                )

            if (
                tp_mesh_name in getattr(device_mesh, "mesh_dim_names", ())
                and device_mesh[tp_mesh_name].size() > 1
            ):
                logger.warning(
                    "%s tensor parallelism requested but this strategy has no "
                    "FLUX-specific TP plan; proceeding with FSDP only.",
                    _LOG_PREFIX,
                )

            # Before sharding: fully_shard indexes parameters, so the module
            # structure has to be final first.
            ac.apply(
                P,
                model,
                _FLUX_BLOCK_ATTRS,
                activation_checkpointing,
                enable_compile=bool(kwargs.get("enable_compile", False)),
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

    P.register_parallel_strategy(name=_FLUX_MODEL_NAME)(FluxParallelizationStrategy)
    logger.info(
        "%s registered a real activation-checkpointing parallelization strategy for %s; "
        "fsdp.activation_checkpointing now takes effect.",
        _LOG_PREFIX,
        _FLUX_MODEL_NAME,
    )
    return True
