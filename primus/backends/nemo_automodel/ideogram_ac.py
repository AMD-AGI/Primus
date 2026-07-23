###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real activation checkpointing for Ideogram-4 in the AutoModel diffusion recipe.

WHY:
  AutoModel picks an FSDP/AC parallelization strategy by the model's class name
  (``PARALLELIZATION_STRATEGIES`` in
  ``nemo_automodel.components.distributed.parallelizer``). ``Ideogram4Transformer2DModel``
  is NOT registered, so it falls back to the default strategy, whose AC path wraps
  each layer's ``self_attn`` / ``mlp`` submodules -- attributes an Ideogram block
  (``Ideogram4TransformerBlock`` in ``model.layers``) does not have -- so
  ``fsdp.activation_checkpointing: true`` is a SILENT NO-OP for Ideogram (same
  failure mode as FLUX). This registers a strategy that actually checkpoints the
  Ideogram blocks.

WHAT (NO diffusers / Automodel fork):
  Registers an Ideogram-specific parallelization strategy via the submodule's own
  ``register_parallel_strategy`` entry point. It wraps each ``model.layers`` block
  in a NON-REENTRANT ``checkpoint_wrapper`` (recompute on backward) BEFORE FSDP2
  sharding, then shards like the in-tree Wan/Hunyuan/FLUX diffusion strategies.
  Checkpointing only happens when the recipe passes ``activation_checkpointing=True``.

  Env-gated by ``PRIMUS_IDEOGRAM_REAL_AC=1`` (default off), mirroring
  ``PRIMUS_FLUX_REAL_AC``. Off = default behavior (Ideogram AC is a no-op).

  TP note: targets the dp/FSDP path (tp_size=1); warns once if a TP mesh is given.

Activation (env, no config schema change):
    PRIMUS_IDEOGRAM_REAL_AC=1   register the real-AC Ideogram parallelization strategy
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}

_IDEOGRAM_MODEL_NAME = "Ideogram4Transformer2DModel"
_IDEOGRAM_BLOCK_ATTR = "layers"


def is_ideogram_real_ac_enabled() -> bool:
    return os.getenv("PRIMUS_IDEOGRAM_REAL_AC", "0") in _TRUTHY


def install() -> bool:
    """Register the real-AC Ideogram parallelization strategy.

    No-op (returns False) unless ``PRIMUS_IDEOGRAM_REAL_AC`` is set. Idempotent
    (returns True if already present). Modifies NO Automodel source.
    """
    if not is_ideogram_real_ac_enabled():
        return False

    import torch

    import nemo_automodel.components.distributed.parallelizer as P

    if _IDEOGRAM_MODEL_NAME in P.PARALLELIZATION_STRATEGIES:
        return True  # already registered (idempotent)

    class Ideogram4ParallelizationStrategy(P.ParallelizationStrategy):
        """FSDP2 + real activation checkpointing for the Ideogram-4 single-stream DiT."""

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
            **kwargs,
        ):
            dp_mesh = P.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

            if tp_mesh_name in getattr(device_mesh, "mesh_dim_names", ()) and device_mesh[tp_mesh_name].size() > 1:
                logger.warning(
                    "[PrimusIdeogramAC] tensor parallelism requested but the Ideogram "
                    "real-AC strategy has no TP plan; proceeding with FSDP only."
                )

            # Normalize false-like strings: some CLI/config paths forward the flag as a
            # raw string, and a non-empty "false" would otherwise be truthy and wrongly
            # enable AC. "true"/"full"/"selective" keep their meaning (see below).
            ac_value = activation_checkpointing
            if isinstance(ac_value, str) and ac_value.strip().lower() in {"false", "0", "off", "no", "none", ""}:
                ac_value = False

            if ac_value:
                blocks = getattr(model, _IDEOGRAM_BLOCK_ATTR, None)
                if blocks is None:
                    logger.warning(
                        "[PrimusIdeogramAC] activation_checkpointing requested but model has no "
                        "'%s' block list; nothing checkpointed.",
                        _IDEOGRAM_BLOCK_ATTR,
                    )
                elif P.is_selective_activation_checkpointing(ac_value):
                    # PARTIAL / selective (TorchTitan-style, op-level) AC: save attention +
                    # half the matmuls + comm collectives, recompute only the cheap ops -> far
                    # less backward recompute than full AC at moderate extra memory. This reuses
                    # the SHARED Automodel selective-AC machinery (the same lever used for FLUX/WAN).
                    # The wrapper is tagged with SELECTIVE_AC_WRAPPER_FLAG so per-layer compile
                    # compiles it OUTER (SAC INNER) and the partitioner honors the recompute tags
                    # (see _apply_per_layer_compile). has_kv_sharing=False: Ideogram is a diffusion
                    # DiT with no KV cache. Blocks are replaced in-place in `model` (and `layers`).
                    layers = list(blocks)
                    P.apply_selective_checkpointing_to_layers(
                        model,
                        layers,
                        False,
                        enable_compile=bool(kwargs.get("enable_compile", False)),
                    )
                    logger.info(
                        "[PrimusIdeogramAC] wrapped %d Ideogram blocks with SELECTIVE (partial) "
                        "activation checkpointing",
                        len(layers),
                    )
                else:
                    # FULL AC: recompute the whole block on backward (max memory saved, ~16-20%
                    # recompute tax). Historical default (activation_checkpointing True / "full").
                    wrapped = 0
                    for idx in range(len(blocks)):
                        blocks[idx] = P.checkpoint_wrapper(
                            blocks[idx],
                            checkpoint_impl=P.CheckpointImpl.NO_REENTRANT,
                        )
                        wrapped += 1
                    logger.info(
                        "[PrimusIdeogramAC] wrapped %d Ideogram blocks with FULL activation checkpointing",
                        wrapped,
                    )

            if not mp_policy:
                mp_policy = P.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                )

            P.apply_fsdp2_sharding_recursively(
                model,
                dp_mesh,
                mp_policy,
                offload_policy,
                kwargs.get("enable_fsdp2_prefetch", True),
                kwargs.get("fsdp2_backward_prefetch_depth", 2),
                kwargs.get("fsdp2_forward_prefetch_depth", 1),
            )

            return P.fully_shard(
                model,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=False,
            )

    P.register_parallel_strategy(name=_IDEOGRAM_MODEL_NAME)(Ideogram4ParallelizationStrategy)
    logger.info(
        "[PrimusIdeogramAC] Registered real activation-checkpointing parallelization "
        "strategy for %s (honors fsdp.activation_checkpointing).",
        _IDEOGRAM_MODEL_NAME,
    )
    return True
