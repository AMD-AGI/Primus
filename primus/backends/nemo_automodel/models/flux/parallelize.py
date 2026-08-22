###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Real activation checkpointing for FLUX in the NeMo AutoModel diffusion recipe.

WHY:
  AutoModel selects an FSDP/AC parallelization strategy by the model's class name
  (``PARALLELIZATION_STRATEGIES`` in
  ``nemo_automodel.components.distributed.parallelizer``). ``FluxTransformer2DModel``
  is NOT registered, so it falls back to the default strategy. The default
  strategy applies activation checkpointing by wrapping each layer's ``self_attn``
  / ``mlp`` submodules -- attributes a FLUX block does not have -- so
  ``fsdp.activation_checkpointing: true`` is a SILENT NO-OP for FLUX: the flag is
  accepted, nothing is checkpointed, and the activation-memory ceiling (hence the
  achievable batch size) is unchanged.

WHAT this does (NO diffusers / Automodel fork):
  Registers a FLUX-specific parallelization strategy for the ``FluxTransformer2DModel``
  class via the submodule's own ``register_parallel_strategy`` entry point. The
  strategy wraps each dual-stream ``transformer_blocks`` block and each
  single-stream ``single_transformer_blocks`` block (19 + 38 = 57) BEFORE FSDP2
  sharding, then shards exactly like the in-tree Wan/Hunyuan diffusion strategies.
  Checkpointing only happens when the recipe passes a truthy
  ``activation_checkpointing``, so an AC-off run is unaffected.

  ``activation_checkpointing`` accepts three settings, matching the Ideogram-4
  strategy so the AC axis means the same thing in both:
    ``true`` / ``full``  whole-block NON-REENTRANT recompute on backward (max
                         memory saved, pays a full recompute tax)
    ``selective``        op-level partial AC via the shared Automodel machinery:
                         keep attention and half the matmuls, recompute the cheap
                         ops (much less recompute, moderate extra memory)
    ``false`` / ``off``  no checkpointing
  False-like STRINGS are normalized, because a raw ``"false"`` arriving from the
  CLI is otherwise truthy and would silently checkpoint an AC-off run.

  Env-gated by ``PRIMUS_FLUX_REAL_AC=1`` (default off). Off = current behavior
  (FLUX AC remains a no-op), which keeps the flag an explicit, reversible A/B lever.
  When on, ``fsdp.activation_checkpointing`` starts doing what it says for FLUX.

  TP note: this strategy targets the dp/FSDP path (tp_size=1). If a TP mesh is
  requested it warns once and proceeds without a FLUX-specific TP plan.

Activation (env, no config schema change):
    PRIMUS_FLUX_REAL_AC=1    register the real-AC FLUX parallelization strategy
                             (default off = no-op)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}

_FLUX_MODEL_NAME = "FluxTransformer2DModel"
_FLUX_BLOCK_ATTRS = ("transformer_blocks", "single_transformer_blocks")


def is_flux_real_ac_enabled() -> bool:
    """Whether the real-AC FLUX strategy should be registered."""
    return os.getenv("PRIMUS_FLUX_REAL_AC", "0") in _TRUTHY


def install() -> bool:
    """Register the real-AC FLUX parallelization strategy.

    No-op (returns False) unless ``PRIMUS_FLUX_REAL_AC`` is set. Registers a
    strategy for ``FluxTransformer2DModel`` in the AutoModel strategy registry;
    idempotent (returns True if already present). Modifies NO Automodel source.
    """
    if not is_flux_real_ac_enabled():
        return False

    import torch

    import nemo_automodel.components.distributed.parallelizer as P

    if _FLUX_MODEL_NAME in P.PARALLELIZATION_STRATEGIES:
        return True  # already registered (idempotent)

    class FluxParallelizationStrategy(P.ParallelizationStrategy):
        """FSDP2 + real activation checkpointing for FLUX MMDiT transformers."""

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

            # Read the hook's state through the module, never a from-import: a
            # from-import binds False at import time and the guard then fires on
            # every healthy run.
            from primus.backends.nemo_automodel.distributed import fsdp2_reshard

            logger.info(
                "[PrimusFluxAC] reshard_after_forward: received=%s applied_by_hook=%s",
                reshard_after_forward,
                fsdp2_reshard.applied_reshard_after_forward,
            )
            if not fsdp2_reshard.patch_installed:
                # The trainer swallows hook exceptions, so a failed install shows up
                # only as one log line and then a full run at ~3P traffic. This is the
                # one unambiguous in-process signal that it happened.
                logger.error(
                    "[PrimusFluxAC] the FSDP2 reshard repair hook is NOT installed, so "
                    "fsdp.reshard_after_forward from YAML was dropped upstream and this run "
                    "will use ZeRO-3-style per-block resharding regardless of config."
                )

            if tp_mesh_name in getattr(device_mesh, "mesh_dim_names", ()) and device_mesh[tp_mesh_name].size() > 1:
                logger.warning(
                    "[PrimusFluxAC] tensor parallelism requested but the FLUX real-AC "
                    "strategy has no FLUX-specific TP plan; proceeding with FSDP only."
                )

            # Real AC: wrap each dual-stream and single-stream block so its forward
            # activations are recomputed on backward. Must happen BEFORE FSDP2
            # sharding so the module structure is stable when fully_shard indexes
            # params. NO_REENTRANT is required for torch.compile compatibility.
            # Normalize false-like strings: some CLI/config paths forward the flag as a
            # raw string, and a non-empty "false" would otherwise be truthy and wrongly
            # enable AC on a run whose whole purpose is to measure AC off.
            # "true"/"full"/"selective" keep their meaning (see below).
            ac_value = activation_checkpointing
            if isinstance(ac_value, str) and ac_value.strip().lower() in {"false", "0", "off", "no", "none", ""}:
                ac_value = False

            if ac_value:
                # FLUX keeps its dual-stream and single-stream blocks in two separate
                # ModuleLists, so both are collected before wrapping. Order is stable
                # (dual then single) to keep the logged count meaningful: 19 + 38 = 57.
                blocks = [
                    (attr, getattr(model, attr))
                    for attr in _FLUX_BLOCK_ATTRS
                    if getattr(model, attr, None) is not None
                ]
                if not blocks:
                    logger.warning(
                        "[PrimusFluxAC] activation_checkpointing requested but model has none of "
                        "the block lists %s; nothing checkpointed.",
                        ", ".join(_FLUX_BLOCK_ATTRS),
                    )
                elif P.is_selective_activation_checkpointing(ac_value):
                    # PARTIAL / selective (TorchTitan-style, op-level) AC: save attention +
                    # half the matmuls + comm collectives, recompute only the cheap ops -> far
                    # less backward recompute than full AC at moderate extra memory. This is the
                    # SHARED Automodel selective-AC machinery, the same lever Ideogram-4 uses,
                    # which is what makes the AC axis comparable between the two studies.
                    # The wrapper is tagged with SELECTIVE_AC_WRAPPER_FLAG so per-layer compile
                    # compiles it OUTER (SAC INNER) and the partitioner honors the recompute
                    # tags. has_kv_sharing=False: FLUX is a diffusion MMDiT with no KV cache.
                    # The helper replaces each block by identity, so passing blocks from two
                    # different ModuleLists in one call is safe.
                    layers = [blk for _, lst in blocks for blk in lst]
                    P.apply_selective_checkpointing_to_layers(
                        model,
                        layers,
                        False,
                        enable_compile=bool(kwargs.get("enable_compile", False)),
                    )
                    logger.info(
                        "[PrimusFluxAC] wrapped %d FLUX blocks with SELECTIVE (partial) "
                        "activation checkpointing",
                        len(layers),
                    )
                else:
                    # FULL AC: recompute the whole block on backward (max memory saved, at the
                    # cost of a recompute tax). Default for activation_checkpointing True / "full".
                    wrapped = 0
                    for _, block_list in blocks:
                        for idx in range(len(block_list)):
                            block_list[idx] = P.checkpoint_wrapper(
                                block_list[idx],
                                checkpoint_impl=P.CheckpointImpl.NO_REENTRANT,
                            )
                            wrapped += 1
                    logger.info(
                        "[PrimusFluxAC] wrapped %d FLUX blocks with FULL activation checkpointing",
                        wrapped,
                    )

            if not mp_policy:
                mp_policy = P.MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=torch.float32,
                )

            # Keyword form throughout: this call previously passed seven positional
            # arguments and stopped one short of reshard_after_forward, silently
            # discarding it. Keywords make that class of bug impossible to repeat.
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

            return P.fully_shard(
                model,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=False,
            )

    P.register_parallel_strategy(name=_FLUX_MODEL_NAME)(FluxParallelizationStrategy)
    logger.info(
        "[PrimusFluxAC] Registered real activation-checkpointing parallelization "
        "strategy for %s (honors fsdp.activation_checkpointing).",
        _FLUX_MODEL_NAME,
    )
    return True
