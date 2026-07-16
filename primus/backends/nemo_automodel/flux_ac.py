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
  single-stream ``single_transformer_blocks`` block in a NON-REENTRANT
  ``checkpoint_wrapper`` (recompute on backward) BEFORE FSDP2 sharding, then shards
  exactly like the in-tree Wan/Hunyuan diffusion strategies. Checkpointing only
  happens when the recipe passes ``activation_checkpointing=True`` (i.e. the config
  flag), so an AC-off run is unaffected.

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
            **kwargs,
        ):
            dp_mesh = P.get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

            if tp_mesh_name in getattr(device_mesh, "mesh_dim_names", ()) and device_mesh[tp_mesh_name].size() > 1:
                logger.warning(
                    "[PrimusFluxAC] tensor parallelism requested but the FLUX real-AC "
                    "strategy has no FLUX-specific TP plan; proceeding with FSDP only."
                )

            # Real AC: wrap each dual-stream and single-stream block so its forward
            # activations are recomputed on backward. Must happen BEFORE FSDP2
            # sharding so the module structure is stable when fully_shard indexes
            # params. NO_REENTRANT is required for torch.compile compatibility.
            if activation_checkpointing:
                wrapped = 0
                for attr in _FLUX_BLOCK_ATTRS:
                    blocks = getattr(model, attr, None)
                    if blocks is None:
                        continue
                    for idx in range(len(blocks)):
                        blocks[idx] = P.checkpoint_wrapper(
                            blocks[idx],
                            checkpoint_impl=P.CheckpointImpl.NO_REENTRANT,
                        )
                        wrapped += 1
                logger.info("[PrimusFluxAC] wrapped %d FLUX blocks with activation checkpointing", wrapped)

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

    P.register_parallel_strategy(name=_FLUX_MODEL_NAME)(FluxParallelizationStrategy)
    logger.info(
        "[PrimusFluxAC] Registered real activation-checkpointing parallelization "
        "strategy for %s (honors fsdp.activation_checkpointing).",
        _FLUX_MODEL_NAME,
    )
    return True
