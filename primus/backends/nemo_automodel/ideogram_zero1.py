###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""DDP + ZeRO-1 (distributed optimizer) for Ideogram-4 in the AutoModel diffusion recipe.

WHY (see 10_PHASE_C_PROFILE.md §3d / 00_PLAN.md sharding workstream):
  The 288 GB MI355X leaves large memory headroom at the compute-bound knee mbs
  (more so after torch.compile freed ~16-18 %). One way to spend it is to shard
  *less* and cut FSDP collectives. Pure **DDP** replicates params+grads (no
  per-layer all-gather) but also replicates the AdamW optimizer state (fp32 m/v ~=
  2x params); **ZeRO-1** shards *only* that optimizer state across the DP ranks so
  we get DDP-like compute with the optimizer memory back.

  IMPORTANT single-node caveat (measured, 10 §2): exposed FSDP comm is only
  0.7-3 % on one 8-GPU node, AND per-layer torch.compile (our +17-30 % lever) is
  wired ONLY on the FSDP2 path (fsdp2.py -> _apply_per_layer_compile; DDPManager
  never compiles). So on a single node DDP+ZeRO-1 (no compile) TRAILS
  FSDP2+compile and cannot recover the gap from comms. This path is the
  **multi-node** lever (where all-gather crosses the slow inter-node network) and
  a reference point; the single-node sharding lever that KEEPS compile is HSDP
  (fsdp.dp_replicate_size>1). This module exists to (a) validate the DDP+ZeRO-1
  path runs with the Ideogram adapter + hd256 + real AC, and (b) measure the
  ZeRO-1 optimizer-memory recovery vs pure DDP.

WHAT (NO diffusers / Automodel fork) — two env-gated monkeypatches:
  1. ZeRO-1 optimizer: wrap the recipe's optimizer
     (nemo_automodel.recipes.diffusion.train._build_optimizer) in
     torch.distributed.optim.ZeroRedundancyOptimizer. Skipped (with a warning) if
     the params are DTensors (i.e. FSDP2 mode) since ZeRO-1 only applies to
     replicated/DDP params.
  2. DDP real activation checkpointing: DDPManager applies AC via
     apply_submodule_checkpointing, which wraps by attribute name
     (mlp/self_attn/norm1...) -- attributes an Ideogram4TransformerBlock does NOT
     have -- so DDP-mode AC is a SILENT NO-OP for Ideogram (same failure mode the
     FSDP strategy has, fixed there by ideogram_ac.py). Without real AC the 9.3B
     DiT OOMs at any real mbs. We patch DDPManager.parallelize to whole-block
     checkpoint-wrap model.layers for Ideogram before DDP wrap.

Activation (env, no config schema change):
    PRIMUS_IDEOGRAM_ZERO1=1   install BOTH patches (ZeRO-1 optimizer + DDP whole-block AC)
"""
from __future__ import annotations

import inspect
import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}

_IDEOGRAM_MODEL_NAME = "Ideogram4Transformer2DModel"
_IDEOGRAM_BLOCK_ATTR = "layers"


def is_zero1_enabled() -> bool:
    return os.getenv("PRIMUS_IDEOGRAM_ZERO1", "0") in _TRUTHY


def is_ddp_enabled() -> bool:
    """Pure-DDP baseline (whole-block AC on the DDP path, no ZeRO-1)."""
    return os.getenv("PRIMUS_IDEOGRAM_DDP", "0") in _TRUTHY


def _looks_like_ideogram(model) -> bool:
    return type(model).__name__ == _IDEOGRAM_MODEL_NAME and hasattr(model, _IDEOGRAM_BLOCK_ATTR)


def _params_are_dtensor(params) -> bool:
    for p in params:
        if type(p).__name__ == "DTensor" or hasattr(p, "_local_tensor"):
            return True
    return False


def _install_zero1_optimizer_patch() -> bool:
    """Wrap the diffusion recipe's optimizer in ZeroRedundancyOptimizer."""
    import nemo_automodel.recipes.diffusion.train as train_mod

    if getattr(train_mod, "_primus_zero1_patched", False):
        return True

    _orig_build_optimizer = train_mod._build_optimizer

    def _build_optimizer_zero1(trainable_params, optimizer_cfg, learning_rate, is_peft: bool = False):
        params = list(trainable_params)
        base = _orig_build_optimizer(params, optimizer_cfg, learning_rate, is_peft=is_peft)
        if not is_zero1_enabled():
            return base
        if _params_are_dtensor(params):
            logger.warning(
                "[PrimusIdeogramZeRO1] PRIMUS_IDEOGRAM_ZERO1 set but params are DTensor (FSDP2 mode); "
                "ZeRO-1 only applies to replicated/DDP params -> keeping the plain (FSDP-sharded) optimizer."
            )
            return base

        import torch.distributed as dist
        from torch.distributed.optim import ZeroRedundancyOptimizer

        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            logger.warning("[PrimusIdeogramZeRO1] no >1-rank process group; keeping the plain optimizer.")
            return base

        optimizer_cls = type(base)
        # base.defaults holds the per-group defaults, but it can contain keys that are
        # NOT accepted by the optimizer's __init__ (e.g. AdamW stores decoupled_weight_decay
        # -- set internally by its Adam parent -- yet AdamW.__init__ doesn't take it). ZeRO
        # forwards these to reconstruct per-rank optimizers, so filter to the ctor signature.
        defaults = dict(base.defaults)
        defaults.pop("lr", None)  # passed explicitly below
        try:
            _sig = inspect.signature(optimizer_cls.__init__)
            _has_varkw = any(p.kind == p.VAR_KEYWORD for p in _sig.parameters.values())
            if not _has_varkw:
                _accepted = set(_sig.parameters)
                _dropped = [k for k in defaults if k not in _accepted]
                for k in _dropped:
                    defaults.pop(k, None)
                if _dropped:
                    logger.info(
                        "[PrimusIdeogramZeRO1] dropped non-ctor optimizer defaults %s for %s",
                        _dropped,
                        optimizer_cls.__name__,
                    )
        except (ValueError, TypeError):
            pass  # signature unavailable (C-impl); pass defaults through unchanged
        try:
            zro = ZeroRedundancyOptimizer(
                params,
                optimizer_class=optimizer_cls,
                lr=learning_rate,
                overlap_with_ddp=False,
                **defaults,
            )
        except Exception as e:  # fall back rather than break the run
            logger.error("[PrimusIdeogramZeRO1] ZeroRedundancyOptimizer build failed (%s); using plain optimizer.", e)
            return base
        logger.info(
            "[PrimusIdeogramZeRO1] wrapped %s in ZeroRedundancyOptimizer over %d ranks "
            "(optimizer state sharded; params/grads replicated by DDP).",
            optimizer_cls.__name__,
            dist.get_world_size(),
        )
        return zro

    train_mod._build_optimizer = _build_optimizer_zero1
    train_mod._primus_zero1_patched = True
    return True


def _install_ddp_ac_patch() -> bool:
    """Whole-block-checkpoint the Ideogram layers on the DDP path (generic sub-module AC no-ops there)."""
    import nemo_automodel.components.distributed.ddp as ddp_mod
    import nemo_automodel.components.distributed.parallelizer as P

    if getattr(ddp_mod, "_primus_ideogram_ddp_ac_patched", False):
        return True

    _orig_parallelize = ddp_mod.DDPManager.parallelize

    def _parallelize_with_ideogram_ac(self, model):
        if getattr(self, "activation_checkpointing", False) and _looks_like_ideogram(model):
            blocks = getattr(model, _IDEOGRAM_BLOCK_ATTR, None)
            wrapped = 0
            if blocks is not None:
                for idx in range(len(blocks)):
                    blocks[idx] = P.checkpoint_wrapper(
                        blocks[idx],
                        checkpoint_impl=P.CheckpointImpl.NO_REENTRANT,
                    )
                    wrapped += 1
            logger.info("[PrimusIdeogramZeRO1] DDP: whole-block checkpoint-wrapped %d Ideogram layers", wrapped)
            # Disable the generic (no-op-for-Ideogram) sub-module AC so it is not attempted again.
            saved = self.activation_checkpointing
            self.activation_checkpointing = False
            try:
                return _orig_parallelize(self, model)
            finally:
                self.activation_checkpointing = saved
        return _orig_parallelize(self, model)

    ddp_mod.DDPManager.parallelize = _parallelize_with_ideogram_ac
    ddp_mod._primus_ideogram_ddp_ac_patched = True
    return True


def install() -> bool:
    """Install DDP / ZeRO-1 patches. No-op (returns False) unless PRIMUS_IDEOGRAM_DDP
    or PRIMUS_IDEOGRAM_ZERO1 is set.

    - PRIMUS_IDEOGRAM_DDP=1   -> whole-block AC on the DDP path (pure-DDP baseline).
    - PRIMUS_IDEOGRAM_ZERO1=1 -> the above + wrap the optimizer in ZeroRedundancyOptimizer.

    The DDP-AC patch monkeypatches DDPManager.parallelize only; it is inert on the
    FSDP2 path (DDPManager is never used there). Idempotent; modifies NO Automodel
    source (module-level monkeypatches only).
    """
    if not (is_zero1_enabled() or is_ddp_enabled()):
        return False
    ok_ac = _install_ddp_ac_patch()
    ok_opt = _install_zero1_optimizer_patch() if is_zero1_enabled() else True
    return bool(ok_ac and ok_opt)
