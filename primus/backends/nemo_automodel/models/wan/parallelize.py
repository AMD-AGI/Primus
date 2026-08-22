###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Repair two silently-dropped FSDP2 knobs on the Wan diffusion path.

WHY:
  Unlike FLUX and Ideogram-4, ``WanTransformer3DModel`` *is* registered in
  Automodel's ``PARALLELIZATION_STRATEGIES``, so activation checkpointing is not
  a no-op here. But the in-tree ``WanParallelizationStrategy`` drops two config
  values on the floor, and both failures are silent:

  1. ``activation_checkpointing`` is consumed as a bare truthiness test::

         if activation_checkpointing and hasattr(model, "blocks"):
             ... checkpoint_wrapper(..., NO_REENTRANT)

     ``"selective"`` is a non-empty string, so it is truthy, so selective AC
     runs the *full* AC branch. ``fsdp.activation_checkpointing: selective`` and
     ``: full`` therefore produce byte-identical execution. (The same test also
     enables AC for the string ``"false"``.)

  2. ``reshard_after_forward`` never reaches PyTorch. The strategy calls
     ``apply_fsdp2_sharding_recursively`` with seven positional arguments and
     stops one short of the ``reshard_after_forward`` parameter, so it always
     gets the default ``None`` and the per-layer heuristic takes over. This is
     the strategy-side half of the bug described in
     ``distributed/fsdp2_reshard.py``; that hook repairs ``FSDP2Manager`` but
     cannot help if the strategy then discards the value. Fixing either half
     alone changes nothing, which is why ``zero2``/``zero3``/``zero3rs4`` were
     indistinguishable on this path.

  Neither failure is visible in the logs: the config echo shows the requested
  value, because the value *was* parsed. It just was never applied.

WHAT (NO diffusers / Automodel fork):
  Subclasses the in-tree strategy and replaces the registry entry.
  ``register_parallel_strategy`` cannot be used -- it asserts the name is not
  already present -- so the entry is overwritten directly. Subclassing (rather
  than reimplementing) keeps the in-tree Wan TP plan authoritative: TP, mixed
  precision and root sharding are still the parent's, and stay correct across a
  submodule bump.

  Registering at hook time is NOT enough. ``_apply_parallelization`` in
  ``_diffusers/auto_diffusion_pipeline.py`` calls ``_init_parallelizer()``
  immediately before parallelizing, and that function unconditionally does::

      PARALLELIZATION_STRATEGIES["WanTransformer3DModel"] = WanParallelizationStrategy()

  which overwrites whatever anyone registered earlier -- silently, and every
  time. So ``_init_parallelizer`` is wrapped to re-apply our entry after it
  runs. This is why FLUX and Ideogram-4 never needed the same treatment: they
  are not in that hard-coded list, so their registrations survive. Hunyuan is,
  and would hit the identical problem.

  Two changes, both narrow:
    * AC is applied here, before delegating, so ``"selective"`` reaches the
      shared Automodel selective-AC machinery -- the same lever FLUX and
      Ideogram use. The parent is then called with ``activation_checkpointing``
      already resolved, so it never double-wraps.
    * ``reshard_after_forward`` is injected into the parent's
      ``apply_fsdp2_sharding_recursively`` call for the duration of that call.

  Env-gated by ``PRIMUS_WAN_PARALLELIZE_FIX=1`` (default off), mirroring
  ``PRIMUS_FLUX_REAL_AC`` / ``PRIMUS_IDEOGRAM_REAL_AC``. Off = stock in-tree
  behaviour. The gate is kept even though the repair is behaviour-neutral for
  ``activation_checkpointing: true`` + unset ``reshard_after_forward``, because
  it swaps a registry entry that other Wan users may rely on.

Activation (env, no config schema change):
    PRIMUS_WAN_PARALLELIZE_FIX=1   install the repaired Wan strategy
"""
from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "True", "yes", "on"}
_FALSE_LIKE = {"false", "0", "off", "no", "none", ""}

_WAN_MODEL_NAME = "WanTransformer3DModel"
_WAN_BLOCK_ATTR = "blocks"

# Marks our wrapper around _init_parallelizer so install() stays idempotent.
_INIT_PATCH_SENTINEL = "_primus_wan_init_parallelizer_patch"

# Built once, from the in-tree class, and reused on every re-apply. Rebuilding it
# per call would eventually subclass our own subclass, since _init_parallelizer
# keeps putting the in-tree instance back and we keep reading the entry.
_STRATEGY_CLS = None


def is_wan_parallelize_fix_enabled() -> bool:
    return os.getenv("PRIMUS_WAN_PARALLELIZE_FIX", "0") in _TRUTHY


def _normalize_ac(value):
    """Map false-like strings to False, leave 'full'/'selective'/bools alone.

    Some CLI/config paths forward the flag as a raw string, and a non-empty
    ``"false"`` would otherwise be truthy and wrongly enable AC.
    """
    if isinstance(value, str) and value.strip().lower() in _FALSE_LIKE:
        return False
    return value


def _build_strategy_cls(P):
    """Build (once) the repaired strategy class on top of the in-tree one."""
    global _STRATEGY_CLS
    if _STRATEGY_CLS is not None:
        return _STRATEGY_CLS

    base_cls = P.WanParallelizationStrategy

    class WanRepairedParallelizationStrategy(base_cls):
        """In-tree Wan strategy with selective AC and reshard_after_forward honored."""

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
            reshard_after_forward=None,
            **kwargs,
        ):
            # Read the sibling hook's state through the module, never a
            # from-import: a from-import binds False at import time and the
            # guard then fires on every healthy run.
            from primus.backends.nemo_automodel.distributed import fsdp2_reshard

            logger.info(
                "[PrimusWanParallelize] reshard_after_forward: received=%s applied_by_hook=%s",
                reshard_after_forward,
                fsdp2_reshard.applied_reshard_after_forward,
            )
            if not fsdp2_reshard.patch_installed:
                # The trainer swallows hook exceptions, so a failed install shows
                # up only as one log line and then a full run at ~3P traffic.
                logger.error(
                    "[PrimusWanParallelize] the FSDP2 reshard repair hook is NOT installed, so "
                    "fsdp.reshard_after_forward from YAML was dropped upstream and this run "
                    "will use ZeRO-3-style per-block resharding regardless of config."
                )

            ac_value = _normalize_ac(activation_checkpointing)
            blocks = getattr(model, _WAN_BLOCK_ATTR, None)

            if ac_value and blocks is None:
                logger.warning(
                    "[PrimusWanParallelize] activation_checkpointing requested but model has no "
                    "'%s' block list; nothing checkpointed.",
                    _WAN_BLOCK_ATTR,
                )
            elif ac_value and P.is_selective_activation_checkpointing(ac_value):
                # PARTIAL / selective (TorchTitan-style, op-level) AC: save attention +
                # half the matmuls + comm collectives, recompute only the cheap ops. This
                # is the branch the in-tree strategy has no code for. The wrapper is
                # tagged with SELECTIVE_AC_WRAPPER_FLAG so per-layer compile compiles it
                # OUTER (SAC INNER) and the partitioner honors the recompute tags.
                # has_kv_sharing=False: Wan is a diffusion DiT with no KV cache.
                layers = list(blocks)
                P.apply_selective_checkpointing_to_layers(
                    model,
                    layers,
                    False,
                    enable_compile=bool(kwargs.get("enable_compile", False)),
                )
                logger.info(
                    "[PrimusWanParallelize] wrapped %d Wan blocks with SELECTIVE (partial) "
                    "activation checkpointing",
                    len(layers),
                )
            elif ac_value:
                # FULL AC: identical to what the in-tree strategy does. Done here rather
                # than delegated so that exactly one place decides, and so the log line
                # below is emitted on both branches.
                wrapped = 0
                for idx in range(len(blocks)):
                    blocks[idx] = P.checkpoint_wrapper(
                        blocks[idx],
                        checkpoint_impl=P.CheckpointImpl.NO_REENTRANT,
                    )
                    wrapped += 1
                logger.info(
                    "[PrimusWanParallelize] wrapped %d Wan blocks with FULL activation checkpointing",
                    wrapped,
                )
            else:
                logger.info("[PrimusWanParallelize] activation checkpointing OFF")

            # The parent hardcodes seven positional arguments to
            # apply_fsdp2_sharding_recursively and so can never pass an eighth. Bind the
            # value for the duration of the delegated call instead of duplicating the
            # parent's TP plan here. Scoped and restored: parallelize() runs once per
            # process on a single thread.
            original = P.apply_fsdp2_sharding_recursively

            @functools.wraps(original)
            def _with_reshard(*args, **kw):
                # Only supply it when the caller did not; a future in-tree fix that
                # passes it explicitly must win, so this degrades to a no-op.
                if len(args) < 8:
                    kw.setdefault("reshard_after_forward", reshard_after_forward)
                return original(*args, **kw)

            logger.info(
                "[PrimusWanParallelize] forwarding reshard_after_forward=%s into "
                "apply_fsdp2_sharding_recursively",
                reshard_after_forward,
            )
            P.apply_fsdp2_sharding_recursively = _with_reshard
            try:
                return super().parallelize(
                    model,
                    device_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    sequence_parallel=sequence_parallel,
                    # Already applied above; must be falsy or the parent re-wraps
                    # every block a second time.
                    activation_checkpointing=False,
                    tp_shard_plan=tp_shard_plan,
                    dp_replicate_mesh_name=dp_replicate_mesh_name,
                    dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
                    tp_mesh_name=tp_mesh_name,
                    **kwargs,
                )
            finally:
                P.apply_fsdp2_sharding_recursively = original

    _STRATEGY_CLS = WanRepairedParallelizationStrategy
    return _STRATEGY_CLS


def _apply_to_registry(P) -> bool:
    """Put our strategy in the registry. True if it had to be (re-)installed."""
    cls = _build_strategy_cls(P)
    if isinstance(P.PARALLELIZATION_STRATEGIES.get(_WAN_MODEL_NAME), cls):
        return False
    P.PARALLELIZATION_STRATEGIES[_WAN_MODEL_NAME] = cls()
    return True


def install() -> bool:
    """Install the repaired Wan parallelization strategy.

    No-op (returns False) unless ``PRIMUS_WAN_PARALLELIZE_FIX`` is set.
    Idempotent. Modifies NO Automodel source.
    """
    if not is_wan_parallelize_fix_enabled():
        return False

    import nemo_automodel.components.distributed.parallelizer as P

    _apply_to_registry(P)

    # _init_parallelizer() resets this entry to the in-tree instance every time
    # the pipeline parallelizes, so the registration above would be undone before
    # it is ever read. Re-apply immediately after it runs.
    from nemo_automodel._diffusers import auto_diffusion_pipeline as adp

    if not getattr(adp._init_parallelizer, _INIT_PATCH_SENTINEL, False):
        original_init = adp._init_parallelizer

        @functools.wraps(original_init)
        def _init_parallelizer_then_repair(*args, **kwargs):
            result = original_init(*args, **kwargs)
            if _apply_to_registry(P):
                logger.info(
                    "[PrimusWanParallelize] re-applied the repaired %s strategy after "
                    "_init_parallelizer reset it to the in-tree one.",
                    _WAN_MODEL_NAME,
                )
            return result

        setattr(_init_parallelizer_then_repair, _INIT_PATCH_SENTINEL, True)
        adp._init_parallelizer = _init_parallelizer_then_repair

    logger.info(
        "[PrimusWanParallelize] Replaced the in-tree %s strategy: selective activation "
        "checkpointing and reshard_after_forward are now honored.",
        _WAN_MODEL_NAME,
    )
    return True
